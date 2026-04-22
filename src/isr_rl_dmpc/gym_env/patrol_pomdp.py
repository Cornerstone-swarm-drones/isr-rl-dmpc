"""
POMDP-style support types for the Phase 1 patrol-risk environment.

These helpers do not implement a full Dec-POMDP solver for the map-scale
multi-drone system. Instead, they expose the current heuristic patrol system in
the standard POMDP language we want to grow into:

* hidden world state
* belief state
* transition model
* observation model
* belief update

The current deterministic patrol baseline remains the policy over that belief
state. Later RL can replace the policy without needing to redesign the other
pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from isr_rl_dmpc.core.belief_grid import BeliefGrid, BeliefGridConfig, LocalBeliefGrid
from isr_rl_dmpc.gym_env.sensor_model import ForwardFOVSensorModel


@dataclass(frozen=True)
class PatrolHiddenWorldState:
    """
    Hidden world state for Phase 1.

    ``patrol_risk_state`` rises under neglect and drops with observation.
    ``persistent_threat_state`` represents latent threat-like structure in the
    world. In the current Phase 1 system it is static heuristic world state, so
    observation can reveal it but does not eliminate it.
    """

    patrol_risk_state: np.ndarray
    persistent_threat_state: np.ndarray
    drone_positions: np.ndarray
    drone_yaws: np.ndarray

    @property
    def combined_risk_state(self) -> np.ndarray:
        return np.maximum(self.patrol_risk_state, self.persistent_threat_state)


@dataclass(frozen=True)
class PatrolBeliefState:
    """
    Belief state for the Phase 1 patrol system.

    ``global_patrol_risk_belief`` is the fused coverage-neglect belief.
    ``global_threat_belief`` is the fused latent-threat belief proxy.
    ``global_threat_confirmation_belief`` is the interpreter-friendly
    persistence score used to confirm persistent threats over repeated looks.
    The local belief lists preserve the decentralized per-drone copies.
    """

    global_patrol_risk_belief: np.ndarray
    global_threat_belief: np.ndarray
    local_patrol_risk_beliefs: tuple[np.ndarray, ...]
    local_threat_beliefs: tuple[np.ndarray, ...]
    global_threat_confirmation_belief: np.ndarray | None = None

    @property
    def combined_global_risk_belief(self) -> np.ndarray:
        return np.maximum(self.global_patrol_risk_belief, self.global_threat_belief)

    @classmethod
    def from_belief_grids(
        cls,
        global_belief: BeliefGrid,
        local_beliefs: Sequence[LocalBeliefGrid],
        *,
        global_threat_confirmation_belief: np.ndarray | None = None,
    ) -> "PatrolBeliefState":
        return cls(
            global_patrol_risk_belief=global_belief.uncertainty.copy(),
            global_threat_belief=global_belief.anomaly_score.copy(),
            local_patrol_risk_beliefs=tuple(local.uncertainty.copy() for local in local_beliefs),
            local_threat_beliefs=tuple(local.anomaly_score.copy() for local in local_beliefs),
            global_threat_confirmation_belief=(
                None
                if global_threat_confirmation_belief is None
                else np.asarray(global_threat_confirmation_belief, dtype=np.float64).copy()
            ),
        )


@dataclass(frozen=True)
class PatrolTransitionModel:
    """
    Transition model for the hidden patrol world state.

    Phase 1 currently has two world-state components:

    * patrol risk dynamics: explicit neglect growth, observation-driven reset
    * persistent threat dynamics: static latent structure for now
    """

    config: BeliefGridConfig

    def advance_patrol_risk_state(self, patrol_risk_state: np.ndarray) -> np.ndarray:
        growth = float(np.exp(self.config.growth_rate))
        patrol_risk_state = np.asarray(patrol_risk_state, dtype=np.float64)
        return np.minimum(
            self.config.uncertainty_max,
            np.maximum(self.config.uncertainty_min, patrol_risk_state) * growth,
        )

    def advance_persistent_threat_state(self, persistent_threat_state: np.ndarray) -> np.ndarray:
        """
        Advance the latent threat field.

        In Phase 1 the latent threat field is static heuristic world state.
        The explicit method keeps the interface ready for later dynamic threats.
        """
        return np.clip(np.asarray(persistent_threat_state, dtype=np.float64), 0.0, 1.0)

    def compose_hidden_risk_state(
        self,
        patrol_risk_state: np.ndarray,
        persistent_threat_state: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            np.asarray(patrol_risk_state, dtype=np.float64),
            np.asarray(persistent_threat_state, dtype=np.float64),
        )


@dataclass(frozen=True)
class PatrolObservationModel:
    """
    Observation model for the Phase 1 patrol system.

    The same visual observation can carry two different meanings:

    * it reduces patrol-risk belief by refreshing stale cells
    * it increases threat belief when the hidden world state contains
      persistent threat evidence
    """

    sensor_model: ForwardFOVSensorModel

    def observe_cells(
        self,
        position_xy: np.ndarray,
        yaw: float,
        cell_centers_xy: np.ndarray,
        *,
        rng: np.random.RandomState | np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.sensor_model.observe_cells(position_xy, yaw, cell_centers_xy, rng=rng)

    def project_threat_belief_evidence(
        self,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
        persistent_threat_state: np.ndarray,
    ) -> np.ndarray:
        if np.asarray(cell_indices).size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.clip(
            np.asarray(persistent_threat_state, dtype=np.float64)[np.asarray(cell_indices, dtype=np.int32)]
            * np.asarray(qualities, dtype=np.float64),
            0.0,
            1.0,
        )


class PatrolBeliefUpdater:
    """Belief update helper for local observation, sharing, fusion, and sync."""

    def apply_local_observation(
        self,
        local_belief: LocalBeliefGrid,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
        threat_belief_evidence: np.ndarray,
        *,
        step: int,
    ) -> None:
        local_belief.observe(
            cell_indices,
            qualities,
            anomaly_scores=threat_belief_evidence,
            step=step,
        )

    def share_neighbor_updates(
        self,
        local_beliefs: Sequence[LocalBeliefGrid],
        adjacency: np.ndarray,
        *,
        step: int,
    ) -> int:
        shared_indices: set[int] = set()
        changed = [local.changed_indices().copy() for local in local_beliefs]
        adjacency = np.asarray(adjacency, dtype=np.int32)

        for i in range(len(local_beliefs)):
            for j in range(i + 1, len(local_beliefs)):
                if adjacency[i, j] <= 0:
                    continue
                if changed[i].size > 0:
                    local_beliefs[j].fuse_from(local_beliefs[i], indices=changed[i], step=step)
                    shared_indices.update(int(idx) for idx in changed[i])
                if changed[j].size > 0:
                    local_beliefs[i].fuse_from(local_beliefs[j], indices=changed[j], step=step)
                    shared_indices.update(int(idx) for idx in changed[j])

        return int(len(shared_indices))

    def rebuild_global_belief(
        self,
        global_belief: BeliefGrid,
        local_beliefs: Sequence[LocalBeliefGrid],
        *,
        step: int,
    ) -> None:
        global_belief.uncertainty.fill(global_belief.config.uncertainty_max)
        global_belief.anomaly_score.fill(0.0)
        global_belief.last_observed_step.fill(-1)
        global_belief.last_fused_step.fill(-1)
        for local in local_beliefs:
            global_belief.fuse_from(local, step=step)

    def apply_global_sync(
        self,
        global_belief: BeliefGrid,
        local_beliefs: Sequence[LocalBeliefGrid],
        *,
        step: int,
    ) -> None:
        for local in local_beliefs:
            local.fuse_from(global_belief, step=step)
