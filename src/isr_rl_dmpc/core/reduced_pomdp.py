"""
Reduced local POMDP for Phase 1 patrol-risk vs persistent-threat reasoning.

This model is intentionally tiny. It does not replace the full simulator or the
multi-drone patrol environment. Instead, it captures one local decision in an
interpretable form:

    "Is this high-value region merely neglected, or is it hiding a persistent
    threat that will not disappear under observation?"

The model is small enough for exact belief updates and a QMDP baseline, which
is sufficient for diagnostics and for preparing the project structure for
belief-based RL later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ReducedPatrolState(IntEnum):
    CLEAR = 0
    NEGLECTED = 1
    THREAT_TRACKED = 2
    THREAT_NEGLECTED = 3


class ReducedPatrolAction(IntEnum):
    ROUTINE_PATROL = 0
    FOCUSED_REVISIT = 1
    ESCALATE_AND_TRACK = 2


class ReducedPatrolObservation(IntEnum):
    QUIET = 0
    ELEVATED = 1
    PERSISTENT = 2


@dataclass(frozen=True)
class ReducedLocalPOMDPConfig:
    gamma: float = 0.95
    threat_birth_prob: float = 0.01
    threat_persistence_prob: float = 0.98


class ReducedLocalPatrolPOMDP:
    """
    Small local POMDP used for approximate planning diagnostics.

    The hidden state factors into:

    * whether the region is neglected or freshly monitored
    * whether a persistent threat is absent or present
    """

    def __init__(self, config: ReducedLocalPOMDPConfig | None = None) -> None:
        self.config = config or ReducedLocalPOMDPConfig()
        self.states = tuple(ReducedPatrolState)
        self.actions = tuple(ReducedPatrolAction)
        self.observations = tuple(ReducedPatrolObservation)

        self.transition_matrix = self._build_transition_matrix()
        self.observation_matrix = self._build_observation_matrix()
        self.reward_matrix = self._build_reward_matrix()

    @staticmethod
    def state_labels() -> list[str]:
        return ["clear", "neglected", "threat_tracked", "threat_neglected"]

    @staticmethod
    def action_labels() -> list[str]:
        return ["routine_patrol", "focused_revisit", "escalate_track"]

    @staticmethod
    def observation_labels() -> list[str]:
        return ["quiet", "elevated", "persistent"]

    @staticmethod
    def state_factors(state: ReducedPatrolState) -> tuple[int, int]:
        threat_present = int(state in (ReducedPatrolState.THREAT_TRACKED, ReducedPatrolState.THREAT_NEGLECTED))
        neglected = int(state in (ReducedPatrolState.NEGLECTED, ReducedPatrolState.THREAT_NEGLECTED))
        return threat_present, neglected

    @staticmethod
    def compose_state(threat_present: int, neglected: int) -> ReducedPatrolState:
        if threat_present:
            return ReducedPatrolState.THREAT_NEGLECTED if neglected else ReducedPatrolState.THREAT_TRACKED
        return ReducedPatrolState.NEGLECTED if neglected else ReducedPatrolState.CLEAR

    def factorized_belief(self, threat_probability: float, neglect_probability: float) -> np.ndarray:
        """Construct a 4-state belief from two interpretable marginals."""
        p_threat = float(np.clip(threat_probability, 0.0, 1.0))
        p_neglect = float(np.clip(neglect_probability, 0.0, 1.0))
        belief = np.array(
            [
                (1.0 - p_threat) * (1.0 - p_neglect),
                (1.0 - p_threat) * p_neglect,
                p_threat * (1.0 - p_neglect),
                p_threat * p_neglect,
            ],
            dtype=np.float64,
        )
        return belief / max(float(np.sum(belief)), 1e-12)

    def predicted_belief(self, belief: np.ndarray, action: ReducedPatrolAction) -> np.ndarray:
        belief = np.asarray(belief, dtype=np.float64)
        predicted = belief @ self.transition_matrix[int(action)]
        return predicted / max(float(np.sum(predicted)), 1e-12)

    def observation_likelihood(self, belief: np.ndarray, action: ReducedPatrolAction) -> np.ndarray:
        predicted = self.predicted_belief(belief, action)
        return predicted @ self.observation_matrix[int(action)]

    def belief_update(
        self,
        belief: np.ndarray,
        action: ReducedPatrolAction,
        observation: ReducedPatrolObservation,
    ) -> np.ndarray:
        predicted = self.predicted_belief(belief, action)
        likelihood = self.observation_matrix[int(action), :, int(observation)]
        posterior = predicted * likelihood
        norm = float(np.sum(posterior))
        if norm <= 1e-12:
            return np.full(len(self.states), 1.0 / len(self.states), dtype=np.float64)
        return posterior / norm

    def _build_transition_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.actions), len(self.states), len(self.states)), dtype=np.float64)

        for action in self.actions:
            for state in self.states:
                threat_present, neglected = self.state_factors(state)

                if action == ReducedPatrolAction.ROUTINE_PATROL:
                    neglect_clear_prob = 0.45
                    neglect_regrow_prob = 0.18
                elif action == ReducedPatrolAction.FOCUSED_REVISIT:
                    neglect_clear_prob = 0.82
                    neglect_regrow_prob = 0.06
                else:
                    neglect_clear_prob = 0.90 if threat_present else 0.45
                    neglect_regrow_prob = 0.04

                next_threat_present_probs = {
                    0: 1.0 - self.config.threat_birth_prob,
                    1: self.config.threat_birth_prob,
                }
                if threat_present:
                    next_threat_present_probs = {
                        0: 1.0 - self.config.threat_persistence_prob,
                        1: self.config.threat_persistence_prob,
                    }

                if neglected:
                    next_neglected_probs = {0: neglect_clear_prob, 1: 1.0 - neglect_clear_prob}
                else:
                    next_neglected_probs = {0: 1.0 - neglect_regrow_prob, 1: neglect_regrow_prob}

                for next_threat_present, threat_prob in next_threat_present_probs.items():
                    for next_neglected, neglect_prob in next_neglected_probs.items():
                        next_state = self.compose_state(next_threat_present, next_neglected)
                        matrix[int(action), int(state), int(next_state)] += threat_prob * neglect_prob

        matrix /= np.sum(matrix, axis=-1, keepdims=True)
        return matrix

    def _build_observation_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.actions), len(self.states), len(self.observations)), dtype=np.float64)

        quiet, elevated, persistent = [int(obs) for obs in self.observations]

        for action in self.actions:
            matrix[int(action), int(ReducedPatrolState.CLEAR)] = [0.88, 0.10, 0.02]
            if action == ReducedPatrolAction.ROUTINE_PATROL:
                matrix[int(action), int(ReducedPatrolState.NEGLECTED)] = [0.12, 0.78, 0.10]
                matrix[int(action), int(ReducedPatrolState.THREAT_TRACKED)] = [0.05, 0.28, 0.67]
                matrix[int(action), int(ReducedPatrolState.THREAT_NEGLECTED)] = [0.03, 0.27, 0.70]
            elif action == ReducedPatrolAction.FOCUSED_REVISIT:
                matrix[int(action), int(ReducedPatrolState.NEGLECTED)] = [0.48, 0.43, 0.09]
                matrix[int(action), int(ReducedPatrolState.THREAT_TRACKED)] = [0.03, 0.15, 0.82]
                matrix[int(action), int(ReducedPatrolState.THREAT_NEGLECTED)] = [0.02, 0.12, 0.86]
            else:
                matrix[int(action), int(ReducedPatrolState.NEGLECTED)] = [0.35, 0.45, 0.20]
                matrix[int(action), int(ReducedPatrolState.THREAT_TRACKED)] = [0.02, 0.08, 0.90]
                matrix[int(action), int(ReducedPatrolState.THREAT_NEGLECTED)] = [0.01, 0.06, 0.93]

        matrix /= np.sum(matrix, axis=-1, keepdims=True)
        return matrix

    def _build_reward_matrix(self) -> np.ndarray:
        rewards = np.array(
            [
                [1.0, 0.3, -1.0],
                [-1.1, 0.8, -0.6],
                [-1.8, 0.9, 1.4],
                [-2.4, 1.1, 1.6],
            ],
            dtype=np.float64,
        )
        return rewards


class QMDPPlanner:
    """Approximate planner that solves the underlying MDP and acts on beliefs."""

    def __init__(
        self,
        pomdp: ReducedLocalPatrolPOMDP,
        *,
        gamma: float | None = None,
        tolerance: float = 1e-8,
        max_iterations: int = 500,
    ) -> None:
        self.pomdp = pomdp
        self.gamma = float(gamma if gamma is not None else pomdp.config.gamma)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.value_function = np.zeros(len(self.pomdp.states), dtype=np.float64)
        self.q_values = np.zeros((len(self.pomdp.states), len(self.pomdp.actions)), dtype=np.float64)
        self._solve_mdp()

    def _solve_mdp(self) -> None:
        values = np.zeros(len(self.pomdp.states), dtype=np.float64)

        for _ in range(self.max_iterations):
            q_values = np.zeros_like(self.q_values)
            for action in self.pomdp.actions:
                q_values[:, int(action)] = (
                    self.pomdp.reward_matrix[:, int(action)]
                    + self.gamma * (self.pomdp.transition_matrix[int(action)] @ values)
                )
            next_values = np.max(q_values, axis=1)
            if np.max(np.abs(next_values - values)) < self.tolerance:
                values = next_values
                self.q_values = q_values
                break
            values = next_values
            self.q_values = q_values

        self.value_function = values

    def action_values(self, belief: np.ndarray) -> np.ndarray:
        belief = np.asarray(belief, dtype=np.float64)
        return belief @ self.q_values

    def select_action(self, belief: np.ndarray) -> ReducedPatrolAction:
        action_idx = int(np.argmax(self.action_values(belief)))
        return ReducedPatrolAction(action_idx)

    def belief_grid(self, resolution: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        threat_axis = np.linspace(0.0, 1.0, int(resolution))
        neglect_axis = np.linspace(0.0, 1.0, int(resolution))
        action_grid = np.zeros((len(neglect_axis), len(threat_axis)), dtype=np.int32)
        value_grid = np.zeros((len(neglect_axis), len(threat_axis)), dtype=np.float64)

        for i, neglect_probability in enumerate(neglect_axis):
            for j, threat_probability in enumerate(threat_axis):
                belief = self.pomdp.factorized_belief(threat_probability, neglect_probability)
                action_values = self.action_values(belief)
                action_grid[i, j] = int(np.argmax(action_values))
                value_grid[i, j] = float(np.max(action_values))

        return threat_axis, neglect_axis, action_grid, value_grid
