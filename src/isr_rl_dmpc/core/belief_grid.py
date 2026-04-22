"""
Belief-grid data structures for Phase 1 belief-based coverage.

The grid tracks two signals per cell:

* ``uncertainty``: lack of recent reliable observation in ``[0, 1]``
* ``anomaly_score``: latent threat-belief proxy in ``[0, 1]``

The design intentionally separates a global fused map from per-drone local
copies so coverage logic can evolve from simple central fusion to more
distributed communication without changing the core data model. The historical
``anomaly_score`` name is kept for compatibility, but in the current Phase 1
POMDP interpretation it plays the role of a threat-belief channel rather than a
confirmed hostile-state estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BeliefGridConfig:
    """Configuration for belief-grid dynamics and thresholds."""

    uncertainty_min: float = 0.05
    uncertainty_max: float = 1.0
    growth_rate: float = 0.03
    anomaly_decay: float = 0.98
    global_sync_steps: int = 25
    report_threshold: float = 0.1
    revisit_threshold: float = 0.4
    alert_threshold: float = 0.7


@dataclass(frozen=True)
class BeliefCellState:
    """Snapshot of one cell's current belief state."""

    uncertainty: float
    anomaly_score: float
    last_observed_step: int
    last_fused_step: int


class BeliefGrid:
    """
    Dense belief-grid storage with vectorized update rules.

    The grid stores belief arrays directly for efficient step-wise updates,
    while :class:`BeliefCellState` provides a convenient typed snapshot when
    callers need an individual cell view.
    """

    def __init__(
        self,
        centers: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        priorities: np.ndarray,
        config: Optional[BeliefGridConfig] = None,
        name: str = "global",
    ) -> None:
        self.config = config or BeliefGridConfig()
        self.name = name

        self.centers = np.asarray(centers, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)
        self.priorities = np.asarray(priorities, dtype=np.float64)

        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("centers must have shape (n_cells, 2)")
        if len(self.rows) != len(self.centers):
            raise ValueError("rows length must match centers")
        if len(self.cols) != len(self.centers):
            raise ValueError("cols length must match centers")
        if len(self.priorities) != len(self.centers):
            raise ValueError("priorities length must match centers")

        self.n_cells = int(self.centers.shape[0])
        self.uncertainty = np.empty(self.n_cells, dtype=np.float64)
        self.anomaly_score = np.empty(self.n_cells, dtype=np.float64)
        self.last_observed_step = np.empty(self.n_cells, dtype=np.int32)
        self.last_fused_step = np.empty(self.n_cells, dtype=np.int32)
        self.reset()

    @classmethod
    def from_cells(
        cls,
        cells: Sequence[object],
        *,
        grid_resolution: float,
        config: Optional[BeliefGridConfig] = None,
        name: str = "global",
    ) -> "BeliefGrid":
        """Build a belief grid from Module 1 grid cells."""
        centers = np.array([np.asarray(cell.center, dtype=np.float64) for cell in cells])
        priorities = np.array([float(cell.priority) for cell in cells], dtype=np.float64)
        rows = np.floor_divide(centers[:, 1], float(grid_resolution)).astype(np.int32)
        cols = np.floor_divide(centers[:, 0], float(grid_resolution)).astype(np.int32)
        return cls(centers, rows, cols, priorities, config=config, name=name)

    def reset(self) -> None:
        """Reset the grid to the maximally uncertain, anomaly-free state."""
        self.uncertainty.fill(self.config.uncertainty_max)
        self.anomaly_score.fill(0.0)
        self.last_observed_step.fill(-1)
        self.last_fused_step.fill(-1)

    def copy(self, *, name: Optional[str] = None) -> "BeliefGrid":
        """Return a deep copy of this grid."""
        copied = BeliefGrid(
            self.centers.copy(),
            self.rows.copy(),
            self.cols.copy(),
            self.priorities.copy(),
            config=self.config,
            name=name or self.name,
        )
        copied.uncertainty = self.uncertainty.copy()
        copied.anomaly_score = self.anomaly_score.copy()
        copied.last_observed_step = self.last_observed_step.copy()
        copied.last_fused_step = self.last_fused_step.copy()
        return copied

    def cell_state(self, cell_id: int) -> BeliefCellState:
        """Return a typed snapshot for one cell."""
        idx = int(cell_id)
        return BeliefCellState(
            uncertainty=float(self.uncertainty[idx]),
            anomaly_score=float(self.anomaly_score[idx]),
            last_observed_step=int(self.last_observed_step[idx]),
            last_fused_step=int(self.last_fused_step[idx]),
        )

    def grow_uncertainty(self) -> None:
        """
        Apply the agreed exponential uncertainty growth.

        Unobserved cells become progressively urgent while remaining capped.
        """
        growth = float(np.exp(self.config.growth_rate))
        self.uncertainty = np.minimum(
            self.config.uncertainty_max,
            np.maximum(self.config.uncertainty_min, self.uncertainty) * growth,
        )

    def decay_anomaly(self) -> None:
        """Apply a light anomaly decay so stale alerts fade without vanishing instantly."""
        self.anomaly_score *= self.config.anomaly_decay
        np.clip(self.anomaly_score, 0.0, 1.0, out=self.anomaly_score)

    def advance(self) -> None:
        """Advance belief dynamics by one step before new observations arrive."""
        self.grow_uncertainty()
        self.decay_anomaly()

    def observe(
        self,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
        *,
        anomaly_scores: Optional[np.ndarray] = None,
        step: int,
    ) -> None:
        """
        Apply visual observations to a set of cells.

        Observation reset rule:
            uncertainty = max(u_min, 1 - quality)
        """
        idx = np.asarray(cell_indices, dtype=np.int32)
        if idx.size == 0:
            return

        q = np.clip(np.asarray(qualities, dtype=np.float64), 0.0, 1.0)
        if q.shape != idx.shape:
            raise ValueError("qualities must match cell_indices shape")

        self.uncertainty[idx] = np.maximum(self.config.uncertainty_min, 1.0 - q)
        self.last_observed_step[idx] = int(step)

        if anomaly_scores is not None:
            scores = np.clip(np.asarray(anomaly_scores, dtype=np.float64), 0.0, 1.0)
            if scores.shape != idx.shape:
                raise ValueError("anomaly_scores must match cell_indices shape")
            self.anomaly_score[idx] = np.maximum(self.anomaly_score[idx], scores)

    def fuse_from(
        self,
        other: "BeliefGrid",
        *,
        indices: Optional[np.ndarray] = None,
        step: int,
    ) -> np.ndarray:
        """Fuse another belief grid into this one using min/max rules."""
        idx = self._normalize_indices(indices)
        self.uncertainty[idx] = np.minimum(self.uncertainty[idx], other.uncertainty[idx])
        self.anomaly_score[idx] = np.maximum(self.anomaly_score[idx], other.anomaly_score[idx])
        self.last_observed_step[idx] = np.maximum(
            self.last_observed_step[idx],
            other.last_observed_step[idx],
        )
        self.last_fused_step[idx] = int(step)
        return idx

    def get_age(self, current_step: int) -> np.ndarray:
        """Return per-cell observation age in steps."""
        observed = self.last_observed_step >= 0
        ages = np.full(self.n_cells, current_step + 1, dtype=np.float64)
        ages[observed] = current_step - self.last_observed_step[observed]
        return np.maximum(ages, 0.0)

    def anomaly_counts(self) -> dict[str, int]:
        """Return counts above the configured anomaly thresholds."""
        return {
            "gt_0_1": int(np.sum(self.anomaly_score > self.config.report_threshold)),
            "gt_0_4": int(np.sum(self.anomaly_score > self.config.revisit_threshold)),
            "gt_0_7": int(np.sum(self.anomaly_score > self.config.alert_threshold)),
        }

    def total_uncertainty(self) -> float:
        return float(np.sum(self.uncertainty))

    def mean_uncertainty(self) -> float:
        return float(np.mean(self.uncertainty))

    def max_uncertainty(self) -> float:
        return float(np.max(self.uncertainty))

    def _normalize_indices(self, indices: Optional[np.ndarray]) -> np.ndarray:
        if indices is None:
            return np.arange(self.n_cells, dtype=np.int32)
        idx = np.unique(np.asarray(indices, dtype=np.int32))
        return idx[(idx >= 0) & (idx < self.n_cells)]


class LocalBeliefGrid(BeliefGrid):
    """Per-drone belief copy that tracks which cells changed recently."""

    def __init__(self, *args, drone_id: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drone_id = int(drone_id)
        self._recently_changed = np.zeros(self.n_cells, dtype=bool)

    @classmethod
    def from_belief_grid(
        cls,
        grid: BeliefGrid,
        *,
        drone_id: int,
        name: Optional[str] = None,
    ) -> "LocalBeliefGrid":
        """Clone a global grid into a per-drone local copy."""
        local = cls(
            grid.centers.copy(),
            grid.rows.copy(),
            grid.cols.copy(),
            grid.priorities.copy(),
            config=grid.config,
            name=name or f"local_{drone_id}",
            drone_id=drone_id,
        )
        local.uncertainty = grid.uncertainty.copy()
        local.anomaly_score = grid.anomaly_score.copy()
        local.last_observed_step = grid.last_observed_step.copy()
        local.last_fused_step = grid.last_fused_step.copy()
        return local

    def observe(
        self,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
        *,
        anomaly_scores: Optional[np.ndarray] = None,
        step: int,
    ) -> None:
        super().observe(cell_indices, qualities, anomaly_scores=anomaly_scores, step=step)
        idx = self._normalize_indices(cell_indices)
        self._recently_changed[idx] = True

    def fuse_from(
        self,
        other: BeliefGrid,
        *,
        indices: Optional[np.ndarray] = None,
        step: int,
    ) -> np.ndarray:
        idx = super().fuse_from(other, indices=indices, step=step)
        self._recently_changed[idx] = True
        return idx

    def changed_indices(self) -> np.ndarray:
        """Return the cells updated since the last clear."""
        return np.flatnonzero(self._recently_changed)

    def clear_recent_changes(self) -> None:
        """Reset the changed-cell tracker after communication finishes."""
        self._recently_changed.fill(False)
