"""
Module 9 - DMPC Analytics: Trajectory Statistics and Parameter Diagnostics

Replaces the former RL/neural-network learning module with a lightweight
analytics component that monitors DMPC controller performance using only
NumPy and SciPy.  No neural networks or external ML frameworks are used.

Key responsibilities
--------------------
- Accumulate per-step statistics (tracking error, solve times, costs).
- Compute running performance metrics (RMSE, mean cost, success rate).
- Provide parameter sensitivity estimates via finite-difference Jacobians.
- Persist and restore diagnostic data to/from NumPy archives.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class StepRecord:
    """Single DMPC step record."""

    state: np.ndarray
    control: np.ndarray
    reference: np.ndarray
    solve_status: str
    solve_time: float
    objective: float
    tracking_error: float


@dataclass
class DMPCAnalytics:
    """
    Lightweight analytics collector for the pure DMPC controller.

    All accumulated data is stored as plain NumPy arrays; no neural
    networks or gradient computation are involved.

    Attributes:
        state_dim:   Dimension of the drone state vector.
        control_dim: Dimension of the control input vector.
    """

    state_dim: int = 11
    control_dim: int = 3

    _records: List[StepRecord] = field(default_factory=list, init=False)
    _episode_rewards: List[float] = field(default_factory=list, init=False)

    # Running statistics
    tracking_error_history: List[float] = field(default_factory=list, init=False)
    solve_time_history: List[float] = field(default_factory=list, init=False)
    objective_history: List[float] = field(default_factory=list, init=False)
    solve_success_history: List[bool] = field(default_factory=list, init=False)

    def record_step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        reference: np.ndarray,
        solve_status: str,
        solve_time: float,
        objective: float,
    ) -> None:
        """Store one DMPC solve step for later analysis."""
        err = float(np.linalg.norm(state[:3] - reference[:3]))
        rec = StepRecord(
            state=np.asarray(state, dtype=np.float64),
            control=np.asarray(control, dtype=np.float64),
            reference=np.asarray(reference, dtype=np.float64),
            solve_status=solve_status,
            solve_time=float(solve_time),
            objective=float(objective) if np.isfinite(objective) else np.nan,
            tracking_error=err,
        )
        self._records.append(rec)
        self.tracking_error_history.append(err)
        self.solve_time_history.append(float(solve_time))
        self.objective_history.append(rec.objective)
        self.solve_success_history.append(
            solve_status in ("optimal", "optimal_inaccurate")
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict:
        """
        Return a snapshot of current performance metrics.

        Returns:
            Dictionary with keys:
            - ``rmse_tracking``: RMS position tracking error [m]
            - ``mean_objective``: Mean MPC cost value
            - ``mean_solve_time_ms``: Mean solver time [ms]
            - ``solve_success_rate``: Fraction of optimal solves ∈ [0, 1]
            - ``total_steps``: Number of recorded steps
        """
        n = len(self._records)
        if n == 0:
            return {
                "rmse_tracking": 0.0,
                "mean_objective": 0.0,
                "mean_solve_time_ms": 0.0,
                "solve_success_rate": 0.0,
                "total_steps": 0,
            }

        errors = np.array(self.tracking_error_history)
        times = np.array(self.solve_time_history)
        objectives = np.array([o for o in self.objective_history if np.isfinite(o)])
        successes = np.array(self.solve_success_history)

        return {
            "rmse_tracking": float(np.sqrt(np.mean(errors ** 2))),
            "mean_objective": float(np.mean(objectives)) if len(objectives) > 0 else np.nan,
            "mean_solve_time_ms": float(np.mean(times) * 1e3),
            "solve_success_rate": float(np.mean(successes)),
            "total_steps": n,
        }

    def reset(self) -> None:
        """Clear all accumulated records."""
        self._records.clear()
        self.tracking_error_history.clear()
        self.solve_time_history.clear()
        self.objective_history.clear()
        self.solve_success_history.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Save analytics data to a NumPy archive (.npz)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            tracking_errors=np.array(self.tracking_error_history),
            solve_times=np.array(self.solve_time_history),
            objectives=np.array(self.objective_history),
            solve_successes=np.array(self.solve_success_history, dtype=bool),
        )

    def load(self, path) -> None:
        """Restore analytics data from a NumPy archive (.npz)."""
        data = np.load(Path(path))
        self.tracking_error_history = list(data["tracking_errors"].tolist())
        self.solve_time_history = list(data["solve_times"].tolist())
        self.objective_history = list(data["objectives"].tolist())
        self.solve_success_history = list(data["solve_successes"].tolist())

    # ------------------------------------------------------------------
    # Parameter sensitivity (finite differences)
    # ------------------------------------------------------------------

    @staticmethod
    def finite_difference_jacobian(
        f,
        params: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        Estimate the Jacobian ∂f/∂θ via central finite differences.

        Args:
            f:      Callable mapping parameter vector θ → scalar cost.
            params: Current parameter vector (n,).
            eps:    Finite-difference step size.

        Returns:
            Jacobian vector of shape ``(n,)``.
        """
        n = len(params)
        jac = np.zeros(n)
        for i in range(n):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            jac[i] = (f(p_plus) - f(p_minus)) / (2.0 * eps)
        return jac
