"""
ADMM Consensus Layer for multi-drone DMPC coordination.

Implements the Alternating Direction Method of Multipliers (ADMM) to enforce
consensus on shared trajectory and collision-avoidance constraints across
per-drone DMPC sub-problems.  Runs for a small number of inner iterations
(typically 3–5) per environment step.

Mathematical formulation
------------------------
Each drone i solves a local QP and produces a local trajectory proposal z_i.
The consensus variable v represents the agreed-upon reference shared by all
drones.  The augmented Lagrangian is:

  L_ρ = Σ_i [f_i(z_i) + y_i^T (z_i - v) + (ρ/2) ‖z_i - v‖²]

ADMM iteration:
  z-update:  z_i ← argmin_{z_i} [f_i(z_i) + y_i^T z_i + (ρ/2)‖z_i - v‖²]
                  (simplified: proximal update towards current v)
  v-update:  v  ← mean_i(z_i + y_i / ρ)
  dual-update: y_i ← y_i + ρ (z_i - v)

References
----------
Boyd, S. et al. (2011).  Distributed optimization and statistical learning via
the alternating direction method of multipliers.  Foundations and Trends in
Machine Learning, 3(1), 1–122.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ADMMConfig:
    """Configuration for the ADMM consensus layer."""

    rho: float = 1.0
    """Augmented Lagrangian penalty parameter ρ > 0."""

    max_iters: int = 5
    """Number of ADMM inner iterations per environment step."""

    primal_tol: float = 1e-3
    """Primal residual tolerance for early stopping."""

    dual_tol: float = 1e-3
    """Dual residual tolerance for early stopping."""


class ADMMConsensus:
    """
    ADMM-based consensus coordinator for a swarm of DMPC drones.

    Each drone submits its local trajectory proposal (z_i), and the
    coordinator runs ADMM iterations to compute a consensus variable (v)
    that is consistent across the swarm.

    Args:
        num_drones:  Number of agents in the swarm.
        dim:         Dimension of each agent's consensus variable
                     (e.g. 3 for a position reference).
        config:      ADMM hyper-parameters.

    Attributes:
        z (np.ndarray):  Local proposals, shape ``(num_drones, dim)``.
        v (np.ndarray):  Consensus variable, shape ``(dim,)``.
        y (np.ndarray):  Dual variables, shape ``(num_drones, dim)``.
        primal_residuals (np.ndarray): Per-drone ‖z_i - v‖ after last step.
    """

    def __init__(
        self,
        num_drones: int,
        dim: int = 3,
        config: Optional[ADMMConfig] = None,
    ) -> None:
        self.num_drones = num_drones
        self.dim = dim
        self.config = config or ADMMConfig()

        self.z = np.zeros((num_drones, dim), dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.y = np.zeros((num_drones, dim), dtype=np.float64)

        self.primal_residuals = np.zeros((num_drones, dim), dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all consensus and dual variables to zero."""
        self.z[:] = 0.0
        self.v[:] = 0.0
        self.y[:] = 0.0
        self.primal_residuals[:] = 0.0

    def step(self, local_proposals: np.ndarray) -> np.ndarray:
        """
        Run ADMM iterations given per-drone local proposals.

        Args:
            local_proposals: Array of shape ``(num_drones, dim)`` containing
                             each drone's unconstrained trajectory proposal.

        Returns:
            v: Consensus variable of shape ``(dim,)``.
        """
        rho = self.config.rho
        z = np.array(local_proposals, dtype=np.float64)

        # z-update: proximal step towards current v (gradient of quadratic)
        # z_i ← (z_i_proposal + ρ * v - y_i) / (1 + ρ)
        # (assumes f_i is a quadratic with unit Hessian after scaling)
        for _ in range(self.config.max_iters):
            v_old = self.v.copy()

            # z-update (parallel across drones)
            z = (local_proposals + rho * self.v[None, :] - self.y) / (1.0 + rho)

            # v-update: average of (z_i + y_i / ρ)
            self.v = np.mean(z + self.y / rho, axis=0)

            # dual-update
            self.y += rho * (z - self.v[None, :])

            # Residuals
            self.primal_residuals = z - self.v[None, :]
            dual_residual = float(np.linalg.norm(rho * (self.v - v_old)))
            primal_residual = float(np.linalg.norm(self.primal_residuals))

            if (
                primal_residual < self.config.primal_tol
                and dual_residual < self.config.dual_tol
            ):
                break

        self.z = z
        return self.v.copy()

    def get_adjusted_references(self) -> np.ndarray:
        """
        Return per-drone consensus-adjusted reference positions.

        Each drone's reference is shifted towards the global consensus
        variable by the magnitude of its primal residual, blending local
        proposals with the shared consensus.

        Returns:
            Array of shape ``(num_drones, dim)``.
        """
        return self.z.copy()

    def get_primal_residuals(self) -> np.ndarray:
        """Return per-drone primal residuals ‖z_i - v‖, shape ``(num_drones, dim)``."""
        return self.primal_residuals.copy()
