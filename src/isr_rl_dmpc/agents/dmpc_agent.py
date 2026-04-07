"""
DMPCAgent - Pure DMPC swarm agent.

Wraps the :class:`~isr_rl_dmpc.modules.dmpc_controller.DMPC` controller and
:class:`~isr_rl_dmpc.modules.attitude_controller.AttitudeController` to provide
a unified agent interface used by mission scripts and evaluation harnesses.

No neural networks, replay buffers, or RL training loops are involved.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from isr_rl_dmpc.modules.dmpc_controller import DMPC, DMPCConfig
from isr_rl_dmpc.modules.attitude_controller import AttitudeController, DroneParameters
from isr_rl_dmpc.modules.learning_module import DMPCAnalytics


class DMPCAgent:
    """
    Single-drone DMPC agent.

    Combines the DMPC trajectory optimiser with the geometric attitude
    controller.  Exposes a simple ``act`` / ``record`` interface so that
    mission scripts and evaluation harnesses do not need to handle the
    two-layer controller directly.

    Args:
        state_dim:       Dimension of the drone state vector (default 11).
        control_dim:     Dimension of the control input (default 3).
        horizon:         MPC prediction horizon.
        dt:              Discretisation step [s].
        Q:               State tracking cost matrix.  Falls back to
                         scaled identity if ``None``.
        R:               Input cost matrix.  Falls back to scaled
                         identity if ``None``.
        accel_max:       Maximum acceleration magnitude [m/s²].
        collision_radius: Minimum inter-drone separation [m].
    """

    def __init__(
        self,
        state_dim: int = 11,
        control_dim: int = 3,
        horizon: int = 20,
        dt: float = 0.02,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        accel_max: float = 10.0,
        collision_radius: float = 5.0,
    ) -> None:
        config = DMPCConfig(
            horizon=horizon,
            dt=dt,
            state_dim=state_dim,
            control_dim=control_dim,
            Q_base=Q if Q is not None else np.eye(state_dim),
            R_base=R if R is not None else np.eye(control_dim) * 0.1,
            accel_max=accel_max,
            collision_radius=collision_radius,
        )
        self.dmpc = DMPC(config)
        self.attitude_ctrl = AttitudeController(DroneParameters())
        self.analytics = DMPCAnalytics(state_dim=state_dim, control_dim=control_dim)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_obs(obs) -> np.ndarray:
        """Flatten a dict observation into a 1-D NumPy array."""
        if isinstance(obs, dict):
            return np.concatenate([
                np.asarray(obs[k], dtype=np.float32).ravel()
                for k in sorted(obs.keys())
            ])
        return np.asarray(obs, dtype=np.float32).ravel()

    def act(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        neighbor_states: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute the next control action.

        Args:
            state:                Current drone state ``(state_dim,)``.
            reference_trajectory: Reference trajectory
                                  ``(horizon+1, state_dim)`` or
                                  ``(horizon, state_dim)``.
            neighbor_states:      List of neighbour state vectors for
                                  collision avoidance.

        Returns:
            Tuple ``(motor_commands, info)`` where *motor_commands* is a
            (4,) array of per-rotor thrust values and *info* contains
            DMPC solve diagnostics.
        """
        u_seq, info = self.dmpc(state, reference_trajectory, neighbor_states)

        # Use the first control step for the attitude loop
        u0 = u_seq[0] if u_seq.shape[0] > 0 else np.zeros(self.dmpc.config.control_dim)

        # Build a reference for the attitude controller (position + accel)
        ref_att = np.zeros(12)
        ref_att[:len(reference_trajectory[0])] = reference_trajectory[0]
        ref_att[6:9] = u0  # inject DMPC acceleration command

        attitude_out = self.attitude_ctrl.control_loop(state, ref_att)

        self.analytics.record_step(
            state=state,
            control=u0,
            reference=reference_trajectory[0],
            solve_status=info.get("status", "unknown"),
            solve_time=info.get("solve_time", 0.0),
            objective=info.get("objective", float("inf")),
        )

        return attitude_out["motor_thrusts"], {**info, **attitude_out}

    def get_metrics(self) -> Dict:
        """Return aggregated DMPC performance metrics."""
        return self.analytics.get_metrics()

    def reset(self) -> None:
        """Reset analytics history (does not change controller parameters)."""
        self.analytics.reset()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save DMPC cost matrices and analytics to *path*."""
        self.dmpc.save_config(path)

    def load(self, path: str) -> None:
        """Restore DMPC cost matrices from *path*."""
        self.dmpc.load_config(path)
