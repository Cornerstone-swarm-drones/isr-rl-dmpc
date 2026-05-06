"""
Minimal shared EKF for one moving threat track.

The filter intentionally stays small and inspectable for Phase 1:

* state: [x, y, vx, vy]
* process model: constant velocity
* measurement model: direct noisy position [x, y]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SharedTrackEKFConfig:
    dt: float
    process_noise_accel: float = 2.5
    measurement_noise_std: float = 2.0
    initial_velocity_std: float = 25.0
    confidence_cov_scale: float = 1400.0
    confidence_age_decay_steps: float = 40.0


class SharedTrackEKF:
    """Constant-velocity EKF for a single shared target track."""

    def __init__(self, config: SharedTrackEKFConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 1e6
        self.last_update_step: Optional[int] = None

    def initialize(self, position_xy: np.ndarray, *, step: int) -> None:
        pos = np.asarray(position_xy, dtype=np.float64).reshape(2)
        self.state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        pos_var = max(self.config.measurement_noise_std ** 2, 1e-6)
        vel_var = max(self.config.initial_velocity_std ** 2, pos_var)
        self.covariance = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
        self.initialized = True
        self.last_update_step = int(step)

    def _transition_matrix(self) -> np.ndarray:
        dt = float(self.config.dt)
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _process_noise(self) -> np.ndarray:
        dt = float(self.config.dt)
        sigma = float(self.config.process_noise_accel)
        q = sigma ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def predict(self) -> None:
        if not self.initialized:
            return
        f = self._transition_matrix()
        q = self._process_noise()
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q

    def update(self, measurement_xy: np.ndarray, *, step: int, measurement_scale: float = 1.0) -> None:
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        if not self.initialized:
            self.initialize(z, step=step)
            return

        h = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        r_base = max(self.config.measurement_noise_std ** 2, 1e-6)
        r_scale = max(float(measurement_scale), 1e-6)
        r = np.eye(2, dtype=np.float64) * r_base * r_scale

        innovation = z - (h @ self.state)
        s = h @ self.covariance @ h.T + r
        s_inv = np.linalg.inv(s)
        k = self.covariance @ h.T @ s_inv
        self.state = self.state + k @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - k @ h) @ self.covariance
        self.last_update_step = int(step)

    def confidence(self, *, step: int) -> float:
        if not self.initialized:
            return 0.0
        pos_trace = float(np.trace(self.covariance[:2, :2]))
        cov_conf = float(np.exp(-pos_trace / max(self.config.confidence_cov_scale, 1e-6)))
        if self.last_update_step is None:
            return cov_conf
        age = max(int(step) - int(self.last_update_step), 0)
        age_conf = float(np.exp(-age / max(self.config.confidence_age_decay_steps, 1e-6)))
        return float(np.clip(cov_conf * age_conf, 0.0, 1.0))
