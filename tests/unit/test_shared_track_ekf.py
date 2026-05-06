from __future__ import annotations

import numpy as np

from isr_rl_dmpc.gym_env.shared_track_ekf import SharedTrackEKF, SharedTrackEKFConfig


def test_shared_track_ekf_predict_and_update() -> None:
    ekf = SharedTrackEKF(
        SharedTrackEKFConfig(
            dt=0.1,
            process_noise_accel=1.0,
            measurement_noise_std=1.5,
        )
    )
    ekf.initialize(np.array([10.0, -4.0], dtype=np.float64), step=0)
    ekf.state[2:] = np.array([2.0, -1.0], dtype=np.float64)
    prev_cov_trace = float(np.trace(ekf.covariance[:2, :2]))

    ekf.predict()
    assert np.allclose(ekf.state[:2], np.array([10.2, -4.1]), atol=1e-3)

    ekf.update(np.array([10.3, -4.0], dtype=np.float64), step=1, measurement_scale=1.0)
    next_cov_trace = float(np.trace(ekf.covariance[:2, :2]))
    assert next_cov_trace < prev_cov_trace


def test_shared_track_ekf_confidence_decays_with_age() -> None:
    ekf = SharedTrackEKF(
        SharedTrackEKFConfig(
            dt=0.05,
            process_noise_accel=1.0,
            measurement_noise_std=1.0,
            confidence_cov_scale=1000.0,
            confidence_age_decay_steps=10.0,
        )
    )
    ekf.initialize(np.array([0.0, 0.0], dtype=np.float64), step=0)
    ekf.update(np.array([0.1, -0.1], dtype=np.float64), step=1, measurement_scale=1.0)

    c_now = ekf.confidence(step=1)
    c_later = ekf.confidence(step=20)
    assert 0.0 <= c_now <= 1.0
    assert 0.0 <= c_later <= 1.0
    assert c_later < c_now
