from __future__ import annotations

import numpy as np

from isr_rl_dmpc.gym_env.sensor_model import ForwardFOVSensorModel, VisionSensorConfig


def test_visible_mask_respects_135_degree_forward_fov() -> None:
    model = ForwardFOVSensorModel(
        VisionSensorConfig(max_range=100.0, noise_std=0.0)
    )
    centers = np.array(
        [
            [40.0, 0.0],    # directly ahead
            [20.0, 34.64],  # ~60 degrees, still visible
            [0.0, 40.0],    # 90 degrees, outside 67.5 half-angle
            [-20.0, 0.0],   # behind
        ],
        dtype=np.float64,
    )

    visible, _, _ = model.visible_mask(
        drone_position_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        cell_centers_xy=centers,
    )

    assert visible.tolist() == [True, True, False, False]


def test_quality_decreases_with_distance_and_angle() -> None:
    model = ForwardFOVSensorModel(
        VisionSensorConfig(max_range=100.0, noise_std=0.0)
    )

    close_quality = model.compute_quality(np.array([10.0]), np.array([0.0]))[0]
    far_quality = model.compute_quality(np.array([80.0]), np.array([0.0]))[0]
    angled_quality = model.compute_quality(
        np.array([10.0]),
        np.array([np.deg2rad(60.0)]),
    )[0]

    assert close_quality > far_quality
    assert close_quality > angled_quality
    assert 0.0 <= far_quality <= 1.0
    assert 0.0 <= angled_quality <= 1.0
