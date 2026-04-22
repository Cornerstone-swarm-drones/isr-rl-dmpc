from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from isr_rl_dmpc.core.belief_grid import BeliefGrid, BeliefGridConfig


def _make_grid() -> BeliefGrid:
    cells = [
        SimpleNamespace(center=np.array([10.0, 10.0]), priority=1.0),
        SimpleNamespace(center=np.array([30.0, 10.0]), priority=0.8),
        SimpleNamespace(center=np.array([10.0, 30.0]), priority=0.6),
    ]
    return BeliefGrid.from_cells(
        cells,
        grid_resolution=20.0,
        config=BeliefGridConfig(
            uncertainty_min=0.05,
            uncertainty_max=1.0,
            growth_rate=0.2,
        ),
    )


def test_uncertainty_growth_increases_and_caps() -> None:
    grid = _make_grid()
    grid.uncertainty[:] = np.array([0.2, 0.95, 1.0], dtype=np.float64)

    grid.grow_uncertainty()

    expected_first = min(1.0, 0.2 * np.exp(0.2))
    assert np.isclose(grid.uncertainty[0], expected_first)
    assert np.isclose(grid.uncertainty[1], 1.0)
    assert np.isclose(grid.uncertainty[2], 1.0)


def test_observation_reset_uses_quality_and_minimum_floor() -> None:
    grid = _make_grid()

    grid.observe(
        np.array([0, 1], dtype=np.int32),
        np.array([0.9, 1.0], dtype=np.float64),
        step=4,
    )

    assert np.isclose(grid.uncertainty[0], 0.1)
    assert np.isclose(grid.uncertainty[1], grid.config.uncertainty_min)
    assert grid.last_observed_step[0] == 4
    assert grid.last_observed_step[1] == 4


def test_fusion_uses_min_uncertainty_and_max_anomaly() -> None:
    left = _make_grid()
    right = _make_grid()
    left.uncertainty[:] = np.array([0.8, 0.7, 0.2], dtype=np.float64)
    left.anomaly_score[:] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    right.uncertainty[:] = np.array([0.4, 0.9, 0.1], dtype=np.float64)
    right.anomaly_score[:] = np.array([0.6, 0.1, 0.7], dtype=np.float64)

    left.fuse_from(right, step=9)

    assert np.allclose(left.uncertainty, np.array([0.4, 0.7, 0.1]))
    assert np.allclose(left.anomaly_score, np.array([0.6, 0.2, 0.7]))
    assert np.all(left.last_fused_step == 9)
