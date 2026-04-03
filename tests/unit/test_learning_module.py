"""
TEST: Module 9 - DMPC Analytics (replaces former RL Learning Module)

Unit tests for the pure-optimisation DMPCAnalytics data collector.
All RL/neural-network tests have been removed together with the
corresponding code.
"""

import pytest
import numpy as np
from pathlib import Path
from isr_rl_dmpc import StepRecord, DMPCAnalytics


class TestStepRecord:
    """Test StepRecord dataclass."""

    def test_creation(self):
        rec = StepRecord(
            state=np.zeros(11),
            control=np.zeros(3),
            reference=np.zeros(11),
            solve_status="optimal",
            solve_time=0.005,
            objective=1.23,
            tracking_error=0.1,
        )
        assert rec.solve_status == "optimal"
        assert rec.tracking_error == 0.1


class TestDMPCAnalytics:
    """Test DMPCAnalytics collector."""

    @pytest.fixture
    def analytics(self):
        return DMPCAnalytics(state_dim=11, control_dim=3)

    def test_initial_empty(self, analytics):
        metrics = analytics.get_metrics()
        assert metrics["total_steps"] == 0
        assert metrics["rmse_tracking"] == 0.0

    def test_record_step(self, analytics):
        analytics.record_step(
            state=np.zeros(11),
            control=np.zeros(3),
            reference=np.zeros(11),
            solve_status="optimal",
            solve_time=0.005,
            objective=1.0,
        )
        assert analytics.get_metrics()["total_steps"] == 1

    def test_metrics_after_steps(self, analytics):
        for _ in range(20):
            state = np.random.randn(11) * 0.1
            analytics.record_step(
                state=state,
                control=np.zeros(3),
                reference=np.zeros(11),
                solve_status="optimal",
                solve_time=np.random.uniform(0.001, 0.01),
                objective=np.random.uniform(0.5, 2.0),
            )
        m = analytics.get_metrics()
        assert m["total_steps"] == 20
        assert m["rmse_tracking"] >= 0
        assert m["solve_success_rate"] == 1.0
        assert m["mean_solve_time_ms"] > 0

    def test_infeasible_steps_tracked(self, analytics):
        analytics.record_step(
            state=np.zeros(11),
            control=np.zeros(3),
            reference=np.zeros(11),
            solve_status="infeasible",
            solve_time=0.0,
            objective=float("inf"),
        )
        m = analytics.get_metrics()
        assert m["solve_success_rate"] == 0.0

    def test_reset(self, analytics):
        analytics.record_step(
            np.zeros(11), np.zeros(3), np.zeros(11), "optimal", 0.005, 1.0
        )
        analytics.reset()
        assert analytics.get_metrics()["total_steps"] == 0

    def test_save_load(self, analytics, tmp_path):
        for _ in range(5):
            analytics.record_step(
                np.random.randn(11), np.zeros(3), np.zeros(11),
                "optimal", 0.003, 0.8,
            )
        path = tmp_path / "analytics.npz"
        analytics.save(path)
        assert path.exists()

        a2 = DMPCAnalytics()
        a2.load(path)
        assert len(a2.tracking_error_history) == 5

    def test_finite_difference_jacobian(self):
        def cost(params):
            return float(np.sum(params ** 2))

        params = np.array([1.0, 2.0, 3.0])
        jac = DMPCAnalytics.finite_difference_jacobian(cost, params)
        # Gradient of sum(x^2) is 2x
        np.testing.assert_allclose(jac, 2 * params, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
