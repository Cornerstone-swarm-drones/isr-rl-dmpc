"""
TEST: Stability Analysis Module

Unit tests for the DMPC stability analyser covering all five checks:
Lyapunov, eigenvalue, ISS, collision barrier, and recursive feasibility.
"""

import pytest
import numpy as np
from isr_rl_dmpc.analysis import (
    DMPCStabilityAnalyzer,
    LyapunovResult,
    EigenvalueResult,
    ISSResult,
    CollisionBarrierResult,
    RecursiveFeasibilityResult,
    SwarmStabilityReport,
)


@pytest.fixture
def analyzer():
    # Use the 9-state translational controllable subsystem [p(3), v(3), a(3)].
    # Yaw and yaw-rate (state indices 9-10 in the 11-state vector) are NOT
    # driven by the translational acceleration inputs [ax, ay, az] and are
    # therefore excluded from this analysis — they are stabilised independently
    # by the geometric SO(3) attitude controller.
    return DMPCStabilityAnalyzer(
        state_dim=9,
        control_dim=3,
        dt=0.02,
        collision_radius=5.0,
    )


class TestDMPCStabilityAnalyzerInit:
    def test_builds_dynamics(self, analyzer):
        # state_dim=9 (translational subsystem)
        assert analyzer.A.shape == (9, 9)
        assert analyzer.B.shape == (9, 3)

    def test_dare_terminal_cost_pd(self, analyzer):
        assert np.all(np.linalg.eigvalsh(analyzer.P) > 0)

    def test_k_lqr_shape(self, analyzer):
        assert analyzer.K_lqr.shape == (3, 9)


class TestLyapunovStability:
    def test_returns_lyapunov_result(self, analyzer):
        result = analyzer.check_lyapunov_stability()
        assert isinstance(result, LyapunovResult)

    def test_stable_system(self, analyzer):
        result = analyzer.check_lyapunov_stability()
        assert result.is_stable

    def test_p_positive_definite(self, analyzer):
        result = analyzer.check_lyapunov_stability()
        assert result.P_positive_definite

    def test_delta_v_negative(self, analyzer):
        result = analyzer.check_lyapunov_stability()
        assert result.delta_V_max < 0

    def test_positive_margin(self, analyzer):
        result = analyzer.check_lyapunov_stability()
        assert result.lyapunov_margin > 0


class TestEigenvalueStability:
    def test_returns_eigenvalue_result(self, analyzer):
        result = analyzer.check_eigenvalue_stability()
        assert isinstance(result, EigenvalueResult)

    def test_eigenvalues_shape(self, analyzer):
        result = analyzer.check_eigenvalue_stability()
        assert result.eigenvalues.shape == (9,)

    def test_spectral_radius_less_than_one(self, analyzer):
        result = analyzer.check_eigenvalue_stability()
        assert result.spectral_radius < 1.0

    def test_is_stable(self, analyzer):
        result = analyzer.check_eigenvalue_stability()
        assert result.is_stable

    def test_positive_stability_margin(self, analyzer):
        result = analyzer.check_eigenvalue_stability()
        assert result.stability_margin > 0


class TestISSAnalysis:
    def test_returns_iss_result(self, analyzer):
        result = analyzer.check_iss()
        assert isinstance(result, ISSResult)

    def test_iss_holds(self, analyzer):
        result = analyzer.check_iss()
        assert result.is_iss

    def test_iss_gain_positive(self, analyzer):
        result = analyzer.check_iss()
        assert result.iss_gain > 0

    def test_max_disturbance_positive(self, analyzer):
        result = analyzer.check_iss()
        assert result.max_disturbance > 0


class TestCollisionBarrier:
    def test_returns_cbf_result(self, analyzer):
        result = analyzer.check_collision_barrier()
        assert isinstance(result, CollisionBarrierResult)

    def test_default_positions_valid(self, analyzer):
        result = analyzer.check_collision_barrier()
        assert result.is_valid

    def test_safety_margin_positive(self, analyzer):
        result = analyzer.check_collision_barrier()
        assert result.safety_margin > 0

    def test_too_close_drones_invalid(self, analyzer):
        # Place two drones closer than collision_radius
        positions = np.array([
            [0.0, 0.0, 10.0],
            [2.0, 0.0, 10.0],  # only 2 m apart, < 5 m radius
        ])
        result = analyzer.check_collision_barrier(drone_positions=positions)
        assert not result.is_valid


class TestRecursiveFeasibility:
    def test_returns_feasibility_result(self, analyzer):
        result = analyzer.check_recursive_feasibility()
        assert isinstance(result, RecursiveFeasibilityResult)

    def test_terminal_set_volume_positive(self, analyzer):
        result = analyzer.check_recursive_feasibility()
        assert result.terminal_set_volume > 0


class TestFullReport:
    def test_returns_report(self, analyzer):
        report = analyzer.full_stability_report(horizon=20)
        assert isinstance(report, SwarmStabilityReport)

    def test_key_stability_properties(self, analyzer):
        """The most critical properties (Lyapunov, eigenvalue, ISS) hold."""
        report = analyzer.full_stability_report(horizon=20)
        assert report.lyapunov.is_stable
        assert report.eigenvalue.is_stable
        assert report.iss.is_iss
        assert report.collision_barrier.is_valid

    def test_summary_nonempty(self, analyzer):
        report = analyzer.full_stability_report(horizon=20)
        assert len(report.summary) > 0
        assert "OVERALL STABLE" in report.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
