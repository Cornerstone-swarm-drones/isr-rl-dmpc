"""
TEST: Module 7 - Pure DMPC Controller (CVXPY/OSQP)

Unit tests for the pure optimisation-based DMPC controller.
Neural-network tests (CostWeightNetwork, DynamicsResidualNetwork,
ValueNetworkMPC) have been removed together with the corresponding code.
"""

import pytest
import numpy as np
from isr_rl_dmpc import (
    DMPCConfig, MPCSolver, DMPC, compute_lqr_terminal_cost,
)


class TestDMPCConfig:
    """Test DMPCConfig initialisation."""

    def test_default_config(self):
        config = DMPCConfig()
        assert config.horizon == 20
        assert config.dt == 0.02
        assert config.state_dim == 11
        assert config.control_dim == 3
        assert config.accel_max == 10.0

    def test_p_base_auto_computed(self):
        """P_base should be set automatically via DARE if not provided."""
        config = DMPCConfig()
        assert config.P_base is not None
        assert config.P_base.shape == (11, 11)
        # P must be positive-definite
        assert np.all(np.linalg.eigvalsh(config.P_base) > 0)

    def test_custom_cost_matrices(self):
        Q = np.eye(11) * 2.0
        R = np.eye(3) * 0.5
        config = DMPCConfig(Q_base=Q, R_base=R)
        assert np.allclose(config.Q_base, Q)
        assert np.allclose(config.R_base, R)


class TestComputeLQRTerminalCost:
    """Test the DARE-based terminal cost helper."""

    def test_returns_pd_matrix(self):
        P = compute_lqr_terminal_cost(
            state_dim=11, control_dim=3,
            Q=np.eye(11), R=np.eye(3) * 0.1, dt=0.02,
        )
        assert P.shape == (11, 11)
        assert np.all(np.linalg.eigvalsh(P) > 0)

    def test_symmetric(self):
        P = compute_lqr_terminal_cost(11, 3, np.eye(11), np.eye(3) * 0.1, 0.02)
        assert np.allclose(P, P.T, atol=1e-10)


class TestMPCSolver:
    """Test CVXPY QP MPC solver."""

    @pytest.fixture
    def solver(self):
        return MPCSolver(DMPCConfig())

    def test_solver_initialization(self, solver):
        assert solver is not None
        assert solver.horizon == 20

    def test_solve_mpc_returns_correct_shape(self, solver):
        x0 = np.zeros(11)
        x_ref = np.zeros((21, 11))
        A = np.eye(11)
        B = np.zeros((11, 3))
        Q = np.eye(11)
        R = np.eye(3)
        P = np.eye(11)

        u_opt, info = solver.solve(x0, x_ref, A, B, Q, R, P)
        assert u_opt.shape == (20, 3)
        assert "status" in info

    def test_solve_with_collision_constraints(self, solver):
        x0 = np.zeros(11)
        x_ref = np.zeros((21, 11))
        A = np.eye(11)
        B = np.zeros((11, 3))
        Q = np.eye(11)
        R = np.eye(3)
        P = np.eye(11)
        neighbor_pos = np.array([2.0, 0.0, 0.0])

        u_opt, info = solver.solve(
            x0, x_ref, A, B, Q, R, P, neighbor_positions=[neighbor_pos]
        )
        assert u_opt.shape == (20, 3)


class TestDMPC:
    """Test pure DMPC controller."""

    @pytest.fixture
    def controller(self):
        return DMPC(DMPCConfig())

    def test_initialization(self, controller):
        assert controller is not None
        assert hasattr(controller, "cvxpy_solver")
        assert hasattr(controller, "Q")
        assert hasattr(controller, "R")
        assert hasattr(controller, "P")
        # Ensure no neural-network attributes exist
        assert not hasattr(controller, "cost_weight_network")
        assert not hasattr(controller, "dynamics_residual")
        assert not hasattr(controller, "terminal_value")

    def test_call_returns_correct_shapes(self, controller):
        x = np.zeros(11)
        x_ref = np.zeros((21, 11))
        u_opt, info = controller(x, x_ref)
        assert u_opt.shape == (20, 3)
        assert "status" in info

    def test_no_weight_scales_key(self, controller):
        """The 'weight_scales' key from NN hybrid controller must not appear."""
        x = np.zeros(11)
        x_ref = np.zeros((21, 11))
        _, info = controller(x, x_ref)
        assert "weight_scales" not in info

    def test_save_load_config(self, controller, tmp_path):
        path = str(tmp_path / "dmpc_config")
        controller.save_config(path)
        import os
        assert os.path.exists(path + ".npz")

        Q_before = controller.Q.copy()
        controller.load_config(path + ".npz")
        assert np.allclose(controller.Q, Q_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
