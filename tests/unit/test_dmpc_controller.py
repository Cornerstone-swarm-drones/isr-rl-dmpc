"""
TEST: Module 7 - Hybrid DMPC Controller (CVXPY + PyTorch)

Unit tests for hybrid DMPC combining convex optimization with learning
Tests MPC solving, cost scaling, and trajectory planning
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock
from isr_rl_dmpc import (
    DMPCConfig, CostWeightNetwork, DynamicsResidualNetwork, 
    ValueNetworkMPC, MPCSolver, DMPC,
)

class TestDMPCConfig:
    """Test DMPCConfig initialization."""
    
    def test_config_initialization(self):
        """DMPCConfig initializes with defaults."""
        
        config = DMPCConfig()
        assert config.horizon == 20
        assert config.dt == 0.02
        assert config.state_dim == 11
        assert config.control_dim == 3
        assert config.accel_max == 10.0


class TestCostWeightNetwork:
    """Test PyTorch cost weight adaptation network."""
    
    @pytest.fixture
    def network(self):
        return CostWeightNetwork(state_dim=11)
    
    def test_network_initialization(self, network):
        """Cost weight network initializes correctly."""
        assert network is not None
    
    def test_forward_pass(self, network):
        """Forward pass produces valid output."""
        state = torch.randn(1, 11)
        
        scales = network(state)
        
        assert scales.shape == (1, 3)
        assert torch.all(scales > 0)  # Scales should be positive
    
    def test_batch_forward(self, network):
        """Network handles batch input."""
        batch_states = torch.randn(32, 11)
        
        scales = network(batch_states)
        
        assert scales.shape == (32, 3)


class TestDynamicsResidualNetwork:
    """Test learned residual dynamics network."""
    
    @pytest.fixture
    def network(self):
        return DynamicsResidualNetwork(state_dim=11, control_dim=3)
    
    def test_network_initialization(self, network):
        """Dynamics residual network initializes."""
        assert network is not None
    
    def test_residual_output(self, network):
        """Network outputs state-dimension residuals."""
        state = torch.randn(1, 11)
        control = torch.randn(1, 3)
        
        residual = network(state, control)
        
        assert residual.shape == (1, 11)
    
    def test_batch_residual(self, network):
        """Network handles batch residuals."""
        batch_states = torch.randn(32, 11)
        batch_controls = torch.randn(32, 3)
        
        residuals = network(batch_states, batch_controls)
        
        assert residuals.shape == (32, 11)


class TestValueNetworkMPC:
    """Test terminal value function for MPC."""
    
    @pytest.fixture
    def network(self):
        return ValueNetworkMPC(state_dim=11)
    
    def test_terminal_value(self, network):
        """Terminal value network outputs scalar values."""
        state = torch.randn(1, 11)
        
        value = network(state)
        
        assert value.shape == (1,)
    
    def test_batch_terminal_values(self, network):
        """Network computes batch terminal values."""
        batch_states = torch.randn(32, 11)
        
        values = network(batch_states)
        
        assert values.shape == (32,)


class TestMPCSolver:
    """Test CVXPY QP MPC solver."""
    
    @pytest.fixture
    def solver(self):
        config = DMPCConfig()
        return MPCSolver(config)
    
    def test_solver_initialization(self, solver):
        """CVXPY solver initializes."""
        assert solver is not None
        assert solver.horizon == 20
    
    def test_solve_mpc(self, solver):
        """Solve MPC problem for given state."""
        x0 = np.random.randn(11)
        x_ref = np.random.randn(20, 11)
        A = np.eye(11)
        B = np.zeros((11, 3))
        Q = np.eye(11)
        R = np.eye(3)
        P = np.eye(11)
        
        u_opt, info = solver.solve(x0, x_ref, A, B, Q, R, P)
        
        assert u_opt.shape == (20, 3)  # 20-step horizon, 3-D control
        assert 'status' in info
    
    def test_collision_constraints(self, solver):
        """MPC respects collision avoidance constraints."""
        x0 = np.zeros(11)
        x_ref = np.zeros((20, 11))
        A = np.eye(11)
        B = np.zeros((11, 3))
        Q = np.eye(11)
        R = np.eye(3)
        P = np.eye(11)
        
        # Neighbor position nearby
        neighbor_pos = np.array([2.0, 0.0, 0.0])
        
        u_opt, info = solver.solve(x0, x_ref, A, B, Q, R, P,
                                   neighbor_positions=[neighbor_pos])
        
        assert u_opt.shape == (20, 3)


class TestDMPC:
    """Test hybrid DMPC with PyTorch and CVXPY."""
    
    @pytest.fixture
    def dmpc(self):
        config = DMPCConfig(device='cpu')
        return DMPC(config)
    
    def test_dmpc_initialization(self, dmpc):
        """ DMPC initializes correctly."""
        assert dmpc is not None
        assert hasattr(dmpc, 'cvxpy_solver')
        assert hasattr(dmpc, 'cost_weight_network')
    
    def test_forward_pass(self, dmpc):
        """Forward pass computes control."""
        x = np.random.randn(11)
        x_ref = np.random.randn(20, 11)
        
        u_opt, info = dmpc(x, x_ref)
        
        assert u_opt.shape == (20, 3)
        assert 'status' in info
        assert 'weight_scales' in info
    
    def test_adaptive_scaling(self, dmpc):
        """Cost matrices scaled adaptively by network."""
        x = np.random.randn(11)
        x_ref = np.random.randn(20, 11)
        
        u_opt, info = dmpc(x, x_ref)
        
        Q_scale, R_scale, P_scale = info['weight_scales']
        assert Q_scale > 0 and R_scale > 0 and P_scale > 0


class TestDMPCLearning:
    """Test learning from trajectories."""
    
    @pytest.fixture
    def dmpc(self):
        config = DMPCConfig(device='cpu')
        return DMPC(config)
    
    def test_learn_from_trajectory(self, dmpc):
        """Learn correction models from experience."""
        states = torch.randn(100, 11)
        actions = torch.randn(100, 3)
        next_states = torch.randn(100, 11)
        rewards = torch.randn(100)
        
        # Should complete without error
        dmpc.learn_from_trajectory(states, actions, next_states, rewards, epochs=1)
    
    def test_checkpoint_save_load(self, dmpc, tmp_path):
        """Save and load checkpoint."""
        # Save
        checkpoint_path = str(tmp_path / "dmpc_checkpoint.pt")
        dmpc.save_checkpoint(checkpoint_path)
        
        # Verify file exists
        import os
        assert os.path.exists(checkpoint_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
