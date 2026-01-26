"""
TEST: Module 8 - Attitude Controller (SO(3) + PyTorch Gain Adaptation)

Unit tests for geometric attitude control with learned gain adaptation
Tests quaternion conversions, control law, and motor mixing
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock
from isr_rl_dmpc import (
    DroneParameters, GainAdaptationNetwork, GeometricController, AttitudeController
)

class TestDroneParametersBasics:
    """Test DroneParameters initialization."""
    
    def test_params_initialization(self):
        """DroneParameters initializes with defaults."""
        
        params = DroneParameters()
        assert params.mass == 1.0
        assert params.Kp_attitude == 4.5
        assert params.Kd_attitude == 1.5
        assert params.inertia is not None
        assert params.inertia.shape == (3, 3)
    
    def test_inertia_matrix(self):
        """Inertia matrix initialized correctly."""
        
        params = DroneParameters()
        inertia = params.inertia
        
        # Should be diagonal
        assert np.allclose(inertia, np.diag(np.diag(inertia)))
        # Positive diagonal elements
        assert np.all(np.diag(inertia) > 0)


class TestGainAdaptationNetwork:
    """Test PyTorch gain adaptation network."""
    
    @pytest.fixture
    def network(self):
        return GainAdaptationNetwork(state_dim=11)
    
    def test_network_initialization(self, network):
        """Gain adaptation network initializes."""
        assert network is not None
    
    def test_forward_pass(self, network):
        """Forward pass produces gain multipliers."""
        state = torch.randn(1, 11)
        
        multipliers = network(state)
        
        assert multipliers.shape == (1, 4)  # 4 gains to adapt
        assert torch.all(multipliers > 0.5)  # Lower bound
        assert torch.all(multipliers < 2.0)  # Upper bound
    
    def test_batch_multipliers(self, network):
        """Network handles batch states."""
        batch_states = torch.randn(32, 11)
        
        multipliers = network(batch_states)
        
        assert multipliers.shape == (32, 4)


class TestQuaternionConversions:
    """Test quaternion to rotation matrix conversions."""
    
    @pytest.fixture
    def controller(self):
        params = DroneParameters()
        return GeometricController(params)
    
    def test_quaternion_to_matrix(self, controller):
        """Convert quaternion to rotation matrix."""
        # Identity quaternion
        q = np.array([1, 0, 0, 0])
        
        R = controller.quaternion_to_matrix(q)
        
        assert R.shape == (3, 3)
        # Should be close to identity for identity quaternion
        assert np.allclose(R, np.eye(3), atol=1e-6)
    
    def test_matrix_to_quaternion(self, controller):
        """Convert rotation matrix to quaternion."""
        # Identity matrix
        R = np.eye(3)
        
        q = controller.matrix_to_quaternion(R)
        
        assert q.shape == (3,)
    
    def test_quaternion_normalization(self, controller):
        """Quaternions are normalized."""
        q_unnorm = np.array([2, 1, 1, 1])
        
        R = controller.quaternion_to_matrix(q_unnorm)
        
        # Check orthogonality
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


class TestAttitudeErrorComputation:
    """Test SO(3) attitude error computation."""
    
    @pytest.fixture
    def controller(self):
        params = DroneParameters()
        return GeometricController(params)
    
    def test_zero_error_identity_attitudes(self, controller):
        """Zero error when attitudes match."""
        R = np.eye(3)
        R_d = np.eye(3)
        
        e_R = controller.attitude_error(R, R_d)
        
        # Error should be close to zero
        assert np.allclose(e_R, np.zeros(3), atol=1e-6)
    
    def test_nonzero_error_different_attitudes(self, controller):
        """Non-zero error for different attitudes."""
        R = np.eye(3)
        # Desired attitude: 90° rotation about z
        angle = np.pi / 2
        R_d = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        e_R = controller.attitude_error(R, R_d)
        
        # Error magnitude should be significant
        assert np.linalg.norm(e_R) > 0.1


class TestDesiredAttitudeFromAcceleration:
    """Test differential flatness for attitude generation."""
    
    @pytest.fixture
    def controller(self):
        params = DroneParameters()
        return GeometricController(params)
    
    def test_hover_attitude(self, controller):
        """Hover (zero acceleration) produces level attitude."""
        a_d = np.zeros(3)
        yaw_d = 0.0
        
        R_d = controller.desired_attitude_from_accel(a_d, yaw_d)
        
        assert R_d.shape == (3, 3)
        # Should be orthogonal
        assert np.allclose(R_d @ R_d.T, np.eye(3), atol=1e-6)
    
    def test_forward_acceleration_attitude(self, controller):
        """Forward acceleration produces forward-tilted attitude."""
        a_d = np.array([5.0, 0.0, 0.0])  # Forward
        yaw_d = 0.0
        
        R_d = controller.desired_attitude_from_accel(a_d, yaw_d)
        
        assert R_d.shape == (3, 3)
        assert np.allclose(R_d @ R_d.T, np.eye(3), atol=1e-6)


class TestSO3ControlLaw:
    """Test SO(3) geometric control law."""
    
    @pytest.fixture
    def controller(self):
        params = DroneParameters()
        return GeometricController(params)
    
    def test_control_law_zero_error(self, controller):
        """Zero control at zero error."""
        R = np.eye(3)
        omega = np.zeros(3)
        R_d = np.eye(3)
        
        tau = controller.control_law(R, omega, R_d)
        
        # Torque should be small
        assert np.linalg.norm(tau) < 0.1
    
    def test_control_law_nonzero_error(self, controller):
        """Non-zero control for non-zero error."""
        R = np.eye(3)
        omega = np.zeros(3)
        # Desired attitude: 45° rotation about z
        angle = np.pi / 4
        R_d = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        tau = controller.control_law(R, omega, R_d)
        
        # Torque should be significant
        assert np.linalg.norm(tau) > 0.1
    
    def test_adaptive_gains(self, controller):
        """Control law uses adaptive gains."""
        R = np.eye(3)
        omega = np.zeros(3)
        R_d = np.array([
            [0.7071, -0.7071, 0],
            [0.7071, 0.7071, 0],
            [0, 0, 1]
        ])
        
        tau_base = controller.control_law(R, omega, R_d, Kp=4.5, Kd=1.5)
        tau_scaled = controller.control_law(R, omega, R_d, Kp=9.0, Kd=3.0)
        
        # Scaled gains should produce larger torque
        assert np.linalg.norm(tau_scaled) > np.linalg.norm(tau_base)


class TestAttitudeController:
    """Test attitude controller."""
    
    @pytest.fixture
    def controller(self):
        from attitude_controller import DroneParameters, AttitudeController
        params = DroneParameters()
        return AttitudeController(params)
    
    def test_controller_initialization(self, controller):
        """Attitude controller initializes."""
        assert controller is not None
        assert hasattr(controller, 'so3_controller')
        assert hasattr(controller, 'gain_network')
    
    def test_control_loop(self, controller):
        """Control loop computes motor thrusts."""
        state = np.random.randn(11)
        reference = np.random.randn(12)
        
        output = controller.control_loop(state, reference, use_adaptation=True)
        
        assert 'motor_thrusts' in output
        assert 'torque' in output
        assert 'gain_multipliers' in output
        assert output['motor_thrusts'].shape == (4,)


class TestAttitudeControllerLearning:
    """Test learning from collected data."""
    
    @pytest.fixture
    def controller(self):
        params = DroneParameters()
        return AttitudeController(params)
    
    def test_learn_gain_adaptation(self, controller):
        """Learn gain adaptation from data."""
        states = torch.randn(100, 11)
        target_gains = torch.randn(100, 4)
        
        # Should complete without error
        controller.learn_gain_adaptation(states, target_gains, epochs=2)
    
    def test_gain_save_load(self, controller, tmp_path):
        """Save and load learned gains."""
        gain_path = str(tmp_path / "gains.pt")
        
        controller.save_gains(gain_path)
        controller.load_gains(gain_path)
        
        # Should complete without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
