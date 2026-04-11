"""
TEST: Module 8 - Attitude Controller (Pure Geometric SO(3) Control)

Unit tests for geometric attitude control with fixed gains.
Neural gain-adaptation tests have been removed together with
GainAdaptationNetwork.
"""

import pytest
import numpy as np
from isr_rl_dmpc import (
    DroneParameters, GeometricController, AttitudeController,
)


class TestDroneParameters:
    """Test DroneParameters initialisation."""

    def test_defaults(self):
        params = DroneParameters()
        assert params.mass == 1.477  # hector_quadrotor airframe
        assert params.Kp_attitude == 4.5
        assert params.Kd_attitude == 1.5
        assert params.inertia is not None
        assert params.inertia.shape == (3, 3)

    def test_inertia_diagonal(self):
        params = DroneParameters()
        assert np.allclose(params.inertia, np.diag(np.diag(params.inertia)))
        assert np.all(np.diag(params.inertia) > 0)

    def test_no_neural_network_attribute(self):
        """GainAdaptationNetwork must have been removed."""
        ctrl = AttitudeController(DroneParameters())
        assert not hasattr(ctrl, "gain_network")
        assert not hasattr(ctrl, "gain_optimizer")


class TestGeometricController:
    """Test SO(3) geometric controller."""

    @pytest.fixture
    def geo(self):
        return GeometricController(DroneParameters())

    def test_quaternion_to_matrix_identity(self, geo):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        R = geo.quaternion_to_matrix(q)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_matrix_to_quaternion_identity(self, geo):
        R = np.eye(3)
        q = geo.matrix_to_quaternion(R)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.abs(q[0]), 1.0, atol=1e-7)

    def test_attitude_error_zero_for_identical(self, geo):
        R = np.eye(3)
        e = geo.attitude_error(R, R)
        np.testing.assert_allclose(e, np.zeros(3), atol=1e-10)

    def test_desired_attitude_from_accel(self, geo):
        a_d = np.array([0.0, 0.0, 0.0])
        R_d = geo.desired_attitude_from_accel(a_d)
        assert R_d.shape == (3, 3)
        # Check orthonormality (relaxed tolerance due to normalisation numerics)
        np.testing.assert_allclose(R_d @ R_d.T, np.eye(3), atol=1e-5)

    def test_control_law_zero_error(self, geo):
        R = np.eye(3)
        tau = geo.control_law(R, np.zeros(3), R)
        np.testing.assert_allclose(tau, np.zeros(3), atol=1e-10)


class TestAttitudeController:
    """Test the full cascade attitude controller."""

    @pytest.fixture
    def ctrl(self):
        return AttitudeController(DroneParameters())

    def test_no_gain_network(self, ctrl):
        """GainAdaptationNetwork must have been removed."""
        assert not hasattr(ctrl, "gain_network")
        assert not hasattr(ctrl, "gain_optimizer")

    def test_control_loop_output_keys(self, ctrl):
        state = np.zeros(11)
        reference = np.zeros(12)
        out = ctrl.control_loop(state, reference)
        assert "motor_thrusts" in out
        assert "desired_attitude" in out
        assert "torque" in out

    def test_motor_thrusts_shape(self, ctrl):
        state = np.random.randn(11).astype(np.float32)
        reference = np.random.randn(12).astype(np.float32)
        out = ctrl.control_loop(state, reference)
        assert out["motor_thrusts"].shape == (4,)

    def test_motor_thrusts_non_negative(self, ctrl):
        state = np.zeros(11)
        reference = np.zeros(12)
        out = ctrl.control_loop(state, reference)
        assert np.all(out["motor_thrusts"] >= 0)

    def test_desired_attitude_orthonormal(self, ctrl):
        state = np.zeros(11)
        reference = np.zeros(12)
        R_d = ctrl.control_loop(state, reference)["desired_attitude"]
        np.testing.assert_allclose(R_d @ R_d.T, np.eye(3), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
