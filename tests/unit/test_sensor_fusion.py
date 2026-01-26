"""
TEST: Module 3 - Sensor Fusion (Multi-Sensor EKF for Drone & Target State Estimation)

Focused unit tests for sensor fusion, drone state, and target tracking
"""

import pytest
import numpy as np
from unittest.mock import Mock
from isr_rl_dmpc import SensorFusionManager


class TestSensorFusionManagerBasics:
    """Sensor fusion manager initialization and drone state."""
    
    def test_sensor_fusion_initializes(self):
        """SensorFusionManager initializes correctly."""
        
        fusion = SensorFusionManager(dt=0.02, n_targets=50)
        
        assert fusion.dt == 0.02
        assert fusion.drone_state is None
    
    def test_drone_state_prediction(self):
        """Drone state predicts from IMU."""
        
        fusion = SensorFusionManager(dt=0.02)
        
        imu_accel = np.array([0.0, 0.0, -9.81])  # Hovering
        imu_gyro = np.array([0.0, 0.0, 0.0])
        
        state = fusion.predict_drone(imu_accel, imu_gyro, power_draw=50.0)
        
        assert state is not None
        assert hasattr(state, 'position')
        assert hasattr(state, 'velocity')


class TestDroneStateEstimation:
    """Drone state estimation workflow."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02)
    
    def test_predict_then_update_gps(self, fusion):
        """Predict and update with GPS measurements."""
        # Predict from IMU
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        # Update with GPS
        gps_pos = np.array([100.0, 100.0, 50.0])
        state = fusion.update_drone_gps(gps_pos)
        
        assert state is not None
    
    def test_update_drone_compass(self, fusion):
        """Update drone heading with magnetometer."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        mag_field = np.array([1.0, 0.0, 0.0])
        state = fusion.update_drone_compass(mag_field)
        
        assert state is not None
    
    def test_get_drone_state_with_covariance(self, fusion):
        """Get drone state with uncertainty."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        state, cov = fusion.get_drone_state()
        
        assert state is not None
        assert cov.shape == (18, 18)
    
    def test_battery_and_health_updates(self, fusion):
        """Update battery and health estimates."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        fusion.update_drone_battery(4000.0)
        fusion.update_drone_health(0.95)
        
        state = fusion.get_drone_state()[0]
        assert state.battery_energy == 4000.0
        assert state.health == 0.95
    
    def test_is_drone_healthy(self, fusion):
        """Check drone state health."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        healthy = fusion.is_drone_healthy()
        assert isinstance(healthy, bool)


class TestTargetTracking:
    """Target state estimation and tracking."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02, n_targets=50)
    
    def test_create_target_track(self, fusion):
        """Create new target track."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        assert target_id is not None
        assert isinstance(target_id, str)
    
    def test_delete_target_track(self, fusion):
        """Delete target track."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        fusion.delete_target_track(target_id)
        
        # Should be removed
        assert target_id not in fusion.target_states
    
    def test_predict_targets(self, fusion):
        """Predict all target states."""
        fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        fusion.create_target_track(np.array([300.0, 300.0, 0.0]))
        
        fusion.predict_targets()  # Should not raise
        assert len(fusion.target_manager.tracks) == 2
    
    def test_get_target_state(self, fusion):
        """Get target state with covariance."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        result = fusion.get_target_state(target_id)
        
        assert result is not None
        state, cov = result
        assert state is not None
        assert cov.shape == (11, 11)
    
    def test_get_all_target_states(self, fusion):
        """Get all tracked target states."""
        for i in range(3):
            fusion.create_target_track(np.array([200 + i*50, 200, 0]))
        
        states = fusion.get_all_target_states()
        
        assert len(states) == 3
    
    def test_get_healthy_targets(self, fusion):
        """Get targets with low uncertainty."""
        fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        healthy = fusion.get_healthy_targets()
        
        assert isinstance(healthy, dict)


class TestMultiSensorFusion:
    """Multi-sensor update handling."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02)
    
    def test_update_target_radar(self, fusion):
        """Update target with radar measurements."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        radar_meas = [Mock(distance=150.0, bearing=0.0)]
        success = fusion.update_target_radar(target_id, radar_meas)
        
        assert isinstance(success, bool)
    
    def test_update_target_optical(self, fusion):
        """Update target with optical measurements."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        optical_meas = [Mock(pixel_x=100, pixel_y=100)]
        success = fusion.update_target_optical(target_id, optical_meas)
        
        assert isinstance(success, bool)
    
    def test_update_target_multi_sensor(self, fusion):
        """Update target with multiple sensor types."""
        target_id = fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        measurements = {
            'radar': [Mock(distance=150.0)],
            'optical': [Mock(pixel_x=100)]
        }
        success = fusion.update_target_multi_sensor(target_id, measurements)
        
        assert isinstance(success, bool)


class TestGlobalStateVector:
    """Global state vector for RL agent."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02)
    
    def test_global_state_vector_drone_only(self, fusion):
        """State vector includes drone state (18D)."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        state_vec = fusion.get_global_state_vector()
        
        # Should be 18D (drone only)
        assert state_vec.shape[0] >= 18
    
    def test_global_state_vector_with_targets(self, fusion):
        """State vector includes targets (11D each)."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        
        for i in range(3):
            fusion.create_target_track(np.array([200 + i*50, 200, 0]))
        
        state_vec = fusion.get_global_state_vector()
        
        # 18D drone + 11D per target
        expected_dim = 18 + 3 * 11
        assert state_vec.shape[0] == expected_dim
    
    def test_global_state_with_uncertainty(self, fusion):
        """Get state vector with covariance matrix."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        state_vec, cov = fusion.get_global_state_with_uncertainty()
        
        # Covariance should be block diagonal
        expected_dim = 18 + 11
        assert cov.shape == (expected_dim, expected_dim)


class TestSensorFusionStep:
    """Complete sensor fusion step."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02)
    
    def test_step_with_all_sensors(self, fusion):
        """Single step with all sensor inputs."""
        status = fusion.step(
            imu_accel=np.array([0, 0, -9.81]),
            imu_gyro=np.zeros(3),
            gps_pos=np.array([100, 100, 50]),
            gps_vel=np.array([5, 0, 0]),
            mag_field=np.array([1, 0, 0]),
            battery_wh=4000.0,
            health=0.95,
            power_draw=50.0
        )
        
        assert status is not None
        assert 'drone' in status
        assert 'targets' in status
    
    def test_step_with_minimal_sensors(self, fusion):
        """Step with only critical sensors."""
        status = fusion.step(
            imu_accel=np.array([0, 0, -9.81]),
            imu_gyro=np.zeros(3)
        )
        
        assert status is not None
    
    def test_step_multiple_times(self, fusion):
        """Multiple sequential steps."""
        for _ in range(10):
            fusion.step(
                imu_accel=np.array([0, 0, -9.81]),
                imu_gyro=np.zeros(3),
                gps_pos=np.array([100, 100, 50])
            )
        
        # Should accumulate state correctly
        state_vec = fusion.get_global_state_vector()
        assert state_vec.shape[0] > 0


class TestSystemStatus:
    """System status reporting."""
    
    @pytest.fixture
    def fusion(self):
        
        return SensorFusionManager(dt=0.02)
    
    def test_get_system_status(self, fusion):
        """Get complete system status dictionary."""
        fusion.predict_drone(np.array([0, 0, -9.81]), np.zeros(3))
        fusion.create_target_track(np.array([200.0, 200.0, 0.0]))
        
        status = fusion.get_system_status()
        
        assert 'drone' in status
        assert 'targets' in status
        assert status['drone']['healthy'] is not None
        assert status['targets']['n_targets'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
