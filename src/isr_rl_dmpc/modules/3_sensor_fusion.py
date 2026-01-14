"""
Integrated sensor fusion manager combining:
1. Drone state estimation (18D)
2. Target tracking (11D per target)

Provides unified interface for ISR-RL-DMPC system.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

from ..core import (
    DroneState, TargetState, DroneStateEstimator, TargetStateEstimator, SensorType
)

logger = logging.getLogger(__name__)


class SensorFusionManager:
    """
    Complete sensor fusion system for ISR-RL-DMPC.
    
    Manages:
    - Own drone state (18D with covariance)
    - Multiple target states (11D each with covariance)
    - Provides global state vector for RL agent
    - Provides state + uncertainty for DMPC control
    """
    
    def __init__(self, dt: float = 0.02, n_targets: int = 50):
        """
        Initialize sensor fusion manager.
        
        Args:
            dt: Time step (seconds)
            n_targets: Maximum number of tracks
        """
        self.dt = dt
        
        # Drone state estimation (18D)
        self.drone_estimator = DroneStateEstimator(dt)
        
        # Target tracking (11D each)
        self.target_manager = TargetStateEstimator(dt, n_targets)
        
        # Current state
        self.drone_state = None
        self.target_states = {}
        
        # Sensor positions (for measurement Jacobians)
        self.radar_pos = np.zeros(3)
        self.optical_pos = np.zeros(3)
        
        logger.info(f"SensorFusionManager initialized (dt={dt}s)")
    
    # ========== DRONE STATE ESTIMATION ==========
    
    def predict_drone(self, imu_accel: np.ndarray, imu_gyro: np.ndarray,
                     power_draw: float = 0.0) -> DroneState:
        """
        Predict drone state from IMU.
        
        Args:
            imu_accel: Accelerometer [ax, ay, az] (m/s²) in body frame
            imu_gyro: Gyro [ωx, ωy, ωz] (rad/s) in body frame
            power_draw: Power consumption (Watts)
        
        Returns:
            Updated drone state (18D)
        """
        self.drone_estimator.predict(imu_accel, imu_gyro, power_draw)
        self.drone_state = self.drone_estimator.get_drone_state()
        return self.drone_state
    
    def update_drone_gps(self, gps_pos: np.ndarray,
                        gps_vel: Optional[np.ndarray] = None,
                        pos_noise: float = 5.0,
                        vel_noise: float = 1.0) -> DroneState:
        """
        Update drone position/velocity with GPS.
        
        Args:
            gps_pos: GPS position [x, y, z]
            gps_vel: GPS velocity [vx, vy, vz] (optional)
            pos_noise: Position noise (m)
            vel_noise: Velocity noise (m/s)
        
        Returns:
            Updated drone state
        """
        self.drone_estimator.update_gps(gps_pos, gps_vel, pos_noise, vel_noise)
        self.drone_state = self.drone_estimator.get_drone_state()
        return self.drone_state
    
    def update_drone_compass(self, mag_field: np.ndarray) -> DroneState:
        """
        Update drone heading with magnetometer.
        
        Args:
            mag_field: Magnetic field [mx, my, mz]
        
        Returns:
            Updated drone state
        """
        self.drone_estimator.update_magnetometer(mag_field)
        self.drone_state = self.drone_estimator.get_drone_state()
        return self.drone_state
    
    def update_drone_battery(self, battery_wh: float) -> None:
        """Update battery charge."""
        self.drone_estimator.update_battery(battery_wh)
    
    def update_drone_health(self, health: float) -> None:
        """Update health estimate."""
        self.drone_estimator.update_health(health)
    
    def get_drone_state(self) -> Tuple[DroneState, np.ndarray]:
        """
        Get drone state with covariance.
        
        Returns:
            (DroneState, 18×18 covariance)
        """
        state = self.drone_estimator.get_drone_state()
        cov = self.drone_estimator.get_covariance_18d()
        return state, cov
    
    def get_drone_uncertainty(self) -> Dict[str, float]:
        """Get uncertainty (1-sigma) for each drone state component."""
        return self.drone_estimator.get_uncertainty()
    
    # ========== TARGET TRACKING ==========
    
    def create_target_track(self, initial_position: np.ndarray) -> str:
        """
        Create new target track.
        
        Args:
            initial_position: Initial position [x, y, z]
        
        Returns:
            Target ID
        """
        # Initialize with zero velocity/acceleration
        initial_state = TargetState(
            position=initial_position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=0.0,
            yaw_rate=0.0
        )
        
        target_id = self.target_manager.create_track(initial_state)
        return target_id
    
    def delete_target_track(self, target_id: str) -> None:
        """Delete target track (target lost)."""
        self.target_manager.delete_track(target_id)
    
    def predict_targets(self) -> None:
        """Predict all target states."""
        self.target_manager.predict_all()
    
    def update_target_radar(self, target_id: str,
                           measurements: list) -> bool:
        """
        Update target with radar measurements.
        
        Args:
            target_id: Target identifier
            measurements: List of RadarMeasurement
        
        Returns:
            True if successful
        """
        meas_dict = {SensorType.RADAR: measurements}
        return self.target_manager.update_track(target_id, meas_dict, self.radar_pos)
    
    def update_target_optical(self, target_id: str,
                             measurements: list) -> bool:
        """Update target with optical measurements."""
        meas_dict = {SensorType.OPTICAL: measurements}
        return self.target_manager.update_track(target_id, meas_dict, self.optical_pos)
    
    def update_target_rf(self, target_id: str,
                        measurements: list) -> bool:
        """Update target with RF fingerprinting measurements."""
        meas_dict = {SensorType.RF_FINGERPRINT: measurements}
        return self.target_manager.update_track(target_id, meas_dict, np.zeros(3))
    
    def update_target_acoustic(self, target_id: str,
                              measurements: list) -> bool:
        """Update target with acoustic TDOA measurements."""
        meas_dict = {SensorType.ACOUSTIC_TDOA: measurements}
        return self.target_manager.update_track(target_id, meas_dict, np.zeros(3))
    
    def update_target_multi_sensor(self, target_id: str,
                                   measurements: Dict[SensorType, list]) -> bool:
        """
        Update target with multi-sensor measurements (adaptive fusion).
        
        Args:
            target_id: Target identifier
            measurements: Dict mapping sensor type to list of measurements
        
        Returns:
            True if successful
        """
        if target_id not in self.target_manager.tracks:
            return False
        
        sensor_pos = self.radar_pos  # Or choose appropriate sensor
        self.target_manager.tracks[target_id].update_adaptive_fusion(
            measurements, sensor_pos
        )
        return True
    
    def get_target_state(self, target_id: str) -> Optional[Tuple[TargetState, np.ndarray]]:
        """
        Get target state with covariance.
        
        Returns:
            (TargetState, 11×11 covariance) or None if not found
        """
        if target_id not in self.target_manager.tracks:
            return None
        
        track = self.target_manager.tracks[target_id]
        return track.get_state(), track.get_covariance()
    
    def get_all_target_states(self) -> Dict[str, TargetState]:
        """Get states for all targets."""
        return self.target_manager.get_all_estimates()
    
    def get_all_target_covariances(self) -> Dict[str, np.ndarray]:
        """Get covariances for all targets."""
        return self.target_manager.get_all_covariances()
    
    def get_healthy_targets(self) -> Dict[str, TargetState]:
        """Get only healthy (low uncertainty) targets."""
        return self.target_manager.get_healthy_tracks()
    
    # ========== INTEGRATED STATE VECTOR ==========
    
    def get_global_state_vector(self) -> np.ndarray:
        """
        Get complete global state vector for RL agent.
        
        Format: [18D drone] + [11D target1] + [11D target2] + ...
        
        Returns:
            Global state vector (18 + 11*N_targets dimensions)
        """
        drone_state = self.drone_estimator.get_drone_state()
        components = [drone_state.to_vector()]
        
        # Add target states (ordered by ID)
        for target_id in sorted(self.target_manager.tracks.keys()):
            target_state = self.target_manager.tracks[target_id].get_state()
            components.append(target_state.to_vector())
        
        return np.concatenate(components)
    
    def get_global_state_with_uncertainty(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get global state vector with concatenated covariances.
        
        Returns:
            (state_vector, concatenated_covariance_matrix)
        """
        state_vec = self.get_global_state_vector()
        
        # Build block-diagonal covariance
        n_targets = len(self.target_manager.tracks)
        total_dim = 18 + 11 * n_targets
        P = np.zeros((total_dim, total_dim))
        
        # Drone covariance (18×18)
        P[0:18, 0:18] = self.drone_estimator.get_covariance_18d()
        
        # Target covariances (11×11 each)
        for i, target_id in enumerate(sorted(self.target_manager.tracks.keys())):
            start_idx = 18 + i * 11
            end_idx = start_idx + 11
            track = self.target_manager.tracks[target_id]
            P[start_idx:end_idx, start_idx:end_idx] = track.get_covariance()
        
        return state_vec, P
    
    def is_drone_healthy(self) -> bool:
        """Check if drone state estimation is healthy."""
        return self.drone_estimator.is_healthy()
    
    def get_system_status(self) -> Dict:
        """
        Get complete system status.
        
        Returns:
            Dictionary with system metrics
        """
        drone_state, drone_cov = self.get_drone_state()
        target_states = self.get_all_target_states()
        
        return {
            'timestamp': None,  # Will be set by caller
            'drone': {
                'healthy': self.is_drone_healthy(),
                'position': drone_state.position,
                'velocity': drone_state.velocity,
                'battery_wh': drone_state.battery_energy,
                'health': drone_state.health,
                'position_uncertainty': np.sqrt(np.mean(np.diag(drone_cov)[0:3]))
            },
            'targets': {
                tid: {
                    'position': state.position,
                    'velocity': state.velocity,
                    'yaw': state.yaw
                }
                for tid, state in target_states.items()
            },
            'n_targets': len(target_states),
            'n_healthy_targets': len(self.target_manager.get_healthy_tracks())
        }
    
    def step(self, imu_accel: np.ndarray, imu_gyro: np.ndarray,
            gps_pos: Optional[np.ndarray] = None,
            gps_vel: Optional[np.ndarray] = None,
            mag_field: Optional[np.ndarray] = None,
            radar_measurements: Optional[Dict[str, list]] = None,
            optical_measurements: Optional[Dict[str, list]] = None,
            battery_wh: Optional[float] = None,
            health: Optional[float] = None,
            power_draw: float = 0.0) -> Dict:
        """
        Single sensor fusion step (main control loop).
        
        Args:
            imu_accel: Accelerometer data
            imu_gyro: Gyro data
            gps_pos: GPS position (optional)
            gps_vel: GPS velocity (optional)
            mag_field: Magnetometer (optional)
            radar_measurements: Dict[target_id] -> list of RadarMeasurement
            optical_measurements: Dict[target_id] -> list of OpticalMeasurement
            battery_wh: Battery charge (optional)
            health: Health estimate (optional)
            power_draw: Power consumption (Watts)
        
        Returns:
            System status dictionary
        """
        self.predict_drone(imu_accel, imu_gyro, power_draw)
        
        if gps_pos is not None:
            self.update_drone_gps(gps_pos, gps_vel)
        
        if mag_field is not None:
            self.update_drone_compass(mag_field)
        
        if battery_wh is not None:
            self.update_drone_battery(battery_wh)
        
        if health is not None:
            self.update_drone_health(health)
        
        self.predict_targets()
        
        if radar_measurements:
            for target_id, meas_list in radar_measurements.items():
                self.update_target_radar(target_id, meas_list)
        
        if optical_measurements:
            for target_id, meas_list in optical_measurements.items():
                self.update_target_optical(target_id, meas_list)
        
        return self.get_system_status()
