"""
Multi-filter drone state estimation system (18D).
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

from ..core import DroneState

logger = logging.getLogger(__name__)


class PositionVelocityEKF:
    """
    6D Extended Kalman Filter for position and velocity.
    
    Fuses GPS measurements with IMU acceleration integration.
    """
    
    def __init__(self, dt: float = 0.02, gps_noise_pos: float = 5.0,
                 gps_noise_vel: float = 1.0):
        self.dt = dt
        
        self.x = np.zeros(6) # State: [x, y, z, vx, vy, vz]
        
        # Covariance (6×6)
        self.P = np.eye(6)
        self.P[0:3, 0:3] *= 100.0  # Position: ±10m
        self.P[3:6, 3:6] *= 10.0   # Velocity: ±3 m/s
        
        # Process noise (6×6)
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.01   # Position drift
        self.Q[3:6, 3:6] *= 0.1    # Velocity uncertainty
        
        # Measurement noise parameters
        self.gps_noise_pos = gps_noise_pos
        self.gps_noise_vel = gps_noise_vel
    
    def predict(self, accel_world: np.ndarray) -> None:
        """
        Predict step using acceleration (IMU).
        
        Args:
            accel_world: Acceleration in world frame [ax, ay, az]
        """
        # State transition matrix F (6×6)
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Predict state
        self.x[0:3] += self.x[3:6] * self.dt + 0.5 * accel_world * (self.dt**2)
        self.x[3:6] += accel_world * self.dt
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure covariance remains positive definite
        self.P = (self.P + self.P.T) / 2
    
    def update_gps(self, gps_pos: np.ndarray, gps_vel: np.ndarray) -> None:
        """
        Update with GPS measurement (position + velocity).
        
        Args:
            gps_pos: GPS position [x, y, z]
            gps_vel: GPS velocity [vx, vy, vz]
        """
        z = np.concatenate([gps_pos, gps_vel])
        
        # Measurement matrix H (6×6) - observe all states
        H = np.eye(6)
        
        # Measurement covariance R (6×6)
        R = np.eye(6)
        R[0:3, 0:3] *= self.gps_noise_pos**2
        R[3:6, 3:6] *= self.gps_noise_vel**2
        
        # Innovation
        z_hat = H @ self.x
        y = z - z_hat
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S + 1e-9 * np.eye(6))
        
        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Ensure positive definite
        self.P = (self.P + self.P.T) / 2
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and velocity."""
        return self.x[0:3].copy(), self.x[3:6].copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get 6×6 covariance."""
        return self.P.copy()


class AttitudeEKF:
    """
    4D Extended Kalman Filter for attitude (quaternion).
    
    Fuses gyro integration with accelerometer and magnetometer corrections.
    """
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        
        # State: quaternion [qw, qx, qy, qz]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Covariance (4×4) - quaternion uncertainty
        self.P = np.eye(4) * 0.01  # ±5.7°
        
        # Process noise
        self.Q = np.eye(4) * 0.001
        
        # Accel correction gains
        self.ka = 0.01  # Accelerometer correction
        self.km = 0.01  # Magnetometer correction
    
    def predict(self, gyro: np.ndarray) -> None:
        """
        Predict quaternion using gyro (angular velocity).
        
        Args:
            gyro: Angular velocity [ωx, ωy, ωz] in rad/s
        """
        # Quaternion derivative: dq/dt = 0.5 * q ⊗ ω
        omega = np.array([0, *gyro])
        dq = 0.5 * self._quat_mult(self.q, omega)
        
        # Update quaternion
        self.q = self.q + dq * self.dt
        
        # Normalize
        self.q = self.q / (np.linalg.norm(self.q) + 1e-9)
        
        # Simplified covariance update (no Jacobian)
        self.P = self.P + self.Q
    
    def update_accel(self, accel: np.ndarray) -> None:
        """
        Correct attitude using accelerometer (roll and pitch only).
        
        Args:
            accel: Acceleration [ax, ay, az] in m/s²
        """
        # Normalize accelerometer
        accel_norm = accel / (np.linalg.norm(accel) + 1e-9)
        
        # Expected direction (down = [0, 0, -1])
        a_expected = np.array([0, 0, 1])
        
        # Error: cross product
        error = np.cross(accel_norm, a_expected)
        
        # Apply correction (only roll and pitch)
        dq = self.ka * np.array([0, *error])
        self.q = self.q + dq
        self.q = self.q / (np.linalg.norm(self.q) + 1e-9)
    
    def update_mag(self, mag: np.ndarray) -> None:
        """
        Correct yaw using magnetometer.
        
        Args:
            mag: Magnetic field [mx, my, mz]
        """
        # Normalize magnetometer
        mag_norm = mag / (np.linalg.norm(mag) + 1e-9)
        
        # Expected direction (north in body frame)
        m_expected = np.array([1, 0, 0])
        
        # Error: cross product
        error = np.cross(mag_norm, m_expected)
        
        # Apply correction (mainly yaw)
        dq = self.km * np.array([0, *error])
        self.q = self.q + dq
        self.q = self.q / (np.linalg.norm(self.q) + 1e-9)
    
    def get_state(self) -> np.ndarray:
        """Get quaternion."""
        return self.q.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get 4×4 covariance."""
        return self.P.copy()
    
    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        qw, qx, qy, qz = self.q
        x, y, z = v
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R @ np.array([x, y, z])


class AngularVelocityFilter:
    """
    Simple filter for angular velocity (direct gyro readings with bias estimation).
    """
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.omega = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.P = np.eye(3) * 0.001  # Bias uncertainty
    
    def predict(self, gyro: np.ndarray) -> None:
        """Update angular velocity estimate."""
        self.omega = gyro - self.gyro_bias
    
    def estimate_bias(self, stationary_gyro: np.ndarray, alpha: float = 0.1) -> None:
        """
        Estimate gyro bias when drone is stationary.
        
        Args:
            stationary_gyro: Gyro reading during stationary period
            alpha: Low-pass filter coefficient
        """
        self.gyro_bias = (1 - alpha) * self.gyro_bias + alpha * stationary_gyro
    
    def get_state(self) -> np.ndarray:
        """Get angular velocity."""
        return self.omega.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get 3×3 covariance."""
        return self.P.copy()


class DroneStateEstimator:
    """
    Complete 18D drone state estimation system.
    
    Aggregates three specialized filters:
    - Position/Velocity (6D EKF)
    - Attitude (4D EKF)
    - Angular Velocity (3D)
    Plus direct measurements:
    - Battery (1D)
    - Health (1D)
    """
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        
        # Three specialized filters
        self.pv_filter = PositionVelocityEKF(dt)
        self.att_filter = AttitudeEKF(dt)
        self.av_filter = AngularVelocityFilter(dt)
        
        # Direct measurements
        self.battery = 5000.0  # Wh
        self.health = 1.0      # normalized [0, 1]
        self.max_battery = 5000.0
        
        # State history for diagnostics
        self.state_history = []
        self.covariance_history = []
    
    def predict(self, imu_accel: np.ndarray, imu_gyro: np.ndarray,
                power_draw: float = 0.0) -> None:
        """
        Prediction step from IMU measurements.
        
        Args:
            imu_accel: Accelerometer reading [ax, ay, az] (m/s²) in body frame
            imu_gyro: Gyro reading [ωx, ωy, ωz] (rad/s) in body frame
            power_draw: Power consumption (Watts)
        """
        try:
            # Rotate acceleration from body to world frame
            accel_world = self.att_filter.rotate_vector(imu_accel)
            accel_world[2] -= 9.81  # Remove gravity
            
            # Update all filters
            self.pv_filter.predict(accel_world)
            self.att_filter.predict(imu_gyro)
            self.av_filter.predict(imu_gyro)
            
            # Update battery (simple discharge model)
            self.battery = max(0, self.battery - power_draw * self.dt / 3600)
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
    
    def update_gps(self, gps_pos: np.ndarray, gps_vel: Optional[np.ndarray] = None,
                   pos_noise: float = 5.0, vel_noise: float = 1.0) -> None:
        """
        Update with GPS measurement.
        
        Args:
            gps_pos: GPS position [x, y, z]
            gps_vel: GPS velocity [vx, vy, vz] (if RTK available)
            pos_noise: Position measurement noise (m)
            vel_noise: Velocity measurement noise (m/s)
        """
        try:
            if gps_vel is None:
                gps_vel = self.pv_filter.x[3:6]  # Use previous estimate
            
            # Update PV filter
            self.pv_filter.update_gps(gps_pos, gps_vel)
            
        except Exception as e:
            logger.error(f"Error in GPS update: {e}")
    
    def update_magnetometer(self, mag_field: np.ndarray) -> None:
        """
        Update attitude with magnetometer.
        
        Args:
            mag_field: Magnetometer reading [mx, my, mz]
        """
        try:
            self.att_filter.update_mag(mag_field)
        except Exception as e:
            logger.error(f"Error in magnetometer update: {e}")
    
    def update_accel_static(self, imu_accel: np.ndarray) -> None:
        """
        Correct roll and pitch using accelerometer (when approximately static).
        
        Args:
            imu_accel: Accelerometer reading [ax, ay, az]
        """
        try:
            self.att_filter.update_accel(imu_accel)
        except Exception as e:
            logger.error(f"Error in accel static update: {e}")
    
    def update_battery(self, battery_charge: float) -> None:
        """
        Update battery from fuel gauge.
        
        Args:
            battery_charge: Battery charge (Wh)
        """
        # Light correction of battery estimate
        self.battery = 0.95 * self.battery + 0.05 * battery_charge
        self.battery = np.clip(self.battery, 0, self.max_battery)
    
    def update_health(self, health_estimate: float) -> None:
        """
        Update health from motor/system sensors.
        
        Args:
            health_estimate: Health estimate [0, 1]
        """
        self.health = np.clip(health_estimate, 0, 1)
    
    def estimate_gyro_bias(self, stationary_gyro: np.ndarray, alpha: float = 0.1) -> None:
        """
        Estimate and remove gyro bias during initialization.
        
        Args:
            stationary_gyro: Gyro reading during stationary calibration
            alpha: Filter coefficient
        """
        self.av_filter.estimate_bias(stationary_gyro, alpha)
    
    def get_drone_state(self) -> DroneState:
        """Get complete 18D drone state."""
        pos, vel = self.pv_filter.get_state()
        q = self.att_filter.get_state()
        omega = self.av_filter.get_state()
        
        # Get acceleration from PV filter state
        accel_world = self.pv_filter.x[3:6]
        
        return DroneState(
            position=pos,
            velocity=vel,
            acceleration=accel_world,
            quaternion=q,
            angular_velocity=omega,
            battery_energy=self.battery,
            health=self.health
        )
    
    def get_covariance_18d(self) -> np.ndarray:
        """Get full 18×18 covariance matrix."""
        P = np.zeros((18, 18))
        
        # Position/Velocity: 6×6
        P[0:6, 0:6] = self.pv_filter.get_covariance()
        
        # Acceleration: assume similar to velocity uncertainty
        P[6:9, 6:9] = self.pv_filter.P[3:6, 3:6] / 10.0
        
        # Attitude (quaternion): 4×4
        P[9:13, 9:13] = self.att_filter.get_covariance()
        
        # Angular Velocity: 3×3
        P[13:16, 13:16] = self.av_filter.get_covariance()
        
        # Battery: 1×1
        P[16, 16] = 10.0  # ±√10 Wh
        
        # Health: 1×1
        P[17, 17] = 0.01  # ±0.1
        
        return P
    
    def get_uncertainty(self) -> Dict[str, float]:
        """Get 1-sigma uncertainty for each component."""
        P = self.get_covariance_18d()
        
        return {
            'position_m': np.sqrt(np.mean(np.diag(P)[0:3])),
            'velocity_ms': np.sqrt(np.mean(np.diag(P)[3:6])),
            'acceleration_ms2': np.sqrt(np.mean(np.diag(P)[6:9])),
            'quaternion_norm': np.sqrt(np.mean(np.diag(P)[9:13])),
            'angular_velocity_rads': np.sqrt(np.mean(np.diag(P)[13:16])),
            'battery_wh': np.sqrt(P[16, 16]),
            'health': np.sqrt(P[17, 17])
        }
    
    def save_state_history(self, max_history: int = 1000) -> None:
        """Store current state in history for analysis."""
        if len(self.state_history) >= max_history:
            self.state_history.pop(0)
            self.covariance_history.pop(0)
        
        self.state_history.append(self.get_drone_state())
        self.covariance_history.append(self.get_covariance_18d())
    
    def is_healthy(self, max_position_uncertainty: float = 50.0) -> bool:
        """
        Check if state estimation is healthy.
        
        Args:
            max_position_uncertainty: Maximum acceptable position uncertainty (m)
        
        Returns:
            True if healthy, False otherwise
        """
        unc = self.get_uncertainty()
        
        checks = [
            unc['position_m'] < max_position_uncertainty,
            unc['velocity_ms'] < 100.0,
            unc['quaternion_norm'] < 0.5,
            self.battery > 0.1 * self.max_battery,
            self.health > 0.1
        ]
        
        return all(checks)
