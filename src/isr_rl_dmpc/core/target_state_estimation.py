"""
Extended Kalman Filter for target tracking (11D).

Multi-sensor fusion for:
- Radar (4D: range, range_rate, azimuth, elevation)
- Optical (2-3D: azimuth, elevation, range)
- RF Fingerprinting (3D: x_est, y_est, z_est)
- Acoustic TDOA (3D: x_est, y_est, z_est)

Adaptive weighting based on sensor confidence.

Result: 11D target state with 11x11 covariance per target
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

from .data_structures import TargetState

class SensorType(Enum):
    """Supported sensor types."""
    RADAR = "radar"
    OPTICAL = "optical"
    RF_FINGERPRINT = "rf_fingerprint"
    ACOUSTIC_TDOA = "acoustic_tdoa"



@dataclass
class RadarMeasurement:
    """Radar measurement (4D)."""
    range: float  # meters
    range_rate: float  # m/s
    azimuth: float  # radians
    elevation: float  # radians
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 4D measurement vector."""
        return np.array([self.range, self.range_rate, self.azimuth, self.elevation])


@dataclass
class OpticalMeasurement:
    """Optical bearing measurement (2D or 3D)."""
    azimuth: float  # radians
    elevation: float  # radians
    range: Optional[float] = None  # meters (optional)
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to measurement vector (2D or 3D)."""
        if self.range is not None:
            return np.array([self.azimuth, self.elevation, self.range])
        else:
            return np.array([self.azimuth, self.elevation])


@dataclass
class RFMeasurement:
    """RF fingerprinting measurement (3D position estimate)."""
    x_est: float  # meters
    y_est: float  # meters
    z_est: float  # meters
    confidence: float  # [0, 1]
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 3D measurement vector."""
        return np.array([self.x_est, self.y_est, self.z_est])


@dataclass
class AcousticMeasurement:
    """Acoustic TDOA measurement (3D position from hyperbolic trilateration)."""
    x_est: float  # meters
    y_est: float  # meters
    z_est: float  # meters
    confidence: float  # [0, 1]
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 3D measurement vector."""
        return np.array([self.x_est, self.y_est, self.z_est])


class TargetTrackingEKF:
    """
    11D Extended Kalman Filter for a single target.
    
    State: [x, y, z, vx, vy, vz, ax, ay, az, ψ, ψ̇]
    Matrices: 11x11
    """
    
    def __init__(self, target_id: str, initial_state: TargetState,
                 dt: float = 0.02, process_noise_q: float = 0.1):
        """
        Initialize target tracking EKF.
        
        Args:
            target_id: Unique target identifier
            initial_state: Initial 11D target state estimate
            dt: Time step (seconds)
            process_noise_q: Process noise standard deviation
        """
        self.target_id = target_id
        self.dt = dt
        self.last_update = 0.0
        
        # State: 11D
        self.x = initial_state.to_vector()
        
        # Covariance: 11×11
        self.P = np.eye(11)
        self._init_covariance()
        
        # Process noise: 11×11
        self.Q = np.eye(11) * (process_noise_q ** 2)
        self._init_process_noise()
        
        # Measurement histories for adaptive weighting
        self.measurement_history = {
            SensorType.RADAR: [],
            SensorType.OPTICAL: [],
            SensorType.RF_FINGERPRINT: [],
            SensorType.ACOUSTIC_TDOA: []
        }
        self.max_history = 20
        
        # Sensor-specific parameters
        self.sensor_params = self._init_sensor_params()
    
    def _init_covariance(self) -> None:
        """Initialize covariance matrix based on state uncertainty."""
        # Position uncertainty (m²)
        self.P[0:3, 0:3] = np.eye(3) * 100.0  # ±10m
        
        # Velocity uncertainty ((m/s)²)
        self.P[3:6, 3:6] = np.eye(3) * 10.0  # ±3 m/s
        
        # Acceleration uncertainty ((m/s²)²)
        self.P[6:9, 6:9] = np.eye(3) * 1.0  # ±1 m/s²
        
        # Yaw uncertainty (rad²)
        self.P[9, 9] = 1.0  # ±1 rad ≈ ±57°
        
        # Yaw rate uncertainty ((rad/s)²)
        self.P[10, 10] = 0.1  # ±0.3 rad/s
    
    def _init_process_noise(self) -> None:
        """Initialize process noise (accounts for unmodeled dynamics)."""
        # Position drift
        self.Q[0:3, 0:3] = np.eye(3) * 0.001
        
        # Velocity uncertainty
        self.Q[3:6, 3:6] = np.eye(3) * 0.01
        
        # Acceleration uncertainty
        self.Q[6:9, 6:9] = np.eye(3) * 0.1
        
        # Yaw uncertainty
        self.Q[9, 9] = 0.001
        
        # Yaw rate uncertainty
        self.Q[10, 10] = 0.0001
    
    def _init_sensor_params(self) -> Dict:
        """Initialize sensor-specific parameters."""
        return {
            SensorType.RADAR: {
                'measurement_noise': np.diag([0.5, 0.5, 0.1, 0.1]),  # (4×4)
                'min_confidence': 0.5,
                'max_range': 10000.0  # meters
            },
            SensorType.OPTICAL: {
                'measurement_noise_2d': np.diag([0.05, 0.05]),  # (2×2)
                'measurement_noise_3d': np.diag([0.05, 0.05, 10.0]),  # (3×3)
                'min_confidence': 0.3,
                'max_range': 5000.0
            },
            SensorType.RF_FINGERPRINT: {
                'measurement_noise': np.eye(3) * 50.0,  # (3×3)
                'min_confidence': 0.4,
                'max_range': 1000.0
            },
            SensorType.ACOUSTIC_TDOA: {
                'measurement_noise': np.eye(3) * 100.0,  # (3×3)
                'min_confidence': 0.3,
                'max_range': 2000.0
            }
        }
    
    def predict(self) -> None:
        """
        Predict step using constant acceleration model.
        
        State transition:
            x̂⁻ = F @ x̂⁺
            P⁻ = F @ P⁺ @ F^T + Q
        """
        # Process matrix F (11×11)
        F = self._compute_process_matrix()
        
        # Update state
        self.x = F @ self.x
        
        # Wrap yaw to [0, 2π]
        self.x[9] = self._normalize_angle(self.x[9])
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure positive definite
        self.P = (self.P + self.P.T) / 2
    
    def _compute_process_matrix(self) -> np.ndarray:
        """Compute 11×11 state transition matrix."""
        F = np.eye(11)
        
        # Position <- velocity and acceleration
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = np.eye(3) * (0.5 * self.dt**2)
        
        # Velocity <- acceleration
        F[3:6, 6:9] = np.eye(3) * self.dt
        
        # Yaw <- yaw_rate
        F[9, 10] = self.dt
        
        return F
    
    def update_radar(self, measurement: RadarMeasurement, sensor_pos: np.ndarray) -> None:
        """
        Update with radar measurement (4D: range, range_rate, azimuth, elevation).
        
        Args:
            measurement: Radar measurement
            sensor_pos: Sensor position [x, y, z]
        """
        z = measurement.to_vector()
        
        # Measurement matrix H (4×11)
        H = self._compute_radar_measurement_matrix(sensor_pos)
        
        # Measurement covariance R (4×4)
        R = self.sensor_params[SensorType.RADAR]['measurement_noise']
        
        # Perform EKF update
        self._ekf_update(z, H, R)
        
        # Store in history
        self._add_to_measurement_history(SensorType.RADAR, measurement)
    
    def update_optical(self, measurement: OpticalMeasurement, sensor_pos: np.ndarray) -> None:
        """
        Update with optical bearing measurement (2D or 3D).
        
        Args:
            measurement: Optical measurement
            sensor_pos: Sensor position [x, y, z]
        """
        z = measurement.to_vector()
        is_3d = measurement.range is not None
        
        # Measurement matrix H (2×11 or 3×11)
        H = self._compute_optical_measurement_matrix(sensor_pos, is_3d)
        
        # Measurement covariance R
        if is_3d:
            R = self.sensor_params[SensorType.OPTICAL]['measurement_noise_3d']
        else:
            R = self.sensor_params[SensorType.OPTICAL]['measurement_noise_2d']
        
        # Perform EKF update
        self._ekf_update(z, H, R)
        
        # Store in history
        self._add_to_measurement_history(SensorType.OPTICAL, measurement)
    
    def update_rf(self, measurement: RFMeasurement) -> None:
        """
        Update with RF fingerprinting position estimate (3D).
        
        Args:
            measurement: RF fingerprinting measurement
        """
        z = measurement.to_vector()
        
        # Measurement matrix H (3×11) - observe position only
        H = np.zeros((3, 11))
        H[0:3, 0:3] = np.eye(3)
        
        # Measurement covariance R (3×3)
        # Scale by confidence
        R = self.sensor_params[SensorType.RF_FINGERPRINT]['measurement_noise']
        R = R / (measurement.confidence + 0.1)
        
        # Perform EKF update
        self._ekf_update(z, H, R)
        
        # Store in history
        self._add_to_measurement_history(SensorType.RF_FINGERPRINT, measurement)
    
    def update_acoustic(self, measurement: AcousticMeasurement) -> None:
        """
        Update with acoustic TDOA position estimate (3D).
        
        Args:
            measurement: Acoustic TDOA measurement
        """
        z = measurement.to_vector()
        
        # Measurement matrix H (3×11) - observe position only
        H = np.zeros((3, 11))
        H[0:3, 0:3] = np.eye(3)
        
        # Measurement covariance R (3×3)
        # Scale by confidence
        R = self.sensor_params[SensorType.ACOUSTIC_TDOA]['measurement_noise']
        R = R / (measurement.confidence + 0.1)
        
        # Perform EKF update
        self._ekf_update(z, H, R)
        
        # Store in history
        self._add_to_measurement_history(SensorType.ACOUSTIC_TDOA, measurement)
    
    def update_adaptive_fusion(self, measurements: Dict[SensorType, list],
                              sensor_pos: np.ndarray) -> None:
        """
        Adaptive multi-sensor fusion with confidence weighting.
        
        Args:
            measurements: Dict mapping sensor type to list of measurements
            sensor_pos: Sensor position [x, y, z]
        """
        # Compute adaptive weights
        weights = self._compute_adaptive_weights(measurements)
        
        # Update with each sensor scaled by weight
        for sensor_type, meas_list in measurements.items():
            if not meas_list or weights[sensor_type] < 1e-6:
                continue
            
            weight = weights[sensor_type]
            
            for measurement in meas_list:
                # Scale measurement covariance by weight
                if sensor_type == SensorType.RADAR:
                    self.update_radar(measurement, sensor_pos)
                elif sensor_type == SensorType.OPTICAL:
                    self.update_optical(measurement, sensor_pos)
                elif sensor_type == SensorType.RF_FINGERPRINT:
                    self.update_rf(measurement)
                elif sensor_type == SensorType.ACOUSTIC_TDOA:
                    self.update_acoustic(measurement)
    
    def _compute_adaptive_weights(self, measurements: Dict[SensorType, list]) -> Dict[SensorType, float]:
        """
        Compute adaptive weights based on sensor confidence and innovation.
        
        Formula: w_s = σ_s^{-2} / Σ σ_s'^{-2}
        
        Returns:
            Dictionary mapping sensor type to normalized weight [0, 1]
        """
        weights = {}
        total_weight = 0
        
        for sensor_type in SensorType:
            if sensor_type not in measurements or not measurements[sensor_type]:
                weights[sensor_type] = 0
                continue
            
            # Recent measurement quality
            history = self.measurement_history.get(sensor_type, [])
            if history:
                # Weight inversely proportional to recent innovation magnitude
                recent_innovation = np.mean([
                    getattr(m, 'confidence', 0.5) for m in history[-3:]
                ])
            else:
                recent_innovation = 0.5
            
            # Inverse variance (higher confidence = higher weight)
            weight = 1.0 / (1.0 - recent_innovation + 0.1)
            weights[sensor_type] = weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for sensor_type in weights:
                weights[sensor_type] /= total_weight
        else:
            weights = {st: 1.0 / len(SensorType) for st in SensorType}
        
        return weights
    
    def _ekf_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """
        Standard EKF update step.
        
        Args:
            z: Measurement vector
            H: Measurement matrix (m×11)
            R: Measurement covariance (m×m)
        """
        # Innovation
        z_hat = H @ self.x
        y = z - z_hat
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-9 * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            logger.warning(f"Singular covariance matrix for target {self.target_id}")
            return
        
        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(11) - K @ H) @ self.P
        
        # Wrap yaw
        self.x[9] = self._normalize_angle(self.x[9])
        
        # Ensure positive definite
        self.P = (self.P + self.P.T) / 2
    
    def _compute_radar_measurement_matrix(self, sensor_pos: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian for radar measurement.
        
        Measurement: [range, range_rate, azimuth, elevation]
        
        Returns:
            4×11 measurement Jacobian
        """
        H = np.zeros((4, 11))
        
        # Relative position
        dx = self.x[0] - sensor_pos[0]
        dy = self.x[1] - sensor_pos[1]
        dz = self.x[2] - sensor_pos[2]
        
        # Range
        r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-9
        
        # ∂r/∂pos
        H[0, 0] = dx / r
        H[0, 1] = dy / r
        H[0, 2] = dz / r
        
        # ∂ṙ/∂pos and ∂ṙ/∂vel
        vx_rel = self.x[3]
        vy_rel = self.x[4]
        vz_rel = self.x[5]
        
        H[1, 0] = -(dx * vx_rel + dy * vy_rel + dz * vz_rel) / (r**2) * (dx / r)
        H[1, 1] = -(dx * vx_rel + dy * vy_rel + dz * vz_rel) / (r**2) * (dy / r)
        H[1, 2] = -(dx * vx_rel + dy * vy_rel + dz * vz_rel) / (r**2) * (dz / r)
        H[1, 3] = dx / r
        H[1, 4] = dy / r
        H[1, 5] = dz / r
        
        # ∂azimuth/∂pos
        rho = np.sqrt(dx**2 + dy**2) + 1e-9
        H[2, 0] = -dy / (rho**2)
        H[2, 1] = dx / (rho**2)
        
        # ∂elevation/∂pos
        H[3, 0] = -dx * dz / (r**2 * rho)
        H[3, 1] = -dy * dz / (r**2 * rho)
        H[3, 2] = rho / (r**2)
        
        return H
    
    def _compute_optical_measurement_matrix(self, sensor_pos: np.ndarray,
                                           is_3d: bool) -> np.ndarray:
        """
        Compute Jacobian for optical measurement.
        
        Measurement: [azimuth, elevation] or [azimuth, elevation, range]
        """
        # Relative position
        dx = self.x[0] - sensor_pos[0]
        dy = self.x[1] - sensor_pos[1]
        dz = self.x[2] - sensor_pos[2]
        
        rho = np.sqrt(dx**2 + dy**2) + 1e-9
        r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-9
        
        if is_3d:
            H = np.zeros((3, 11))
        else:
            H = np.zeros((2, 11))
        
        # ∂azimuth/∂pos
        H[0, 0] = -dy / (rho**2)
        H[0, 1] = dx / (rho**2)
        
        # ∂elevation/∂pos
        H[1, 0] = -dx * dz / (r**2 * rho)
        H[1, 1] = -dy * dz / (r**2 * rho)
        H[1, 2] = rho / (r**2)
        
        # ∂range/∂pos (if 3D)
        if is_3d:
            H[2, 0] = dx / r
            H[2, 1] = dy / r
            H[2, 2] = dz / r
        
        return H
    
    def _add_to_measurement_history(self, sensor_type: SensorType, measurement) -> None:
        """Store measurement in history for analysis."""
        history = self.measurement_history[sensor_type]
        if len(history) >= self.max_history:
            history.pop(0)
        history.append(measurement)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [0, 2π]."""
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        return angle
    
    def get_state(self) -> TargetState:
        """Get 11D target state."""
        return TargetState.from_vector(self.x)
    
    def get_covariance(self) -> np.ndarray:
        """Get 11×11 covariance."""
        return self.P.copy()
    
    def get_uncertainty(self) -> Dict[str, float]:
        """Get 1-sigma uncertainty for each component."""
        return {
            'position_m': np.sqrt(np.mean(np.diag(self.P)[0:3])),
            'velocity_ms': np.sqrt(np.mean(np.diag(self.P)[3:6])),
            'acceleration_ms2': np.sqrt(np.mean(np.diag(self.P)[6:9])),
            'yaw_rad': np.sqrt(self.P[9, 9]),
            'yaw_rate_rads': np.sqrt(self.P[10, 10])
        }
    
    def is_track_healthy(self, max_pos_unc: float = 200.0) -> bool:
        """Check if track is in healthy state."""
        unc = self.get_uncertainty()
        return (
            unc['position_m'] < max_pos_unc and
            unc['velocity_ms'] < 100.0 and
            self.P[9, 9] < 1.0
        )


class TargetStateEstimator:
    """
    Manages tracking of multiple targets.
    
    Maintains separate 11D EKF for each detected target.
    """
    
    def __init__(self, dt: float = 0.02, max_targets: int = 50):
        self.dt = dt
        self.max_targets = max_targets
        self.tracks: Dict[str, TargetTrackingEKF] = {}
        self.track_counter = 0
    
    def create_track(self, initial_state: TargetState) -> str:
        """
        Create new track for detected target.
        
        Returns:
            Target ID
        """
        if len(self.tracks) >= self.max_targets:
            logger.warning(f"Maximum targets ({self.max_targets}) reached")
            return None
        
        target_id = f"target_{self.track_counter:03d}"
        self.track_counter += 1
        
        self.tracks[target_id] = TargetTrackingEKF(
            target_id, initial_state, self.dt
        )
        
        logger.info(f"Created track: {target_id}")
        return target_id
    
    def delete_track(self, target_id: str) -> None:
        """Delete track (target lost)."""
        if target_id in self.tracks:
            del self.tracks[target_id]
            logger.info(f"Deleted track: {target_id}")
    
    def predict_all(self) -> None:
        """Predict step for all targets."""
        for track in self.tracks.values():
            track.predict()
    
    def update_track(self, target_id: str, measurements: Dict[SensorType, list],
                    sensor_pos: np.ndarray) -> bool:
        """
        Update specific track with measurements.
        
        Returns:
            True if successful, False if track not found
        """
        if target_id not in self.tracks:
            return False
        
        self.tracks[target_id].update_adaptive_fusion(measurements, sensor_pos)
        return True
    
    def get_all_estimates(self) -> Dict[str, TargetState]:
        """Get state estimates for all targets."""
        return {
            tid: track.get_state()
            for tid, track in self.tracks.items()
        }
    
    def get_healthy_tracks(self) -> Dict[str, TargetState]:
        """Get only healthy tracks."""
        return {
            tid: track.get_state()
            for tid, track in self.tracks.items()
            if track.is_track_healthy()
        }
    
    def get_all_covariances(self) -> Dict[str, np.ndarray]:
        """Get covariance for all targets."""
        return {
            tid: track.get_covariance()
            for tid, track in self.tracks.items()
        }
