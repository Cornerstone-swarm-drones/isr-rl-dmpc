"""
Core data structures for ISR system.

Defines DroneState, TargetState, and MissionState classes.
Integrates with utility modules for quaternion operations and visualizations.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class DroneState:
    """
    State Representation of a single drone in the swarm.

    Attributes:
        position (np.ndarray): 3D position [x, y, z] (m)
        velocity (np.ndarray): 3D velocity [vx, vy, vz] (m/s)
        acceleration (np.ndarray): 3D acceleration [ax, ay, az] (m/s^2)
        quaternion (np.ndarray): Unit quaternion for attitude [qw, qx, qy, qz]
        angular_velocity (np.ndarray): 3D angular velocity [wx, wy, wz] (rad/s)
        battery_energy (float): Available energy [0, E_max] (Wh)
        health (float): Structural health [0, H_max] (normalized 0-1)
        last_update (float): Simulation time of the last update (s)
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery_energy: float = 5000.0
    health: float = 1.0
    last_update: float = 0.0

    def __post_init__(self):
        """Validate and normalize parameters."""
        assert self.position.shape == (3,), f"Position shape must be (3,), got {self.position.shape}"
        assert self.velocity.shape == (3,), f"Velocity shape must be (3,), got {self.velocity.shape}"
        assert self.acceleration.shape == (3,), f"Acceleration shape must be (3,), got {self.acceleration.shape}"
        assert self.quaternion.shape == (4,), f"Quaternion shape must be (4,), got {self.quaternion.shape}"
        assert self.angular_velocity.shape == (3,), f"Angular velocity shape must be (3,), got {self.angular_velocity.shape}"

        # Normalize unit quaternion
        q_norm = np.linalg.norm(self.quaternion)
        if q_norm > 0:
            self.quaternion = self.quaternion / q_norm

        # Clip battery energy and health to valid ranges
        self.battery_energy = np.clip(self.battery_energy, 0, 5000)
        self.health = np.clip(self.health, 0, 1)

    def to_vector(self, normalize: bool = False) -> np.ndarray:
        """
        Flatten the state into a single 1D array for RL algorithms.

        Args:
            normalize: If True, scales values to approximately [-1, 1] for better convergence

        Returns:
            State Vector (18,): [pos(3), vel(3), acc(3), q(4), ang_vel(3), battery(1), health(1)]
        """
        state_vector = np.concatenate([
            self.position,
            self.velocity,
            self.acceleration,
            self.quaternion,
            self.angular_velocity,
            np.array([self.battery_energy, self.health]),
        ])

        if normalize:
            state_vector_norm = np.zeros_like(state_vector)
            state_vector_norm[0:3] = np.clip(self.position / 500., -1., 1.)
            state_vector_norm[3:6] = np.clip(self.velocity / 20., -1., 1.)
            state_vector_norm[6:9] = np.clip(self.acceleration / 10., -1., 1.)
            state_vector_norm[9] = 2 * (self.battery_energy / 5000.) - 1.
            state_vector_norm[10] = 2 * self.health - 1.
            state_vector_norm[11:15] = self.quaternion  # Quaternion already normalized
            state_vector_norm[15:18] = np.clip(self.angular_velocity / 2., -1., 1.)
            return state_vector_norm

        return state_vector

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization and logging."""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "quaternion": self.quaternion.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "battery_energy": float(self.battery_energy),
            "health": float(self.health),
            "last_update": float(self.last_update)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DroneState":
        """Reconstruct DroneState from dictionary."""
        return cls(
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            acceleration=np.array(data["acceleration"]),
            quaternion=np.array(data["quaternion"]),
            angular_velocity=np.array(data["angular_velocity"]),
            battery_energy=data["battery_energy"],
            health=data["health"],
            last_update=data["last_update"],
        )

    def __eq__(self, other: "DroneState") -> bool:
        """Check equality for floating point with tolerance."""
        if not isinstance(other, DroneState):
            return False

        tol = 1e-6
        return (
            np.allclose(self.position, other.position, atol=tol) and
            np.allclose(self.velocity, other.velocity, atol=tol) and
            np.allclose(self.acceleration, other.acceleration, atol=tol) and
            np.allclose(self.quaternion, other.quaternion, atol=tol) and
            np.allclose(self.angular_velocity, other.angular_velocity, atol=tol) and
            np.isclose(self.battery_energy, other.battery_energy, atol=tol) and
            np.isclose(self.health, other.health, atol=tol) and
            np.isclose(self.last_update, other.last_update, atol=tol)
        )

    def __repr__(self) -> str:
        """String Representation."""
        return (
            f"DroneState(pos={self.position}, vel={self.velocity}, "
            f"E={self.battery_energy:.1f}Wh, H={self.health:.2f}, t={self.last_update:.2f}s)"
        )

    def copy(self) -> "DroneState":
        """Deep copy of drone state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            battery_energy=self.battery_energy,
            health=self.health,
            last_update=self.last_update,
        )


@dataclass
class TargetState:
    """
    State Representation of a detected target.

    Attributes:
        position (np.ndarray): 3D position [x, y, z] (m)
        velocity (np.ndarray): 3D velocity [vx, vy, vz] (m/s)
        acceleration (np.ndarray): 3D acceleration [ax, ay, az] (m/s^2)
        yaw_angle (float): Heading angle in radians, normalized to [0, 2π)
        yaw_rate (float): Rate of change of heading (rad/s)
        classification_confidence (float): Confidence score [-1, 1] (-1: Hostile, 1: Friendly)
        target_id (str): Identifier for the target (hostile, friendly, or unknown)
        threat_score (float): Computed threat level based on classification and dynamics
        covariance (np.ndarray): 11x11 Covariance matrix from Extended Kalman Filter (EKF)
        last_update (float): Timestamp of the last sensor reading (s)
        tracked_duration (float): Total duration the target has been tracked (s)
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_angle: float = 0.0
    yaw_rate: float = 0.0
    classification_confidence: float = 0.0
    target_id: str = "unknown"
    threat_score: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(11))
    last_update: float = 0.0
    tracked_duration: float = 0.0

    def __post_init__(self):
        """Validate shapes and ranges."""
        assert self.position.shape == (3,), f"Position shape must be (3,), got {self.position.shape}"
        assert self.velocity.shape == (3,), f"Velocity shape must be (3,), got {self.velocity.shape}"
        assert self.acceleration.shape == (3,), f"Acceleration shape must be (3,), got {self.acceleration.shape}"
        assert self.covariance.shape == (11, 11), f"Covariance shape must be (11,11), got {self.covariance.shape}"

        # Normalize parameters
        self.classification_confidence = np.clip(self.classification_confidence, -1., 1.)
        self.yaw_angle = self.yaw_angle % (2 * np.pi)

    def to_vector(self) -> np.ndarray:
        """
        Flatten the state (excluding covariance) into a 11-dimensional vector for RL.

        Returns:
            State Vector (12,): [pos(3), vel(3), acc(3), yaw(1), yaw_rate(1)]
        """
        return np.concatenate([
            self.position,
            self.velocity,
            self.acceleration,
            np.array([self.yaw_angle, self.yaw_rate])
        ])
    
    @classmethod
    def from_vector(cls, state: np.ndarray) -> "TargetState":
        """Reconstruct TargetState from an 11D EKF state vector."""
        assert state.shape == (11,), f"Expected 11D state, got {state.shape}"

        return cls(
            position=state[0:3].copy(),
            velocity=state[3:6].copy(),
            acceleration=state[6:9].copy(),
            yaw_angle=float(state[9]),
            yaw_rate=float(state[10]),

            # The EKF does not propagate these; keep sensible defaults.
            classification_confidence=0.0,
            target_id="unknown",
            threat_score=0.0,
            covariance=np.eye(11),
            last_update=0.0,
            tracked_duration=0.0,
        )


    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "yaw_angle": float(self.yaw_angle),
            "yaw_rate": float(self.yaw_rate),
            "classification_confidence": float(self.classification_confidence),
            "target_id": self.target_id,
            "threat_score": float(self.threat_score),
            "covariance": self.covariance.tolist(),
            "last_update": float(self.last_update),
            "tracked_duration": float(self.tracked_duration),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TargetState":
        """Reconstruct TargetState from dictionary."""
        return cls(
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            acceleration=np.array(data["acceleration"]),
            yaw_angle=data["yaw_angle"],
            yaw_rate=data["yaw_rate"],
            classification_confidence=data["classification_confidence"],
            target_id=data["target_id"],
            threat_score=data["threat_score"],
            covariance=np.array(data["covariance"]),
            last_update=data["last_update"],
            tracked_duration=data["tracked_duration"],
        )

    def __eq__(self, other: "TargetState") -> bool:
        """Check equality for floating point with tolerance."""
        if not isinstance(other, TargetState):
            return False

        tol = 1e-6
        return (
            np.allclose(self.position, other.position, atol=tol) and
            np.allclose(self.velocity, other.velocity, atol=tol) and
            np.allclose(self.acceleration, other.acceleration, atol=tol) and
            np.isclose(self.yaw_angle, other.yaw_angle, atol=tol) and
            np.isclose(self.yaw_rate, other.yaw_rate, atol=tol) and
            np.isclose(self.classification_confidence, other.classification_confidence, atol=tol) and
            self.target_id == other.target_id and
            np.isclose(self.threat_score, other.threat_score, atol=tol) and
            np.allclose(self.covariance, other.covariance, atol=tol) and
            np.isclose(self.last_update, other.last_update, atol=tol) and
            np.isclose(self.tracked_duration, other.tracked_duration, atol=tol)
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TargetState(pos={self.position}, conf={self.classification_confidence:.2f}, "
            f"id={self.target_id}, threat={self.threat_score:.1f})"
        )

    def copy(self) -> "TargetState":
        """Deep copy of TargetState."""
        return TargetState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            yaw_angle=self.yaw_angle,
            yaw_rate=self.yaw_rate,
            classification_confidence=self.classification_confidence,
            target_id=self.target_id,
            threat_score=self.threat_score,
            covariance=self.covariance.copy(),
            last_update=self.last_update,
            tracked_duration=self.tracked_duration,
        )


@dataclass
class MissionState:
    """
    Global mission state including area coverage and objectives.

    Attributes:
        area_boundary (np.ndarray): Polygon vertices of the search area (Nx2)
        waypoints (List[np.ndarray]): List of 2D/3D points the swarm must visit
        coverage_matrix (np.ndarray): Binary array representing grid cells (True = Covered)
        elapsed_time (float): Time since mission start (s)
        mission_duration (float): Maximum allowed time for the mission (s)
        coverage_efficiency (float): Measure of coverage effectiveness (0-1)
    """

    area_boundary: np.ndarray = field(
        default_factory=lambda: np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    )
    waypoints: List[np.ndarray] = field(default_factory=list)
    coverage_matrix: np.ndarray = field(default_factory=lambda: np.zeros(100, dtype=bool))
    elapsed_time: float = 0.0
    mission_duration: float = 3600.0
    coverage_efficiency: float = 0.0

    def __post_init__(self):
        """Validate shapes."""
        assert self.area_boundary.ndim == 2 and self.area_boundary.shape[1] == 2, \
            f"area_boundary must be Nx2, got shape {self.area_boundary.shape}"
        assert self.coverage_matrix.ndim == 1, \
            f"coverage_matrix must be 1D, got shape {self.coverage_matrix.shape}"
        assert self.coverage_matrix.dtype == bool, \
            f"coverage_matrix must be bool dtype, got {self.coverage_matrix.dtype}"

        self.elapsed_time = max(0, self.elapsed_time)
        self.mission_duration = max(1, self.mission_duration)
        self.coverage_efficiency = np.clip(self.coverage_efficiency, 0, 1)

    def get_coverage_percentage(self) -> float:
        """
        Calculate current coverage percentage.

        Returns:
            Coverage percentage (0-100)
        """
        if len(self.coverage_matrix) == 0:
            return 0.0
        return 100.0 * np.sum(self.coverage_matrix) / len(self.coverage_matrix)

    def get_mission_progress(self) -> float:
        """
        Calculate mission time progress (0-1).

        Returns:
            Mission progress ratio
        """
        if self.mission_duration <= 0:
            return 0.0
        return np.clip(self.elapsed_time / self.mission_duration, 0, 1)

    def add_waypoint(self, waypoint: np.ndarray):
        """
        Add a new navigation goal to the mission.

        Args:
            waypoint: 2D or 3D coordinate
        """
        assert waypoint.shape in [(2,), (3,)], f"Waypoint shape must be (2,) or (3,), got {waypoint.shape}"
        self.waypoints.append(waypoint.copy())

    def mark_cell_covered(self, cell_idx: int):
        """
        Sets a specific index in the coverage_matrix to True.

        Args:
            cell_idx: Index of the grid cell to mark as covered
        """
        if 0 <= cell_idx < len(self.coverage_matrix):
            self.coverage_matrix[cell_idx] = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "area_boundary": self.area_boundary.tolist(),
            "waypoints": [w.tolist() for w in self.waypoints],
            "coverage_matrix": self.coverage_matrix.tolist(),
            "elapsed_time": float(self.elapsed_time),
            "mission_duration": float(self.mission_duration),
            "coverage_efficiency": float(self.coverage_efficiency),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MissionState":
        """Reconstruct MissionState from dictionary."""
        return cls(
            area_boundary=np.array(data["area_boundary"]),
            waypoints=[np.array(w) for w in data.get("waypoints", [])],
            coverage_matrix=np.array(data["coverage_matrix"], dtype=bool),
            elapsed_time=data["elapsed_time"],
            mission_duration=data["mission_duration"],
            coverage_efficiency=data.get("coverage_efficiency", 0.0),
        )

    def __eq__(self, other: "MissionState") -> bool:
        """Check equality for floating point with tolerance."""
        if not isinstance(other, MissionState):
            return False

        tol = 1e-6
        return (
            np.allclose(self.area_boundary, other.area_boundary, atol=tol) and
            len(self.waypoints) == len(other.waypoints) and
            all(np.allclose(w1, w2, atol=tol) for w1, w2 in zip(self.waypoints, other.waypoints)) and
            np.array_equal(self.coverage_matrix, other.coverage_matrix) and
            np.isclose(self.elapsed_time, other.elapsed_time, atol=tol) and
            np.isclose(self.mission_duration, other.mission_duration, atol=tol) and
            np.isclose(self.coverage_efficiency, other.coverage_efficiency, atol=tol)
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MissionState(area=[{self.area_boundary.shape[0]} vertices], "
            f"waypoints={len(self.waypoints)}, coverage={self.get_coverage_percentage():.1f}%, "
            f"time={self.elapsed_time:.1f}s/{self.mission_duration:.1f}s, eff={self.coverage_efficiency:.2f})"
        )

    def copy(self) -> "MissionState":
        """Deep copy of MissionState."""
        return MissionState(
            area_boundary=self.area_boundary.copy(),
            waypoints=[w.copy() for w in self.waypoints],
            coverage_matrix=self.coverage_matrix.copy(),
            elapsed_time=self.elapsed_time,
            mission_duration=self.mission_duration,
            coverage_efficiency=self.coverage_efficiency,
        )
