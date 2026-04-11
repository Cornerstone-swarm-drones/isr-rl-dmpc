"""
Physics-based simulator for ISR-RL-DMPC autonomous swarm system.

Provides 6-DOF rigid body dynamics simulation for multi-drone, multi-target
coordination. Includes realistic effects: gravity, wind, battery depletion,
collision detection, and geofence enforcement.

Classes:
    DronePhysics: Individual drone dynamics (rigid body, 4 motors)
    TargetPhysics: Target motion models (constant velocity, acceleration)
    EnvironmentSimulator: Multi-agent coordination and physics simulation
    WindModel: Dryden turbulence wind disturbances
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# For imports in actual implementation
# from isr_rl_dmpc.core import DroneState, TargetState, MissionState
# from isr_rl_dmpc.utils import MathUtils, UnitConversions


class TargetType(Enum):
    """Target classification types."""
    UNKNOWN = 0
    FRIENDLY = 1
    HOSTILE = 2
    NEUTRAL = 3


@dataclass
class DroneConfig:
    """Drone physical parameters — tuned for hector_quadrotor airframe.

    Reference: hector_quadrotor (TU Darmstadt)
    https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor

    mass       = 1.477 kg  (URDF base_link)
    arm_length = 0.215 m
    Ixx = Iyy  = 0.01152 kg·m²
    Izz        = 0.02180 kg·m²
    """
    mass: float = 1.477              # kg
    gravity: float = 9.81            # m/s²
    max_thrust: float = 9.5          # N per motor (4× → 38 N peak)
    max_angular_velocity: float = 2.0  # rad/s
    max_linear_velocity: float = 15.0  # m/s
    battery_capacity: float = 5000.0 * 14.8 * 3.6  # J  (5 Ah × 14.8 V × 3600)
    battery_energy_rate: float = 64.0  # W (hover power estimate)
    min_battery_voltage: float = 12.0  # V (3.0 V/cell × 4S)
    hover_thrust: float = 1.477 * 9.81 / 4  # N per motor ≈ 3.62 N
    inertia_matrix: np.ndarray = None
    motor_constant: float = 8.27e-6  # rad/s per Newton thrust

    def __post_init__(self):
        """Initialize derived parameters."""
        if self.inertia_matrix is None:
            # hector_quadrotor URDF inertia values
            self.inertia_matrix = np.array([
                [0.01152, 0.0, 0.0],
                [0.0, 0.01152, 0.0],
                [0.0, 0.0, 0.02180]
            ])


@dataclass
class TargetConfig:
    """Target motion parameters."""
    max_velocity: float = 20.0  # m/s (for moving targets)
    max_acceleration: float = 2.0  # m/s² (for maneuvering)
    min_detection_distance: float = 100.0  # m
    max_detection_distance: float = 500.0  # m
    classification_error_rate: float = 0.05  # 5% misclassification


@dataclass
class EnvironmentConfig:
    """Environment simulation parameters."""
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³ (sea level)
    wind_speed_mean: float = 5.0  # m/s (avg wind)
    wind_gust_frequency: float = 0.1  # Hz (gust oscillations)
    grid_resolution: float = 100.0  # m (coverage grid cell size)
    geofence_boundary: Tuple[float, float, float] = (2000.0, 2000.0, 500.0)  # x, y, z (m)
    timestep: float = 0.02  # s (50 Hz)
    physics_substeps: int = 10  # RK4 integration substeps


class WindModel:
    """Dryden turbulence wind model for atmospheric disturbances."""

    def __init__(self, config: EnvironmentConfig, seed: Optional[int] = None):
        """
        Initialize wind model.

        Args:
            config: Environment configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Wind state (continuous)
        self.wind_north = 0.0  # m/s
        self.wind_east = 0.0  # m/s
        self.wind_up = 0.0  # m/s
        
        # Wind rates (for Dryden model)
        self.wind_rate_north = 0.0
        self.wind_rate_east = 0.0
        self.wind_rate_up = 0.0
        
        # Dryden filter parameters
        self.L_u = 200.0  # m (length scale)
        self.sigma_u = 2.0  # m/s (intensity)

    def update(self, dt: float) -> np.ndarray:
        """
        Update wind field using Dryden turbulence model.

        Args:
            dt: Time step (s)

        Returns:
            Wind velocity vector [u_wind, v_wind, w_wind] (m/s)
        """
        # Dryden model: continuous shaping filter
        a = -self.config.wind_gust_frequency
        b = self.config.wind_gust_frequency * self.rng.randn()
        
        # Update wind state
        self.wind_north = self.wind_north + a * self.wind_north * dt + b * dt
        self.wind_east = self.wind_east + a * self.wind_east * dt + b * dt
        self.wind_up = self.wind_up + a * self.wind_up * dt + b * dt
        
        # Add mean wind component
        mean_wind = self.config.wind_speed_mean
        wind_velocity = np.array([
            mean_wind + self.wind_north,
            self.wind_east,
            self.wind_up
        ])
        
        return np.clip(wind_velocity, -50.0, 50.0)  # Clip to realistic bounds


class DronePhysics:
    """6-DOF rigid body physics for individual drone."""

    def __init__(self, drone_id: int, config: DroneConfig, seed: Optional[int] = None):
        """
        Initialize drone physics engine.

        Args:
            drone_id: Unique drone identifier
            config: Drone configuration parameters
            seed: Random seed for reproducibility
        """
        self.drone_id = drone_id
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # State variables
        self.position = np.zeros(3)  # x, y, z (m)
        self.velocity = np.zeros(3)  # dx, dy, dz (m/s)
        self.acceleration = np.zeros(3)  # ddx, ddy, ddz (m/s²)
        
        # Attitude (quaternion)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.angular_velocity = np.zeros(3)  # p, q, r (rad/s)
        self.angular_acceleration = np.zeros(3)  # rad/s²
        
        # Motor states
        self.motor_speeds = np.zeros(4)  # rad/s for 4 motors
        self.motor_thrusts = np.zeros(4)  # N for 4 motors
        
        # Battery state
        self.battery_energy = config.battery_capacity  # J
        self.battery_health = 1.0  # 0-1 scale
        self.battery_cycles = 0
        
        # Health tracking
        self.health = 1.0  # 0-1 scale
        self.is_active = True
        
        # Sensor noise
        self.last_wind_effect = np.zeros(3)

    def motor_command_to_thrust(self, commands: np.ndarray) -> None:
        """
        Convert motor commands to thrust forces.

        Args:
            commands: Motor PWM commands [0, 1] for 4 motors
        """
        # Clamp commands
        commands = np.clip(commands, 0.0, 1.0)
        
        # Max thrust per motor
        thrust_per_motor = commands * self.config.max_thrust
        
        # Store motor speeds (proportional to thrust)
        self.motor_speeds = np.sqrt(thrust_per_motor / self.config.motor_constant)
        self.motor_thrusts = thrust_per_motor

    def compute_thrust_vector(self) -> np.ndarray:
        """
        Compute net thrust force from motor thrusts.

        Motor configuration (quad copter):
        - Motor 0: Front-Right
        - Motor 1: Rear-Left
        - Motor 2: Front-Left
        - Motor 3: Rear-Right

        Returns:
            Total thrust force [Fx, Fy, Fz] (N)
        """
        # Total vertical thrust (all motors)
        total_thrust = np.sum(self.motor_thrusts)
        
        # Compute torques from differential motor speeds
        # Assume motor configuration for standard quad
        front_right = self.motor_thrusts[0]
        rear_left = self.motor_thrusts[1]
        front_left = self.motor_thrusts[2]
        rear_right = self.motor_thrusts[3]
        
        # Roll torque (right-left differential)
        tau_roll = (front_right + rear_right - front_left - rear_left) * 0.1
        
        # Pitch torque (front-back differential)
        tau_pitch = (front_right + front_left - rear_left - rear_right) * 0.1
        
        # Yaw torque (rotor drag)
        tau_yaw = (front_right + rear_left - front_left - rear_right) * 0.05
        
        return np.array([total_thrust, tau_roll, tau_pitch, tau_yaw])

    def normalize_quaternion(self) -> None:
        """Normalize quaternion to unit magnitude."""
        q_norm = np.linalg.norm(self.q)
        if q_norm > 0:
            self.q = self.q / q_norm

    def quaternion_to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = self.q
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def update_attitude(self, angular_rates: np.ndarray, dt: float) -> None:
        """
        Update attitude using quaternion integration.

        Args:
            angular_rates: Angular velocity [p, q, r] (rad/s)
            dt: Time step (s)
        """
        p, q, r = angular_rates
        w, x, y, z = self.q
        
        # Quaternion derivative (standard kinematic equation)
        dq_w = 0.5 * (-p*x - q*y - r*z)
        dq_x = 0.5 * (p*w + r*y - q*z)
        dq_y = 0.5 * (q*w - r*x + p*z)
        dq_z = 0.5 * (r*w + q*x - p*y)
        
        # Integrate
        self.q += np.array([dq_w, dq_x, dq_y, dq_z]) * dt
        self.normalize_quaternion()
        self.angular_velocity = angular_rates.copy()

    def step(self, motor_commands: np.ndarray, wind: np.ndarray, dt: float) -> None:
        """
        Advance drone physics by one time step.

        Args:
            motor_commands: PWM commands [0, 1] for 4 motors
            wind: Wind velocity [u, v, w] (m/s)
            dt: Time step (s)
        """
        if not self.is_active:
            return
        
        # Convert motor commands to thrust
        self.motor_command_to_thrust(motor_commands)
        
        # Compute thrust vector
        thrust_torque = self.compute_thrust_vector()
        total_thrust = thrust_torque[0]
        torques = thrust_torque[1:]
        
        # Get rotation matrix
        R = self.quaternion_to_rotation_matrix()
        
        # Gravity vector
        gravity_vec = np.array([0.0, 0.0, -self.config.gravity])
        
        # Thrust in body frame -> world frame
        thrust_body = np.array([0.0, 0.0, total_thrust])
        thrust_world = R @ thrust_body
        
        # Compute acceleration (F = ma)
        self.acceleration = (thrust_world + self.config.mass * gravity_vec) / self.config.mass
        
        # Add drag (proportional to velocity)
        drag_coeff = 0.1
        self.acceleration -= drag_coeff * self.velocity
        
        # Integrate velocity and position
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Clamp velocity
        vel_magnitude = np.linalg.norm(self.velocity)
        if vel_magnitude > self.config.max_linear_velocity:
            self.velocity = (self.velocity / vel_magnitude) * self.config.max_linear_velocity
        
        # Angular dynamics (Euler's equation for rigid body)
        I = self.config.inertia_matrix
        self.angular_acceleration = np.linalg.inv(I) @ (
            torques - np.cross(self.angular_velocity, I @ self.angular_velocity)
        )
        
        # Integrate angular velocity
        angular_velocity_new = self.angular_velocity + self.angular_acceleration * dt
        
        # Clamp angular velocity
        ang_vel_magnitude = np.linalg.norm(angular_velocity_new)
        if ang_vel_magnitude > self.config.max_angular_velocity:
            angular_velocity_new = (
                angular_velocity_new / ang_vel_magnitude
            ) * self.config.max_angular_velocity
        
        # Update attitude
        self.update_attitude(angular_velocity_new, dt)
        
        # Battery depletion using a physics-motivated model:
        # P_total = P_hover + P_extra
        # P_hover: baseline power to maintain altitude (from config)
        # P_extra: additional mechanical power for maneuvering, proportional
        #          to thrust above hover level times velocity magnitude
        hover_thrust_total = self.config.hover_thrust * 4  # 4 motors × 3.62 N ≈ 14.48 N
        total_thrust = float(np.sum(self.motor_thrusts))
        extra_thrust = max(0.0, total_thrust - hover_thrust_total)
        maneuvering_power = extra_thrust * (np.linalg.norm(self.velocity) + 1.0)
        power_consumed = self.config.battery_energy_rate + maneuvering_power
        self.battery_energy -= power_consumed * dt
        
        # Check battery status
        if self.battery_energy <= 0:
            self.is_active = False
            self.velocity = np.zeros(3)
            self.battery_energy = 0

    def get_state_vector(self) -> np.ndarray:
        """
        Get full 18D state vector for observation.

        Returns:
            [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
             q_w, q_x, q_y, q_z, 
             ang_vel_p, ang_vel_q, ang_vel_r,
             battery_energy, health, active, motor_speed_avg, temperature]
        """
        return np.array([
            *self.position,
            *self.velocity,
            *self.q,
            *self.angular_velocity,
            self.battery_energy / self.config.battery_capacity,
            self.health,
            float(self.is_active),
            np.mean(self.motor_speeds) / 1000.0,  # Normalized
            1.0  # Temperature (simplified, always 1.0)
        ])


class TargetPhysics:
    """Motion model for surveillance targets."""

    def __init__(self, target_id: int, config: TargetConfig, seed: Optional[int] = None):
        """
        Initialize target physics.

        Args:
            target_id: Unique target identifier
            config: Target configuration
            seed: Random seed for reproducibility
        """
        self.target_id = target_id
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        
        # Classification
        self.target_type = TargetType.UNKNOWN
        self.confidence = 0.5
        self.is_detected = False
        
        # Health/threat
        self.threat_level = 0.0  # 0-1 scale
        self.priority = 0.0  # 0-1 scale

    def step(self, dt: float, control_acceleration: Optional[np.ndarray] = None) -> None:
        """
        Advance target motion.

        Args:
            dt: Time step (s)
            control_acceleration: Optional commanded acceleration (m/s²)
        """
        if control_acceleration is not None:
            self.acceleration = np.clip(
                control_acceleration,
                -self.config.max_acceleration,
                self.config.max_acceleration
            )
        
        # Integrate kinematics
        self.velocity += self.acceleration * dt
        
        # Clamp velocity
        vel_magnitude = np.linalg.norm(self.velocity)
        if vel_magnitude > self.config.max_velocity:
            self.velocity = (self.velocity / vel_magnitude) * self.config.max_velocity
        
        # Update position
        self.position += self.velocity * dt

    def get_state_vector(self) -> np.ndarray:
        """
        Get 12D target state for observation.

        Returns:
            [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
             threat_level, priority, type_id, confidence, is_detected, reserved]
        """
        return np.array([
            *self.position,
            *self.velocity,
            self.threat_level,
            self.priority,
            float(self.target_type.value),
            self.confidence,
            float(self.is_detected),
            0.0  # Reserved
        ])


class EnvironmentSimulator:
    """
    Multi-drone, multi-target physics simulation.
    
    Integrates DronePhysics and TargetPhysics for coordinated swarm simulation
    with collision detection, geofence enforcement, and wind disturbances.
    """

    def __init__(
        self,
        num_drones: int,
        max_targets: int,
        drone_config: Optional[DroneConfig] = None,
        target_config: Optional[TargetConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize environment simulator.

        Args:
            num_drones: Number of drones in swarm
            max_targets: Maximum number of targets
            drone_config: Drone configuration
            target_config: Target configuration
            env_config: Environment configuration
            seed: Random seed for reproducibility
        """
        self.num_drones = num_drones
        self.max_targets = max_targets
        self.num_targets = 0
        
        # Configs
        self.drone_config = drone_config or DroneConfig()
        self.target_config = target_config or TargetConfig()
        self.env_config = env_config or EnvironmentConfig()
        
        self.rng = np.random.RandomState(seed)
        
        # Create physics objects
        self.drones = [
            DronePhysics(i, self.drone_config, seed=seed)
            for i in range(num_drones)
        ]
        
        self.targets = [
            TargetPhysics(i, self.target_config, seed=seed)
            for i in range(max_targets)
        ]
        
        # Wind model
        self.wind_model = WindModel(self.env_config, seed=seed)
        self.current_wind = np.zeros(3)
        
        # Statistics
        self.collision_count = 0
        self.geofence_violations = 0
        self.simulation_time = 0.0

    def add_target(self, position: np.ndarray, target_type: TargetType) -> None:
        """
        Add a target to the simulation.

        Args:
            position: Initial position [x, y, z] (m)
            target_type: Target classification
        """
        if self.num_targets < self.max_targets:
            target = self.targets[self.num_targets]
            target.position = position.copy()
            target.target_type = target_type
            target.is_detected = False
            self.num_targets += 1

    def set_drone_initial_state(
        self,
        drone_id: int,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None
    ) -> None:
        """
        Set drone initial state.

        Args:
            drone_id: Drone ID
            position: Initial position [x, y, z] (m)
            velocity: Initial velocity [vx, vy, vz] (m/s)
            quaternion: Initial attitude [w, x, y, z]
        """
        drone = self.drones[drone_id]
        drone.position = position.copy()
        if velocity is not None:
            drone.velocity = velocity.copy()
        if quaternion is not None:
            drone.q = quaternion.copy()
            drone.normalize_quaternion()

    def check_collision(self, drone_id: int, min_distance: float = 1.0) -> bool:
        """
        Check collision with other drones.

        Args:
            drone_id: Drone to check
            min_distance: Minimum safe distance (m)

        Returns:
            True if collision detected
        """
        drone = self.drones[drone_id]
        
        for other_id, other_drone in enumerate(self.drones):
            if other_id != drone_id and other_drone.is_active:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < min_distance:
                    self.collision_count += 1
                    return True
        
        return False

    def check_geofence(self, drone_id: int) -> bool:
        """
        Check if drone violates geofence.

        Args:
            drone_id: Drone to check

        Returns:
            True if outside geofence
        """
        drone = self.drones[drone_id]
        bounds = self.env_config.geofence_boundary
        
        if (abs(drone.position[0]) > bounds[0] or
            abs(drone.position[1]) > bounds[1] or
            drone.position[2] < 0 or drone.position[2] > bounds[2]):
            
            self.geofence_violations += 1
            return True
        
        return False

    def update_target_detections(self) -> None:
        """Update target detection status based on drone proximity."""
        for target in self.targets[:self.num_targets]:
            min_distance = float('inf')
            
            for drone in self.drones:
                if drone.is_active:
                    distance = np.linalg.norm(drone.position - target.position)
                    min_distance = min(min_distance, distance)
            
            # Detection based on distance
            if (min_distance >= self.target_config.min_detection_distance and
                min_distance <= self.target_config.max_detection_distance):
                target.is_detected = True

    def step(self, motor_commands: np.ndarray) -> None:
        """
        Advance simulation by one control step.

        Motor commands shape: (num_drones, 4)

        Args:
            motor_commands: Motor PWM commands for all drones
        """
        dt = self.env_config.timestep
        substep_dt = dt / self.env_config.physics_substeps
        
        # Update wind
        self.current_wind = self.wind_model.update(dt)
        
        # Physics sub-stepping (RK4-style)
        for _ in range(self.env_config.physics_substeps):
            # Update drone physics
            for drone_id, drone in enumerate(self.drones):
                if drone_id < len(motor_commands):
                    commands = motor_commands[drone_id]
                    drone.step(commands, self.current_wind, substep_dt)
            
            # Update target physics
            for target in self.targets[:self.num_targets]:
                target.step(substep_dt)
        
        # Detect collisions
        for drone_id in range(self.num_drones):
            if self.check_collision(drone_id):
                self.drones[drone_id].is_active = False
        
        # Check geofence
        for drone_id in range(self.num_drones):
            if self.check_geofence(drone_id):
                self.drones[drone_id].is_active = False
        
        # Update target detections
        self.update_target_detections()
        
        # Update simulation time
        self.simulation_time += dt

    def get_drone_states(self) -> np.ndarray:
        """
        Get all drone states (N, 18).

        Returns:
            Drone state matrix
        """
        states = np.array([drone.get_state_vector() for drone in self.drones])
        return states

    def get_target_states(self) -> np.ndarray:
        """
        Get target states (M, 12).

        Returns:
            Target state matrix (padded to max_targets)
        """
        states = np.zeros((self.max_targets, 12))
        for i in range(self.num_targets):
            states[i] = self.targets[i].get_state_vector()
        return states

    def get_statistics(self) -> Dict:
        """
        Get simulation statistics.

        Returns:
            Dictionary with metrics
        """
        active_drones = sum(1 for d in self.drones if d.is_active)
        avg_battery = np.mean([d.battery_energy for d in self.drones])
        
        return {
            'simulation_time': self.simulation_time,
            'active_drones': active_drones,
            'total_drones': self.num_drones,
            'collision_count': self.collision_count,
            'geofence_violations': self.geofence_violations,
            'avg_battery_energy': avg_battery,
            'current_wind': self.current_wind.tolist(),
        }

    def reset(self) -> None:
        """Reset simulator state."""
        for drone in self.drones:
            drone.position = np.zeros(3)
            drone.velocity = np.zeros(3)
            drone.q = np.array([1.0, 0.0, 0.0, 0.0])
            drone.angular_velocity = np.zeros(3)
            drone.battery_energy = drone.config.battery_capacity
            drone.is_active = True
        
        for target in self.targets:
            target.position = np.zeros(3)
            target.velocity = np.zeros(3)
            target.is_detected = False
        
        self.collision_count = 0
        self.geofence_violations = 0
        self.simulation_time = 0.0
        self.num_targets = 0
        self.current_wind = np.zeros(3)
        # Reset wind model state for deterministic behavior
        self.wind_model.wind_north = 0.0
        self.wind_model.wind_east = 0.0
        self.wind_model.wind_up = 0.0
        self.wind_model.rng = self.rng
