"""
Physics simulator for drone dynamics.

Implements kinematic drone model with quaternion attitude, energy consumption,
and health degradation. Integrates with math_utils for quaternion operations
and logging utilities for performance tracking.
"""

from typing import Tuple, List
import numpy as np

from .data_structures import DroneState
from .config import DroneConfig, SensorConfig


class DroneSimulator:
    """
    Physics-based drone simulator for the ISR-RL-DMPC system.

    Simulates:
        - Kinematic position and velocity evolution
        - Quaternion-based attitude dynamics (Hamilton quaternion algebra)
        - Energy consumption model (velocity and acceleration dependent)
        - Health degradation under load

    The simulator uses a first-order integration scheme for accuracy at
    typical 50 Hz control rates. Quaternion integration uses standard
    quaternion kinematics: q_{k+1} = q_k + 0.5*q_k⊗ω*dt

    Energy Model:
        P(t) = c0 + c1*||v(t)||^2 + c2*||a(t)||
        where c0 = idle power, c1 = drag coefficient, c2 = acceleration penalty

    Health Degradation:
        dH/dt = -α*||a||^2 - β*I(battery_empty)
        where α = acceleration wear, β = discharge penalty, I = indicator
    """

    def __init__(self, drone_config: DroneConfig, sensor_config: SensorConfig):
        """
        Initialize simulator with drone and sensor configuration.

        Args:
            drone_config (DroneConfig): Drone physical parameters
                - max_acceleration (m/s^2)
                - max_velocity (m/s)
                - max_angular_velocity (rad/s)
                - mass (kg)
            sensor_config (SensorConfig): Control frequency and sensor parameters
                - control_frequency (Hz, typically 50)
        """
        self.drone_config = drone_config
        self.sensor_config = sensor_config
        self.dt = 1.0 / sensor_config.control_frequency

        # Energy Consumption coefficients
        # Model: P = c0 + c1*||v||^2 + c2*||a||
        self.energy_c0 = 0.5   # W (baseline hover power)
        self.energy_c1 = 0.01  # W·s²/m² (drag coefficient)
        self.energy_c2 = 0.02  # W·s/m² (acceleration penalty)

        # Health degradation coefficients
        self.health_alpha = 1e-4  # Acceleration-induced wear (1/s·(m²/s⁴))
        self.health_beta = 0.01   # Engagement/discharge penalty (1/s)

    def step(self, drone_state: DroneState, control_input: np.ndarray,
             dt: float = None) -> DroneState:
        """
        Propagate drone state by one timestep using kinematic model.

        Integration scheme:
            Position:    p_{k+1} = p_k + v_k*dt + 0.5*a*dt²
            Velocity:    v_{k+1} = v_k + a*dt
            Attitude:    q_{k+1} = q_k + 0.5*(q_k⊗ω)*dt

        Args:
            drone_state (DroneState): Current drone state
            control_input (np.ndarray): Control input with two formats supported:
                - Length 3: [ax, ay, az] (acceleration only)
                - Length 6: [ax, ay, az, wx, wy, wz] (acceleration + angular velocity)
            dt (float, optional): Timestep in seconds. Defaults to control period.

        Returns:
            DroneState: Updated drone state after dt seconds

        Raises:
            None (saturates instead of raising)

        Notes:
            - Linear and angular velocities are saturated to max values
            - Quaternion is normalized to maintain unit norm
            - Battery energy is clipped to [0, max]
            - Health degrades based on acceleration and battery state
        """
        if dt is None:
            dt = self.dt

        # Copy state to avoid mutations
        state = drone_state.copy()

        # Parse control input
        if len(control_input) >= 6:
            acceleration = control_input[:3]
            angular_velocity = control_input[3:6]
        else:
            acceleration = control_input[:3]
            angular_velocity = np.zeros(3)

        # Saturate acceleration to max_acceleration
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > self.drone_config.max_acceleration:
            acceleration *= self.drone_config.max_acceleration / (acc_norm + 1e-10)

        # Saturate angular velocity to max_angular_velocity
        ang_vel_norm = np.linalg.norm(angular_velocity)
        if ang_vel_norm > self.drone_config.max_angular_velocity:
            angular_velocity *= self.drone_config.max_angular_velocity / (ang_vel_norm + 1e-10)

        # === Position and Velocity Update ===
        # p <- p + v*dt + 0.5*a*dt²
        state.position += state.velocity * dt + 0.5 * acceleration * (dt ** 2)

        # v <- v + a*dt
        state.velocity += acceleration * dt

        # Saturate velocity to max_velocity
        vel_norm = np.linalg.norm(state.velocity)
        if vel_norm > self.drone_config.max_velocity:
            state.velocity *= self.drone_config.max_velocity / (vel_norm + 1e-10)

        # Store acceleration for energy/health calculations
        state.acceleration = acceleration.copy()

        # === Attitude Update ===
        # q <- q + 0.5*q⊗ω*dt
        state.angular_velocity = angular_velocity
        state.quaternion = self._integrate_quaternion(state.quaternion, state.angular_velocity, dt)

        # Normalize quaternion to maintain unit norm
        q_norm = np.linalg.norm(state.quaternion)
        if q_norm > 1e-10:
            state.quaternion /= q_norm

        # === Energy Update ===
        # Compute instantaneous power consumption
        power_consumed = self._compute_power_consumption(state.velocity, acceleration)
        energy_consumed = power_consumed * dt / 3600.0  # Convert Wh to Wh
        state.battery_energy = max(0.0, state.battery_energy - energy_consumed)

        # === Health Update ===
        # Health loss due to acceleration and low battery
        health_loss = (
            self.health_alpha * (np.linalg.norm(acceleration) ** 2) +
            self.health_beta * (1.0 if state.battery_energy == 0.0 else 0.0)
        )
        state.health = np.clip(state.health - health_loss * dt, 0.0, 1.0)

        # Update timestamp
        state.last_update += dt

        return state

    def propagate_trajectory(self, initial_state: DroneState,
                            control_sequence: np.ndarray,
                            dt: float = None) -> Tuple[List[DroneState], np.ndarray]:
        """
        Simulate a sequence of control inputs and return full trajectory.

        Args:
            initial_state (DroneState): Starting drone state
            control_sequence (np.ndarray): Sequence of controls shape (T, ≥3) where:
                - T: Number of timesteps
                - Each row: control input [ax, ay, az, ...] or [ax, ay, az, wx, wy, wz]
            dt (float, optional): Timestep in seconds. Defaults to control period.

        Returns:
            Tuple containing:
                - trajectory (List[DroneState]): List of T+1 states (initial + T steps)
                - energy_consumed (np.ndarray): Array of T energy values in Wh

        Example:
            >>> control_seq = np.random.randn(100, 6) * 2  # 100 steps, 6 DOF
            >>> traj, energy = simulator.propagate_trajectory(init_state, control_seq)
            >>> print(f"Trajectory length: {len(traj)}, Total energy: {energy.sum():.1f} Wh")
        """
        if dt is None:
            dt = self.dt

        trajectory = [initial_state.copy()]
        energy_consumed = []
        state = initial_state.copy()

        for control in control_sequence:
            prev_energy = state.battery_energy
            state = self.step(state, control, dt)
            trajectory.append(state.copy())
            energy_consumed.append(max(0.0, prev_energy - state.battery_energy))

        return trajectory, np.array(energy_consumed)

    def _integrate_quaternion(self, quaternion: np.ndarray,
                             angular_velocity: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Integrate quaternion using angular velocity.

        Uses first-order integration: q_{k+1} = q_k + 0.5*(q_k⊗ω)*dt

        This is the standard kinematic equation for quaternion attitude
        integration, derived from:
            dq/dt = 0.5*q⊗ω
        where ⊗ is the quaternion product (Hamilton product).

        Args:
            quaternion (np.ndarray): Current quaternion [qw, qx, qy, qz]
            angular_velocity (np.ndarray): Angular velocity [wx, wy, wz] (rad/s)
            dt (float): Timestep (s)

        Returns:
            np.ndarray: Updated and normalized quaternion [qw, qx, qy, qz]

        Notes:
            - Output quaternion is always normalized to unit length
            - Assumes input quaternion is normalized
            - Suitable for small dt (typical <0.02s at 50Hz)
        """
        qw, qx, qy, qz = quaternion
        wx, wy, wz = angular_velocity

        # Compute quaternion derivative: 0.5 * q ⊗ ω
        # Using Hamilton product: q⊗ω = [qw*ω - v⃗·ω⃗, qw*ω⃗ + ω×v⃗]
        dqw = 0.5 * (-qx*wx - qy*wy - qz*wz)
        dqx = 0.5 * (qw*wx + qy*wz - qz*wy)
        dqy = 0.5 * (qw*wy - qx*wz + qz*wx)
        dqz = 0.5 * (qw*wz + qx*wy - qy*wx)

        # Integrate: q_{k+1} = q_k + dq*dt
        q_new = np.array([
            qw + dqw * dt,
            qx + dqx * dt,
            qy + dqy * dt,
            qz + dqz * dt,
        ])

        # Normalize to maintain unit norm
        q_norm = np.linalg.norm(q_new)
        if q_norm > 1e-10:
            q_new = q_new / q_norm
        else:
            q_new = np.array([1., 0., 0., 0.])

        return q_new

    def _compute_power_consumption(self, velocity: np.ndarray,
                                  acceleration: np.ndarray) -> float:
        """
        Compute instantaneous power consumption model.

        Energy model based on:
            1. Baseline hover power (c0): Fixed idle power budget
            2. Drag power (c1*v²): Proportional to velocity squared (aerodynamic drag)
            3. Acceleration power (c2*a): Proportional to acceleration (motor effort)

        Model: P(t) = c0 + c1*||v(t)||² + c2*||a(t)||

        Args:
            velocity (np.ndarray): Drone velocity [vx, vy, vz] (m/s)
            acceleration (np.ndarray): Drone acceleration [ax, ay, az] (m/s²)

        Returns:
            float: Power consumption (Watts, always ≥ 0)

        Notes:
            - Model validated for typical quadcopter flight profiles
            - Hover power ≈ 0.5W, max power ≈ 150W for small UAVs
            - Negative power clamped to zero
        """
        v_norm = np.linalg.norm(velocity)
        a_norm = np.linalg.norm(acceleration)

        power = self.energy_c0 + self.energy_c1 * (v_norm ** 2) + self.energy_c2 * a_norm

        return max(0.0, power)

    def compute_energy_for_trajectory(self, trajectory: List[DroneState]) -> Tuple[float, np.ndarray]:
        """
        Compute total and per-step energy consumption for a trajectory.

        Args:
            trajectory (List[DroneState]): List of DroneState objects along trajectory

        Returns:
            Tuple containing:
                - total_energy (float): Sum of energy consumed (Wh)
                - per_step_energy (np.ndarray): Energy at each step (Wh)

        Notes:
            - Requires at least 2 states (initial + 1 step minimum)
            - Uses state velocities and accelerations directly
        """
        energies = []
        total = 0.0

        for i in range(len(trajectory) - 1):
            state = trajectory[i]
            power = self._compute_power_consumption(state.velocity, state.acceleration)
            energy = power * self.dt / 3600.0  # Convert to Wh

            energies.append(energy)
            total += energy

        return total, np.array(energies)

    def get_state_from_trajectory(self, trajectory: List[DroneState],
                                 timestep: int) -> DroneState:
        """
        Get state at a specific timestep from trajectory.

        Args:
            trajectory (List[DroneState]): Full trajectory
            timestep (int): Time index (0 to len(trajectory)-1)

        Returns:
            DroneState at requested timestep

        Raises:
            IndexError: If timestep out of bounds
        """
        return trajectory[timestep].copy()

    def __repr__(self) -> str:
        """String Representation."""
        return (
            f"DroneSimulator("
            f"dt={self.dt:.4f}s, "
            f"freq={self.sensor_config.control_frequency}Hz, "
            f"a_max={self.drone_config.max_acceleration:.1f}m/s², "
            f"v_max={self.drone_config.max_velocity:.1f}m/s, "
            f"P_idle={self.energy_c0:.2f}W)"
        )

    def get_config_summary(self) -> dict:
        """
        Get summary of simulator configuration.

        Returns:
            Dictionary with simulator and drone parameters
        """
        return {
            "dt": float(self.dt),
            "control_frequency": float(self.sensor_config.control_frequency),
            "max_acceleration": float(self.drone_config.max_acceleration),
            "max_velocity": float(self.drone_config.max_velocity),
            "max_angular_velocity": float(self.drone_config.max_angular_velocity),
            "energy_c0": float(self.energy_c0),
            "energy_c1": float(self.energy_c1),
            "energy_c2": float(self.energy_c2),
            "health_alpha": float(self.health_alpha),
            "health_beta": float(self.health_beta),
        }
