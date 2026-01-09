"""
Physics simulator for drone dynamics.

Implements kinematic drone model with quaternion attitude, energy consumption,
and health degradation.
"""

from typing import Tuple, List
import numpy as np
from ..core import DroneState
from ..config import DroneConfig, SensorConfig

class DroneSimulator:
    """
    Physics-based drone simulator for the ISR-RL-DMPC system.
    
    Simulates:
    - Kinematic position and velocity evolution
    - Quaternion-based attitude dynamics
    - Energy consumption model
    - Health degradation under load
    """

    def __init__(self, drone_config: DroneConfig, sensor_config: SensorConfig):
        """Initialize simulator.
        
        Arguments:
            drone_config: Drone physical parameters
            sensor_config: Control frequency and sensor parameters
        """
        self.drone_config = drone_config
        self.sensor_config = sensor_config
        self.dt = 1.0/sensor_config.control_frequency # Control timestep

        # Energy Consumption coefficients
        # P = c0 + c1*||v||^2 + c2*||a||
        self.energy_c0 = 0.5 # W (baseline hover)
        self.energy_c1 = 0.01 # W.s^2/m^2
        self.energy_c2 = 0.02 # W.s/m/s^2

        # Health degradation coefficients
        self.health_alpha = 1e-4 # Acceleration-induced wear
        self.health_beta = 0.01 # Engagement penalty

    def step(self, drone_state: DroneState, control_input: np.ndarray, dt: float = None) -> DroneState:
        """
        Propagate drone state by one timestep.

        Arguments:
            drone_state (DroneState): Current drone state
            control_input (np.ndarray): Control input [ax, ay, az, psi_ref, psi_rate]
                        or [ax, ay, ax, wx, wy, wz]
            dt (float): Timestep (default: control period)

        Returns:
            Updated drone state after dt seconds
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
        
        # Saturate acceleration
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > self.drone_config.max_acceleration:
            acceleration *= self.drone_config.max_acceleration/acc_norm

        # Saturate angular velocity
        ang_vel_norm = np.linalg.norm(angular_velocity)
        if ang_vel_norm > self.drone_config.max_angular_velocity:
            angular_velocity *= self.drone_config.max_angular_velocity / ang_vel_norm

        # Update position: p <- p + v*dt + 0.5*a*dt^2
        state.position += state.velocity * dt + 0.5*acceleration* (dt**2)

        # Update velocity: v <- v + a*dt
        state.velocity += acceleration*dt

        # Saturate velocity
        vel_norm = np.linalg.norm(state.velocity)
        if vel_norm > self.drone_config.max_velocity:
            state.velocity *= self.drone_config.max_velocity / vel_norm

        # Update Quaternion: q <- q + 0.5*q⊗(ang_vel)*dt
        state.angular_velocity = angular_velocity
        state.quaternion = self._integrate_quaternion(state.quaternion, state.angular_velocity, dt)

        # Normalize Quaternion
        q_norm = np.linalg.norm(state.quaternion)
        if q_norm > 0:
            state.quaternion /= q_norm

        # Compute and apply energy consumption
        power_consumed = self._compute_power_consumption(state.velocity, acceleration)
        energy_consumed = power_consumed * dt /3600.0 # Wh
        state.battery_energy = max(0.0, state.battery_energy - energy_consumed)

        # Update health based on acceleration and battery
        health_loss = (self.health_alpha * (np.linalg.norm(acceleration)**2) + self.health_beta * 
                       (1.0 if state.battery_energy==0.0 else 0.0))
        state.health = np.clip(state.health - health_loss * dt, 0.0, 1.0)

        # Update timestamp
        state.timestamp += dt

        return state
    
    def propagate_trajectory(self, initial_state: DroneState, control_sequence: np.ndarray, 
                            dt: float = None) -> Tuple[List[DroneState], np.ndarray]:
        """
        Simulate a sequence of control inputs.

        Arguments:
            initial_state (DroneState): Starting drone state
            control_sequence (np.ndarray): Sequence of controls (T, 6) where T is number of steps
            dt (float): Timestep

        Returns:
            (trajectory, energy_consumed)
            - trajectory: List of T+1 DroneState objects
            - energy_consumed: Array of T energy values (Wh)
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
    
    def _integrate_quaternion(self, quaternion: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate quaternion using angular velocity.
        
        Uses: q_{k+1} = q_k + 0.5 * q_k ⊗ w * dt
        
        Args:
            quaternion: Current quaternion [qw, qx, qy, qz]
            angular_velocity: Angular velocity [wx, wy, wz]
            dt: Timestep
            
        Returns:
            Updated quaternion
        """
        qw, qx, qy, qz = quaternion
        wx, wy, wz = angular_velocity

        # Quaternion derivation: 0.5 * q_k ⊗ w
        dqw = 0.5*(-qx*wx - qy*wy - qz*wz)
        dqx = 0.5*(qw*wx + qy*wz - qz*wy)
        dqy = 0.5*(qw*wy - qx*wz + qz*wx)
        dqz = 0.5*(qw*wz + qx*wy - qy*wx)

        # Integrate
        q_new = np.array([
            qw + dqw * dt,
            qx + dqx * dt,
            qy + dqy * dt,
            qz + dqz * dt,
        ])

        # Normalize
        q_norm = np.linalg.norm(q_new)
        if q_norm > 0:
            q_new = q_new / q_norm

        return q_new
    
    def _compute_power_consumption(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
        """
        Compute instantaneous power consumption.

        Model: P = c0 + c1*||v||^2 + c2*||a||

        Arguments:
            velocity (np.ndarray): Drone velocity (m/s)
            acceleration (np.ndarray): Drone acceleration (m/s^2)

        Returns:
            Power consumption (Watts)
        """
        v_norm = np.linalg.norm(velocity)
        a_norm = np.linalg.norm(acceleration)

        power = self.energy_c0 + self.energy_c1 * (v_norm**2) + self.energy_c2*a_norm

        return max(0.0, power)
    
    def compute_energy_for_trajectory(self, trajectory: List[DroneState]) -> Tuple[float, np.ndarray]:
        """
        Compute total energy and per-step energy for a trajectory.
        
        Args:
            trajectory: List of DroneState objects along trajectory
            
        Returns:
            (total_energy, per_step_energy)
        """
        energies = []
        total = 0.0

        for i in range(len(trajectory)):
            state = trajectory[i]
            next_state = trajectory[i+1]

            power = self._compute_power_consumption(state.velocity, state.acceleration)
            energy = power * self.dt / 3600.0

            energies.append(energy)
            total += energy
        
        return total, np.array(energies)
    
    def __repr__(self) -> str:
        """String Representation"""

        return (
            f"DroneSimulator(dt={self.dt:.3f}s, "
            f"f_ctrl={self.sensor_config.control_frequency}Hz, "
            f"P={self.energy_c0 + self.energy_c1:.3f}W model)"
        )
