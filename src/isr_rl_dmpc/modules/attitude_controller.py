"""
Attitude Controller: Geometric Control + PyTorch Gain Adaptation
Real-time SO(3) control with learned parameter adjustment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


@dataclass
class DroneParameters:
    """Physical parameters of quadrotor drone"""
    mass: float = 1.0  # kg
    inertia: np.ndarray = None  # 3x3 inertia matrix
    
    motor_speed_max: float = 800.0  # rad/s
    motor_constant: float = 8.27e-6  # N/(rad/s)²
    thrust_max_per_rotor: float = 25.0  # N
    
    arm_length: float = 0.215  # meters
    
    # Base control gains (numpy/fixed)
    Kp_attitude: float = 4.5
    Kd_attitude: float = 1.5
    Kp_position: float = 2.0
    Kd_position: float = 1.5
    
    device: str = "cpu"
    
    def __post_init__(self):
        if self.inertia is None:
            Ixx, Iyy, Izz = 0.0083, 0.0083, 0.0166
            self.inertia = np.diag([Ixx, Iyy, Izz])


class GainAdaptationNetwork(nn.Module):
    """PyTorch: Learns multiplicative gain adjustments"""
    
    def __init__(self, state_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # Multipliers for Kp_att, Kd_att, Kp_pos, Kd_pos
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State [batch, state_dim]
        
        Returns:
            multipliers: [batch, 4] gain adjustment factors (1.0 = no change)
        """
        factors = self.network(x)
        # Ensure positive, bounded in [0.5, 2.0]
        return torch.sigmoid(factors) * 1.5 + 0.5


class GeometricController:
    """
    Geometric attitude control on SO(3) manifold
    Pure numpy for real-time execution (no autograd overhead)
    """
    
    def __init__(self, params: DroneParameters):
        self.params = params
        self.inertia = params.inertia  # 3x3
        self.inertia_inv = np.linalg.inv(self.inertia)
    
    def quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix SO(3)"""
        q = q / (np.linalg.norm(q) + 1e-10)  # Normalize
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])
    
    def attitude_error(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        """Compute attitude error on SO(3) manifold"""
        R_e = R_d.T @ R
        e_R = np.array([
            R_e[2, 1] - R_e[1, 2],
            R_e[0, 2] - R_e[2, 0],
            R_e[1, 0] - R_e[0, 1]
        ]) / 2.0
        return e_R
    
    def desired_attitude_from_accel(self, a_d: np.ndarray, yaw_d: float = 0.0) -> np.ndarray:
        """Differential flatness: desired accel → desired attitude"""
        g = 9.81
        
        # z_body = normalize(a_d + g*[0,0,1])
        z_body = a_d + np.array([0, 0, g])
        z_body = z_body / (np.linalg.norm(z_body) + 1e-6)
        
        # x_body from desired yaw
        x_body_desired = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])
        
        # y_body = z_body × x_body_desired
        y_body = np.cross(z_body, x_body_desired)
        y_body = y_body / (np.linalg.norm(y_body) + 1e-6)
        
        # x_body = y_body × z_body
        x_body = np.cross(y_body, z_body)
        
        # R_d = [x_body, y_body, z_body]^T
        R_d = np.stack([x_body, y_body, z_body])
        
        return R_d
    
    def control_law(self, R: np.ndarray, omega: np.ndarray, R_d: np.ndarray,
                   omega_d: Optional[np.ndarray] = None,
                   Kp: float = None, Kd: float = None) -> np.ndarray:
        """
        Geometric control law: τ = -Kp*e_R - Kd*e_ω + ω×(J*ω)
        
        Args:
            R: Current attitude (3×3)
            omega: Current angular velocity (3,)
            R_d: Desired attitude (3×3)
            omega_d: Desired angular velocity (3,), default zero
            Kp, Kd: Gain multipliers from adaptation network
        
        Returns:
            tau: Control torque (3,)
        """
        if omega_d is None:
            omega_d = np.zeros(3)
        
        if Kp is None:
            Kp = self.params.Kp_attitude
        if Kd is None:
            Kd = self.params.Kd_attitude
        
        # Attitude error
        e_R = self.attitude_error(R, R_d)
        
        # Angular velocity error
        e_omega = omega - omega_d
        
        # Control law
        tau_R = -Kp * e_R
        tau_omega = -Kd * e_omega
        
        # Gyroscopic compensation: ω × (J*ω)
        J_omega = self.inertia @ omega
        tau_gyro = np.cross(omega, J_omega)
        
        tau = tau_R + tau_omega + tau_gyro
        
        return tau


class AttitudeController:
    """
    Hybrid attitude control: SO(3) math (Geometric Controller) + PyTorch gain adaptation
    """
    
    def __init__(self, params: DroneParameters):
        self.params = params
        self.device = torch.device(params.device)
        
        # Pure SO(3) controller (numpy, real-time)
        self.so3_controller = GeometricController(params)
        
        # PyTorch gain adaptation network
        self.gain_network = GainAdaptationNetwork(state_dim=11).to(self.device)
        self.gain_optimizer = optim.Adam(self.gain_network.parameters(), lr=1e-4)
        
        # Motor mixer (X-quad configuration)
        self._setup_motor_mixer()
    
    def _setup_motor_mixer(self):
        """Setup motor mixer for X-quad"""
        # Motor placement angles: 45°, 135°, 225°, 315°
        angles = np.array([45, 135, 225, 315]) * np.pi / 180
        L = self.params.arm_length / np.sqrt(2)
        
        # Mixer matrix (4x4)
        mixer = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-L*np.sin(angles[0]), -L*np.sin(angles[1]), -L*np.sin(angles[2]), -L*np.sin(angles[3])],
            [L*np.cos(angles[0]), L*np.cos(angles[1]), L*np.cos(angles[2]), L*np.cos(angles[3])],
            [-1.0, 1.0, -1.0, 1.0]
        ])
        
        self.mixer_inv = np.linalg.pinv(mixer)
    
    def control_loop(self, state: np.ndarray, reference: np.ndarray,
                    use_adaptation: bool = True) -> Dict:
        """
        Complete control loop: position → attitude → motor commands
        
        Args:
            state: [11] Current state [p(3), v(3), a(3), ψ, ψ̇]
            reference: [12] Reference state [p(3), v(3), a(3), ψ, (unused)]
            use_adaptation: Whether to use learned gain adaptation
        
        Returns:
            Dictionary with motor_thrusts, torque, etc.
        """
        p = state[0:3]
        v = state[3:6]
        a = state[6:9]
        psi = state[9]
        psi_dot = state[10]
        
        p_d = reference[0:3]
        v_d = reference[3:6]
        a_d = reference[6:9]
        psi_d = reference[9]
        
        # Step 1: Position PD control → desired acceleration
        e_p = p - p_d
        e_v = v - v_d
        a_des = a_d - self.params.Kp_position * e_p - self.params.Kd_position * e_v
        
        # Step 2: Differential flatness → desired attitude
        R_d = self.so3_controller.desired_attitude_from_accel(a_des, psi_d)
        
        # Step 3: Current attitude (from Euler angles in state)
        R = self._euler_to_matrix(np.array([0.0, 0.0, psi]))
        omega = np.array([0.0, 0.0, psi_dot])
        
        # Step 4: PyTorch - Get adaptive gains
        Kp_mult, Kd_mult, _, _ = self._get_gain_multipliers(state, use_adaptation)
        Kp = self.params.Kp_attitude * Kp_mult
        Kd = self.params.Kd_attitude * Kd_mult
        
        # Step 5: SO(3) control law → torque
        tau = self.so3_controller.control_law(R, omega, R_d, Kp=Kp, Kd=Kd)
        
        # Step 6: Total force from desired acceleration
        g = 9.81
        F_total = self.params.mass * (a_des[2] + g)
        F_total = np.clip(F_total, 0, 4 * self.params.thrust_max_per_rotor)
        
        # Step 7: Motor mixing [F, τ_x, τ_y, τ_z] → [ω₁, ω₂, ω₃, ω₄]
        force_torque = np.array([F_total, tau[0], tau[1], tau[2]])
        motor_thrusts = self.mixer_inv @ force_torque
        motor_thrusts = np.clip(motor_thrusts, 0, self.params.thrust_max_per_rotor)
        
        return {
            'motor_thrusts': motor_thrusts,
            'desired_attitude': R_d,
            'desired_accel': a_des,
            'total_force': F_total,
            'torque': tau,
            'gain_multipliers': (Kp_mult, Kd_mult)
        }
    
    def _get_gain_multipliers(self, state: np.ndarray, 
                             use_adaptation: bool = True) -> Tuple[float, float, float, float]:
        """Get gain adjustment factors from network"""
        if not use_adaptation:
            return 1.0, 1.0, 1.0, 1.0
        
        state_torch = torch.from_numpy(state.astype(np.float32)).to(self.device)
        with torch.no_grad():
            multipliers = self.gain_network(state_torch.unsqueeze(0)).squeeze(0)
        
        return tuple(multipliers.cpu().numpy())
    
    def _euler_to_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to SO(3)"""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        return Rz @ Ry @ Rx
    
    def learn_gain_adaptation(self, states: torch.Tensor, target_gains: torch.Tensor,
                             batch_size: int = 32, epochs: int = 10):
        """
        Train gain adaptation network
        
        Args:
            states: [N, 11] collected states
            target_gains: [N, 4] desired gain adjustments
        """
        states = states.to(self.device)
        target_gains = target_gains.to(self.device)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(0, len(states), batch_size):
                batch_end = min(i + batch_size, len(states))
                batch_states = states[i:batch_end]
                batch_targets = target_gains[i:batch_end]
                
                pred_gains = self.gain_network(batch_states)
                loss = ((pred_gains - batch_targets) ** 2).mean()
                
                self.gain_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gain_network.parameters(), 1.0)
                self.gain_optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/max(1, len(states)//batch_size):.6f}")
    
    def save_gains(self, path: str):
        """Save learned gains"""
        torch.save(self.gain_network.state_dict(), path)
    
    def load_gains(self, path: str):
        """Load learned gains"""
        self.gain_network.load_state_dict(torch.load(path, map_location=self.device))


# Usage example
if __name__ == "__main__":
    params = DroneParameters(device="cpu")
    controller = AttitudeController(params)
    
    # Example execution
    state = np.random.randn(11).astype(np.float32)
    reference = np.random.randn(12).astype(np.float32)
    
    output = controller.control_loop(state, reference, use_adaptation=True)
    
    print(f"✓ Hybrid Attitude Control")
    print(f"  Motor thrusts: {output['motor_thrusts']}")
    print(f"  Torque: {output['torque']}")
    print(f"  Gain multipliers: {output['gain_multipliers']}")
