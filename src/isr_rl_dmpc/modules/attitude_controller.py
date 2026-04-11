"""
Module 8 - Attitude Controller: Geometric SO(3) Control

Implements real-time attitude and position control for quadrotor
drones using geometric control on the SO(3) manifold.  All learned
gain-adaptation neural networks have been removed.  Control gains are
fixed at construction time via :class:`DroneParameters` and can be
updated manually if needed.

Mathematical overview
---------------------
Position loop (PD):
    a_des = a_ref − Kp_pos · (p − p_ref) − Kd_pos · (v − v_ref)

Attitude loop (geometric, SO(3)):
    e_R  = ½ vec( R_d^T R − R^T R_d )
    e_ω  = ω − R^T R_d ω_d
    τ    = −Kp_att · e_R − Kd_att · e_ω  + ω × (J ω)

Motor mixing (X-quad):
    [F, τ_x, τ_y, τ_z]  →  [T₁, T₂, T₃, T₄]  via pseudo-inverse
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R_tool


@dataclass
class DroneParameters:
    """Physical parameters of quadrotor drone."""

    mass: float = 1.477  # kg  (hector_quadrotor)
    inertia: np.ndarray = None  # 3×3 inertia matrix

    motor_speed_max: float = 800.0  # rad/s
    motor_constant: float = 8.27e-6  # N/(rad/s)²
    thrust_max_per_rotor: float = 9.5  # N  (hector_quadrotor: 4×9.5 = 38 N peak)

    arm_length: float = 0.215  # metres

    # Control gains (fixed; no NN adaptation)
    Kp_attitude: float = 4.5
    Kd_attitude: float = 1.5
    Kp_position: float = 2.0
    Kd_position: float = 1.5

    def __post_init__(self) -> None:
        if self.inertia is None:
            # hector_quadrotor URDF inertia values
            Ixx, Iyy, Izz = 0.01152, 0.01152, 0.02180
            self.inertia = np.diag([Ixx, Iyy, Izz])


class GeometricController:
    """
    Geometric attitude control on the SO(3) manifold.

    Pure NumPy implementation for real-time execution.
    """

    def __init__(self, params: DroneParameters) -> None:
        self.params = params
        self.inertia = params.inertia
        self.inertia_inv = np.linalg.inv(self.inertia)

    def quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert scalar-first quaternion [w, x, y, z] to rotation matrix."""
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
        return R_tool.from_quat(q_scipy).as_matrix()

    def matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to scalar-first quaternion [w, x, y, z]."""
        q = R_tool.from_matrix(R).as_quat()  # [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]])

    def attitude_error(self, R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
        """Compute attitude error vector on the SO(3) manifold."""
        R_e = R_d.T @ R
        return np.array([
            R_e[2, 1] - R_e[1, 2],
            R_e[0, 2] - R_e[2, 0],
            R_e[1, 0] - R_e[0, 1],
        ]) / 2.0

    def desired_attitude_from_accel(
        self, a_d: np.ndarray, yaw_d: float = 0.0
    ) -> np.ndarray:
        """
        Compute desired rotation matrix from desired acceleration (differential
        flatness).
        """
        g = 9.81
        z_body = a_d + np.array([0.0, 0.0, g])
        z_body /= np.linalg.norm(z_body) + 1e-6

        x_body_desired = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        y_body = np.cross(z_body, x_body_desired)
        y_body /= np.linalg.norm(y_body) + 1e-6
        x_body = np.cross(y_body, z_body)

        return np.column_stack([x_body, y_body, z_body])

    def control_law(
        self,
        R: np.ndarray,
        omega: np.ndarray,
        R_d: np.ndarray,
        omega_d: Optional[np.ndarray] = None,
        Kp: Optional[float] = None,
        Kd: Optional[float] = None,
    ) -> np.ndarray:
        """
        Geometric control law: τ = −Kp·e_R − Kd·e_ω + ω × (J ω).

        Args:
            R:      Current rotation matrix (3×3).
            omega:  Current angular velocity (3,).
            R_d:    Desired rotation matrix (3×3).
            omega_d: Desired angular velocity (3,); defaults to zero.
            Kp:     Proportional gain override; uses ``params.Kp_attitude`` if None.
            Kd:     Derivative gain override; uses ``params.Kd_attitude`` if None.

        Returns:
            Control torque vector (3,).
        """
        if omega_d is None:
            omega_d = np.zeros(3)
        Kp = Kp if Kp is not None else self.params.Kp_attitude
        Kd = Kd if Kd is not None else self.params.Kd_attitude

        e_R = self.attitude_error(R, R_d)
        e_omega = omega - R.T @ R_d @ omega_d
        J_omega = self.inertia @ omega

        return -Kp * e_R - Kd * e_omega + np.cross(omega, J_omega)


class AttitudeController:
    """
    Attitude controller using SO(3) geometric control with fixed gains.

    Processes the full cascade from position reference to motor thrust
    commands in a single :meth:`control_loop` call.
    """

    def __init__(self, params: DroneParameters) -> None:
        self.params = params
        self.controller = GeometricController(params)
        self._setup_motor_mixer()

    def _setup_motor_mixer(self) -> None:
        """Build motor mixer for X-quad configuration."""
        angles = np.array([45, 135, 225, 315]) * np.pi / 180.0
        L = self.params.arm_length / np.sqrt(2.0)
        mixer = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-L * np.sin(angles[0]), -L * np.sin(angles[1]),
             -L * np.sin(angles[2]), -L * np.sin(angles[3])],
            [ L * np.cos(angles[0]),  L * np.cos(angles[1]),
              L * np.cos(angles[2]),  L * np.cos(angles[3])],
            [-1.0, 1.0, -1.0, 1.0],
        ])
        self.mixer_inv = np.linalg.pinv(mixer)

    def control_loop(
        self, state: np.ndarray, reference: np.ndarray
    ) -> Dict:
        """
        Full cascade: position → attitude → motor commands.

        Args:
            state:     Current state vector of shape ``(11,)``:
                       ``[p(3), v(3), a(3), yaw, yaw_rate]``.
            reference: Reference state of shape ``(≥10,)``:
                       ``[p_ref(3), v_ref(3), a_ref(3), yaw_ref, ...]``.

        Returns:
            Dictionary containing:
            - ``motor_thrusts`` (4,) – per-rotor thrust [N]
            - ``desired_attitude`` (3×3) – target rotation matrix
            - ``desired_accel`` (3,) – commanded acceleration [m/s²]
            - ``total_force`` float – total thrust [N]
            - ``torque`` (3,) – control torque [N·m]
        """
        p, v = state[0:3], state[3:6]
        psi, psi_dot = state[9], state[10]

        p_d, v_d, a_d = reference[0:3], reference[3:6], reference[6:9]
        psi_d = reference[9]

        # Position PD → desired acceleration
        e_p = p - p_d
        e_v = v - v_d
        a_des = (
            a_d
            - self.params.Kp_position * e_p
            - self.params.Kd_position * e_v
        )

        # Differential flatness → desired attitude
        R_d = self.controller.desired_attitude_from_accel(a_des, psi_d)

        # Current attitude from yaw angle
        R = self._euler_to_matrix(np.array([0.0, 0.0, psi]))
        omega = np.array([0.0, 0.0, psi_dot])

        # SO(3) control law → torque
        tau = self.controller.control_law(R, omega, R_d)

        # Total thrust
        g = 9.81
        z_axis = R[:, 2]
        F_total = float(
            self.params.mass * (a_des + np.array([0.0, 0.0, g])) @ z_axis
        )
        F_total = np.clip(F_total, 0.0, 4.0 * self.params.thrust_max_per_rotor)

        # Motor mixing
        force_torque = np.array([F_total, tau[0], tau[1], tau[2]])
        motor_thrusts = np.clip(
            self.mixer_inv @ force_torque,
            0.0,
            self.params.thrust_max_per_rotor,
        )

        return {
            "motor_thrusts": motor_thrusts,
            "desired_attitude": R_d,
            "desired_accel": a_des,
            "total_force": F_total,
            "torque": tau,
        }

    @staticmethod
    def _euler_to_matrix(euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) in XYZ convention to SO(3)."""
        return R_tool.from_euler("xyz", euler).as_matrix()


if __name__ == "__main__":
    params = DroneParameters()
    controller = AttitudeController(params)

    state = np.random.randn(11).astype(np.float32)
    reference = np.random.randn(12).astype(np.float32)

    output = controller.control_loop(state, reference)
    print("✓ Pure Geometric Attitude Control")
    print(f"  Motor thrusts: {output['motor_thrusts']}")
    print(f"  Torque:        {output['torque']}")
