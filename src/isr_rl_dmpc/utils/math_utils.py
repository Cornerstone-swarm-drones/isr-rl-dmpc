"""
Mathematical utilities for ISR-RL-DMPC.

Provides quaternion operations, matrix utilities, and coordinate transformations
for drone attitude and position calculations.
"""

import numpy as np
from typing import Tuple, Union


class QuaternionOps:
    """Quaternion algebra operations for SO(3) rotations."""

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit magnitude.

        Args:
            q: Quaternion [qw, qx, qy, qz]

        Returns:
            Normalized quaternion
        """
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            return q / q_norm
        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Quaternion multiplication q1 ⊗ q2.

        Args:
            q1, q2: Quaternions [qw, qx, qy, qz]

        Returns:
            Product q1 ⊗ q2
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """
        Quaternion conjugate q*.

        Args:
            q: Quaternion [qw, qx, qy, qz]

        Returns:
            Conjugate [qw, -qx, -qy, -qz]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def inverse(q: np.ndarray) -> np.ndarray:
        """
        Quaternion inverse q^{-1}.

        Args:
            q: Unit quaternion

        Returns:
            Inverse (same as conjugate for unit quaternions)
        """
        return QuaternionOps.conjugate(q)

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate 3D vector by quaternion: v' = q v q*.

        Args:
            q: Unit quaternion [qw, qx, qy, qz]
            v: 3D vector [vx, vy, vz]

        Returns:
            Rotated vector
        """
        q_conj = QuaternionOps.conjugate(q)
        v_quat = np.array([0.0, v[0], v[1], v[2]])

        # q v q*
        temp = QuaternionOps.multiply(q, v_quat)
        result = QuaternionOps.multiply(temp, q_conj)

        return result[1:4]

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create quaternion from axis-angle representation.

        Args:
            axis: Rotation axis (normalized)
            angle: Rotation angle (radians)

        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        half_angle = angle / 2.0

        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ])

    @staticmethod
    def to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle representation.

        Args:
            q: Unit quaternion [qw, qx, qy, qz]

        Returns:
            (axis, angle) where axis is normalized 3D vector and angle in radians
        """
        q = QuaternionOps.normalize(q)
        qw = np.clip(q[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(qw)

        sin_half_angle = np.sqrt(1.0 - qw**2)
        if sin_half_angle < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = q[1:4] / sin_half_angle

        return axis, angle

    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        Args:
            q: Unit quaternion [qw, qx, qy, qz]

        Returns:
            3x3 rotation matrix R
        """
        q = QuaternionOps.normalize(q)
        qw, qx, qy, qz = q

        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

        return QuaternionOps.normalize(np.array([qw, qx, qy, qz]))

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between quaternions.

        Args:
            q1, q2: Unit quaternions
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion
        """
        q1 = QuaternionOps.normalize(q1)
        q2 = QuaternionOps.normalize(q2)

        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        if np.abs(theta) < 1e-6:
            return QuaternionOps.normalize(q1 + t * (q2 - q1))

        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        return QuaternionOps.normalize(w1 * q1 + w2 * q2)


class MatrixOps:
    """Matrix operations for linear algebra."""

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric (cross-product) matrix from vector.

        Args:
            v: 3D vector [vx, vy, vz]

        Returns:
            3x3 skew-symmetric matrix [v]_x
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    @staticmethod
    def vee_map(S: np.ndarray) -> np.ndarray:
        """
        Extract vector from skew-symmetric matrix (inverse of skew_symmetric).

        Args:
            S: 3x3 skew-symmetric matrix

        Returns:
            3D vector
        """
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    @staticmethod
    def pseudoinverse(A: np.ndarray, rcond: float = 1e-5) -> np.ndarray:
        """
        Compute Moore-Penrose pseudoinverse.

        Args:
            A: Matrix
            rcond: Relative condition number threshold

        Returns:
            Pseudoinverse A^+
        """
        return np.linalg.pinv(A, rcond=rcond)

    @staticmethod
    def null_space(A: np.ndarray, rtol: float = 1e-5) -> np.ndarray:
        """
        Find orthonormal basis for null space of A.

        Args:
            A: Matrix
            rtol: Relative tolerance for singular values

        Returns:
            Basis vectors as columns of matrix
        """
        _, s, Vh = np.linalg.svd(A)
        null_mask = s <= rtol * s[0]
        return Vh[null_mask].T

    @staticmethod
    def lyapunov_equation(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Solve continuous Lyapunov equation: A P + P A^T = -Q.

        Args:
            A: System matrix
            Q: Symmetric positive definite matrix

        Returns:
            Solution matrix P
        """
        return scipy.linalg.solve_lyapunov(A, -Q)

    @staticmethod
    def matrix_exponential(A: np.ndarray, t: float) -> np.ndarray:
        """
        Compute matrix exponential e^(At).

        Args:
            A: Matrix
            t: Time (scalar)

        Returns:
            Matrix exponential
        """
        try:
            from scipy.linalg import expm
            return expm(A * t)
        except ImportError:
            # Fallback: use eigendecomposition
            eigvals, eigvecs = np.linalg.eig(A)
            exp_eigvals = np.diag(np.exp(eigvals * t))
            return eigvecs @ exp_eigvals @ np.linalg.inv(eigvecs)


class CoordinateTransform:
    """Coordinate system transformations."""

    @staticmethod
    def body_to_world(position: np.ndarray, velocity_body: np.ndarray,
                      q: np.ndarray) -> np.ndarray:
        """
        Transform velocity from body frame to world frame.

        Args:
            position: 3D position in world frame (not used but kept for API consistency)
            velocity_body: 3D velocity in body frame
            q: Quaternion representing attitude (body to world)

        Returns:
            Velocity in world frame
        """
        R = QuaternionOps.to_rotation_matrix(q)
        return R @ velocity_body

    @staticmethod
    def world_to_body(position: np.ndarray, velocity_world: np.ndarray,
                      q: np.ndarray) -> np.ndarray:
        """
        Transform velocity from world frame to body frame.

        Args:
            position: 3D position in world frame (not used)
            velocity_world: 3D velocity in world frame
            q: Quaternion representing attitude

        Returns:
            Velocity in body frame
        """
        R = QuaternionOps.to_rotation_matrix(q)
        return R.T @ velocity_world

    @staticmethod
    def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF.

        Args:
            lat: Latitude (radians)
            lon: Longitude (radians)
            alt: Altitude (meters)

        Returns:
            ECEF position [x, y, z]
        """
        # WGS84 parameters
        a = 6378137.0  # Semi-major axis (m)
        e2 = 0.00669438  # First eccentricity squared

        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt) * np.sin(lat)

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_geodetic(ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF to geodetic coordinates.

        Args:
            ecef: ECEF position [x, y, z]

        Returns:
            (latitude, longitude, altitude) in radians and meters
        """
        x, y, z = ecef
        a = 6378137.0
        e2 = 0.00669438

        lon = np.arctan2(y, x)

        p = np.sqrt(x**2 + y**2)
        lat_0 = np.arctan2(z, p * (1 - e2))

        # Iterative refinement
        for _ in range(10):
            N = a / np.sqrt(1 - e2 * np.sin(lat_0)**2)
            lat = np.arctan2(z + e2 * N * np.sin(lat_0), p)
            if np.abs(lat - lat_0) < 1e-12:
                break
            lat_0 = lat

        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N

        return lat, lon, alt

    @staticmethod
    def ned_to_enu(ned: np.ndarray) -> np.ndarray:
        """
        Convert NED (North-East-Down) to ENU (East-North-Up).

        Args:
            ned: NED coordinates [N, E, D]

        Returns:
            ENU coordinates [E, N, U]
        """
        N, E, D = ned
        return np.array([E, N, -D])

    @staticmethod
    def enu_to_ned(enu: np.ndarray) -> np.ndarray:
        """
        Convert ENU to NED.

        Args:
            enu: ENU coordinates [E, N, U]

        Returns:
            NED coordinates [N, E, D]
        """
        E, N, U = enu
        return np.array([N, E, -U])


class GeometryOps:
    """Geometric computations."""

    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Check if 2D point is inside polygon using ray casting.

        Args:
            point: 2D point [x, y]
            polygon: Nx2 array of polygon vertices

        Returns:
            True if point is inside, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def closest_point_on_segment(point: np.ndarray, seg_start: np.ndarray,
                                  seg_end: np.ndarray) -> np.ndarray:
        """
        Find closest point on line segment to given point.

        Args:
            point: 2D or 3D point
            seg_start, seg_end: Segment endpoints

        Returns:
            Closest point on segment
        """
        t = max(0, min(1, np.dot(point - seg_start, seg_end - seg_start) /
                       np.dot(seg_end - seg_start, seg_end - seg_start)))
        return seg_start + t * (seg_end - seg_start)

    @staticmethod
    def distance_point_to_segment(point: np.ndarray, seg_start: np.ndarray,
                                   seg_end: np.ndarray) -> float:
        """
        Compute distance from point to line segment.

        Args:
            point: 2D or 3D point
            seg_start, seg_end: Segment endpoints

        Returns:
            Minimum distance
        """
        closest = GeometryOps.closest_point_on_segment(point, seg_start, seg_end)
        return np.linalg.norm(point - closest)

    @staticmethod
    def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
        """
        Compute centroid of polygon.

        Args:
            polygon: Nx2 array of vertices

        Returns:
            2D centroid
        """
        return np.mean(polygon, axis=0)

    @staticmethod
    def polygon_area(polygon: np.ndarray) -> float:
        """
        Compute area of 2D polygon using shoelace formula.

        Args:
            polygon: Nx2 array of vertices

        Returns:
            Area
        """
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i, 0] * polygon[j, 1]
            area -= polygon[j, 0] * polygon[i, 1]
        return abs(area) / 2.0


class NumericalOps:
    """Numerical computation utilities."""

    @staticmethod
    def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Clip vector norm to maximum value.

        Args:
            v: Vector
            max_norm: Maximum allowed norm

        Returns:
            Clipped vector
        """
        norm = np.linalg.norm(v)
        if norm > max_norm:
            return v * (max_norm / norm)
        return v

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-π, π].

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    @staticmethod
    def angle_diff(angle1: float, angle2: float) -> float:
        """
        Compute shortest angular difference.

        Args:
            angle1, angle2: Angles in radians

        Returns:
            Shortest difference (magnitude ≤ π)
        """
        diff = angle2 - angle1
        return NumericalOps.normalize_angle(diff)

    @staticmethod
    def smooth_step(x: float, x_min: float = 0, x_max: float = 1) -> float:
        """
        Smoothstep function for smooth transitions.

        Args:
            x: Input value
            x_min, x_max: Transition range

        Returns:
            Smoothly interpolated value in [0, 1]
        """
        if x <= x_min:
            return 0.0
        if x >= x_max:
            return 1.0
        t = (x - x_min) / (x_max - x_min)
        return t**2 * (3 - 2 * t)

    @staticmethod
    def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute moving average.

        Args:
            values: 1D array of values
            window_size: Window size

        Returns:
            Moving average array
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(values, kernel, mode='valid')
