"""
Unit conversions and coordinate transformations for ISR-RL-DMPC.

Provides utilities for converting between different unit systems and
coordinate representations commonly used in robotics.
"""

import numpy as np
from typing import Tuple, Union, List


class UnitConversions:
    """Unit conversion utilities."""

    # Conversion factors
    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEG = 180.0 / np.pi
    KMH_TO_MS = 1.0 / 3.6
    MS_TO_KMH = 3.6
    FEET_TO_M = 0.3048
    M_TO_FEET = 1.0 / 0.3048
    KNOTS_TO_MS = 0.51444
    MS_TO_KNOTS = 1.0 / 0.51444
    LBS_TO_KG = 0.453592
    KG_TO_LBS = 1.0 / 0.453592
    DEG_C_TO_K = 273.15
    DEG_F_TO_C = 5.0 / 9.0

    @staticmethod
    def deg_to_rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert degrees to radians."""
        return deg * UnitConversions.DEG_TO_RAD

    @staticmethod
    def rad_to_deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert radians to degrees."""
        return rad * UnitConversions.RAD_TO_DEG

    @staticmethod
    def kmh_to_ms(kmh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert km/h to m/s."""
        return kmh * UnitConversions.KMH_TO_MS

    @staticmethod
    def ms_to_kmh(ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert m/s to km/h."""
        return ms * UnitConversions.MS_TO_KMH

    @staticmethod
    def feet_to_m(feet: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert feet to meters."""
        return feet * UnitConversions.FEET_TO_M

    @staticmethod
    def m_to_feet(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert meters to feet."""
        return m * UnitConversions.M_TO_FEET

    @staticmethod
    def knots_to_ms(knots: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert nautical knots to m/s."""
        return knots * UnitConversions.KNOTS_TO_MS

    @staticmethod
    def ms_to_knots(ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert m/s to nautical knots."""
        return ms * UnitConversions.MS_TO_KNOTS

    @staticmethod
    def lbs_to_kg(lbs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert pounds to kilograms."""
        return lbs * UnitConversions.LBS_TO_KG

    @staticmethod
    def kg_to_lbs(kg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert kilograms to pounds."""
        return kg * UnitConversions.KG_TO_LBS

    @staticmethod
    def celsius_to_kelvin(celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Celsius to Kelvin."""
        return celsius + UnitConversions.DEG_C_TO_K

    @staticmethod
    def kelvin_to_celsius(kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Kelvin to Celsius."""
        return kelvin - UnitConversions.DEG_C_TO_K

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32.0) * UnitConversions.DEG_F_TO_C

    @staticmethod
    def celsius_to_fahrenheit(celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Celsius to Fahrenheit."""
        return celsius * (9.0 / 5.0) + 32.0

    @staticmethod
    def dbm_to_watts(dbm: float) -> float:
        """Convert dBm to watts."""
        return 10.0 ** ((dbm - 30.0) / 10.0)

    @staticmethod
    def watts_to_dbm(watts: float) -> float:
        """Convert watts to dBm."""
        return 10.0 * np.log10(watts * 1000.0)

    @staticmethod
    def db_to_linear(db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert decibels to linear ratio."""
        return 10.0 ** (db / 10.0)

    @staticmethod
    def linear_to_db(linear: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert linear ratio to decibels."""
        return 10.0 * np.log10(linear)


class AttitudeConversions:
    """Convert between different attitude representations."""

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to quaternion.

        Args:
            roll: Roll angle (x-axis rotation) in radians
            pitch: Pitch angle (y-axis rotation) in radians
            yaw: Yaw angle (z-axis rotation) in radians

        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        # Half angles
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])

    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles.

        Args:
            q: Quaternion [qw, qx, qy, qz]

        Returns:
            (roll, pitch, yaw) in radians
        """
        qw, qx, qy, qz = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def dcm_to_quaternion(dcm: np.ndarray) -> np.ndarray:
        """
        Convert Direction Cosine Matrix (3x3 rotation matrix) to quaternion.

        Args:
            dcm: 3x3 rotation matrix

        Returns:
            Quaternion [qw, qx, qy, qz]
        """
        trace = np.trace(dcm)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (dcm[2, 1] - dcm[1, 2]) * s
            qy = (dcm[0, 2] - dcm[2, 0]) * s
            qz = (dcm[1, 0] - dcm[0, 1]) * s
        elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
            qw = (dcm[2, 1] - dcm[1, 2]) / s
            qx = 0.25 * s
            qy = (dcm[0, 1] + dcm[1, 0]) / s
            qz = (dcm[0, 2] + dcm[2, 0]) / s
        elif dcm[1, 1] > dcm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
            qw = (dcm[0, 2] - dcm[2, 0]) / s
            qx = (dcm[0, 1] + dcm[1, 0]) / s
            qy = 0.25 * s
            qz = (dcm[1, 2] + dcm[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
            qw = (dcm[1, 0] - dcm[0, 1]) / s
            qx = (dcm[0, 2] + dcm[2, 0]) / s
            qy = (dcm[1, 2] + dcm[2, 1]) / s
            qz = 0.25 * s

        # Normalize
        q_norm = np.linalg.norm([qw, qx, qy, qz])
        return np.array([qw, qx, qy, qz]) / q_norm

    @staticmethod
    def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Direction Cosine Matrix.

        Args:
            q: Quaternion [qw, qx, qy, qz]

        Returns:
            3x3 rotation matrix
        """
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q

        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])

    @staticmethod
    def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Convert axis-angle to quaternion.

        Args:
            axis: 3D rotation axis (normalized)
            angle: Rotation angle in radians

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
    def quaternion_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle.

        Args:
            q: Quaternion [qw, qx, qy, qz]

        Returns:
            (axis, angle) where axis is normalized 3D vector
        """
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        qw = np.clip(q[0], -1.0, 1.0)

        angle = 2.0 * np.arccos(qw)
        sin_half_angle = np.sqrt(1.0 - qw**2)

        if sin_half_angle < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = q[1:4] / sin_half_angle

        return axis, angle


class PositionProjections:
    """Project positions between different coordinate systems."""

    @staticmethod
    def spherical_to_cartesian(azimuth: float, elevation: float,
                              radius: float) -> np.ndarray:
        """
        Convert spherical coordinates to Cartesian.

        Args:
            azimuth: Azimuth angle in radians (bearing from north)
            elevation: Elevation angle in radians (from horizon)
            radius: Radial distance in meters

        Returns:
            3D Cartesian position [x, y, z]
        """
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.cos(elevation) * np.cos(azimuth)
        z = radius * np.sin(elevation)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical.

        Args:
            x, y, z: Cartesian coordinates in meters

        Returns:
            (azimuth, elevation, radius) in radians and meters
        """
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(x, y)
        elevation = np.arcsin(z / (radius + 1e-10))
        return azimuth, elevation, radius

    @staticmethod
    def cylindrical_to_cartesian(theta: float, rho: float, z: float) -> np.ndarray:
        """
        Convert cylindrical coordinates to Cartesian.

        Args:
            theta: Azimuth angle in radians
            rho: Radial distance in xy-plane
            z: Height in meters

        Returns:
            3D Cartesian position [x, y, z]
        """
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return np.array([x, y, z])

    @staticmethod
    def cartesian_to_cylindrical(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to cylindrical.

        Args:
            x, y, z: Cartesian coordinates

        Returns:
            (theta, rho, z) where theta is azimuth, rho is radial distance
        """
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return theta, rho, z

    @staticmethod
    def relative_to_absolute(relative_pos: np.ndarray, reference_pos: np.ndarray,
                            reference_orientation: np.ndarray) -> np.ndarray:
        """
        Convert position relative to a reference frame to absolute coordinates.

        Args:
            relative_pos: Position relative to reference frame
            reference_pos: Position of reference frame in absolute coordinates
            reference_orientation: Orientation of reference frame (quaternion)

        Returns:
            Absolute position
        """
        from . import math_utils
        # Rotate relative position by reference orientation
        rotated = math_utils.QuaternionOps.rotate_vector(reference_orientation, relative_pos)
        # Add reference position
        return reference_pos + rotated

    @staticmethod
    def absolute_to_relative(absolute_pos: np.ndarray, reference_pos: np.ndarray,
                            reference_orientation: np.ndarray) -> np.ndarray:
        """
        Convert absolute position to position relative to a reference frame.

        Args:
            absolute_pos: Absolute position
            reference_pos: Position of reference frame
            reference_orientation: Orientation of reference frame (quaternion)

        Returns:
            Position relative to reference frame
        """
        from . import math_utils
        # Translate to reference origin
        translated = absolute_pos - reference_pos
        # Inverse rotate by reference orientation
        inverse_q = math_utils.QuaternionOps.conjugate(reference_orientation)
        return math_utils.QuaternionOps.rotate_vector(inverse_q, translated)


class BearingDistance:
    """Convert between Cartesian and bearing-distance representations."""

    @staticmethod
    def cartesian_to_bearing_distance(position: np.ndarray,
                                     reference_pos: np.ndarray = np.array([0, 0, 0]),
                                     reference_heading: float = 0) -> Tuple[float, float]:
        """
        Convert position to bearing and distance from reference.

        Args:
            position: Target position [x, y, z]
            reference_pos: Reference position [x, y, z]
            reference_heading: Reference heading in radians (0 = North)

        Returns:
            (bearing, distance) where bearing is in radians [0, 2π]
        """
        relative = position[:2] - reference_pos[:2]
        distance = np.linalg.norm(relative)

        # Bearing relative to North (positive y-axis)
        bearing = np.arctan2(relative[0], relative[1]) - reference_heading

        # Normalize bearing to [0, 2π]
        bearing = bearing % (2 * np.pi)

        return bearing, distance

    @staticmethod
    def bearing_distance_to_cartesian(bearing: float, distance: float,
                                     reference_pos: np.ndarray = np.array([0, 0, 0]),
                                     reference_heading: float = 0,
                                     altitude: float = 0) -> np.ndarray:
        """
        Convert bearing and distance to Cartesian position.

        Args:
            bearing: Bearing angle in radians
            distance: Distance in meters
            reference_pos: Reference position [x, y, z]
            reference_heading: Reference heading in radians
            altitude: Altitude above reference

        Returns:
            Cartesian position [x, y, z]
        """
        # Adjust bearing by reference heading
        absolute_bearing = bearing + reference_heading

        x = reference_pos[0] + distance * np.sin(absolute_bearing)
        y = reference_pos[1] + distance * np.cos(absolute_bearing)
        z = reference_pos[2] + altitude

        return np.array([x, y, z])


class TimeConversions:
    """Time-related conversions."""

    @staticmethod
    def seconds_to_hhmmss(seconds: float) -> str:
        """
        Convert seconds to HH:MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def hhmmss_to_seconds(time_str: str) -> float:
        """
        Convert HH:MM:SS format to seconds.

        Args:
            time_str: Formatted string

        Returns:
            Time in seconds
        """
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def days_to_seconds(days: float) -> float:
        """Convert days to seconds."""
        return days * 24 * 3600

    @staticmethod
    def hours_to_seconds(hours: float) -> float:
        """Convert hours to seconds."""
        return hours * 3600

    @staticmethod
    def minutes_to_seconds(minutes: float) -> float:
        """Convert minutes to seconds."""
        return minutes * 60
