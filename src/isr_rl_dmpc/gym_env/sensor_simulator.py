"""
Synthetic sensor data generator for the ISR-RL-DMPC simulation.

Complements the physics-based ``simulator.py`` by producing realistic noisy
sensor readings (radar, optical, RF fingerprinting, acoustic TDOA) that feed
into the target tracking EKF.

Noise models are range-dependent and weather-aware so that training
environments can explore degraded-sensor conditions.

Classes:
    SensorNoiseModel: Configurable noise sampler with range/weather scaling.
    SensorSimulator: Multi-sensor measurement generator.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

from ..core import (
    DroneState,
    TargetState,
    SensorType,
    RadarMeasurement,
    OpticalMeasurement,
    RFMeasurement,
    AcousticMeasurement,
)

logger = logging.getLogger(__name__)


class SensorNoiseModel:
    """
    Configurable noise sampler for a single sensor modality.

    The noise standard deviation grows linearly with distance when
    ``range_dependent`` is enabled and is scaled by a weather factor
    (1.0 = clear conditions, >1.0 = degraded).

    Noise formula:
        σ_eff = base_noise * (1 + distance / 1000) * weather_factor
        noise ~ N(0, σ_eff)
    """

    def __init__(
        self,
        base_noise: float,
        range_dependent: bool = True,
        weather_factor: float = 1.0,
    ):
        """
        Initialize noise model.

        Args:
            base_noise: Base standard deviation of the noise.
            range_dependent: Scale noise with target distance.
            weather_factor: Multiplicative weather degradation (≥1.0).
        """
        self.base_noise = base_noise
        self.range_dependent = range_dependent
        self.weather_factor = max(weather_factor, 0.0)

    def sample(self, distance: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Draw a single noise sample.

        Args:
            distance: Distance to target in metres.
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            Scalar noise value drawn from N(0, σ_eff).
        """
        sigma = self.base_noise * self.weather_factor
        if self.range_dependent:
            sigma *= 1.0 + distance / 1000.0
        if rng is not None:
            return float(rng.normal(0.0, sigma))
        return float(np.random.normal(0.0, sigma))


class SensorSimulator:
    """
    Multi-sensor synthetic measurement generator.

    Produces ``RadarMeasurement``, ``OpticalMeasurement``,
    ``RFMeasurement``, and ``AcousticMeasurement`` objects with
    configurable noise that can be fed directly into the
    ``TargetTrackingEKF``.

    Weather conditions (visibility, precipitation, wind) are
    mapped to per-sensor weather factors for realistic degradation.
    """

    def __init__(
        self,
        radar_noise: float = 0.5,
        optical_noise: float = 0.01,
        rf_noise: float = 1.0,
        acoustic_noise: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Initialize sensor simulator.

        Args:
            radar_noise: Base noise std for radar range (metres).
            optical_noise: Base noise std for optical angles (radians).
            rf_noise: Base noise std for RF position estimate (metres).
            acoustic_noise: Base noise std for acoustic position (metres).
            seed: Random seed for reproducibility (``None`` = non-deterministic).
        """
        self._rng = np.random.default_rng(seed)
        self._timestamp: float = 0.0

        self.radar_model = SensorNoiseModel(radar_noise)
        self.optical_model = SensorNoiseModel(optical_noise)
        self.rf_model = SensorNoiseModel(rf_noise)
        self.acoustic_model = SensorNoiseModel(acoustic_noise)

        # Weather state
        self._visibility: float = 1.0  # [0, 1]
        self._precipitation: float = 0.0  # [0, 1]
        self._wind_speed: float = 0.0  # m/s

        logger.info(
            "SensorSimulator initialized (radar=%.2f, optical=%.3f, "
            "rf=%.2f, acoustic=%.3f, seed=%s)",
            radar_noise,
            optical_noise,
            rf_noise,
            acoustic_noise,
            seed,
        )

    # ------------------------------------------------------------------
    # Weather configuration
    # ------------------------------------------------------------------

    def set_weather_conditions(
        self,
        visibility: float = 1.0,
        precipitation: float = 0.0,
        wind_speed: float = 0.0,
    ) -> None:
        """
        Update weather state and propagate to sensor noise models.

        Args:
            visibility: Visibility factor in [0, 1] (1 = clear).
            precipitation: Precipitation intensity in [0, 1].
            wind_speed: Wind speed in m/s.
        """
        self._visibility = np.clip(visibility, 0.0, 1.0)
        self._precipitation = np.clip(precipitation, 0.0, 1.0)
        self._wind_speed = max(wind_speed, 0.0)

        # Radar is degraded by precipitation
        self.radar_model.weather_factor = 1.0 + 2.0 * self._precipitation

        # Optical is degraded by low visibility and precipitation
        vis_degradation = 1.0 / max(self._visibility, 0.01)
        self.optical_model.weather_factor = vis_degradation + self._precipitation

        # RF is relatively weather-agnostic
        self.rf_model.weather_factor = 1.0 + 0.1 * self._precipitation

        # Acoustic is degraded by wind
        self.acoustic_model.weather_factor = 1.0 + 0.5 * self._wind_speed / 10.0

        logger.debug(
            "Weather updated: vis=%.2f, precip=%.2f, wind=%.1f m/s",
            self._visibility,
            self._precipitation,
            self._wind_speed,
        )

    # ------------------------------------------------------------------
    # Individual sensor generators
    # ------------------------------------------------------------------

    def generate_radar(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> RadarMeasurement:
        """
        Generate a noisy radar measurement.

        Computes true range, range-rate, azimuth, and elevation from
        geometry and adds sensor noise.

        Args:
            drone_pos: Drone position (3,).
            target_pos: Target position (3,).
            target_vel: Target velocity (3,).

        Returns:
            ``RadarMeasurement`` with noisy range, range_rate, azimuth,
            elevation.
        """
        diff = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
        distance = float(np.linalg.norm(diff))
        distance = max(distance, 1e-6)

        # True values
        unit = diff / distance
        true_range = distance
        true_range_rate = float(np.dot(target_vel, unit))
        true_azimuth = float(np.arctan2(diff[1], diff[0]))
        true_elevation = float(np.arctan2(diff[2], np.linalg.norm(diff[:2])))

        # Add noise
        noisy_range = true_range + self.radar_model.sample(distance, self._rng)
        noisy_range_rate = true_range_rate + self.radar_model.sample(distance, self._rng) * 0.5
        noisy_azimuth = true_azimuth + self.radar_model.sample(distance, self._rng) * 0.01
        noisy_elevation = true_elevation + self.radar_model.sample(distance, self._rng) * 0.01

        return RadarMeasurement(
            range=max(noisy_range, 0.0),
            range_rate=noisy_range_rate,
            azimuth=noisy_azimuth,
            elevation=noisy_elevation,
            timestamp=self._timestamp,
        )

    def generate_optical(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> OpticalMeasurement:
        """
        Generate a noisy optical bearing measurement.

        Args:
            drone_pos: Drone position (3,).
            target_pos: Target position (3,).

        Returns:
            ``OpticalMeasurement`` with noisy azimuth and elevation.
        """
        diff = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
        distance = float(np.linalg.norm(diff))
        distance = max(distance, 1e-6)

        true_azimuth = float(np.arctan2(diff[1], diff[0]))
        true_elevation = float(np.arctan2(diff[2], np.linalg.norm(diff[:2])))

        noisy_azimuth = true_azimuth + self.optical_model.sample(distance, self._rng)
        noisy_elevation = true_elevation + self.optical_model.sample(distance, self._rng)

        return OpticalMeasurement(
            azimuth=noisy_azimuth,
            elevation=noisy_elevation,
            timestamp=self._timestamp,
        )

    def generate_rf(
        self,
        target_pos: np.ndarray,
        signal_power: float = -30.0,
    ) -> RFMeasurement:
        """
        Generate a noisy RF fingerprinting position estimate.

        Args:
            target_pos: Target position (3,).
            signal_power: Received signal power in dBm (higher → better SNR).

        Returns:
            ``RFMeasurement`` with noisy position and confidence.
        """
        pos = np.asarray(target_pos, dtype=np.float64)
        # Stronger signal → less noise
        snr_factor = max(1.0 - (signal_power + 50.0) / 40.0, 0.1)
        noise_x = self.rf_model.sample(0.0, self._rng) * snr_factor
        noise_y = self.rf_model.sample(0.0, self._rng) * snr_factor
        noise_z = self.rf_model.sample(0.0, self._rng) * snr_factor

        confidence = float(np.clip(1.0 / (1.0 + snr_factor), 0.0, 1.0))

        return RFMeasurement(
            x_est=pos[0] + noise_x,
            y_est=pos[1] + noise_y,
            z_est=pos[2] + noise_z,
            confidence=confidence,
            timestamp=self._timestamp,
        )

    def generate_acoustic(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> AcousticMeasurement:
        """
        Generate a noisy acoustic TDOA position estimate.

        Args:
            drone_pos: Drone position (3,).
            target_pos: Target position (3,).

        Returns:
            ``AcousticMeasurement`` with noisy position and confidence.
        """
        pos = np.asarray(target_pos, dtype=np.float64)
        diff = pos - np.asarray(drone_pos, dtype=np.float64)
        distance = float(np.linalg.norm(diff))

        noise_x = self.acoustic_model.sample(distance, self._rng)
        noise_y = self.acoustic_model.sample(distance, self._rng)
        noise_z = self.acoustic_model.sample(distance, self._rng)

        # Confidence decays with distance
        confidence = float(np.clip(np.exp(-distance / 500.0), 0.0, 1.0))

        return AcousticMeasurement(
            x_est=pos[0] + noise_x,
            y_est=pos[1] + noise_y,
            z_est=pos[2] + noise_z,
            confidence=confidence,
            timestamp=self._timestamp,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def generate_all_sensors(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        target_vel: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generate measurements from all sensor modalities.

        Args:
            drone_pos: Drone position (3,).
            target_pos: Target position (3,).
            target_vel: Target velocity (3,). Defaults to zero if ``None``.

        Returns:
            Dictionary mapping sensor name to its measurement dataclass::

                {
                    "radar": RadarMeasurement,
                    "optical": OpticalMeasurement,
                    "rf": RFMeasurement,
                    "acoustic": AcousticMeasurement,
                }
        """
        if target_vel is None:
            target_vel = np.zeros(3)

        self._timestamp += 1.0

        return {
            "radar": self.generate_radar(drone_pos, target_pos, target_vel),
            "optical": self.generate_optical(drone_pos, target_pos),
            "rf": self.generate_rf(target_pos),
            "acoustic": self.generate_acoustic(drone_pos, target_pos),
        }
