"""
Simple 2.5D forward-FOV sensing for belief-based coverage.

This Phase 1 model intentionally stays lightweight:

* fixed-altitude planar coverage
* 135 degree forward field of view
* distance and heading-aware observation quality
* small Gaussian noise, but no ray tracing or camera rendering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VisionSensorConfig:
    """Configuration for the forward FOV sensor model."""

    fov_deg: float = 135.0
    max_range: float = 120.0
    distance_decay: float = 1.2
    angular_decay: float = 1.0
    noise_std: float = 0.02
    min_quality: float = 0.0

    @property
    def half_fov_rad(self) -> float:
        return np.deg2rad(self.fov_deg * 0.5)


class ForwardFOVSensorModel:
    """Compute visible cells and observation quality from pose and yaw."""

    def __init__(self, config: Optional[VisionSensorConfig] = None) -> None:
        self.config = config or VisionSensorConfig()

    def visible_mask(
        self,
        drone_position_xy: np.ndarray,
        yaw: float,
        cell_centers_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return visibility mask, horizontal distances, and absolute angular offsets.
        """
        drone_xy = np.asarray(drone_position_xy, dtype=np.float64)
        centers = np.asarray(cell_centers_xy, dtype=np.float64)
        rel = centers - drone_xy[None, :]
        distances = np.linalg.norm(rel, axis=1)

        bearings = np.arctan2(rel[:, 1], rel[:, 0])
        angular_offset = np.abs(np.arctan2(np.sin(bearings - yaw), np.cos(bearings - yaw)))

        visible = (
            (distances <= self.config.max_range)
            & (angular_offset <= self.config.half_fov_rad)
        )
        return visible, distances, angular_offset

    def compute_quality(
        self,
        distances: np.ndarray,
        angular_offsets: np.ndarray,
        *,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Compute noisy observation quality in ``[0, 1]``."""
        dist = np.asarray(distances, dtype=np.float64)
        ang = np.asarray(angular_offsets, dtype=np.float64)

        range_norm = np.clip(dist / max(self.config.max_range, 1e-6), 0.0, 1.0)
        angle_norm = np.clip(ang / max(self.config.half_fov_rad, 1e-6), 0.0, 1.0)

        distance_term = np.exp(-self.config.distance_decay * range_norm ** 2)
        angle_term = np.cos(ang)
        angle_term = np.clip(angle_term, 0.0, 1.0) ** self.config.angular_decay

        quality = distance_term * angle_term

        if rng is not None and self.config.noise_std > 0.0:
            quality = quality + rng.normal(0.0, self.config.noise_std, size=quality.shape)

        return np.clip(quality, self.config.min_quality, 1.0)

    def observe_cells(
        self,
        drone_position_xy: np.ndarray,
        yaw: float,
        cell_centers_xy: np.ndarray,
        *,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return visible cell indices, quality, distances, and angular offsets.
        """
        visible, distances, angular_offsets = self.visible_mask(
            drone_position_xy,
            yaw,
            cell_centers_xy,
        )
        indices = np.flatnonzero(visible)
        if indices.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return indices.astype(np.int32), empty, empty, empty

        quality = self.compute_quality(
            distances[indices],
            angular_offsets[indices],
            rng=rng,
        )
        return (
            indices.astype(np.int32),
            quality.astype(np.float64),
            distances[indices].astype(np.float64),
            angular_offsets[indices].astype(np.float64),
        )
