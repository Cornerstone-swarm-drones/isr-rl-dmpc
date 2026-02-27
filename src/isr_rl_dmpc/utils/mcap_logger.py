"""
MCAP recording logger for ISR-RL-DMPC simulation data.

Records simulation data in the MCAP format for offline playback and analysis
in Foxglove Studio. MCAP is a high-performance, indexed log format designed
for multimodal robotics data.

Usage:
    recorder = MCAPRecorder("mission_recording.mcap")
    recorder.start()

    # During simulation loop:
    recorder.record_drone_states(drone_positions, timestamp_ns)
    recorder.record_metrics(info, reward, timestamp_ns)

    recorder.stop()

Playback:
    Open the .mcap file in Foxglove Studio for full playback with timeline
    scrubbing, synchronized visualization, and data inspection.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from mcap.writer import Writer

logger = logging.getLogger(__name__)


def _now_ns() -> int:
    """Get current time in nanoseconds."""
    return int(time.time() * 1e9)


class MCAPRecorder:
    """
    Record simulation data to an MCAP file for Foxglove Studio playback.

    Creates indexed, compressed MCAP files containing drone states, target
    states, coverage grids, and mission metrics — all synchronized by
    timestamp for frame-accurate playback.

    Args:
        filepath: Path to the output .mcap file
    """

    def __init__(self, filepath: str = "recording.mcap"):
        self.filepath = Path(filepath)
        self._writer: Optional[Writer] = None
        self._file = None
        self._channels: Dict[str, int] = {}
        self._schemas: Dict[str, int] = {}
        self._started = False

    def start(self) -> None:
        """Open the MCAP file and register schemas and channels."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "wb")
        self._writer = Writer(self._file)
        self._writer.start(profile="", library="isr-rl-dmpc")

        # Register schemas
        self._schemas["scene"] = self._writer.register_schema(
            name="foxglove.SceneUpdate",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "deletions": {"type": "array"},
                    "entities": {"type": "array"},
                },
            }).encode("utf-8"),
        )

        self._schemas["metrics"] = self._writer.register_schema(
            name="isr.SwarmMetrics",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "coverage": {"type": "number"},
                    "avg_battery": {"type": "number"},
                    "active_drones": {"type": "integer"},
                    "total_drones": {"type": "integer"},
                    "collisions": {"type": "integer"},
                    "reward": {"type": "number"},
                },
            }).encode("utf-8"),
        )

        self._schemas["coverage"] = self._writer.register_schema(
            name="isr.CoverageGrid",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "grid_size": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "coverage_map": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "coverage_percentage": {"type": "number"},
                },
            }).encode("utf-8"),
        )

        self._schemas["mission_info"] = self._writer.register_schema(
            name="isr.MissionInfo",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "elapsed_time": {"type": "number"},
                    "mission_duration": {"type": "number"},
                    "progress": {"type": "number"},
                    "coverage_efficiency": {"type": "number"},
                    "num_waypoints": {"type": "integer"},
                },
            }).encode("utf-8"),
        )

        # Register channels
        self._channels["scene"] = self._writer.register_channel(
            topic="/swarm/scene",
            message_encoding="json",
            schema_id=self._schemas["scene"],
        )

        self._channels["metrics"] = self._writer.register_channel(
            topic="/swarm/metrics",
            message_encoding="json",
            schema_id=self._schemas["metrics"],
        )

        self._channels["coverage"] = self._writer.register_channel(
            topic="/mission/coverage",
            message_encoding="json",
            schema_id=self._schemas["coverage"],
        )

        self._channels["mission_info"] = self._writer.register_channel(
            topic="/mission/info",
            message_encoding="json",
            schema_id=self._schemas["mission_info"],
        )

        self._started = True
        logger.info("MCAP recording started: %s", self.filepath)

    def stop(self) -> None:
        """Finalize and close the MCAP file."""
        if self._writer is not None:
            self._writer.finish()
        if self._file is not None:
            self._file.close()
        self._started = False
        logger.info("MCAP recording saved: %s", self.filepath)

    @property
    def is_recording(self) -> bool:
        """Check if the recorder is actively recording."""
        return self._started

    def _write_message(
        self, channel_key: str, data: Dict[str, Any], timestamp_ns: int
    ) -> None:
        """Write a JSON message to the MCAP file."""
        if not self._started or self._writer is None:
            return
        chan_id = self._channels.get(channel_key)
        if chan_id is None:
            return
        payload = json.dumps(data).encode("utf-8")
        self._writer.add_message(
            channel_id=chan_id,
            log_time=timestamp_ns,
            data=payload,
            publish_time=timestamp_ns,
        )

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_scene(
        self,
        drone_positions: np.ndarray,
        drone_quaternions: Optional[np.ndarray] = None,
        drone_batteries: Optional[np.ndarray] = None,
        target_positions: Optional[Dict[str, np.ndarray]] = None,
        target_classifications: Optional[Dict[str, str]] = None,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Record a 3D scene snapshot with drone and target markers.

        Args:
            drone_positions: (N, 3) array of drone positions
            drone_quaternions: (N, 4) array [qw, qx, qy, qz] (optional)
            drone_batteries: (N,) array of battery levels (optional)
            target_positions: dict of target_id -> (3,) position
            target_classifications: dict of target_id -> classification
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        ts = {"sec": int(timestamp_ns // 1_000_000_000),
              "nsec": int(timestamp_ns % 1_000_000_000)}

        entities = []

        # Drone models
        drone_models = []
        drone_texts = []
        for i in range(len(drone_positions)):
            pos = drone_positions[i]
            if drone_batteries is not None and len(drone_batteries) > i:
                batt_frac = float(np.clip(drone_batteries[i] / 5000.0, 0, 1))
                col = {"r": 1.0 - batt_frac, "g": batt_frac, "b": 0.2, "a": 0.9}
            else:
                col = {"r": 0.2, "g": 0.6, "b": 1.0, "a": 0.9}

            orient = {"x": 0, "y": 0, "z": 0, "w": 1}
            if drone_quaternions is not None and len(drone_quaternions) > i:
                q = drone_quaternions[i]
                orient = {"x": float(q[1]), "y": float(q[2]),
                          "z": float(q[3]), "w": float(q[0])}

            drone_models.append({
                "pose": {
                    "position": {"x": float(pos[0]), "y": float(pos[1]),
                                 "z": float(pos[2])},
                    "orientation": orient,
                },
                "scale": {"x": 2.0, "y": 2.0, "z": 2.0},
                "url": "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
                       "Apps/SampleData/models/CesiumDrone/CesiumDrone.glb",
                "media_type": "model/gltf-binary",
                "override_color": False,
                "color": col,
            })

            label = f"D{i}"
            if drone_batteries is not None and len(drone_batteries) > i:
                label += f" [{drone_batteries[i]:.0f}Wh]"
            drone_texts.append({
                "pose": {
                    "position": {"x": float(pos[0]), "y": float(pos[1]),
                                 "z": float(pos[2]) + 4},
                    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                },
                "billboard": True,
                "font_size": 14.0,
                "scale_invariant": True,
                "color": {"r": 1, "g": 1, "b": 1, "a": 1},
                "text": label,
            })

        entities.append({
            "timestamp": ts, "frame_id": "world", "id": "drones",
            "lifetime": {"sec": 0, "nsec": 0}, "frame_locked": False,
            "metadata": [], "arrows": [], "cubes": [], "spheres": [],
            "cylinders": [], "lines": [], "triangles": [],
            "texts": drone_texts, "models": drone_models,
        })

        # Target models
        if target_positions:
            _target_model_urls = {
                "hostile": (
                    "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
                    "Apps/SampleData/models/CesiumMilkTruck/CesiumMilkTruck.glb"
                ),
                "friendly": (
                    "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
                    "Apps/SampleData/models/CesiumAir/Cesium_Air.glb"
                ),
                "unknown": (
                    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/"
                    "master/2.0/Box/glTF-Binary/Box.glb"
                ),
            }
            target_models = []
            target_texts = []
            for tid, tpos in target_positions.items():
                cls_name = (target_classifications.get(tid, "unknown")
                            if target_classifications else "unknown")
                if cls_name == "hostile":
                    col = {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.8}
                elif cls_name == "friendly":
                    col = {"r": 0.0, "g": 1.0, "b": 0.0, "a": 0.8}
                else:
                    col = {"r": 1.0, "g": 1.0, "b": 0.0, "a": 0.8}

                model_url = _target_model_urls.get(
                    cls_name, _target_model_urls["unknown"]
                )

                target_models.append({
                    "pose": {
                        "position": {"x": float(tpos[0]), "y": float(tpos[1]),
                                     "z": float(tpos[2])},
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                    "scale": {"x": 3.0, "y": 3.0, "z": 3.0},
                    "url": model_url,
                    "media_type": "model/gltf-binary",
                    "override_color": False,
                    "color": col,
                })
                target_texts.append({
                    "pose": {
                        "position": {"x": float(tpos[0]), "y": float(tpos[1]),
                                     "z": float(tpos[2]) + 5},
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                    "billboard": True, "font_size": 12.0, "scale_invariant": True,
                    "color": {"r": 1, "g": 1, "b": 1, "a": 1},
                    "text": f"{tid} ({cls_name})",
                })

            entities.append({
                "timestamp": ts, "frame_id": "world", "id": "targets",
                "lifetime": {"sec": 0, "nsec": 0}, "frame_locked": False,
                "metadata": [], "arrows": [], "cubes": [], "spheres": [],
                "cylinders": [], "lines": [], "triangles": [],
                "texts": target_texts, "models": target_models,
            })

        self._write_message("scene", {"deletions": [], "entities": entities},
                            timestamp_ns)

    def record_metrics(
        self,
        info: Dict[str, Any],
        reward: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Record swarm metrics for time-series analysis.

        Args:
            info: Info dict from ISRGridEnv.step()
            reward: Reward value
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        self._write_message("metrics", {
            "step": int(info.get("step", 0)),
            "coverage": float(info.get("coverage", 0.0)),
            "avg_battery": float(info.get("avg_battery", 0.0)),
            "active_drones": int(info.get("active_drones", 0)),
            "total_drones": int(info.get("total_drones", 0)),
            "collisions": int(info.get("collisions", 0)),
            "reward": float(reward),
        }, timestamp_ns)

    def record_coverage(
        self,
        coverage_map: np.ndarray,
        grid_size: tuple,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Record coverage grid state.

        Args:
            coverage_map: 1D array of coverage values
            grid_size: (rows, cols) tuple
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        total = len(coverage_map)
        covered = float(np.sum(coverage_map > 0))
        pct = (covered / total * 100.0) if total > 0 else 0.0

        self._write_message("coverage", {
            "grid_size": list(grid_size),
            "coverage_map": coverage_map.tolist(),
            "coverage_percentage": pct,
        }, timestamp_ns)

    def record_mission_info(
        self,
        elapsed_time: float = 0.0,
        mission_duration: float = 3600.0,
        coverage_efficiency: float = 0.0,
        num_waypoints: int = 0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Record mission state information.

        Args:
            elapsed_time: Time since mission start (s)
            mission_duration: Total mission duration (s)
            coverage_efficiency: Coverage effectiveness [0, 1]
            num_waypoints: Number of remaining waypoints
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        progress = elapsed_time / mission_duration if mission_duration > 0 else 0.0
        self._write_message("mission_info", {
            "elapsed_time": float(elapsed_time),
            "mission_duration": float(mission_duration),
            "progress": float(np.clip(progress, 0, 1)),
            "coverage_efficiency": float(coverage_efficiency),
            "num_waypoints": int(num_waypoints),
        }, timestamp_ns)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
