"""
Foxglove Studio WebSocket bridge for real-time ISR-RL-DMPC visualization.

Provides a WebSocket server that streams simulation data to Foxglove Studio
for interactive 3D visualization, metrics monitoring, and mission analysis.

Usage:
    bridge = FoxgloveBridge(host="0.0.0.0", port=8765)
    await bridge.start()

    # During simulation loop:
    bridge.publish_drone_states(drone_states, timestamp_ns)
    bridge.publish_target_states(target_states, timestamp_ns)
    bridge.publish_mission_state(mission_state, timestamp_ns)

    await bridge.stop()

Foxglove Studio Connection:
    Open Foxglove Studio and connect via:
    ws://localhost:8765
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np

from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelWithoutId

logger = logging.getLogger(__name__)

# JSON schemas for Foxglove panels
_SCENE_UPDATE_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "deletions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "integer"},
                    "timestamp": {"type": "object"},
                },
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "object"},
                    "frame_id": {"type": "string"},
                    "id": {"type": "string"},
                    "lifetime": {"type": "object"},
                    "frame_locked": {"type": "boolean"},
                    "metadata": {"type": "array"},
                    "arrows": {"type": "array"},
                    "cubes": {"type": "array"},
                    "spheres": {"type": "array"},
                    "cylinders": {"type": "array"},
                    "lines": {"type": "array"},
                    "triangles": {"type": "array"},
                    "texts": {"type": "array"},
                    "models": {"type": "array"},
                },
            },
        },
    },
})

_LOCATION_FIX_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "altitude": {"type": "number"},
    },
})


def _timestamp_obj(ns: int) -> Dict[str, int]:
    """Convert nanosecond timestamp to Foxglove timestamp object."""
    return {"sec": int(ns // 1_000_000_000), "nsec": int(ns % 1_000_000_000)}


def _now_ns() -> int:
    """Get current time in nanoseconds."""
    return int(time.time() * 1e9)


def _color(r: float, g: float, b: float, a: float = 1.0) -> Dict[str, float]:
    """Create RGBA color dict."""
    return {"r": r, "g": g, "b": b, "a": a}


def _vec3(x: float, y: float, z: float) -> Dict[str, float]:
    """Create 3D vector dict."""
    return {"x": float(x), "y": float(y), "z": float(z)}


def _quat(x: float, y: float, z: float, w: float) -> Dict[str, float]:
    """Create quaternion dict (Foxglove uses x,y,z,w order)."""
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


_TARGET_TYPE_MAP = {0: "unknown", 1: "friendly", 2: "hostile", 3: "neutral"}


def extract_targets_from_obs(
    targets_obs: np.ndarray,
) -> tuple:
    """Extract target positions and classifications from the env observation.

    The ``targets`` observation is shaped ``(max_targets, 12)`` where each row
    contains:
        [0:3] position (x, y, z)
        [8]   target_type enum (0=unknown, 1=friendly, 2=hostile, 3=neutral)

    Rows that are all-zero are padding and are skipped.

    Returns:
        (target_positions, target_classifications) dicts keyed by target id.
    """
    positions: Dict[str, np.ndarray] = {}
    classifications: Dict[str, str] = {}
    for i in range(len(targets_obs)):
        row = targets_obs[i]
        # Skip zero-padded (inactive) targets
        if np.allclose(row[:3], 0.0):
            continue
        tid = f"T{i}"
        positions[tid] = row[:3].copy()
        ttype = int(round(float(row[8])))
        classifications[tid] = _TARGET_TYPE_MAP.get(ttype, "unknown")
    return positions, classifications


class _BridgeListener(FoxgloveServerListener):
    """Handles Foxglove client subscription events."""

    def __init__(self) -> None:
        self.subscribed_channels: set = set()

    async def on_subscribe(self, server: FoxgloveServer, channel_id: int) -> None:
        self.subscribed_channels.add(channel_id)

    async def on_unsubscribe(self, server: FoxgloveServer, channel_id: int) -> None:
        self.subscribed_channels.discard(channel_id)


class FoxgloveBridge:
    """
    WebSocket bridge for streaming ISR-RL-DMPC simulation data to Foxglove Studio.

    Publishes drone positions, target states, coverage grids, and mission metrics
    as Foxglove-compatible JSON messages over WebSocket.

    Channels:
        /swarm/scene       - 3D scene with drone and target markers
        /swarm/metrics     - Scalar metrics (coverage, battery, reward)
        /mission/coverage  - Coverage grid overlay
        /mission/info      - Mission state and progress

    Args:
        host: WebSocket server host (default: "0.0.0.0")
        port: WebSocket server port (default: 8765)
        server_name: Name shown in Foxglove Studio
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        server_name: str = "ISR-RL-DMPC Simulation",
    ):
        self.host = host
        self.port = port
        self.server_name = server_name

        self._server: Optional[FoxgloveServer] = None
        self._listener = _BridgeListener()
        self._channels: Dict[str, int] = {}
        self._started = False

    async def start(self) -> None:
        """Start the WebSocket server and register channels."""
        self._server = FoxgloveServer(
            self.host,
            self.port,
            self.server_name,
            capabilities=["time"],
            supported_encodings=["json"],
        )
        self._server.set_listener(self._listener)

        await self._server.start()
        self._started = True

        # Register channels
        self._channels["scene"] = await self._server.add_channel(
            ChannelWithoutId(
                topic="/swarm/scene",
                encoding="json",
                schemaName="foxglove.SceneUpdate",
                schema=_SCENE_UPDATE_SCHEMA,
                schemaEncoding="jsonschema",
            )
        )

        self._channels["metrics"] = await self._server.add_channel(
            ChannelWithoutId(
                topic="/swarm/metrics",
                encoding="json",
                schemaName="isr.SwarmMetrics",
                schema=json.dumps({
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
                }),
                schemaEncoding="jsonschema",
            )
        )

        self._channels["coverage"] = await self._server.add_channel(
            ChannelWithoutId(
                topic="/mission/coverage",
                encoding="json",
                schemaName="isr.CoverageGrid",
                schema=json.dumps({
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
                }),
                schemaEncoding="jsonschema",
            )
        )

        self._channels["mission_info"] = await self._server.add_channel(
            ChannelWithoutId(
                topic="/mission/info",
                encoding="json",
                schemaName="isr.MissionInfo",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "elapsed_time": {"type": "number"},
                        "mission_duration": {"type": "number"},
                        "progress": {"type": "number"},
                        "coverage_efficiency": {"type": "number"},
                        "num_waypoints": {"type": "integer"},
                    },
                }),
                schemaEncoding="jsonschema",
            )
        )

        logger.info(
            "Foxglove bridge started on ws://%s:%d — "
            "open Foxglove Studio and connect to this address",
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server is not None:
            await self._server.close()
            self._started = False
            logger.info("Foxglove bridge stopped")

    @property
    def is_running(self) -> bool:
        """Check if the bridge server is running."""
        return self._started

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------

    def _send(self, channel_key: str, data: Dict[str, Any], timestamp_ns: int) -> None:
        """Send a JSON message on a channel (fire-and-forget)."""
        if not self._started or self._server is None:
            return
        chan_id = self._channels.get(channel_key)
        if chan_id is None:
            return
        payload = json.dumps(data).encode("utf-8")
        asyncio.ensure_future(
            self._async_send(chan_id, timestamp_ns, payload)
        )

    async def _async_send(
        self, chan_id: int, timestamp_ns: int, payload: bytes
    ) -> None:
        """Async wrapper for send_message."""
        if self._server is not None:
            await self._server.send_message(chan_id, timestamp_ns, payload)

    # ------------------------------------------------------------------
    # High-level publish methods
    # ------------------------------------------------------------------

    def publish_scene(
        self,
        drone_positions: np.ndarray,
        drone_quaternions: Optional[np.ndarray] = None,
        drone_batteries: Optional[np.ndarray] = None,
        target_positions: Optional[Dict[str, np.ndarray]] = None,
        target_classifications: Optional[Dict[str, str]] = None,
        timestamp_ns: Optional[int] = None,
        grid_extent: Optional[float] = None,
    ) -> None:
        """
        Publish a 3D scene update with drone and target markers.

        Args:
            drone_positions: (N, 3) array of drone XYZ positions
            drone_quaternions: (N, 4) array [qw, qx, qy, qz] (optional)
            drone_batteries: (N,) array of battery levels (optional)
            target_positions: dict mapping target_id -> (3,) position
            target_classifications: dict mapping target_id -> classification string
            timestamp_ns: Timestamp in nanoseconds (default: current time)
            grid_extent: Size of the ground plane in meters (optional)
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        ts = _timestamp_obj(timestamp_ns)

        entities: List[Dict[str, Any]] = []

        # --- Ground plane entity ---
        if grid_extent is not None and grid_extent > 0:
            half = grid_extent / 2.0
            entities.append({
                "timestamp": ts,
                "frame_id": "world",
                "id": "ground_plane",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": False,
                "metadata": [],
                "arrows": [],
                "cubes": [{
                    "pose": {
                        "position": _vec3(half, half, -0.25),
                        "orientation": _quat(0, 0, 0, 1),
                    },
                    "size": _vec3(grid_extent, grid_extent, 0.5),
                    "color": _color(0.15, 0.15, 0.15, 0.4),
                }],
                "spheres": [],
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": [],
                "models": [],
            })

        # --- Drone entities ---
        n_drones = len(drone_positions)
        drone_cubes: List[Dict[str, Any]] = []
        drone_texts: List[Dict[str, Any]] = []

        for i in range(n_drones):
            pos = drone_positions[i]

            # Color by battery level (green = full, red = empty)
            if drone_batteries is not None and len(drone_batteries) > i:
                batt_frac = float(np.clip(drone_batteries[i] / 5000.0, 0, 1))
                col = _color(1.0 - batt_frac, batt_frac, 0.2, 0.9)
            else:
                col = _color(0.2, 0.6, 1.0, 0.9)

            # Orientation
            if drone_quaternions is not None and len(drone_quaternions) > i:
                q = drone_quaternions[i]
                orient = _quat(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
            else:
                orient = _quat(0, 0, 0, 1)

            drone_cubes.append({
                "pose": {
                    "position": _vec3(pos[0], pos[1], pos[2]),
                    "orientation": orient,
                },
                "size": _vec3(2.0, 2.0, 0.5),
                "color": col,
            })

            label = f"D{i}"
            if drone_batteries is not None and len(drone_batteries) > i:
                label += f" [{drone_batteries[i]:.0f}Wh]"
            drone_texts.append({
                "pose": {
                    "position": _vec3(pos[0], pos[1], pos[2] + 3),
                    "orientation": _quat(0, 0, 0, 1),
                },
                "billboard": True,
                "font_size": 14.0,
                "scale_invariant": True,
                "color": _color(1, 1, 1, 1),
                "text": label,
            })

        entities.append({
            "timestamp": ts,
            "frame_id": "world",
            "id": "drones",
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": False,
            "metadata": [],
            "arrows": [],
            "cubes": drone_cubes,
            "spheres": [],
            "cylinders": [],
            "lines": [],
            "triangles": [],
            "texts": drone_texts,
            "models": [],
        })

        # --- Target entities ---
        if target_positions:
            target_spheres: List[Dict[str, Any]] = []
            target_texts: List[Dict[str, Any]] = []

            for tid, tpos in target_positions.items():
                classification = (
                    target_classifications.get(tid, "unknown")
                    if target_classifications
                    else "unknown"
                )
                if classification == "hostile":
                    col = _color(1.0, 0.0, 0.0, 0.8)
                elif classification == "friendly":
                    col = _color(0.0, 1.0, 0.0, 0.8)
                else:
                    col = _color(1.0, 1.0, 0.0, 0.8)

                target_spheres.append({
                    "pose": {
                        "position": _vec3(tpos[0], tpos[1], tpos[2]),
                        "orientation": _quat(0, 0, 0, 1),
                    },
                    "size": _vec3(3.0, 3.0, 3.0),
                    "color": col,
                })

                target_texts.append({
                    "pose": {
                        "position": _vec3(tpos[0], tpos[1], tpos[2] + 5),
                        "orientation": _quat(0, 0, 0, 1),
                    },
                    "billboard": True,
                    "font_size": 12.0,
                    "scale_invariant": True,
                    "color": _color(1, 1, 1, 1),
                    "text": f"{tid} ({classification})",
                })

            entities.append({
                "timestamp": ts,
                "frame_id": "world",
                "id": "targets",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": False,
                "metadata": [],
                "arrows": [],
                "cubes": [],
                "spheres": target_spheres,
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": target_texts,
                "models": [],
            })

        scene_update = {"deletions": [], "entities": entities}
        self._send("scene", scene_update, timestamp_ns)

    def publish_metrics(
        self,
        info: Dict[str, Any],
        reward: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Publish swarm metrics for Foxglove's Plot panel.

        Args:
            info: Info dict from ISRGridEnv.step()
            reward: Reward value from step
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        metrics_msg = {
            "step": int(info.get("step", 0)),
            "coverage": float(info.get("coverage", 0.0)),
            "avg_battery": float(info.get("avg_battery", 0.0)),
            "active_drones": int(info.get("active_drones", 0)),
            "total_drones": int(info.get("total_drones", 0)),
            "collisions": int(info.get("collisions", 0)),
            "reward": float(reward),
        }
        self._send("metrics", metrics_msg, timestamp_ns)

    def publish_coverage(
        self,
        coverage_map: np.ndarray,
        grid_size: tuple,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Publish coverage grid state.

        Args:
            coverage_map: 1D array of coverage values
            grid_size: (rows, cols) of the grid
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        total = len(coverage_map)
        covered = float(np.sum(coverage_map > 0))
        pct = (covered / total * 100.0) if total > 0 else 0.0

        coverage_msg = {
            "grid_size": list(grid_size),
            "coverage_map": coverage_map.tolist(),
            "coverage_percentage": pct,
        }
        self._send("coverage", coverage_msg, timestamp_ns)

    def publish_mission_info(
        self,
        elapsed_time: float = 0.0,
        mission_duration: float = 3600.0,
        coverage_efficiency: float = 0.0,
        num_waypoints: int = 0,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Publish mission state information.

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
        mission_msg = {
            "elapsed_time": float(elapsed_time),
            "mission_duration": float(mission_duration),
            "progress": float(np.clip(progress, 0, 1)),
            "coverage_efficiency": float(coverage_efficiency),
            "num_waypoints": int(num_waypoints),
        }
        self._send("mission_info", mission_msg, timestamp_ns)

    # ------------------------------------------------------------------
    # Convenience: publish from StateManager
    # ------------------------------------------------------------------

    def publish_from_state_manager(
        self,
        state_manager: Any,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Publish all data directly from a StateManager instance.

        Args:
            state_manager: ISR-RL-DMPC StateManager
            timestamp_ns: Timestamp in nanoseconds
        """
        if timestamp_ns is None:
            timestamp_ns = _now_ns()

        positions = state_manager.get_swarm_positions()
        quaternions = state_manager.get_swarm_quaternions()
        batteries = state_manager.get_swarm_battery_levels()

        target_positions = state_manager.get_target_positions()
        target_classifications = {}
        all_targets = state_manager.get_all_targets()
        for tid, ts in all_targets.items():
            if ts.classification_confidence > 0.3:
                target_classifications[tid] = "friendly"
            elif ts.classification_confidence < -0.3:
                target_classifications[tid] = "hostile"
            else:
                target_classifications[tid] = "unknown"

        self.publish_scene(
            drone_positions=positions,
            drone_quaternions=quaternions,
            drone_batteries=batteries,
            target_positions=target_positions,
            target_classifications=target_classifications,
            timestamp_ns=timestamp_ns,
        )

        # Publish metrics from statistics
        stats = state_manager.get_statistics()
        self.publish_metrics(
            info={
                "coverage": stats.get("coverage_percentage", 0.0),
                "avg_battery": stats.get("avg_battery", 0.0),
                "active_drones": stats.get("active_drones", 0),
                "total_drones": stats.get("total_drones", 0),
            },
            timestamp_ns=timestamp_ns,
        )

        # Publish mission info
        mission = state_manager.get_mission_state()
        if mission is not None:
            self.publish_mission_info(
                elapsed_time=mission.elapsed_time,
                mission_duration=mission.mission_duration,
                coverage_efficiency=mission.coverage_efficiency,
                num_waypoints=len(mission.waypoints),
                timestamp_ns=timestamp_ns,
            )
