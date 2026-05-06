"""
pybullet_recorder.py — Lightweight PyBullet visualiser and video recorder.

Provides :class:`PyBulletVisualizer`, a drop-in companion for the
``run_dmpc.py`` / ``run_dmpc_rl.py`` episode runners.  It mirrors drone and
target positions from the ``MARLDMPCEnv`` physics simulation into a live
PyBullet GUI window, and can save an MP4 video of the session.

Usage
-----
::

    from pybullet_sim.pybullet_recorder import PyBulletVisualizer

    viz = PyBulletVisualizer(
        num_drones=4,
        scenario="area_surveillance",
        record=True,
        output_dir="data/videos/dmpc/area_surveillance",
    )
    viz.reset(drone_positions, target_positions)
    for step in episode:
        viz.sync(drone_positions, target_positions)
    viz.close()

Video files are saved as::

    <output_dir>/<scenario>_<timestamp>.mp4
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Optional PyBullet import ─────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

# Make the package importable whether run from repo root or pybullet_sim/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from isr_rl_dmpc.models.hector_quadrotor import get_urdf_path
    from isr_rl_dmpc.models.targets import get_target_urdf_path
    from isr_rl_dmpc.gym_env.simulator import TargetType
    _ISR_PKG_AVAILABLE = True
except ImportError:
    _ISR_PKG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_DRONE_COLORS: List[Tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),
    (0.20, 0.63, 0.17),
    (0.89, 0.10, 0.11),
    (1.00, 0.50, 0.00),
    (0.42, 0.24, 0.60),
    (0.65, 0.34, 0.16),
    (0.97, 0.51, 0.75),
    (0.74, 0.74, 0.13),
]
_WAITING_DRONE_COLOR: Tuple[float, float, float] = (0.6, 0.6, 0.6)
_WAITING_DRONE_ALPHA: float = 0.5

_TARGET_COLORS: Dict[int, Tuple[float, float, float]] = {
    2: (1.0, 0.1, 0.1),   # HOSTILE
    3: (1.0, 0.85, 0.0),  # NEUTRAL
    1: (0.1, 0.9, 0.2),   # FRIENDLY
    0: (0.7, 0.7, 0.7),   # UNKNOWN
}
_DEFAULT_TARGET_COLOR: Tuple[float, float, float] = (0.7, 0.7, 0.7)
_TARGET_VISUAL_ALPHA: float = 0.85

_DRONE_URDF_SCALE: float = 8.0
_FALLBACK_DRONE_HALF_EXTENTS = [2.5, 2.5, 0.4]
_TARGET_URDF_SCALE: float = 2.0
_LABEL_HEIGHT_OFFSET: float = 4.5
_LABEL_TEXT_SIZE: float = 1.8
_TRAJ_HISTORY: int = 200

_HOME_ALTITUDE: float = 30.0  # default altitude for launched drones
_GRID_SIZE: float = 300.0
_GRID_SPACING: float = 50.0
_GRID_COLOR: Tuple[float, float, float] = (0.3, 0.3, 0.3)
_HOME_MARKER_RADIUS: float = 5.0
_HOME_MARKER_COLOR: Tuple[float, float, float] = (0.9, 0.8, 0.1)

_MIN_CAMERA_DISTANCE: float = 30.0
_CAMERA_DISTANCE_MULTIPLIER: float = 3.5
_CAMERA_SMOOTHING_ALPHA: float = 0.1


def _drone_color(idx: int) -> Tuple[float, float, float]:
    return _DRONE_COLORS[idx % len(_DRONE_COLORS)]


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> List[float]:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]


class PyBulletVisualizer:
    """
    PyBullet companion visualiser for DMPC / DMPC-RL episode runners.

    Instantiate once before the episode loop, call :meth:`reset` with the
    initial drone and target positions, then call :meth:`sync` after every
    environment step.  Call :meth:`close` when done to flush the video file.

    Parameters
    ----------
    num_drones:
        Number of drones to visualise.
    num_targets:
        Number of target spheres to show (can be 0).
    scenario:
        Name of the mission scenario — used for the video output sub-folder.
    record:
        If ``True``, save an MP4 video via ``p.startStateLogging``.
        Only effective in GUI mode.
    output_dir:
        Directory in which to save videos.  Defaults to
        ``<repo_root>/data/videos/dmpc/<scenario>/``.
    method:
        Recorder tag (e.g. ``"dmpc"`` or ``"dmpc_rl"``).  Used in the video
        filename so multiple methods can be compared.
    spawn_interval_steps:
        Steps between consecutive drone launches (mirrors the env setting).
        Used to colour waiting drones grey.
    """

    def __init__(
        self,
        num_drones: int = 4,
        num_targets: int = 0,
        scenario: str = "area_surveillance",
        record: bool = False,
        output_dir: Optional[str] = None,
        method: str = "dmpc",
        spawn_interval_steps: int = 50,
    ) -> None:
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.scenario = scenario
        self.record = record
        self.method = method
        self.spawn_interval_steps = spawn_interval_steps

        _root = _REPO_ROOT
        self._output_dir = Path(output_dir) if output_dir else (
            _root / "data" / "videos" / method / scenario
        )

        self._pb_client: int = -1
        self._drone_ids: List[int] = []
        self._target_ids: List[int] = []
        self._label_ids: List[int] = []
        self._traj: List[deque] = [deque(maxlen=_TRAJ_HISTORY) for _ in range(num_drones)]
        self._traj_line_ids: List[List[int]] = [[] for _ in range(num_drones)]
        self._video_log_id: int = -1
        self._step: int = 0
        self._drone_launched: List[bool] = [False] * num_drones

        # Camera state
        self._cam_target = np.array([0.0, 0.0, _HOME_ALTITUDE], dtype=float)
        self._cam_yaw: float = 45.0
        self._cam_pitch: float = -40.0
        self._cam_dist: float = 80.0

        if not _PYBULLET_AVAILABLE:
            print("[PyBulletVisualizer] pybullet not installed — visualisation disabled.")
            return

        self._connect()

    # ------------------------------------------------------------------
    # Connection / scene setup
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        self._pb_client = p.connect(p.GUI, options="--mouse_wheel_multiplier=1")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target.tolist(),
        )
        self._draw_ground_grid()
        self._draw_home_marker()

    def _draw_ground_grid(self) -> None:
        half = _GRID_SIZE
        spacing = _GRID_SPACING
        cr, cg, cb = _GRID_COLOR
        coords = list(np.arange(-half, half + spacing, spacing))
        for c in coords:
            p.addUserDebugLine([c, -half, 0.05], [c, half, 0.05],
                               lineColorRGB=[cr, cg, cb], lineWidth=0.8, lifeTime=0)
            p.addUserDebugLine([-half, c, 0.05], [half, c, 0.05],
                               lineColorRGB=[cr, cg, cb], lineWidth=0.8, lifeTime=0)

    def _draw_home_marker(self) -> None:
        r = _HOME_MARKER_RADIUS
        n_pts = 32
        for k in range(n_pts):
            a0 = 2.0 * math.pi * k / n_pts
            a1 = 2.0 * math.pi * (k + 1) / n_pts
            p.addUserDebugLine(
                [r * math.cos(a0), r * math.sin(a0), 0.1],
                [r * math.cos(a1), r * math.sin(a1), 0.1],
                lineColorRGB=list(_HOME_MARKER_COLOR), lineWidth=3.0, lifeTime=0,
            )
        p.addUserDebugText("HOME", [0.0, 0.0, 2.5],
                           textColorRGB=list(_HOME_MARKER_COLOR), textSize=2.0)

    def _set_drone_color(self, drone_id: int, drone_idx: int, launched: bool) -> None:
        if launched:
            r, g, b = _drone_color(drone_idx)
            alpha = 1.0
        else:
            r, g, b = _WAITING_DRONE_COLOR
            alpha = _WAITING_DRONE_ALPHA
        for link_idx in range(-1, p.getNumJoints(drone_id)):
            p.changeVisualShape(drone_id, link_idx, rgbaColor=[r, g, b, alpha])

    def _load_drone(self, pos: List[float]) -> int:
        if _ISR_PKG_AVAILABLE:
            try:
                urdf_path = get_urdf_path()
                if os.path.isfile(urdf_path):
                    return p.loadURDF(urdf_path, basePosition=pos, useFixedBase=False,
                                      globalScaling=_DRONE_URDF_SCALE,
                                      flags=p.URDF_USE_INERTIA_FROM_FILE)
            except Exception:
                pass
        # Fallback box
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=_FALLBACK_DRONE_HALF_EXTENTS)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=_FALLBACK_DRONE_HALF_EXTENTS,
                                  rgbaColor=[0.6, 0.6, 0.6, 0.5])
        return p.createMultiBody(baseMass=1.477, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos)

    def _load_target(self, pos: List[float], type_val: int) -> int:
        color = _TARGET_COLORS.get(type_val, _DEFAULT_TARGET_COLOR)
        r, g, b = color
        if _ISR_PKG_AVAILABLE:
            type_map = {2: "hostile", 3: "neutral", 1: "friendly", 0: "unknown"}
            try:
                urdf_path = get_target_urdf_path(type_map.get(type_val, "unknown"))
                if os.path.isfile(urdf_path):
                    tgt = p.loadURDF(urdf_path, basePosition=pos, useFixedBase=True,
                                     globalScaling=_TARGET_URDF_SCALE)
                    for link_idx in range(-1, p.getNumJoints(tgt)):
                        p.changeVisualShape(tgt, link_idx,
                                            rgbaColor=[r, g, b, _TARGET_VISUAL_ALPHA])
                    return tgt
            except Exception:
                pass
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.5)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=1.5,
                                  rgbaColor=[r, g, b, _TARGET_VISUAL_ALPHA])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        drone_positions: np.ndarray,
        target_positions: Optional[np.ndarray] = None,
        target_types: Optional[Sequence[int]] = None,
        episode: int = 1,
    ) -> None:
        """
        (Re)initialise the PyBullet scene for a new episode.

        Parameters
        ----------
        drone_positions:
            Array of shape ``(num_drones, 3)`` with [x, y, z] positions.
        target_positions:
            Optional array of shape ``(num_targets, 3)``.
        target_types:
            Optional list of target type integers (0=UNKNOWN, 1=FRIENDLY,
            2=HOSTILE, 3=NEUTRAL).
        episode:
            Episode number — used in the video filename when recording.
        """
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return

        # Remove any existing bodies from a previous episode reset
        for bid in self._drone_ids + self._target_ids:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        # Remove old debug labels/trajectories
        for lid in self._label_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        for line_list in self._traj_line_ids:
            for lid in line_list:
                try:
                    p.removeUserDebugItem(lid)
                except Exception:
                    pass

        self._drone_ids = []
        self._target_ids = []
        self._label_ids = []
        self._traj = [deque(maxlen=_TRAJ_HISTORY) for _ in range(self.num_drones)]
        self._traj_line_ids = [[] for _ in range(self.num_drones)]
        self._drone_launched = [False] * self.num_drones
        self._step = 0

        # Stop any previous recording and start a new one if requested
        if self._video_log_id >= 0:
            try:
                p.stopStateLogging(self._video_log_id)
            except Exception:
                pass
            self._video_log_id = -1

        if self.record:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{self.method}_{self.scenario}_ep{episode}_{ts}.mp4"
            video_path = str(self._output_dir / fname)
            self._video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
            print(f"[PyBulletVisualizer] Recording → {video_path}")

        # Load drone bodies at home position
        positions = np.asarray(drone_positions)
        for i in range(self.num_drones):
            pos = positions[i].tolist() if i < len(positions) else [0.0, 0.0, 5.0]
            did = self._load_drone(pos)
            self._drone_ids.append(did)
            self._set_drone_color(did, i, launched=False)
            r, g, b = _WAITING_DRONE_COLOR
            lid = p.addUserDebugText(
                f"D{i} [wait]",
                [pos[0], pos[1], pos[2] + _LABEL_HEIGHT_OFFSET],
                textColorRGB=[r, g, b],
                textSize=_LABEL_TEXT_SIZE,
            )
            self._label_ids.append(lid)

        # Load target visuals
        if target_positions is not None:
            tgt_types = target_types or [0] * len(target_positions)
            for j, tpos in enumerate(target_positions):
                tid = self._load_target(tpos.tolist(), int(tgt_types[j]))
                self._target_ids.append(tid)

        # Reset camera to home
        p.resetDebugVisualizerCamera(
            cameraDistance=80.0,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=[0.0, 0.0, _HOME_ALTITUDE],
        )
        self._cam_target = np.array([0.0, 0.0, _HOME_ALTITUDE], dtype=float)
        self._cam_dist = 80.0

    def sync(
        self,
        drone_positions: np.ndarray,
        target_positions: Optional[np.ndarray] = None,
        drone_yaws: Optional[np.ndarray] = None,
    ) -> None:
        """
        Sync drone and target positions to PyBullet and advance one frame.

        Parameters
        ----------
        drone_positions:
            Array ``(num_drones, 3)`` with current [x, y, z] positions.
        target_positions:
            Optional array ``(num_targets, 3)`` with target positions.
        drone_yaws:
            Optional array ``(num_drones,)`` with yaw angles [rad].
        """
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return

        positions = np.asarray(drone_positions)
        yaws = np.asarray(drone_yaws) if drone_yaws is not None else np.zeros(self.num_drones)

        # Update drone launch status
        for i in range(self.num_drones):
            if not self._drone_launched[i] and self._step >= i * self.spawn_interval_steps:
                self._drone_launched[i] = True
                if i < len(self._drone_ids):
                    self._set_drone_color(self._drone_ids[i], i, launched=True)

        for i in range(self.num_drones):
            if i >= len(self._drone_ids):
                break
            pos = positions[i].tolist() if i < len(positions) else [0.0, 0.0, 5.0]
            yaw = float(yaws[i]) if i < len(yaws) else 0.0
            quat = _euler_to_quat_xyzw(0.0, 0.0, yaw)
            p.resetBasePositionAndOrientation(self._drone_ids[i], pos, quat)

            # Update label
            if i < len(self._label_ids):
                launched = self._drone_launched[i]
                if launched:
                    r, g, b = _drone_color(i)
                    label_text = f"D{i}"
                else:
                    r, g, b = _WAITING_DRONE_COLOR
                    steps_left = i * self.spawn_interval_steps - self._step
                    label_text = f"D{i} [T-{steps_left}]"
                self._label_ids[i] = p.addUserDebugText(
                    label_text,
                    [pos[0], pos[1], pos[2] + _LABEL_HEIGHT_OFFSET],
                    textColorRGB=[r, g, b],
                    textSize=_LABEL_TEXT_SIZE,
                    replaceItemUniqueId=self._label_ids[i],
                )

            # Trajectory trail for launched drones
            if self._drone_launched[i]:
                self._traj[i].append(pos)
                if len(self._traj[i]) > 1:
                    r, g, b = _drone_color(i)
                    pts = list(self._traj[i])
                    line_id = p.addUserDebugLine(
                        pts[-2], pts[-1],
                        lineColorRGB=[r, g, b], lineWidth=2.0, lifeTime=0,
                    )
                    self._traj_line_ids[i].append(line_id)
                    if len(self._traj_line_ids[i]) > _TRAJ_HISTORY:
                        old = self._traj_line_ids[i].pop(0)
                        p.removeUserDebugItem(old)

        # Sync targets
        if target_positions is not None:
            for j, tpos in enumerate(target_positions):
                if j < len(self._target_ids):
                    p.resetBasePositionAndOrientation(
                        self._target_ids[j], tpos.tolist(), [0, 0, 0, 1]
                    )

        # Auto-follow camera
        self._update_camera(positions)

        p.stepSimulation()
        self._step += 1

    def _update_camera(self, positions: np.ndarray) -> None:
        launched_pos = [positions[i] for i in range(self.num_drones)
                        if self._drone_launched[i] and i < len(positions)]
        if not launched_pos:
            launched_pos = [positions[i] for i in range(min(1, len(positions)))]
        if not launched_pos:
            return
        pts = np.array(launched_pos)
        centroid = pts.mean(axis=0)
        spread = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if len(pts) > 1 else 0.0
        target_dist = max(_MIN_CAMERA_DISTANCE, spread * _CAMERA_DISTANCE_MULTIPLIER)
        alpha = _CAMERA_SMOOTHING_ALPHA
        self._cam_target = (1.0 - alpha) * self._cam_target + alpha * centroid
        self._cam_dist = (1.0 - alpha) * self._cam_dist + alpha * target_dist
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target.tolist(),
        )

    def close(self) -> None:
        """Stop recording (if active) and disconnect from PyBullet."""
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return
        try:
            if self._video_log_id >= 0:
                p.stopStateLogging(self._video_log_id)
                self._video_log_id = -1
                print("[PyBulletVisualizer] Video recording stopped.")
            p.disconnect(self._pb_client)
            self._pb_client = -1
        except Exception as exc:
            print(f"[PyBulletVisualizer] Warning during close: {exc}")

    @property
    def is_connected(self) -> bool:
        """True if the PyBullet window is still open."""
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return False
        try:
            return bool(p.isConnected(self._pb_client))
        except Exception:
            return False
