"""
swarm_pybullet_sim.py — PyBullet simulation for the ISR-DMPC swarm.

Replaces the former ROS2/RViz2 stack with a self-contained PyBullet
simulation that requires only the ``isr_rl_dmpc`` Python package and
``pybullet``.

For each simulation step the script:
1. Steps the 6-DOF ``EnvironmentSimulator`` (rigid-body physics, wind, battery).
2. Runs the ``DMPCAgent`` for every *launched* drone to compute optimal accelerations.
3. Applies the commanded accelerations and advances the physics engine.
4. Syncs every drone body and target marker in the PyBullet scene.
5. Draws incremental trajectory trails using debug lines.
6. Prints a one-line status snapshot every second of simulated time.

Staggered launch
----------------
All drones start at the same home position ``[0, 0, HOME_ALTITUDE]`` and are
launched at intervals of ``spawn_interval`` steps (default 50 = 1 s at 50 Hz).
This guarantees the DMPC no-collision constraints are always satisfied at
launch because the previous drone has already moved far enough away.

Visualisation
-------------
By default the simulation opens an interactive OpenGL window (``--gui``).
Pass ``--no-gui`` for headless / CI use (DIRECT mode — no window).

Video recording
---------------
Pass ``--record`` to save an MP4 video of the PyBullet GUI window.  Videos
are saved under ``data/videos/swarm/<scenario>/<timestamp>.mp4``.
Requires GUI mode (ignored in headless mode).

Usage
-----
    # Interactive (default): opens a PyBullet window
    python pybullet_sim/swarm_pybullet_sim.py

    # Headless (CI / server)
    python pybullet_sim/swarm_pybullet_sim.py --no-gui

    # Override parameters
    python pybullet_sim/swarm_pybullet_sim.py \\
        --n-drones 6 --n-targets 3 --horizon 20 --dt 0.02

    # Run for a fixed number of steps then exit
    python pybullet_sim/swarm_pybullet_sim.py --max-steps 5000

    # Record video
    python pybullet_sim/swarm_pybullet_sim.py --record --scenario area_surveillance
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── ISR-DMPC package imports ─────────────────────────────────────────────────
# Resolve the package regardless of CWD by inserting the repo src/ path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from isr_rl_dmpc.gym_env.simulator import (
    DroneConfig,
    EnvironmentConfig,
    EnvironmentSimulator,
    TargetConfig,
    TargetType,
)
from isr_rl_dmpc.agents import DMPCAgent
from isr_rl_dmpc.models.hector_quadrotor import get_urdf_path
from isr_rl_dmpc.models.targets import get_target_urdf_path


# ── Optional PyBullet import ─────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Per-drone RGB colours (up to 8 distinct drones)
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

# Target colours by type
_TARGET_COLORS = {
    TargetType.HOSTILE:  (1.0, 0.1, 0.1),
    TargetType.NEUTRAL:  (1.0, 0.85, 0.0),
    TargetType.FRIENDLY: (0.1, 0.9, 0.2),
    TargetType.UNKNOWN:  (0.7, 0.7, 0.7),
}
_DEFAULT_TARGET_COLOR: Tuple[float, float, float] = (0.7, 0.7, 0.7)
_TARGET_VISUAL_ALPHA: float = 0.85

_TRAJ_ALPHA: float = 0.6
_TRAJ_HISTORY: int = 200  # trajectory trail length per drone

# Per-drone floating label appearance
_LABEL_HEIGHT_OFFSET: float = 4.5   # metres above the drone body
_LABEL_TEXT_SIZE: float = 1.8

# Drone visual scaling
_DRONE_URDF_SCALE: float = 8.0  # globalScaling factor applied when loading the URDF
_FALLBACK_DRONE_HALF_EXTENTS = [2.5, 2.5, 0.4]  # [x, y, z] half-extents for box fallback [m]

# Target visual scaling
_TARGET_URDF_SCALE: float = 2.0  # globalScaling factor applied when loading target URDFs

# Mapping from TargetType enum to the target URDF type string
_TARGET_TYPE_TO_URDF = {
    TargetType.HOSTILE:  "hostile",
    TargetType.NEUTRAL:  "neutral",
    TargetType.FRIENDLY: "friendly",
    TargetType.UNKNOWN:  "unknown",
}

# Auto-camera tracking tuning
_CAMERA_SMOOTHING_ALPHA: float = 0.15   # exponential-smoothing weight per step
_MIN_CAMERA_DISTANCE: float = 25.0     # metres — never zoom closer than this
_CAMERA_DISTANCE_MULTIPLIER: float = 3.0  # camera distance = multiplier × max spread

# Staggered spawning
_HOME_ALTITUDE: float = 5.0     # metres — all drones start here at home
_DEFAULT_SPAWN_INTERVAL: int = 50  # steps between consecutive launches (1 s at 50 Hz)

# Waiting-drone visual (grey, partially transparent)
_WAITING_DRONE_COLOR: Tuple[float, float, float] = (0.6, 0.6, 0.6)
_WAITING_DRONE_ALPHA: float = 0.5

# Ground grid
_GRID_SIZE: float = 200.0   # total extent of the visualised grid (±200 m each side)
_GRID_SPACING: float = 50.0  # metres between grid lines
_GRID_COLOR: Tuple[float, float, float] = (0.3, 0.3, 0.3)
_GRID_LINE_WIDTH: float = 0.8

# Home-position marker (cylinder on ground)
_HOME_MARKER_RADIUS: float = 4.0  # metres
_HOME_MARKER_COLOR: Tuple[float, float, float, float] = (0.9, 0.8, 0.1, 0.9)  # yellow

# Path to the drone URDF from the canonical src/models location
_URDF_PATH = get_urdf_path()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> List[float]:
    """Return quaternion [x, y, z, w] from ZYX Euler angles (rad)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]


def _drone_color(drone_id: int) -> Tuple[float, float, float]:
    return _DRONE_COLORS[drone_id % len(_DRONE_COLORS)]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SwarmPyBulletSim:
    """
    PyBullet-based ISR-DMPC swarm simulation.

    The simulation loop runs at ``1/dt`` Hz in wall-clock time (with optional
    real-time pacing).  Each iteration:

    1. Runs DMPCAgent for every drone.
    2. Steps the custom 6-DOF physics engine.
    3. Syncs PyBullet body poses for visualisation.
    4. Draws trajectory trails via ``addUserDebugLine``.
    5. Updates target sphere positions.
    """

    def __init__(
        self,
        n_drones: int = 4,
        n_targets: int = 2,
        dt: float = 0.02,
        horizon: int = 20,
        accel_max: float = 8.0,
        collision_radius: float = 3.0,
        seed: int = 42,
        traj_length: int = _TRAJ_HISTORY,
        gui: bool = True,
        realtime: bool = False,
        auto_camera: bool = True,
        spawn_interval: int = _DEFAULT_SPAWN_INTERVAL,
        record: bool = False,
        scenario: str = "swarm",
        output_dir: Optional[str] = None,
    ) -> None:
        self.n_drones = n_drones
        self.n_targets = n_targets
        self.dt = dt
        self.horizon = horizon
        self.accel_max = accel_max
        self.collision_radius = collision_radius
        self.seed = seed
        self.traj_length = traj_length
        self.gui = gui
        self.realtime = realtime
        self.auto_camera = auto_camera
        self.spawn_interval = spawn_interval
        self.record = record
        self.scenario = scenario

        # Determine video output directory
        _repo_root = Path(__file__).resolve().parents[1]
        self._video_dir = Path(output_dir) if output_dir else (
            _repo_root / "data" / "videos" / "swarm" / scenario
        )

        # Staggered-launch state: drone i launches at step i * spawn_interval
        self._drone_launched: List[bool] = [False] * n_drones
        # Home position for all drones
        self._home_pos: List[float] = [0.0, 0.0, _HOME_ALTITUDE]

        # Camera tracking state (smoothed each step when auto_camera=True)
        self._cam_target = np.array([0.0, 0.0, 20.0], dtype=float)
        self._cam_yaw: float = 45.0
        self._cam_pitch: float = -40.0
        self._cam_dist: float = 80.0

        # PyBullet video logging handle
        self._video_log_id: int = -1

        # ── Physics simulator ──────────────────────────────────────────────
        env_cfg = EnvironmentConfig(timestep=dt)
        self._sim = EnvironmentSimulator(
            num_drones=n_drones,
            max_targets=n_targets,
            drone_config=DroneConfig(),
            target_config=TargetConfig(),
            env_config=env_cfg,
            seed=seed,
        )
        self._setup_initial_positions()

        # ── DMPC agents (one per drone) ────────────────────────────────────
        self._agents: List[DMPCAgent] = [
            DMPCAgent(
                horizon=horizon,
                dt=dt,
                accel_max=accel_max,
                collision_radius=collision_radius,
            )
            for _ in range(n_drones)
        ]

        # ── PyBullet setup ─────────────────────────────────────────────────
        self._pb_drone_ids: List[int] = []
        self._pb_target_ids: List[int] = []
        self._pb_client: int = -1

        # Trajectory history per drone [[x,y,z], ...]
        self._traj: List[deque] = [
            deque(maxlen=traj_length) for _ in range(n_drones)
        ]
        # Last debug-line IDs per drone (one per segment)
        self._traj_line_ids: List[List[int]] = [[] for _ in range(n_drones)]
        # Per-drone floating label IDs (updated each step via replaceItemUniqueId)
        self._drone_label_ids: List[int] = []

        if _PYBULLET_AVAILABLE:
            self._init_pybullet()
        else:
            print(
                "[WARNING] pybullet not installed — running headless without 3-D "
                "visualisation.  Install with:  pip install pybullet"
            )

        self._step: int = 0
        self._t0: float = time.monotonic()

        print(
            f"[SwarmPyBulletSim] started — "
            f"{n_drones} drones, {n_targets} targets, "
            f"dt={dt}s, horizon={horizon}, gui={gui and _PYBULLET_AVAILABLE}, "
            f"spawn_interval={spawn_interval} steps ({spawn_interval * dt:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_initial_positions(self) -> None:
        """Place all drones at the shared home position and add target objects.

        All drones start at ``[0, 0, HOME_ALTITUDE]``.  They are launched
        sequentially at intervals of ``spawn_interval`` steps so that the
        DMPC no-collision constraints are always satisfied at launch time
        (the previously launched drone has already flown far enough away).
        """
        rng = np.random.RandomState(self.seed)

        # Every drone starts at the same home position
        home = np.array(self._home_pos, dtype=float)
        for i in range(self.n_drones):
            self._sim.set_drone_initial_state(i, position=home.copy())

        target_types = [TargetType.HOSTILE, TargetType.NEUTRAL, TargetType.FRIENDLY]
        for j in range(self.n_targets):
            pos = rng.uniform(low=-100.0, high=100.0, size=3)
            pos[2] = 0.0  # targets on the ground
            t_type = target_types[j % len(target_types)]
            self._sim.add_target(pos, t_type)

    def _init_pybullet(self) -> None:
        """Connect to PyBullet, load the scene, drones, and target markers."""
        mode = p.GUI if self.gui else p.DIRECT
        self._pb_client = p.connect(mode, options="--mouse_wheel_multiplier=1")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # Gravity handled by the ISR physics engine
        p.setRealTimeSimulation(0)  # We drive the clock manually

        # Ground plane
        p.loadURDF("plane.urdf")

        if self.gui:
            # Improve visual quality
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            # Seed camera on the home position
            self._cam_target = np.array(self._home_pos, dtype=float)
            self._cam_target[2] = 0.0
            self._cam_dist = max(_MIN_CAMERA_DISTANCE, 60.0)

            p.resetDebugVisualizerCamera(
                cameraDistance=self._cam_dist,
                cameraYaw=self._cam_yaw,
                cameraPitch=self._cam_pitch,
                cameraTargetPosition=self._cam_target.tolist(),
            )

            # Draw ground grid for spatial reference
            self._draw_ground_grid()

            # Draw home-position marker
            self._draw_home_marker()

            # Start video recording if requested
            if self.record:
                self._video_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = str(self._video_dir / f"{self.scenario}_{ts}.mp4")
                self._video_log_id = p.startStateLogging(
                    p.STATE_LOGGING_VIDEO_MP4, video_path
                )
                print(f"[SwarmPyBulletSim] Recording video → {video_path}")
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        # ── Load drone bodies ──────────────────────────────────────────────
        urdf_exists = os.path.isfile(_URDF_PATH)
        for i in range(self.n_drones):
            pos = self._sim.drones[i].position.tolist()
            if urdf_exists:
                try:
                    drone_id = p.loadURDF(
                        _URDF_PATH,
                        basePosition=pos,
                        useFixedBase=False,
                        globalScaling=_DRONE_URDF_SCALE,
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                    )
                except Exception as exc:
                    print(f"[WARNING] Failed to load URDF '{_URDF_PATH}': {exc} — using box fallback.")
                    drone_id = self._create_drone_visual(i, pos)
            else:
                drone_id = self._create_drone_visual(i, pos)
            self._pb_drone_ids.append(drone_id)

            # All drones start in "waiting" grey colour
            self._set_drone_color(drone_id, i, launched=False)

            # Floating ID label above each drone
            r, g, b = _WAITING_DRONE_COLOR
            label_id = p.addUserDebugText(
                f"D{i} [wait]",
                [pos[0], pos[1], pos[2] + _LABEL_HEIGHT_OFFSET],
                textColorRGB=[r, g, b],
                textSize=_LABEL_TEXT_SIZE,
            )
            self._drone_label_ids.append(label_id)

        # ── Create target sphere visuals ───────────────────────────────────
        for j in range(self._sim.num_targets):
            target = self._sim.targets[j]
            tgt_id = self._create_target_visual(j, target.position.tolist(), target.target_type)
            self._pb_target_ids.append(tgt_id)

    def _draw_ground_grid(self) -> None:
        """Draw a flat reference grid on the ground plane."""
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return
        half = _GRID_SIZE
        spacing = _GRID_SPACING
        cr, cg, cb = _GRID_COLOR
        coords = list(np.arange(-half, half + spacing, spacing))
        for c in coords:
            # Lines parallel to X axis
            p.addUserDebugLine(
                [c, -half, 0.05], [c, half, 0.05],
                lineColorRGB=[cr, cg, cb], lineWidth=_GRID_LINE_WIDTH, lifeTime=0,
            )
            # Lines parallel to Y axis
            p.addUserDebugLine(
                [-half, c, 0.05], [half, c, 0.05],
                lineColorRGB=[cr, cg, cb], lineWidth=_GRID_LINE_WIDTH, lifeTime=0,
            )

    def _draw_home_marker(self) -> None:
        """Draw a glowing circle on the ground at the home position."""
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return
        hx, hy, _ = self._home_pos
        r = _HOME_MARKER_RADIUS
        n_pts = 32
        for k in range(n_pts):
            a0 = 2.0 * math.pi * k / n_pts
            a1 = 2.0 * math.pi * (k + 1) / n_pts
            p.addUserDebugLine(
                [hx + r * math.cos(a0), hy + r * math.sin(a0), 0.1],
                [hx + r * math.cos(a1), hy + r * math.sin(a1), 0.1],
                lineColorRGB=list(_HOME_MARKER_COLOR[:3]),
                lineWidth=3.0,
                lifeTime=0,
            )
        # Label
        p.addUserDebugText(
            "HOME",
            [hx, hy, 2.5],
            textColorRGB=list(_HOME_MARKER_COLOR[:3]),
            textSize=2.0,
        )

    def _set_drone_color(self, drone_id: int, drone_idx: int, launched: bool) -> None:
        """Apply the correct colour to all links of a drone body."""
        if launched:
            r, g, b = _drone_color(drone_idx)
            alpha = 1.0
        else:
            r, g, b = _WAITING_DRONE_COLOR
            alpha = _WAITING_DRONE_ALPHA
        for link_idx in range(-1, p.getNumJoints(drone_id)):
            p.changeVisualShape(drone_id, link_idx, rgbaColor=[r, g, b, alpha])

    def _create_drone_visual(self, drone_id: int, pos: List[float]) -> int:
        """Create a prominent flat-disc box visual when the URDF cannot be loaded."""
        r, g, b = _drone_color(drone_id)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=_FALLBACK_DRONE_HALF_EXTENTS)
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=_FALLBACK_DRONE_HALF_EXTENTS,
            rgbaColor=[r, g, b, 1.0],
        )
        return p.createMultiBody(
            baseMass=1.477,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=pos,
        )

    def _create_target_visual(
        self, target_id: int, pos: List[float], target_type: TargetType
    ) -> int:
        """Create a 3-D model visual for a target using a URDF.

        Falls back to a coloured sphere when the URDF cannot be loaded.
        """
        urdf_key = _TARGET_TYPE_TO_URDF.get(target_type, "unknown")
        try:
            urdf_path = get_target_urdf_path(urdf_key)
        except ValueError:
            urdf_path = None

        if urdf_path and os.path.isfile(urdf_path):
            try:
                tgt_body = p.loadURDF(
                    urdf_path,
                    basePosition=pos,
                    useFixedBase=True,
                    globalScaling=_TARGET_URDF_SCALE,
                )
                # Apply the canonical target colour to every link
                r, g, b = _TARGET_COLORS.get(target_type, _DEFAULT_TARGET_COLOR)
                for link_idx in range(-1, p.getNumJoints(tgt_body)):
                    p.changeVisualShape(
                        tgt_body, link_idx,
                        rgbaColor=[r, g, b, _TARGET_VISUAL_ALPHA],
                    )
                return tgt_body
            except Exception as exc:
                print(
                    f"[WARNING] Failed to load target URDF '{urdf_path}': {exc} "
                    "— using sphere fallback."
                )

        # Sphere fallback (original behaviour)
        r, g, b = _TARGET_COLORS.get(target_type, _DEFAULT_TARGET_COLOR)
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0)
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=1.0,
            rgbaColor=[r, g, b, _TARGET_VISUAL_ALPHA],
        )
        return p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=pos,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _build_state_vector(self, drone_id: int) -> np.ndarray:
        """Assemble the 11-dimensional state vector for a drone."""
        drone = self._sim.drones[drone_id]
        state = np.zeros(11, dtype=np.float64)
        state[0:3] = drone.position
        state[3:6] = drone.velocity
        state[6:9] = drone.acceleration if hasattr(drone, "acceleration") else np.zeros(3)
        q = drone.q  # [w, x, y, z]
        yaw = math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )
        state[9] = yaw
        state[10] = drone.angular_velocity[2]
        return state

    def _build_reference(self, drone_id: int) -> np.ndarray:
        """Constant hold-position reference (hover at current pose)."""
        state = self._build_state_vector(drone_id)
        ref = state.copy()
        ref[3:9] = 0.0
        return np.tile(ref, (self.horizon + 1, 1))

    # ------------------------------------------------------------------
    # PyBullet sync
    # ------------------------------------------------------------------

    def _sync_pybullet(self, states: List[np.ndarray]) -> None:
        """Push latest positions and attitudes to PyBullet bodies."""
        if not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return

        for i, drone in enumerate(self._sim.drones):
            if i >= len(self._pb_drone_ids):
                break
            pos = drone.position.tolist()
            yaw = float(states[i][9])
            quat = _euler_to_quat_xyzw(0.0, 0.0, yaw)
            p.resetBasePositionAndOrientation(self._pb_drone_ids[i], pos, quat)

            launched = self._drone_launched[i]

            # Floating ID label — shows countdown for waiting drones
            if i < len(self._drone_label_ids):
                if launched:
                    r, g, b = _drone_color(i)
                    label_text = f"D{i}"
                else:
                    r, g, b = _WAITING_DRONE_COLOR
                    steps_until_launch = i * self.spawn_interval - self._step
                    launch_s = steps_until_launch * self.dt
                    label_text = f"D{i} [T-{launch_s:.1f}s]"
                self._drone_label_ids[i] = p.addUserDebugText(
                    label_text,
                    [pos[0], pos[1], pos[2] + _LABEL_HEIGHT_OFFSET],
                    textColorRGB=[r, g, b],
                    textSize=_LABEL_TEXT_SIZE,
                    replaceItemUniqueId=self._drone_label_ids[i],
                )

            # Trajectory trail — only for launched drones
            if launched:
                self._traj[i].append(pos)
                if len(self._traj[i]) > 1:
                    r, g, b = _drone_color(i)
                    pts = list(self._traj[i])
                    line_id = p.addUserDebugLine(
                        pts[-2], pts[-1],
                        lineColorRGB=[r, g, b],
                        lineWidth=2.0,
                        lifeTime=0,  # persistent
                    )
                    self._traj_line_ids[i].append(line_id)
                    # Remove oldest line segment to keep trail bounded
                    if len(self._traj_line_ids[i]) > self.traj_length:
                        old_id = self._traj_line_ids[i].pop(0)
                        p.removeUserDebugItem(old_id)

        for j, target_id in enumerate(self._pb_target_ids):
            target = self._sim.targets[j]
            p.resetBasePositionAndOrientation(
                target_id, target.position.tolist(), [0, 0, 0, 1]
            )

        # Auto-follow camera — tracks swarm centroid with adaptive zoom
        self._update_camera()

    # ------------------------------------------------------------------
    # Auto-follow camera
    # ------------------------------------------------------------------

    def _update_camera(self) -> None:
        """Smoothly recentre the camera on the swarm centroid with adaptive zoom.

        Only active in GUI mode when ``auto_camera=True``.  The camera target
        and distance are exponentially smoothed each step so motion is fluid
        rather than jarring.  The zoom distance is proportional to the maximum
        drone-to-centroid spread, ensuring all drones stay in frame.
        """
        if not self.auto_camera or not self.gui or not _PYBULLET_AVAILABLE or self._pb_client < 0:
            return

        # Track only launched drones; fall back to all drones if none launched yet
        active_positions = [
            drone.position
            for i, drone in enumerate(self._sim.drones)
            if drone.is_active and (self._drone_launched[i] or not any(self._drone_launched))
        ]
        if not active_positions:
            return

        positions = np.array(active_positions)
        centroid = positions.mean(axis=0)

        # Adaptive distance: keep all drones comfortably in frame
        if len(positions) > 1:
            spread = float(np.max(np.linalg.norm(positions - centroid, axis=1)))
        else:
            spread = 0.0
        target_dist = max(_MIN_CAMERA_DISTANCE, spread * _CAMERA_DISTANCE_MULTIPLIER)

        # Exponential smoothing (α ≈ 0.15 → time-constant ~4 steps ≈ 0.08 s at 50 Hz)
        alpha = _CAMERA_SMOOTHING_ALPHA
        self._cam_target = (1.0 - alpha) * self._cam_target + alpha * centroid
        self._cam_dist = (1.0 - alpha) * self._cam_dist + alpha * target_dist

        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self._cam_target.tolist(),
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """
        Advance the simulation by one time step.

        Returns:
            Status dict with step count, simulation time, solve times, etc.
        """
        # ── Check which drones should now be launched ──────────────────────
        for i in range(self.n_drones):
            if not self._drone_launched[i] and self._step >= i * self.spawn_interval:
                self._drone_launched[i] = True
                print(
                    f"[SwarmPyBulletSim] Drone D{i} launched at step {self._step} "
                    f"(t={self._step * self.dt:.2f}s)"
                )
                # Switch drone body to its assigned colour
                if _PYBULLET_AVAILABLE and self._pb_client >= 0 and i < len(self._pb_drone_ids):
                    self._set_drone_color(self._pb_drone_ids[i], i, launched=True)

        # ── Collect current states ─────────────────────────────────────────
        states: List[np.ndarray] = [
            self._build_state_vector(i) for i in range(self.n_drones)
        ]

        # ── Run DMPC only for launched drones ─────────────────────────────
        solve_times: List[float] = []
        motor_commands: List[np.ndarray] = []
        # Neighbour states for DMPC: only launched neighbours
        launched_indices = [j for j in range(self.n_drones) if self._drone_launched[j]]

        for i in range(self.n_drones):
            if not self._drone_launched[i]:
                # Not yet launched — hold at home, skip DMPC
                solve_times.append(0.0)
                motor_commands.append(np.array([0.5, 0.5, 0.5, 0.5]))  # hover
                continue

            neighbor_states = [
                states[j] for j in launched_indices if j != i
            ]
            ref = self._build_reference(i)
            motor_thrusts, info = self._agents[i].act(state=states[i], ref=ref, neighbor_states=neighbor_states)
            solve_times.append(float(info.get("solve_time", 0.0)))
            motor_commands.append(np.asarray(motor_thrusts, dtype=float))

        # ── Step physics ───────────────────────────────────────────────────
        wind = self._sim.wind_model.update(self.dt)
        for i, drone in enumerate(self._sim.drones):
            if drone.is_active:
                if self._drone_launched[i]:
                    drone.step(motor_commands[i], wind, self.dt)
                else:
                    # Freeze at home — override position in case physics drifted
                    drone.position = np.array(self._home_pos, dtype=float)
                    drone.velocity = np.zeros(3)

        self._sim.simulation_time += self.dt
        self._sim.update_target_detections()
        self._step += 1

        # ── Sync PyBullet visuals ──────────────────────────────────────────
        self._sync_pybullet(states)

        # ── Step PyBullet engine (collision / GUI update) ──────────────────
        if _PYBULLET_AVAILABLE and self._pb_client >= 0:
            p.stepSimulation()

        n_launched = sum(self._drone_launched)
        return {
            "step": self._step,
            "sim_time_s": self._sim.simulation_time,
            "wall_time_s": time.monotonic() - self._t0,
            "collisions": self._sim.collision_count,
            "geofence_violations": self._sim.geofence_violations,
            "mean_solve_ms": float(np.mean(solve_times)) * 1e3 if solve_times else 0.0,
            "n_launched": n_launched,
        }

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, max_steps: int = 0) -> None:
        """
        Run the simulation loop.

        Args:
            max_steps: Maximum steps to run (0 = run until Ctrl-C or window closed).
        """
        steps_per_status = max(1, int(round(1.0 / self.dt)))  # ~1 s of sim time

        try:
            while True:
                if max_steps > 0 and self._step >= max_steps:
                    break

                # Real-time pacing
                if self.realtime:
                    step_start = time.monotonic()

                status = self.step()

                if self.realtime:
                    elapsed = time.monotonic() - step_start
                    sleep_time = self.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if status["step"] % steps_per_status == 0:
                    print(
                        f"[t={status['sim_time_s']:7.2f}s | step={status['step']:6d}] "
                        f"launched={status['n_launched']}/{self.n_drones}  "
                        f"wall={status['wall_time_s']:6.1f}s  "
                        f"collisions={status['collisions']}  "
                        f"mean_solve={status['mean_solve_ms']:.2f}ms"
                    )

                # Exit if PyBullet window was closed
                if (
                    _PYBULLET_AVAILABLE
                    and self.gui
                    and self._pb_client >= 0
                    and not p.isConnected(self._pb_client)
                ):
                    print("[SwarmPyBulletSim] PyBullet window closed — exiting.")
                    break

        except KeyboardInterrupt:
            print("\n[SwarmPyBulletSim] interrupted by user.")
        finally:
            self.close()

    def close(self) -> None:
        """Disconnect from PyBullet and stop any active video recording."""
        if _PYBULLET_AVAILABLE and self._pb_client >= 0:
            try:
                if self._video_log_id >= 0:
                    p.stopStateLogging(self._video_log_id)
                    self._video_log_id = -1
                    print("[SwarmPyBulletSim] Video recording stopped.")
                p.disconnect(self._pb_client)
            except Exception as exc:
                print(f"[WARNING] PyBullet disconnect failed: {exc}")
        print(
            f"[SwarmPyBulletSim] finished — "
            f"{self._step} steps, {self._sim.simulation_time:.2f}s simulated."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISR-DMPC swarm simulation using PyBullet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-drones",         type=int,   default=4,    help="Number of drones")
    parser.add_argument("--n-targets",        type=int,   default=2,    help="Number of targets")
    parser.add_argument("--dt",               type=float, default=0.02, help="Simulation time step [s]")
    parser.add_argument("--horizon",          type=int,   default=20,   help="DMPC prediction horizon")
    parser.add_argument("--accel-max",        type=float, default=8.0,  help="Max acceleration [m/s²]")
    parser.add_argument("--collision-radius", type=float, default=3.0,  help="Min inter-drone separation [m]")
    parser.add_argument("--seed",             type=int,   default=42,   help="Random seed")
    parser.add_argument("--traj-length",      type=int,   default=200,  help="Trajectory trail length")
    parser.add_argument("--max-steps",        type=int,   default=0,    help="Steps to run (0 = unlimited)")
    parser.add_argument("--spawn-interval",   type=int,   default=_DEFAULT_SPAWN_INTERVAL,
                        help="Steps between consecutive drone launches (0 = all at once)")
    parser.add_argument("--scenario",         type=str,   default="swarm",
                        help="Scenario name — used for video subfolder naming")
    parser.add_argument("--no-gui",           action="store_true",      help="Headless mode (no window)")
    parser.add_argument("--realtime",         action="store_true",      help="Pace simulation to real time")
    parser.add_argument("--auto-camera",      action="store_true",      help="Disable auto-follow camera (use manual PyBullet navigation)")
    parser.add_argument("--record",           action="store_true",
                        help="Save PyBullet GUI video to data/videos/swarm/<scenario>/ (requires --gui)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sim = SwarmPyBulletSim(
        n_drones=args.n_drones,
        n_targets=args.n_targets,
        dt=args.dt,
        horizon=args.horizon,
        accel_max=args.accel_max,
        collision_radius=args.collision_radius,
        seed=args.seed,
        traj_length=args.traj_length,
        gui=not args.no_gui,
        realtime=args.realtime,
        auto_camera=args.auto_camera,
        spawn_interval=args.spawn_interval,
        record=args.record,
        scenario=args.scenario,
    )
    sim.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
