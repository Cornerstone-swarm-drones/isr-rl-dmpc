"""
swarm_pybullet_sim.py — PyBullet simulation for the ISR-DMPC swarm.

Replaces the former ROS2/RViz2 stack with a self-contained PyBullet
simulation that requires only the ``isr_rl_dmpc`` Python package and
``pybullet``.

For each simulation step the script:
1. Steps the 6-DOF ``EnvironmentSimulator`` (rigid-body physics, wind, battery).
2. Runs the ``DMPCAgent`` for every active drone to compute optimal accelerations.
3. Applies the commanded accelerations and advances the physics engine.
4. Syncs every drone body and target marker in the PyBullet scene.
5. Draws incremental trajectory trails using debug lines.
6. Prints a one-line status snapshot every second of simulated time.

Visualisation
-------------
By default the simulation opens an interactive OpenGL window (``--gui``).
Pass ``--no-gui`` for headless / CI use (DIRECT mode — no window).

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
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
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

_TRAJ_ALPHA: float = 0.6
_TRAJ_HISTORY: int = 200  # trajectory trail length per drone

# Path to the drone URDF relative to this file
_URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "drone.urdf")


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
            f"dt={dt}s, horizon={horizon}, gui={gui and _PYBULLET_AVAILABLE}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_initial_positions(self) -> None:
        """Place drones in a grid and add target objects to the simulator."""
        rng = np.random.RandomState(self.seed)
        side = max(1, int(math.ceil(math.sqrt(self.n_drones))))
        spacing = 30.0  # metres between drones

        for i in range(self.n_drones):
            row, col = divmod(i, side)
            pos = np.array([col * spacing, row * spacing, 20.0], dtype=float)
            self._sim.set_drone_initial_state(i, position=pos)

        target_types = [TargetType.HOSTILE, TargetType.NEUTRAL, TargetType.FRIENDLY]
        for j in range(self.n_targets):
            pos = rng.uniform(low=-100.0, high=100.0, size=3)
            pos[2] = 0.0  # targets on the ground
            t_type = target_types[j % len(target_types)]
            self._sim.add_target(pos, t_type)

    def _init_pybullet(self) -> None:
        """Connect to PyBullet, load the scene, drones, and target markers."""
        mode = p.GUI if self.gui else p.DIRECT
        self._pb_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # Gravity handled by the ISR physics engine
        p.setRealTimeSimulation(0)  # We drive the clock manually

        # Ground plane
        p.loadURDF("plane.urdf")

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=120,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[30.0, 30.0, 20.0],
            )
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
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                    )
                except Exception as exc:
                    print(f"[WARNING] Failed to load URDF '{_URDF_PATH}': {exc} — using box fallback.")
                    drone_id = self._create_drone_visual(i, pos)
            else:
                drone_id = self._create_drone_visual(i, pos)
            self._pb_drone_ids.append(drone_id)

            # Recolour the body to the drone's assigned colour
            r, g, b = _drone_color(i)
            for link_idx in range(-1, p.getNumJoints(drone_id)):
                p.changeVisualShape(
                    drone_id, link_idx,
                    rgbaColor=[r, g, b, 1.0],
                )

        # ── Create target sphere visuals ───────────────────────────────────
        for j in range(self._sim.num_targets):
            target = self._sim.targets[j]
            tgt_id = self._create_target_visual(j, target.position.tolist(), target.target_type)
            self._pb_target_ids.append(tgt_id)

    def _create_drone_visual(self, drone_id: int, pos: List[float]) -> int:
        """Create a simple box visual when the URDF cannot be loaded."""
        r, g, b = _drone_color(drone_id)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.275, 0.275, 0.075])
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.275, 0.275, 0.075],
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
        """Create a sphere visual for a target."""
        r, g, b = _TARGET_COLORS.get(target_type, (0.7, 0.7, 0.7))
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0)
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=1.0,
            rgbaColor=[r, g, b, 0.85],
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

            # Trajectory trail
            self._traj[i].append(pos)
            if len(self._traj[i]) > 1:
                r, g, b = _drone_color(i)
                pts = list(self._traj[i])
                line_id = p.addUserDebugLine(
                    pts[-2], pts[-1],
                    lineColorRGB=[r, g, b],
                    lineWidth=1.5,
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

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """
        Advance the simulation by one time step.

        Returns:
            Status dict with step count, simulation time, solve times, etc.
        """
        # ── Collect current states ─────────────────────────────────────────
        states: List[np.ndarray] = [
            self._build_state_vector(i) for i in range(self.n_drones)
        ]

        # ── Run DMPC for each drone ────────────────────────────────────────
        solve_times: List[float] = []
        for i, (agent, state) in enumerate(zip(self._agents, states)):
            neighbor_states = [states[j] for j in range(self.n_drones) if j != i]
            ref = self._build_reference(i)
            _, info = agent.act(state, ref, neighbor_states=neighbor_states)
            solve_times.append(float(info.get("solve_time", 0.0)))
            dmpc_u = info.get("u0", np.zeros(3))
            self._sim.drones[i].acceleration = np.asarray(dmpc_u, dtype=float)

        # ── Step physics ───────────────────────────────────────────────────
        wind = self._sim.wind_model.update(self.dt)
        for drone in self._sim.drones:
            if drone.is_active:
                motor_thrusts = np.full(4, drone.config.hover_thrust)
                drone.step(motor_thrusts, wind, self.dt)

        self._sim.simulation_time += self.dt
        self._sim.update_target_detections()
        self._step += 1

        # ── Sync PyBullet visuals ──────────────────────────────────────────
        self._sync_pybullet(states)

        # ── Step PyBullet engine (collision / GUI update) ──────────────────
        if _PYBULLET_AVAILABLE and self._pb_client >= 0:
            p.stepSimulation()

        return {
            "step": self._step,
            "sim_time_s": self._sim.simulation_time,
            "wall_time_s": time.monotonic() - self._t0,
            "collisions": self._sim.collision_count,
            "geofence_violations": self._sim.geofence_violations,
            "mean_solve_ms": float(np.mean(solve_times)) * 1e3 if solve_times else 0.0,
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
        """Disconnect from PyBullet."""
        if _PYBULLET_AVAILABLE and self._pb_client >= 0:
            try:
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
    parser.add_argument("--no-gui",           action="store_true",      help="Headless mode (no window)")
    parser.add_argument("--realtime",         action="store_true",      help="Pace simulation to real time")
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
    )
    sim.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
