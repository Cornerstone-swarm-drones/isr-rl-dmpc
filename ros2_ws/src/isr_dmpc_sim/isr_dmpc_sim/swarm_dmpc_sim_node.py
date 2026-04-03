"""
swarm_dmpc_sim_node.py — ROS2 node: physics simulation + DMPC control loop.

Runs the ISR-DMPC swarm simulation at 50 Hz using the existing
``isr_rl_dmpc`` Python package.  For each simulation step the node:

1. Steps the 6-DOF ``EnvironmentSimulator`` (rigid-body physics, wind, battery).
2. Runs the ``DMPCAgent`` for every active drone to compute optimal accelerations.
3. Publishes the updated drone and target states so that ``rviz_bridge_node``
   (and any other ROS2 subscriber) can consume them.

Published topics
----------------
/swarm/poses          geometry_msgs/PoseArray   — drone poses (world frame)
/swarm/states         std_msgs/Float64MultiArray — full 11-dim state per drone
/targets/poses        geometry_msgs/PoseArray   — target positions
/dmpc/metrics         std_msgs/Float64MultiArray — per-drone DMPC diagnostics
/simulation/status    std_msgs/String            — JSON status snapshot

Parameters
----------
n_drones    (int, default 4)  — number of drones in the swarm
n_targets   (int, default 2)  — number of targets to track
dt          (float, default 0.02) — physics / DMPC time step [s]
horizon     (int, default 20) — DMPC prediction horizon
accel_max   (float, default 10.0) — maximum acceleration [m/s²]
collision_radius (float, default 5.0) — minimum inter-drone separation [m]
seed        (int, default 42) — random seed for reproducibility
"""

from __future__ import annotations

import json
import math
import time
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

# ── ISR-DMPC package imports ────────────────────────────────────────────────
from isr_rl_dmpc.gym_env.simulator import (
    EnvironmentSimulator,
    DroneConfig,
    TargetConfig,
    EnvironmentConfig,
    TargetType,
)
from isr_rl_dmpc.agents import DMPCAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert ZYX Euler angles (rad) to a ROS Quaternion message."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _float64_array(data: List[float], label: str = "") -> Float64MultiArray:
    """Pack a flat list of floats into a Float64MultiArray message."""
    msg = Float64MultiArray()
    dim = MultiArrayDimension()
    dim.label = label
    dim.size = len(data)
    dim.stride = len(data)
    msg.layout.dim = [dim]
    msg.data = [float(v) for v in data]
    return msg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SwarmDMPCSimNode(Node):
    """
    ROS2 node that drives the ISR-DMPC physics simulation and DMPC controller.

    The timer callback runs at ``1/dt`` Hz and performs one full simulation
    step for all drones.
    """

    def __init__(self) -> None:
        super().__init__("swarm_dmpc_sim_node")

        # ── Declare & read parameters ──────────────────────────────────────
        self.declare_parameter("n_drones", 4)
        self.declare_parameter("n_targets", 2)
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("horizon", 20)
        self.declare_parameter("accel_max", 10.0)
        self.declare_parameter("collision_radius", 5.0)
        self.declare_parameter("seed", 42)

        self._n_drones: int = self.get_parameter("n_drones").value
        self._n_targets: int = self.get_parameter("n_targets").value
        self._dt: float = self.get_parameter("dt").value
        self._horizon: int = self.get_parameter("horizon").value
        self._accel_max: float = self.get_parameter("accel_max").value
        self._collision_radius: float = self.get_parameter("collision_radius").value
        self._seed: int = self.get_parameter("seed").value

        # ── Physics simulator ──────────────────────────────────────────────
        env_cfg = EnvironmentConfig(timestep=self._dt)
        self._sim = EnvironmentSimulator(
            num_drones=self._n_drones,
            max_targets=self._n_targets,
            drone_config=DroneConfig(),
            target_config=TargetConfig(),
            env_config=env_cfg,
            seed=self._seed,
        )
        self._setup_initial_positions()

        # ── DMPC agents (one per drone) ────────────────────────────────────
        self._agents: List[DMPCAgent] = [
            DMPCAgent(
                horizon=self._horizon,
                dt=self._dt,
                accel_max=self._accel_max,
                collision_radius=self._collision_radius,
            )
            for _ in range(self._n_drones)
        ]

        # ── Publishers ─────────────────────────────────────────────────────
        qos = rclpy.qos.QoSProfile(depth=10)

        self._pub_drone_poses = self.create_publisher(PoseArray, "/swarm/poses", qos)
        self._pub_drone_states = self.create_publisher(
            Float64MultiArray, "/swarm/states", qos
        )
        self._pub_target_poses = self.create_publisher(
            PoseArray, "/targets/poses", qos
        )
        self._pub_metrics = self.create_publisher(
            Float64MultiArray, "/dmpc/metrics", qos
        )
        self._pub_status = self.create_publisher(String, "/simulation/status", qos)

        # ── Main simulation timer ──────────────────────────────────────────
        timer_period = self._dt  # seconds
        self._timer = self.create_timer(timer_period, self._sim_step_callback)

        # Step counter / wall-clock reference
        self._step: int = 0
        self._t0: float = time.monotonic()

        self.get_logger().info(
            f"SwarmDMPCSimNode started — "
            f"{self._n_drones} drones, {self._n_targets} targets, "
            f"dt={self._dt}s, horizon={self._horizon}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_initial_positions(self) -> None:
        """Place drones in a grid and add target objects to the simulator."""
        rng = np.random.RandomState(self._seed)
        side = max(1, int(math.ceil(math.sqrt(self._n_drones))))
        spacing = 30.0  # metres between drones

        for i in range(self._n_drones):
            row, col = divmod(i, side)
            pos = np.array([col * spacing, row * spacing, 20.0], dtype=float)
            self._sim.set_drone_initial_state(i, position=pos)

        target_types = [TargetType.HOSTILE, TargetType.NEUTRAL, TargetType.FRIENDLY]
        for j in range(self._n_targets):
            pos = rng.uniform(low=-100.0, high=100.0, size=3)
            pos[2] = 0.0  # targets on the ground
            t_type = target_types[j % len(target_types)]
            self._sim.add_target(pos, t_type)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def _build_state_vector(self, drone_id: int) -> np.ndarray:
        """
        Assemble the 11-dimensional state vector for a drone from the simulator.

        State layout: [p(3), v(3), a(3), yaw, yaw_rate]
        """
        drone = self._sim.drones[drone_id]
        state = np.zeros(11, dtype=np.float64)
        state[0:3] = drone.position
        state[3:6] = drone.velocity
        state[6:9] = drone.acceleration if hasattr(drone, "acceleration") else np.zeros(3)
        # Derive yaw from quaternion [w, x, y, z]
        q = drone.q
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                         1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        state[9] = yaw
        state[10] = drone.angular_velocity[2] if hasattr(drone, "angular_velocity") else 0.0
        return state

    def _build_reference(self, drone_id: int) -> np.ndarray:
        """
        Build a simple constant reference trajectory: hover at current position.

        A more sophisticated planner (MissionPlanner, waypoints) can replace
        this by publishing reference trajectories on a separate topic.
        """
        state = self._build_state_vector(drone_id)
        # Zero velocity / acceleration reference — just hold position
        ref = state.copy()
        ref[3:9] = 0.0  # zero velocity and acceleration
        return np.tile(ref, (self._horizon + 1, 1))  # (horizon+1, 11)

    def _sim_step_callback(self) -> None:
        """Timer callback: advance physics, run DMPC, and publish results."""
        stamp = self.get_clock().now().to_msg()

        # ── Collect current states ─────────────────────────────────────────
        states: List[np.ndarray] = [
            self._build_state_vector(i) for i in range(self._n_drones)
        ]

        # ── Run DMPC for each drone ────────────────────────────────────────
        solve_times: List[float] = []
        for i, (agent, state) in enumerate(zip(self._agents, states)):
            neighbor_states = [states[j] for j in range(self._n_drones) if j != i]
            ref = self._build_reference(i)
            _, info = agent.act(state, ref, neighbor_states=neighbor_states)
            solve_times.append(float(info.get("solve_time", 0.0)))

            # Apply the commanded acceleration to the physics engine.
            # The attitude controller inside DMPCAgent converts the DMPC
            # acceleration to motor thrusts; here we directly set the drone
            # acceleration for simplicity (the simulator uses it in RK4).
            dmpc_u = info.get("u0", np.zeros(3))
            drone = self._sim.drones[i]
            if hasattr(drone, "acceleration"):
                drone.acceleration = np.asarray(dmpc_u, dtype=float)

        # ── Step physics ───────────────────────────────────────────────────
        wind = self._sim.wind_model.update(self._dt)
        for drone in self._sim.drones:
            if drone.is_active:
                motor_thrusts = np.full(4, drone.config.hover_thrust)
                drone.step(motor_thrusts, wind, self._dt)

        self._sim.simulation_time += self._dt
        self._sim.update_target_detections()
        self._step += 1

        # ── Publish drone poses ────────────────────────────────────────────
        drone_pose_array = PoseArray()
        drone_pose_array.header.stamp = stamp
        drone_pose_array.header.frame_id = "world"

        flat_states: List[float] = []
        for i, drone in enumerate(self._sim.drones):
            pose = Pose()
            pose.position = Point(
                x=float(drone.position[0]),
                y=float(drone.position[1]),
                z=float(drone.position[2]),
            )
            pose.orientation = _euler_to_quat(0.0, 0.0, float(states[i][9]))
            drone_pose_array.poses.append(pose)
            flat_states.extend(states[i].tolist())

        self._pub_drone_poses.publish(drone_pose_array)
        self._pub_drone_states.publish(
            _float64_array(flat_states, label="drone_states")
        )

        # ── Publish target poses ───────────────────────────────────────────
        target_pose_array = PoseArray()
        target_pose_array.header.stamp = stamp
        target_pose_array.header.frame_id = "world"

        for target in self._sim.targets[: self._sim.num_targets]:
            pose = Pose()
            pose.position = Point(
                x=float(target.position[0]),
                y=float(target.position[1]),
                z=float(target.position[2]),
            )
            pose.orientation = Quaternion(w=1.0)
            target_pose_array.poses.append(pose)

        self._pub_target_poses.publish(target_pose_array)

        # ── Publish DMPC metrics ───────────────────────────────────────────
        metrics_data: List[float] = []
        for i, agent in enumerate(self._agents):
            m = agent.get_metrics()
            metrics_data.extend([
                float(m.get("rmse_tracking", 0.0)),
                float(m.get("mean_solve_time_ms", 0.0)),
                float(m.get("solve_success_rate", 0.0)),
            ])

        self._pub_metrics.publish(
            _float64_array(metrics_data, label="dmpc_metrics_per_drone")
        )

        # ── Publish status (low-rate: every 50 steps = 1 s) ───────────────
        if self._step % 50 == 0:
            status = {
                "step": self._step,
                "sim_time_s": round(self._sim.simulation_time, 3),
                "wall_time_s": round(time.monotonic() - self._t0, 3),
                "collisions": self._sim.collision_count,
                "geofence_violations": self._sim.geofence_violations,
                "mean_solve_ms": round(
                    float(np.mean(solve_times)) * 1e3, 3
                ) if solve_times else 0.0,
            }
            msg = String()
            msg.data = json.dumps(status)
            self._pub_status.publish(msg)
            self.get_logger().info(f"Status: {msg.data}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmDMPCSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
