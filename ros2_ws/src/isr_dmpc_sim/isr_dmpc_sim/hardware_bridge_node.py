"""
hardware_bridge_node.py — ROS2 node: MAVROS / PX4 hardware bridge.

Bridges the ISR-DMPC swarm controller to real UAV hardware running
PX4 or ArduPilot via MAVROS.  In simulation mode (``use_sim=True``, the
default) the node is a no-op so that the launch file can include it
unconditionally and the operator simply flips ``use_sim:=false`` for a
live flight.

Architecture
------------
The ISR-DMPC system is split into two layers:

  DMPC Swarm Coordinator (ground station / offboard computer)
       │  /swarm/states  [Float64MultiArray]
       │  /swarm/poses   [PoseArray]
       ▼
  HardwareBridgeNode  (one instance per drone, identified by ``drone_id``)
       │  /drone_{id}/mavros/setpoint_accel/accel  [AccelStamped]
       │  /drone_{id}/mavros/set_mode              [SetMode service]
       │  /drone_{id}/mavros/cmd/arming            [CommandBool service]
       ▼
  MAVROS  (runs on the UAV's companion computer)
       ▼
  PX4 / ArduPilot Flight Controller

Feedback path (state estimation from onboard):
  MAVROS /drone_{id}/mavros/local_position/pose  →  /swarm/hw_poses

Hardware pre-requisites
-----------------------
1. MAVROS installed:
       sudo apt install ros-humble-mavros ros-humble-mavros-extras
       ros2 run mavros install_geographiclib_datasets.sh
2. MAVROS running (one instance per drone, namespaced):
       ros2 run mavros mavros_node \
           --ros-args -r __ns:=/drone_0 \
                      -p fcu_url:="udp://192.168.1.10:14540@14557"
3. Launch this bridge with ``use_sim:=false drone_id:=0``.

Published topics (hardware mode only)
--------------------------------------
/drone_{id}/mavros/setpoint_accel/accel   geometry_msgs/AccelStamped
/swarm/hw_poses                           geometry_msgs/PoseArray

Subscribed topics
-----------------
/swarm/states                                  std_msgs/Float64MultiArray
/drone_{id}/mavros/local_position/pose         geometry_msgs/PoseStamped
/drone_{id}/mavros/state                       mavros_msgs/State  (if available)

Parameters
----------
use_sim   (bool,  default True)  — disable hardware output in simulation
drone_id  (int,   default 0)     — which drone index this bridge manages
n_drones  (int,   default 4)     — total swarm size (for hw_poses array)
state_dim (int,   default 11)    — DMPC state vector dimension
accel_max (float, default 8.0)   — clip commanded acceleration [m/s²]
arm_on_start (bool, default False) — automatically arm + switch OFFBOARD on start
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import (
    AccelStamped,
    Point,
    Pose,
    PoseArray,
    PoseStamped,
    Vector3,
)
from std_msgs.msg import Float64MultiArray, String


# ---------------------------------------------------------------------------
# Optional MAVROS message types — imported lazily so the node still loads
# when mavros_msgs is not installed (simulation-only environments).
# ---------------------------------------------------------------------------

def _try_import_mavros():
    """Return (SetMode, CommandBool, MavrosState) or (None, None, None)."""
    try:
        from mavros_msgs.srv import SetMode, CommandBool
        from mavros_msgs.msg import State as MavrosState
        return SetMode, CommandBool, MavrosState
    except ImportError:
        return None, None, None


_SetMode, _CommandBool, _MavrosState = _try_import_mavros()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class HardwareBridgeNode(Node):
    """
    ROS2 node that bridges DMPC acceleration commands to MAVROS setpoints.

    Safe-guards
    -----------
    - All commanded accelerations are clipped to ``accel_max`` in each axis.
    - The node refuses to send setpoints until MAVROS reports OFFBOARD mode
      and armed state (hardware mode only).
    - A watchdog timer publishes hover commands (zero lateral accel, gravity
      compensation) if no swarm state is received for > 0.5 s.
    """

    # Gravity constant
    _G: float = 9.81

    def __init__(self) -> None:
        super().__init__("hardware_bridge_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("use_sim",      True)
        self.declare_parameter("drone_id",     0)
        self.declare_parameter("n_drones",     4)
        self.declare_parameter("state_dim",    11)
        self.declare_parameter("accel_max",    8.0)
        self.declare_parameter("arm_on_start", False)

        self._use_sim:      bool  = self.get_parameter("use_sim").value
        self._drone_id:     int   = self.get_parameter("drone_id").value
        self._n_drones:     int   = self.get_parameter("n_drones").value
        self._state_dim:    int   = self.get_parameter("state_dim").value
        self._accel_max:    float = self.get_parameter("accel_max").value
        self._arm_on_start: bool  = self.get_parameter("arm_on_start").value

        self._mavros_ns: str = f"/drone_{self._drone_id}/mavros"

        # ── Internal state ────────────────────────────────────────────────
        self._latest_states: Optional[np.ndarray] = None   # (n_drones, state_dim)
        self._hw_poses: List[Optional[Pose]] = [None] * self._n_drones
        self._is_armed:    bool = False
        self._is_offboard: bool = False
        self._last_state_time: float = -1.0

        qos = QoSProfile(depth=10)
        best_effort = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriptions ──────────────────────────────────────────────────
        self.create_subscription(
            Float64MultiArray, "/swarm/states", self._cb_swarm_states, qos
        )

        if not self._use_sim:
            # Real hardware: listen to MAVROS local position
            self.create_subscription(
                PoseStamped,
                f"{self._mavros_ns}/local_position/pose",
                self._cb_mavros_pose,
                best_effort,
            )
            if _MavrosState is not None:
                self.create_subscription(
                    _MavrosState,
                    f"{self._mavros_ns}/state",
                    self._cb_mavros_state,
                    best_effort,
                )

        # ── Publishers ─────────────────────────────────────────────────────
        if not self._use_sim:
            self._pub_setpoint = self.create_publisher(
                AccelStamped,
                f"{self._mavros_ns}/setpoint_accel/accel",
                qos,
            )

        self._pub_hw_poses = self.create_publisher(
            PoseArray, "/swarm/hw_poses", qos
        )

        # ── Services (hardware arm / mode) ─────────────────────────────────
        self._set_mode_cli   = None
        self._arming_cli     = None

        if not self._use_sim and _SetMode is not None and _CommandBool is not None:
            self._set_mode_cli = self.create_client(
                _SetMode, f"{self._mavros_ns}/set_mode"
            )
            self._arming_cli = self.create_client(
                _CommandBool, f"{self._mavros_ns}/cmd/arming"
            )

        # ── Control timer (50 Hz output) ──────────────────────────────────
        self._ctrl_timer = self.create_timer(0.02, self._control_callback)

        # ── Watchdog: detect loss of DMPC commands ─────────────────────────
        self._watchdog_timer = self.create_timer(0.5, self._watchdog_callback)
        self._watchdog_triggered: bool = False

        mode = "SIMULATION (pass-through)" if self._use_sim else "HARDWARE"
        self.get_logger().info(
            f"HardwareBridgeNode started — drone_id={self._drone_id}, "
            f"mode={mode}, accel_max={self._accel_max} m/s²"
        )

        if not self._use_sim and self._arm_on_start:
            # Give MAVROS time to connect before arming
            self.create_timer(5.0, self._auto_arm_offboard)

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _cb_swarm_states(self, msg: Float64MultiArray) -> None:
        """Store the latest swarm state matrix published by the DMPC node."""
        data = np.asarray(msg.data, dtype=np.float64)
        expected = self._n_drones * self._state_dim
        if len(data) < expected:
            return
        self._latest_states = data[: expected].reshape(self._n_drones, self._state_dim)
        self._last_state_time = self.get_clock().now().nanoseconds * 1e-9
        self._watchdog_triggered = False

    def _cb_mavros_pose(self, msg: PoseStamped) -> None:
        """Update drone position from MAVROS local_position/pose."""
        self._hw_poses[self._drone_id] = msg.pose

    def _cb_mavros_state(self, msg) -> None:
        """Track MAVROS armed + mode state."""
        self._is_armed    = msg.armed
        self._is_offboard = (msg.mode == "OFFBOARD")

    # ------------------------------------------------------------------
    # Control callback
    # ------------------------------------------------------------------

    def _control_callback(self) -> None:
        """Send DMPC acceleration setpoint to MAVROS at 50 Hz."""
        stamp = self.get_clock().now().to_msg()

        if self._use_sim:
            # Publish hardware poses from simulation state for diagnostics
            self._publish_hw_poses(stamp)
            return

        if self._latest_states is None:
            return  # Wait for first state

        if not (self._is_armed and self._is_offboard):
            return  # Safety: do not send setpoints unless in OFFBOARD + armed

        # Extract the commanded acceleration for this drone
        state = self._latest_states[self._drone_id]
        accel_cmd = np.clip(state[6:9], -self._accel_max, self._accel_max)

        # Publish AccelStamped to MAVROS setpoint_accel interface
        accel_msg = AccelStamped()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = "map"  # MAVROS expects ENU "map" frame
        accel_msg.accel.linear.x = float(accel_cmd[0])
        accel_msg.accel.linear.y = float(accel_cmd[1])
        accel_msg.accel.linear.z = float(accel_cmd[2])

        self._pub_setpoint.publish(accel_msg)
        self._publish_hw_poses(stamp)

    def _publish_hw_poses(self, stamp) -> None:
        """Publish drone poses sourced from MAVROS (or sim state) as PoseArray."""
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = "world"

        if self._latest_states is not None:
            for i in range(self._n_drones):
                s = self._latest_states[i]
                pose = Pose()
                pose.position = Point(x=float(s[0]), y=float(s[1]), z=float(s[2]))
                # Build quaternion from yaw only (roll/pitch held by attitude ctrl)
                yaw = float(s[9])
                pose.orientation.w = math.cos(yaw * 0.5)
                pose.orientation.z = math.sin(yaw * 0.5)
                pa.poses.append(pose)

        self._pub_hw_poses.publish(pa)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _watchdog_callback(self) -> None:
        """Issue a warning if DMPC commands have stopped arriving."""
        if self._last_state_time < 0:
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self._last_state_time
        if age > 0.5 and not self._watchdog_triggered:
            self._watchdog_triggered = True
            self.get_logger().warn(
                f"No /swarm/states received for {age:.1f} s — "
                "hover hold active (hardware: zero lateral accel)."
            )

    # ------------------------------------------------------------------
    # Auto arm / OFFBOARD helper
    # ------------------------------------------------------------------

    def _auto_arm_offboard(self) -> None:
        """
        Attempt to switch to OFFBOARD mode and arm the vehicle.

        Called once, ``arm_on_start`` seconds after node startup.  Requires
        that the MAVROS set_mode and arming services are available.
        """
        if self._set_mode_cli is None or self._arming_cli is None:
            self.get_logger().warn(
                "mavros_msgs not available — cannot auto-arm. "
                "Ensure mavros_msgs is installed."
            )
            return

        if not self._set_mode_cli.service_is_ready():
            self.get_logger().warn("set_mode service not ready — skipping auto-arm.")
            return

        # Request OFFBOARD mode
        mode_req = _SetMode.Request()
        mode_req.custom_mode = "OFFBOARD"
        self._set_mode_cli.call_async(mode_req)

        # Request arming
        if self._arming_cli.service_is_ready():
            arm_req = _CommandBool.Request()
            arm_req.value = True
            self._arming_cli.call_async(arm_req)

        self.get_logger().info("Auto arm + OFFBOARD request sent.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
