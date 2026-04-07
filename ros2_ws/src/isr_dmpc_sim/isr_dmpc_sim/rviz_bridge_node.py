"""
rviz_bridge_node.py — ROS2 node: RViz2 visualisation bridge.

Subscribes to drone and target state topics published by
``swarm_dmpc_sim_node`` and converts them into RViz2-renderable messages:

- ``visualization_msgs/MarkerArray`` for drone bodies (MESH_RESOURCE using the
  hector_quadrotor .dae model), trajectory history (LINE_STRIP), and target
  spheres.
- ``tf2_ros`` TF frames for each drone (``drone_<i>`` in the ``world`` frame).
- ``nav_msgs/Path`` per-drone trajectory paths for the Path display.

Drone visual: the open-source hector_quadrotor mesh
(``package://isr_dmpc_sim/meshes/quadrotor_base.dae``) replaces the
primitive cylinder/disc representation.  The mesh is shipped with the
``isr_dmpc_sim`` package under ``share/isr_dmpc_sim/meshes/``.

Update rate: ~15 Hz (configurable via ``viz_rate`` parameter) — sub-sampled
relative to the 50 Hz simulation to avoid overloading RViz2.

Subscribed topics
-----------------
/swarm/poses       geometry_msgs/PoseArray   — drone poses
/targets/poses     geometry_msgs/PoseArray   — target poses

Published topics
----------------
/viz/drone_markers    visualization_msgs/MarkerArray
/viz/target_markers   visualization_msgs/MarkerArray
/viz/drone_paths      nav_msgs/Path  (one per drone, namespaced as /viz/drone_<i>_path)
/tf                   tf2_msgs/TFMessage

Parameters
----------
n_drones          (int,   default 4)    — expected number of drones
n_targets         (int,   default 2)    — expected number of targets
viz_rate          (float, default 15.0) — visualisation update rate [Hz]
trajectory_length (int,   default 200)  — max stored trajectory points per drone
drone_color       (str,   default "blue")  — drone marker colour
target_color      (str,   default "red")   — target marker colour
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    PoseStamped,
    TransformStamped,
    Vector3,
)
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time


# ---------------------------------------------------------------------------
# Visualisation constants
# ---------------------------------------------------------------------------

# Drone mesh resource URI — served from the installed ROS2 package share dir.
# The hector_quadrotor .dae model has a rotor span of ~0.55 m at 1 : 1 scale.
# A scale of 2.0 gives a ~1.1 m diameter visual that is clearly visible at the
# 30 m inter-drone spacing used in the default simulation.
_DRONE_MESH_URI: str = "package://isr_dmpc_sim/meshes/quadrotor_base.dae"
_DRONE_MESH_SCALE: float = 2.0  # uniform scale applied to the mesh (metres)

# Trajectory line width
_TRAJ_LINE_WIDTH: float = 0.15
_TRAJ_ALPHA: float = 0.6

# Text label z-offset above the body
_LABEL_Z_OFFSET: float = 1.5
_LABEL_SCALE: float = 1.0

# Target sphere diameter and label offset
_TARGET_SPHERE_DIAM: float = 2.0
_TARGET_LABEL_Z_OFFSET: float = 2.5
_TARGET_LABEL_SCALE: float = 1.2


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_PALETTE = {
    "blue":   ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.9),
    "red":    ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9),
    "green":  ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.9),
    "yellow": ColorRGBA(r=1.0, g=0.9, b=0.1, a=0.9),
    "orange": ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.9),
    "white":  ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8),
}

# Per-drone hue cycle (distinct colours for up to 8 drones)
_DRONE_COLORS = [
    ColorRGBA(r=0.12, g=0.47, b=0.71, a=0.9),
    ColorRGBA(r=0.20, g=0.63, b=0.17, a=0.9),
    ColorRGBA(r=0.89, g=0.10, b=0.11, a=0.9),
    ColorRGBA(r=1.00, g=0.50, b=0.00, a=0.9),
    ColorRGBA(r=0.42, g=0.24, b=0.60, a=0.9),
    ColorRGBA(r=0.65, g=0.34, b=0.16, a=0.9),
    ColorRGBA(r=0.97, g=0.51, b=0.75, a=0.9),
    ColorRGBA(r=0.74, g=0.74, b=0.13, a=0.9),
]


def _drone_color(drone_id: int) -> ColorRGBA:
    return _DRONE_COLORS[drone_id % len(_DRONE_COLORS)]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RVizBridgeNode(Node):
    """
    ROS2 node that converts simulation state topics into RViz2 visuals.
    """

    def __init__(self) -> None:
        super().__init__("rviz_bridge_node")

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("n_drones", 4)
        self.declare_parameter("n_targets", 2)
        self.declare_parameter("viz_rate", 15.0)
        self.declare_parameter("trajectory_length", 200)

        self._n_drones: int = self.get_parameter("n_drones").value
        self._n_targets: int = self.get_parameter("n_targets").value
        self._viz_rate: float = self.get_parameter("viz_rate").value
        self._traj_len: int = self.get_parameter("trajectory_length").value

        # ── State cache (updated by subscriptions) ─────────────────────────
        self._drone_poses: Optional[PoseArray] = None
        self._target_poses: Optional[PoseArray] = None

        # Per-drone trajectory history for LINE_STRIP markers and Path msgs
        self._drone_trajectories: List[Deque[Point]] = [
            deque(maxlen=self._traj_len) for _ in range(self._n_drones)
        ]

        # ── TF broadcaster ─────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── Subscriptions ──────────────────────────────────────────────────
        qos = rclpy.qos.QoSProfile(depth=10)
        self.create_subscription(PoseArray, "/swarm/poses", self._cb_drone_poses, qos)
        self.create_subscription(PoseArray, "/targets/poses", self._cb_target_poses, qos)

        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_drone_markers = self.create_publisher(
            MarkerArray, "/viz/drone_markers", qos
        )
        self._pub_target_markers = self.create_publisher(
            MarkerArray, "/viz/target_markers", qos
        )
        # One Path publisher per drone
        self._pub_paths: List[rclpy.publisher.Publisher] = [
            self.create_publisher(Path, f"/viz/drone_{i}_path", qos)
            for i in range(self._n_drones)
        ]

        # ── Visualisation timer (~15 Hz) ───────────────────────────────────
        self._viz_timer = self.create_timer(1.0 / self._viz_rate, self._viz_callback)

        self.get_logger().info(
            f"RVizBridgeNode started — "
            f"{self._n_drones} drones, viz_rate={self._viz_rate} Hz"
        )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _cb_drone_poses(self, msg: PoseArray) -> None:
        self._drone_poses = msg
        # Append latest positions to trajectory history
        for i, pose in enumerate(msg.poses[: self._n_drones]):
            pt = Point(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z,
            )
            self._drone_trajectories[i].append(pt)

    def _cb_target_poses(self, msg: PoseArray) -> None:
        self._target_poses = msg

    # ------------------------------------------------------------------
    # Visualisation callback
    # ------------------------------------------------------------------

    def _viz_callback(self) -> None:
        """Publish markers, TF transforms, and path messages at ~15 Hz."""
        stamp = self.get_clock().now().to_msg()

        if self._drone_poses is not None:
            self._publish_drone_markers(stamp)
            self._publish_tf(stamp)
            self._publish_paths(stamp)

        if self._target_poses is not None:
            self._publish_target_markers(stamp)

    # ------------------------------------------------------------------
    # Drone markers
    # ------------------------------------------------------------------

    def _publish_drone_markers(self, stamp: Time) -> None:
        """Publish hector_quadrotor mesh + LINE_STRIP trajectory per drone."""
        marker_array = MarkerArray()

        for i, pose in enumerate(self._drone_poses.poses[: self._n_drones]):
            color = _drone_color(i)
            ns_mesh  = "drone_mesh"
            ns_traj  = "drone_trajectory"
            ns_label = "drone_label"

            # ── Body: MESH_RESOURCE (hector_quadrotor .dae) ─────────────
            mesh = Marker()
            mesh.header.stamp = stamp
            mesh.header.frame_id = "world"
            mesh.ns = ns_mesh
            mesh.id = i
            mesh.type = Marker.MESH_RESOURCE
            mesh.action = Marker.ADD
            mesh.pose = pose
            mesh.scale = Vector3(
                x=_DRONE_MESH_SCALE,
                y=_DRONE_MESH_SCALE,
                z=_DRONE_MESH_SCALE,
            )
            # Use per-drone colour tint; mesh_use_embedded_materials=True
            # preserves the .dae material colours when set to True.
            mesh.color = color
            mesh.mesh_resource = _DRONE_MESH_URI
            mesh.mesh_use_embedded_materials = True
            mesh.lifetime = Duration(seconds=0).to_msg()  # persistent
            marker_array.markers.append(mesh)

            # ── Trajectory: LINE_STRIP ───────────────────────────────────
            if len(self._drone_trajectories[i]) > 1:
                traj = Marker()
                traj.header.stamp = stamp
                traj.header.frame_id = "world"
                traj.ns = ns_traj
                traj.id = i
                traj.type = Marker.LINE_STRIP
                traj.action = Marker.ADD
                traj.scale.x = _TRAJ_LINE_WIDTH
                line_color = _drone_color(i)
                line_color.a = _TRAJ_ALPHA
                traj.color = line_color
                traj.points = list(self._drone_trajectories[i])
                traj.lifetime = mesh.lifetime
                marker_array.markers.append(traj)

            # ── Text label ───────────────────────────────────────────────
            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = "world"
            label.ns = ns_label
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = pose.position.x
            label.pose.position.y = pose.position.y
            label.pose.position.z = pose.position.z + _LABEL_Z_OFFSET
            label.pose.orientation.w = 1.0
            label.scale.z = _LABEL_SCALE
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"D{i}"
            label.lifetime = mesh.lifetime
            marker_array.markers.append(label)

        self._pub_drone_markers.publish(marker_array)

    # ------------------------------------------------------------------
    # Target markers
    # ------------------------------------------------------------------

    def _publish_target_markers(self, stamp: Time) -> None:
        """Publish a sphere marker for each target."""
        marker_array = MarkerArray()

        for j, pose in enumerate(self._target_poses.poses):
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = "world"
            sphere.ns = "target"
            sphere.id = j
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose
            sphere.scale = Vector3(x=_TARGET_SPHERE_DIAM, y=_TARGET_SPHERE_DIAM, z=_TARGET_SPHERE_DIAM)
            sphere.color = ColorRGBA(r=1.0, g=0.2, b=0.1, a=0.85)
            sphere.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(sphere)

            # Text label
            lbl = Marker()
            lbl.header = sphere.header
            lbl.ns = "target_label"
            lbl.id = j
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = pose.position.x
            lbl.pose.position.y = pose.position.y
            lbl.pose.position.z = pose.position.z + _TARGET_LABEL_Z_OFFSET
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = _TARGET_LABEL_SCALE
            lbl.color = ColorRGBA(r=1.0, g=0.9, b=0.2, a=1.0)
            lbl.text = f"T{j}"
            lbl.lifetime = sphere.lifetime
            marker_array.markers.append(lbl)

        self._pub_target_markers.publish(marker_array)

    # ------------------------------------------------------------------
    # TF transforms
    # ------------------------------------------------------------------

    def _publish_tf(self, stamp: Time) -> None:
        """Broadcast world→drone_<i> transforms for every drone."""
        transforms: List[TransformStamped] = []

        for i, pose in enumerate(self._drone_poses.poses[: self._n_drones]):
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = "world"
            tf.child_frame_id = f"drone_{i}"
            tf.transform.translation.x = pose.position.x
            tf.transform.translation.y = pose.position.y
            tf.transform.translation.z = pose.position.z
            tf.transform.rotation = pose.orientation
            transforms.append(tf)

        self._tf_broadcaster.sendTransform(transforms)

    # ------------------------------------------------------------------
    # Per-drone Path messages
    # ------------------------------------------------------------------

    def _publish_paths(self, stamp: Time) -> None:
        """Publish a ``nav_msgs/Path`` message for each drone's history."""
        for i in range(self._n_drones):
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = "world"

            for pt in self._drone_trajectories[i]:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position = pt
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)

            self._pub_paths[i].publish(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = RVizBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
