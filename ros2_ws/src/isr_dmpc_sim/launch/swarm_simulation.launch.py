"""
swarm_simulation.launch.py — ROS2 launch file for the ISR-DMPC simulation.

Starts:
  1. swarm_dmpc_sim_node   — physics simulation + DMPC control loop (50 Hz)
  2. rviz_bridge_node      — marker / TF / Path visualisation publisher (~15 Hz)
  3. hardware_bridge_node  — MAVROS bridge (simulation pass-through by default;
                             set use_sim:=false for live hardware flights)
  4. rviz2                 — 3-D visualiser with a pre-configured layout

Usage
-----
  ros2 launch isr_dmpc_sim swarm_simulation.launch.py

Optional launch arguments (all have defaults):
  n_drones         int   4     — number of drones
  n_targets        int   2     — number of targets
  dt               float 0.02  — simulation time step [s]
  horizon          int   20    — DMPC prediction horizon
  accel_max        float 8.0   — maximum acceleration [m/s²]
  collision_radius float 3.0   — minimum inter-drone separation [m]
  seed             int   42    — random seed
  viz_rate         float 15.0  — RViz bridge update rate [Hz]
  trajectory_length int  200   — stored trajectory points per drone
  rviz_config      str   <pkg>/config/simulation.rviz
  use_sim          bool  true  — false → activate live MAVROS hardware output
  drone_id         int   0     — drone index managed by the hardware bridge
  arm_on_start     bool  false — auto-arm + OFFBOARD when use_sim:=false

Examples
--------
  # Simulation only (default)
  ros2 launch isr_dmpc_sim swarm_simulation.launch.py n_drones:=6 n_targets:=3

  # Live hardware flight (drone 0, PX4 via MAVROS)
  # WARNING: arm_on_start:=true will arm the vehicle and switch to OFFBOARD mode
  # 5 seconds after launch.  Ensure the drone is in a safe, flight-ready state
  # on a clear launch pad before using this flag.
  ros2 launch isr_dmpc_sim swarm_simulation.launch.py \\
      use_sim:=false drone_id:=0 arm_on_start:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("isr_dmpc_sim")
    default_rviz_config = os.path.join(pkg_share, "config", "simulation.rviz")

    # ── Declare launch arguments ───────────────────────────────────────────
    args = [
        DeclareLaunchArgument("n_drones",         default_value="4"),
        DeclareLaunchArgument("n_targets",         default_value="2"),
        DeclareLaunchArgument("dt",                default_value="0.02"),
        DeclareLaunchArgument("horizon",           default_value="20"),
        DeclareLaunchArgument("accel_max",         default_value="8.0"),
        DeclareLaunchArgument("collision_radius",  default_value="3.0"),
        DeclareLaunchArgument("seed",              default_value="42"),
        DeclareLaunchArgument("viz_rate",          default_value="15.0"),
        DeclareLaunchArgument("trajectory_length", default_value="200"),
        DeclareLaunchArgument("rviz_config",       default_value=default_rviz_config),
        # Hardware bridge arguments
        DeclareLaunchArgument("use_sim",       default_value="true"),
        DeclareLaunchArgument("drone_id",      default_value="0"),
        DeclareLaunchArgument("arm_on_start",  default_value="false"),
    ]

    # ── Simulation node ────────────────────────────────────────────────────
    sim_node = Node(
        package="isr_dmpc_sim",
        executable="swarm_dmpc_sim_node",
        name="swarm_dmpc_sim_node",
        output="screen",
        parameters=[{
            "n_drones":         LaunchConfiguration("n_drones"),
            "n_targets":        LaunchConfiguration("n_targets"),
            "dt":               LaunchConfiguration("dt"),
            "horizon":          LaunchConfiguration("horizon"),
            "accel_max":        LaunchConfiguration("accel_max"),
            "collision_radius": LaunchConfiguration("collision_radius"),
            "seed":             LaunchConfiguration("seed"),
        }],
    )

    # ── RViz bridge node ───────────────────────────────────────────────────
    bridge_node = Node(
        package="isr_dmpc_sim",
        executable="rviz_bridge_node",
        name="rviz_bridge_node",
        output="screen",
        parameters=[{
            "n_drones":          LaunchConfiguration("n_drones"),
            "n_targets":         LaunchConfiguration("n_targets"),
            "viz_rate":          LaunchConfiguration("viz_rate"),
            "trajectory_length": LaunchConfiguration("trajectory_length"),
        }],
    )

    # ── Hardware bridge node ───────────────────────────────────────────────
    # Always launched; in simulation mode (use_sim=true) it is a no-op.
    # Set use_sim:=false on the command line to activate live MAVROS output.
    hw_bridge_node = Node(
        package="isr_dmpc_sim",
        executable="hardware_bridge_node",
        name="hardware_bridge_node",
        output="screen",
        parameters=[{
            "use_sim":       LaunchConfiguration("use_sim"),
            "drone_id":      LaunchConfiguration("drone_id"),
            "n_drones":      LaunchConfiguration("n_drones"),
            "accel_max":     LaunchConfiguration("accel_max"),
            "arm_on_start":  LaunchConfiguration("arm_on_start"),
        }],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    return LaunchDescription(args + [sim_node, bridge_node, hw_bridge_node, rviz_node])
