"""
swarm_simulation.launch.py — ROS2 launch file for the ISR-DMPC simulation.

Starts:
  1. swarm_dmpc_sim_node   — physics simulation + DMPC control loop (50 Hz)
  2. rviz_bridge_node      — marker / TF / Path visualisation publisher (~15 Hz)
  3. rviz2                 — 3-D visualiser with a pre-configured layout

Usage
-----
  ros2 launch isr_dmpc_sim swarm_simulation.launch.py

Optional launch arguments (all have defaults):
  n_drones         int   4     — number of drones
  n_targets        int   2     — number of targets
  dt               float 0.02  — simulation time step [s]
  horizon          int   20    — DMPC prediction horizon
  accel_max        float 10.0  — maximum acceleration [m/s²]
  collision_radius float 5.0   — minimum inter-drone separation [m]
  seed             int   42    — random seed
  viz_rate         float 15.0  — RViz bridge update rate [Hz]
  trajectory_length int  200   — stored trajectory points per drone
  rviz_config      str   <pkg>/config/simulation.rviz

Example
-------
  ros2 launch isr_dmpc_sim swarm_simulation.launch.py n_drones:=6 n_targets:=3
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
        DeclareLaunchArgument("accel_max",         default_value="10.0"),
        DeclareLaunchArgument("collision_radius",  default_value="5.0"),
        DeclareLaunchArgument("seed",              default_value="42"),
        DeclareLaunchArgument("viz_rate",          default_value="15.0"),
        DeclareLaunchArgument("trajectory_length", default_value="200"),
        DeclareLaunchArgument("rviz_config",       default_value=default_rviz_config),
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

    # ── RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    return LaunchDescription(args + [sim_node, bridge_node, rviz_node])
