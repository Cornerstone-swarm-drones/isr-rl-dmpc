# ISR-DMPC

**Autonomous Multi-Drone Swarm System with Pure Distributed Model Predictive Control**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble%2FJazzy-brightgreen)](https://docs.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ISR-DMPC is an autonomous Intelligence, Surveillance & Reconnaissance (ISR) platform for
multi-drone swarm coordination powered by a **purely optimisation-based Distributed Model
Predictive Controller (DMPC)**.  The system performs grid-based coverage planning,
real-time target tracking, and threat assessment using a modular 6-module architecture.
Simulation and visualisation run in **ROS2 + RViz2** — no third-party simulator required.

## Key Features

- **Pure DMPC Control** — CVXPY/OSQP solves a constrained QP at 50 Hz with hard
  collision-avoidance constraints and an LQR terminal cost computed from the DARE.
- **6 Mission Modules** — Mission planning, formation control, sensor fusion, target
  classification, threat assessment, and task allocation.
- **Geometric Attitude Controller** — SO(3) control with fixed LQR-tuned gains.
- **Gymnasium-Compatible Environment** — `ISRGridEnv` with Dict observation spaces and
  multi-objective rewards for offline algorithm validation.
- **6-DOF Physics Simulation** — Rigid-body dynamics with wind, battery depletion, and
  collision detection, tuned for the hector_quadrotor airframe (1.477 kg).
- **Open-Source Drone Model** — Uses the [hector_quadrotor](https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor)
  COLLADA mesh and self-contained URDF (`urdf/drone.urdf`) for both RViz2 visualisation
  and hardware deployment.  Replaces primitive cylinder shapes with a realistic 3-D model.
- **ROS2 / RViz2 Simulation** — A ready-to-launch ROS2 Python package
  (`ros2_ws/src/isr_dmpc_sim`) publishes drone poses, target positions, DMPC metrics,
  per-drone trajectory paths, and interactive TF frames, all rendered live in RViz2.
- **Hardware Transfer Ready** — `hardware_bridge_node` translates DMPC acceleration
  commands to MAVROS `setpoint_accel` topics (PX4 / ArduPilot).  Switch from
  simulation to live hardware with a single launch argument (`use_sim:=false`).
- **Stability Analysis** — Lyapunov, eigenvalue, ISS, CBF, and recursive feasibility tools.
- **Math / Control Optimisation Guide** — See [`docs/MATH_OPTIMIZATION.md`](docs/MATH_OPTIMIZATION.md)
  for a comprehensive guide to improving DMPC performance on real hardware.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git
cd isr-rl-dmpc

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core Python dependencies
pip install -r requirements/dev.txt

# Install the package in development mode
pip install -e .

# Run tests
pytest tests/

# Run a DMPC mission (non-ROS, headless)
python scripts/run_mission.py --config config/dmpc_config.yaml
```

### ROS2 / RViz2 Simulation

Prerequisites: ROS2 Humble or Jazzy installed and sourced.

```bash
# Source ROS2
source /opt/ros/humble/setup.bash    # or jazzy

# Build the ROS2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Launch the full simulation (physics + DMPC + RViz2)
ros2 launch isr_dmpc_sim swarm_simulation.launch.py

# Optional overrides:
ros2 launch isr_dmpc_sim swarm_simulation.launch.py \
    n_drones:=6 n_targets:=3 horizon:=20 dt:=0.02

# Live hardware flight (PX4 via MAVROS, drone 0):
ros2 launch isr_dmpc_sim swarm_simulation.launch.py \
    use_sim:=false drone_id:=0 arm_on_start:=true
```

RViz2 opens automatically with a pre-configured layout showing:
- 3D hector_quadrotor mesh models (open-source COLLADA, TU Darmstadt) and trajectory ribbons
- Target spheres colour-coded by threat level
- TF frames for every drone
- Live DMPC solve-time and tracking-error overlays

## Project Structure

```
isr-rl-dmpc/
├── src/isr_rl_dmpc/           # Core Python package
│   ├── agents/                # DMPCAgent (pure DMPC, no RL)
│   ├── analysis/              # Stability analysis tools
│   ├── core/                  # Data structures (DroneState, TargetState, MissionState)
│   ├── gym_env/               # Gymnasium environment + 6-DOF physics simulator
│   ├── models/
│   │   └── meshes/hector_quadrotor/  # Open-source drone mesh source files
│   ├── modules/               # 6 ISR modules + DMPC + attitude controller + analytics
│   └── utils/                 # Math, logging, visualization, unit conversions
├── ros2_ws/                   # ROS2 workspace
│   └── src/isr_dmpc_sim/      # ROS2 Python package
│       ├── isr_dmpc_sim/
│       │   ├── swarm_dmpc_sim_node.py   # Physics sim + DMPC loop (50 Hz)
│       │   ├── rviz_bridge_node.py      # Mesh marker / TF / Path publisher (~15 Hz)
│       │   └── hardware_bridge_node.py  # MAVROS hardware bridge (PX4/ArduPilot)
│       ├── meshes/
│       │   ├── quadrotor_base.dae       # hector_quadrotor COLLADA mesh (RViz2)
│       │   ├── quadrotor_base.stl       # STL for collision geometry
│       │   └── LICENSE.txt
│       ├── urdf/
│       │   └── drone.urdf               # Self-contained URDF (hardware deployment)
│       ├── launch/
│       │   └── swarm_simulation.launch.py
│       └── config/
│           └── simulation.rviz
├── scripts/                   # Standalone mission and evaluation scripts
├── config/                    # YAML configuration files
│   ├── drone_specs.yaml       # hector_quadrotor-aligned physical parameters
│   ├── dmpc_config.yaml       # Hardware-appropriate DMPC tuning
│   └── mission_scenarios.yaml # Real-world ISR applications
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter tutorials
└── docs/
    ├── MATH_OPTIMIZATION.md   # Guide to optimising DMPC math & control
    └── ...                    # Other documentation
```

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design patterns |
| [MODULE_SPECS.md](docs/MODULE_SPECS.md) | Detailed module specifications |
| [STABILITY_ANALYSIS.md](docs/STABILITY_ANALYSIS.md) | DMPC stability analysis |
| [MATH_OPTIMIZATION.md](docs/MATH_OPTIMIZATION.md) | Guide to optimising DMPC math & control |
| [PHASE_GUIDE.md](docs/PHASE_GUIDE.md) | Project phase descriptions |
| [GYM_DESIGN.md](docs/GYM_DESIGN.md) | Gymnasium environment design |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API documentation |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [WORKFLOW.md](docs/WORKFLOW.md) | Git workflow guide |

## Technology Stack

| Component | Technology |
|---|---|
| Convex Optimisation | [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/) |
| Terminal Cost | Discrete Algebraic Riccati Equation (SciPy DARE) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Physics Simulation | 6-DOF rigid-body (hector_quadrotor airframe, no Gazebo) |
| Drone Model | [hector_quadrotor](https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor) COLLADA mesh + self-contained URDF |
| ROS2 Simulation | rclpy, geometry_msgs, visualization_msgs, tf2, nav_msgs |
| Hardware Interface | MAVROS (PX4 / ArduPilot) via `hardware_bridge_node` |
| Visualisation | RViz2, Matplotlib |
| Configuration | YAML with dataclass validation |

## Mission Scenarios

Three pre-defined real-world ISR scenarios are available in `config/mission_scenarios.yaml`:

| Scenario | Area | Drones | Targets | Formation | Duration |
|---|---|---|---|---|---|
| Area Surveillance | 400×400 m | 4 | 0 | Grid | 20 min |
| Threat Response | 250×250 m | 4 | 2 | Wedge | 10 min |
| Search & Track | 600×600 m | 4 | 3 | Line | 20 min |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Team

| Name | Email |
|---|---|
| Jivesh Kesar | jrb252026@iitd.ac.in |
| Harsh | jrb252049@iitd.ac.in |
| Rohit Shankar Sinha | jrb252051@iitd.ac.in |

