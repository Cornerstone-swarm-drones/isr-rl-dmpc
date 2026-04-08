# ISR-RL-DMPC

**Autonomous Multi-Drone ISR Swarm using MARL-Adaptive Distributed MPC**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-brightgreen)](https://docs.ros.org/en/jazzy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ISR-RL-DMPC is an autonomous Intelligence, Surveillance & Reconnaissance (ISR) platform for
multi-drone swarm coordination.  It implements a **Multi-Agent Reinforcement Learning (MARL)
framework that uses MAPPO (Multi-Agent Proximal Policy Optimization) to dynamically tune the
cost parameters of a Distributed Model Predictive Control (DMPC) layer**, where coordination
between drones is achieved through **ADMM (Alternating Direction Method of Multipliers)** to
ensure consensus on shared constraints and objectives.

The MARL policy learns *which cost weights* (Q, R scales) to assign to each drone at every
control step; the DMPC then solves the resulting constrained QP in real time using
CVXPY/OSQP, and ADMM drives inter-drone consensus on reference trajectories and collision
margins.  Simulation and visualisation run in **ROS2 + RViz2** — no third-party simulator
required.

## Key Features

- **MARL + MAPPO** — Centralised training / decentralised execution via Stable-Baselines3
  PPO.  Each drone's actor conditions on a 40-D local observation; a shared critic uses the
  full joint state for variance reduction.
- **ADMM Consensus Layer** — Alternating Direction Method of Multipliers enforces
  consistency between each drone's local DMPC sub-problem and the swarm's shared
  trajectory / collision constraints.
- **Adaptive DMPC** — MAPPO outputs per-drone Q and R scale vectors; the DMPC cost
  becomes `Q_eff = Q ⊙ diag(q_scale)`, allowing the RL policy to prioritise position
  tracking, energy efficiency, or formation keeping dynamically.
- **CVXPY / OSQP Solver** — Constrained QP solved at 50 Hz with hard collision-avoidance
  constraints and an LQR terminal cost computed from the DARE.
- **6 Mission Modules** — Mission planning, formation control, sensor fusion, target
  classification, threat assessment, and task allocation.
- **Geometric Attitude Controller** — SO(3) control with fixed LQR-tuned gains.
- **MARL Gymnasium Environment** — `MARLDMPCEnv` with 40-D per-agent observations and
  14-D continuous action spaces (Q/R scale vectors) for MAPPO training.
- **6-DOF Physics Simulation** — Rigid-body dynamics with wind, battery depletion, and
  collision detection, tuned for the hector_quadrotor airframe (1.477 kg).
- **Open-Source Drone Model** — Uses the [hector_quadrotor](https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor)
  COLLADA mesh and self-contained URDF (`urdf/drone.urdf`) for both RViz2 visualisation
  and hardware deployment.
- **ROS2 / RViz2 Simulation** — A ready-to-launch ROS2 Python package
  (`ros2_ws/src/isr_dmpc_sim`) publishes drone poses, target positions, DMPC metrics,
  per-drone trajectory paths, and interactive TF frames, all rendered live in RViz2.
- **Hardware Transfer Ready** — `hardware_bridge_node` translates DMPC acceleration
  commands to MAVROS `setpoint_accel` topics (PX4 / ArduPilot).  Switch from
  simulation to live hardware with a single launch argument (`use_sim:=false`).
- **Stability Analysis** — Lyapunov, eigenvalue, ISS, CBF, and recursive feasibility tools.
- **Math Reference** — See [`math_docs/`](math_docs/) for full derivations of all algorithms.
- **Math / Control Optimisation Guide** — See [`docs/MATH_OPTIMIZATION.md`](docs/MATH_OPTIMIZATION.md)
  for a comprehensive guide to improving DMPC and MARL performance on real hardware.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git
cd isr-rl-dmpc

# ── Option A: Conda (recommended for GPU training) ──────────────────────────
conda env create -f environment.yml
conda activate isr-rl-dmpc

# ── Option B: Python venv (CPU / lightweight) ────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt

# Install the package in development mode (both options)
pip install -e .

# Run tests
pytest tests/

# Train MAPPO policy (headless)
python scripts/train_mappo.py --config config/mappo_config.yaml

# ── Run a scenario with pure DMPC (no RL) ───────────────────────────────────
python scripts/run_dmpc.py --scenario area_surveillance
python scripts/run_dmpc.py --scenario threat_response   --episodes 5
python scripts/run_dmpc.py --scenario search_and_track  --num-drones 4

# ── Run the same scenario with DMPC-RL (MAPPO-adaptive cost weights) ─────────
python scripts/run_dmpc_rl.py --scenario area_surveillance
python scripts/run_dmpc_rl.py --scenario threat_response  --model models/mappo_dmpc/final
python scripts/run_dmpc_rl.py --scenario search_and_track --episodes 5

# Open the comparison notebook to analyse both methods side-by-side
jupyter notebook notebooks/05_comparison_analysis.ipynb
```

### ROS2 / RViz2 Simulation

Prerequisites: ROS2 Jazzy installed and sourced (Ubuntu 24.04 Noble).

```bash
# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

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
# WARNING: arm_on_start:=true arms the vehicle 5 s after launch.
# Ensure the drone is in a safe, flight-ready state before using this flag.
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
│   ├── agents/                # MAPPOAgent (SB3 PPO), DMPCAgent
│   ├── analysis/              # Stability analysis tools
│   ├── core/                  # Data structures (DroneState, TargetState, MissionState)
│   ├── gym_env/               # MARLDMPCEnv (40-D obs, 14-D action) + 6-DOF physics
│   ├── models/
│   │   └── meshes/hector_quadrotor/  # Open-source drone mesh source files
│   ├── modules/               # 6 ISR modules + DMPC + ADMM + attitude controller + analytics
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
├── scripts/                   # Standalone mission, training and evaluation scripts
│   ├── run_dmpc.py            # Pure DMPC runner for all 3 task scenarios
│   ├── run_dmpc_rl.py         # DMPC-RL (MAPPO) runner for all 3 task scenarios
│   ├── train_mappo.py         # Train the MAPPO adaptive-DMPC policy
│   ├── evaluate_swarm_tasks.py# Cornerstone-score evaluation suite
│   ├── benchmark.py           # Module benchmarking (timing, memory)
│   └── visualize_results.py   # Trajectory & metric visualisation
├── config/                    # YAML configuration files
│   ├── drone_specs.yaml       # hector_quadrotor-aligned physical parameters
│   ├── dmpc_config.yaml       # DMPC tuning (horizon, Q, R, ADMM rho)
│   ├── mappo_config.yaml      # MAPPO hyperparameters (lr, clip, entropy coeff)
│   └── mission_scenarios.yaml # Real-world ISR applications
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter tutorials
├── math_docs/                 # Mathematical reference documentation
│   ├── README.md              # Index and notation guide
│   ├── 01_DRONE_STATE_SPACE.md
│   ├── 02_EXTENDED_KALMAN_FILTER.md
│   ├── 03_DMPC_FORMULATION.md
│   ├── 04_LYAPUNOV_AND_STABILITY.md
│   ├── 05_FORMATION_CONSENSUS.md
│   ├── 06_TASK_ALLOCATION.md
│   ├── 07_GEOMETRIC_ATTITUDE_CONTROL.md
│   ├── 08_COVERAGE_PLANNING.md
│   ├── 09_MAPPO_AGENT.md      # MARL/MAPPO math and training
│   └── 10_ADMM_CONSENSUS.md   # ADMM consensus derivation
└── docs/
    ├── MATH_OPTIMIZATION.md   # Guide to optimising MARL-DMPC math & control
    └── ...                    # Other documentation
```

## Documentation

| Document | Description |
| :--- | :--- |
| [math_docs/](math_docs/) | **Mathematical reference** — full derivations for all algorithms |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture — MARL + ADMM + DMPC layers |
| [MODULE_SPECS.md](docs/MODULE_SPECS.md) | Detailed module specifications including MAPPO and ADMM |
| [STABILITY_ANALYSIS.md](docs/STABILITY_ANALYSIS.md) | DMPC stability analysis (Lyapunov, ISS, CBF) |
| [MATH_OPTIMIZATION.md](docs/MATH_OPTIMIZATION.md) | Guide to optimising MARL-DMPC math & control |
| [PHASE_GUIDE.md](docs/PHASE_GUIDE.md) | Mission execution and integration phases |
| [GYM_DESIGN.md](docs/GYM_DESIGN.md) | MARL Gymnasium environment design (MARLDMPCEnv) |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API documentation |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [WORKFLOW.md](docs/WORKFLOW.md) | Git workflow guide |

## Technology Stack

| Component | Technology |
| :--- | :--- |
| MARL Training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO (MAPPO) |
| Consensus Layer | ADMM (Alternating Direction Method of Multipliers) |
| Convex Optimisation | [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/) |
| Terminal Cost | Discrete Algebraic Riccati Equation (SciPy DARE) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Physics Simulation | 6-DOF rigid-body (hector_quadrotor airframe, no Gazebo) |
| Drone Model | [hector_quadrotor](https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor) COLLADA mesh + self-contained URDF |
| ROS2 Simulation | rclpy, geometry_msgs, visualization_msgs, tf2, nav_msgs |
| Hardware Interface | MAVROS (PX4 / ArduPilot) via `hardware_bridge_node` |
| Visualisation | RViz2, TensorBoard, Matplotlib |
| Configuration | YAML with dataclass validation |

## Mission Scenarios

Three pre-defined real-world ISR scenarios are available in `config/mission_scenarios.yaml`:

| Scenario | Area | Drones | Targets | Formation | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `area_surveillance` | 400×400 m | 4 | 0 | Grid | 20 min |
| `threat_response` | 250×250 m | 4 | 2 | Wedge | 10 min |
| `search_and_track` | 600×600 m | 4 | 3 | Line | 20 min |

### Running Scenarios

| Task | Command |
| :--- | :--- |
| Pure DMPC — area surveillance | `python scripts/run_dmpc.py --scenario area_surveillance` |
| Pure DMPC — threat response | `python scripts/run_dmpc.py --scenario threat_response` |
| Pure DMPC — search & track | `python scripts/run_dmpc.py --scenario search_and_track` |
| DMPC-RL — area surveillance | `python scripts/run_dmpc_rl.py --scenario area_surveillance` |
| DMPC-RL — threat response | `python scripts/run_dmpc_rl.py --scenario threat_response` |
| DMPC-RL — search & track | `python scripts/run_dmpc_rl.py --scenario search_and_track` |
| Train MAPPO policy | `python scripts/train_mappo.py` |

Results are saved as JSON files in `data/results/dmpc/` and `data/results/dmpc_rl/`
respectively.  Open `notebooks/05_comparison_analysis.ipynb` for a head-to-head comparison.

### Notebooks

| Notebook | Description |
| :--- | :--- |
| `01_system_overview.ipynb` | Package structure and module inventory |
| `02_module_testing.ipynb` | Interactive module unit tests |
| `03_learning_curves.ipynb` | MAPPO training metrics (reward, entropy, losses) |
| `04_mission_analysis.ipynb` | Per-scenario metrics for both Pure DMPC and DMPC-RL |
| `05_comparison_analysis.ipynb` | Head-to-head comparison (reward, battery, solve time) |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Team

| Name | Email |
| :--- | :--- |
| Jivesh Kesar | jrb252026@iitd.ac.in |
| Harsh | jrb252049@iitd.ac.in |
| Rohit Shankar Sinha | jrb252051@iitd.ac.in |
