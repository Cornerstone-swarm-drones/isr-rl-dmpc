# ISR-DMPC

**Autonomous Multi-Drone Swarm System with Distributed Model Predictive Control**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ISR-DMPC is an autonomous Intelligence, Surveillance & Reconnaissance (ISR)
platform for multi-drone swarm coordination powered by a **purely
optimisation-based Distributed Model Predictive Controller (DMPC)**.  The
system performs grid-based coverage planning, real-time target tracking, and
threat assessment using a modular 6-module architecture together with a
Gymnasium-compatible simulation environment.

> **Migration note (v2.0):** All reinforcement-learning and neural-network
> components have been removed.  The controller now relies entirely on
> CVXPY/OSQP convex optimisation with an LQR-derived terminal cost.  See
> [ARCHITECTURE.md](docs/ARCHITECTURE.md) and
> [STABILITY_ANALYSIS.md](docs/STABILITY_ANALYSIS.md) for details.

## Key Features

- **Pure DMPC Control** — CVXPY/OSQP solves a constrained QP at 50 Hz with
  hard collision-avoidance constraints and an analytically computed terminal
  cost (DARE).
- **6 Mission Modules** — Mission planning, formation control, sensor fusion,
  target classification, threat assessment, and task allocation.
- **Geometric Attitude Controller** — SO(3) control with fixed LQR-tuned gains
  (no neural adaptation).
- **Gymnasium-Compatible Environment** — `ISRGridEnv` with Dict observation
  spaces, multi-objective rewards, and vectorised training support.
- **6-DOF Physics Simulation** — Realistic rigid-body dynamics with wind,
  battery depletion, and collision detection.
- **Stability Analysis** — Lyapunov, eigenvalue, ISS, CBF, and recursive
  feasibility tools in `isr_rl_dmpc.analysis`.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git
cd isr-rl-dmpc

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt

# Install the package in development mode
pip install -e .

# Run tests
pytest tests/

# Run a DMPC mission simulation
python scripts/run_mission.py --config config/dmpc_config.yaml

# Run stability analysis
python -c "
from isr_rl_dmpc.analysis import DMPCStabilityAnalyzer
r = DMPCStabilityAnalyzer().full_stability_report()
print(r.summary)
"

# Launch Jupyter notebook
jupyter notebook notebooks/01_system_overview.ipynb
```

## Project Structure

```
isr-rl-dmpc/
├── src/isr_rl_dmpc/           # Main package
│   ├── agents/                # DMPCAgent (pure DMPC, no RL)
│   ├── analysis/              # Stability analysis tools
│   ├── core/                  # Data structures (DroneState, TargetState, MissionState)
│   ├── gym_env/               # Gymnasium environment (ISRGridEnv, simulator, rewards)
│   ├── models/                # Checkpoint utilities (no NN model files)
│   ├── modules/               # 6 core ISR modules + DMPC + attitude + analytics
│   ├── utils/                 # Math, logging, visualization, and conversions
│   └── config.py              # Configuration system with dataclass validation
├── scripts/                   # Mission execution and evaluation scripts
│   ├── run_mission.py         # Main mission runner
│   ├── foxglove_visualize.py  # Foxglove Studio visualization
│   ├── benchmark.py           # Performance benchmarking
│   ├── test_mission.py        # Mission validation
│   ├── calibrate_sensors.py   # Sensor calibration
│   └── visualize_results.py   # Results visualization
├── config/                    # YAML configuration files
│   ├── default_config.yaml    # Default parameters
│   ├── dmpc_config.yaml       # DMPC-specific parameters
│   ├── drone_specs.yaml       # Drone physical specifications
│   ├── sensor_specs.yaml      # Sensor specifications
│   └── mission_scenarios.yaml # Pre-defined mission scenarios
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter tutorials
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture details
│   ├── MODULE_SPECS.md        # Detailed module specifications
│   ├── STABILITY_ANALYSIS.md  # DMPC stability analysis
│   ├── PHASE_GUIDE.md         # Project phase descriptions
│   ├── GYM_DESIGN.md          # Gym environment design
│   ├── API_REFERENCE.md       # Function/class documentation
│   ├── TROUBLESHOOTING.md     # Common issues and solutions
│   └── WORKFLOW.md            # Git workflow guide
├── requirements/              # Dependency files
├── environment.yml            # Conda environment
├── setup.py                   # Package setup
└── requirements.txt           # pip dependencies
```

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design patterns |
| [MODULE_SPECS.md](docs/MODULE_SPECS.md) | Detailed specifications for all modules |
| [STABILITY_ANALYSIS.md](docs/STABILITY_ANALYSIS.md) | DMPC stability analysis for ISR |
| [PHASE_GUIDE.md](docs/PHASE_GUIDE.md) | Project phase descriptions and workflows |
| [GYM_DESIGN.md](docs/GYM_DESIGN.md) | Gymnasium environment design and usage |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Function and class API documentation |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [FOXGLOVE_INTEGRATION.md](docs/FOXGLOVE_INTEGRATION.md) | Foxglove Studio visualization setup |
| [WORKFLOW.md](docs/WORKFLOW.md) | Git workflow guide for contributors |

## Technology Stack

| Component | Technology |
|---|---|
| Convex Optimisation | [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/) |
| Terminal Cost | Discrete Algebraic Riccati Equation (SciPy DARE) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Simulation | Gymnasium, 6-DOF rigid-body physics |
| Visualisation | Matplotlib, [Foxglove Studio](https://foxglove.dev/) |
| Configuration | YAML with dataclass validation |

## Mission Scenarios

Three pre-defined scenarios are available in `config/mission_scenarios.yaml`:

| Scenario | Area | Drones | Targets | Formation | Duration |
|---|---|---|---|---|---|
| Area Surveillance | 500×500 m | 4 | 0 | Grid | 30 min |
| Threat Response | 300×300 m | 6 | 3 | Wedge | 10 min |
| Search & Track | 800×800 m | 5 | 4 | Line | 20 min |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Team

| Name | Email |
|---|---|
| Jivesh Kesar | jrb252026@iitd.ac.in |
| Harsh | jrb252049@iitd.ac.in |
| Rohit Shankar Sinha | jrb252051@iitd.ac.in |
