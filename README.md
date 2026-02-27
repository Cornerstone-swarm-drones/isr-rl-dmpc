# ISR-RL-DMPC

**Autonomous Multi-Drone Swarm System with Reinforcement Learning-Based Distributed Model Predictive Control**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ISR-RL-DMPC is an autonomous Intelligence, Surveillance & Reconnaissance (ISR) platform that combines **Reinforcement Learning (RL)** with **Distributed Model Predictive Control (DMPC)** for multi-drone swarm coordination. The system performs grid-based coverage planning, real-time target tracking, and threat assessment using a modular 9-component architecture with a Gymnasium-compatible RL environment.

## Key Features

- **Hybrid RL + DMPC Control** — Neural networks learn optimal DMPC cost function parameters while CVXPY ensures constraint satisfaction
- **9 Integrated Modules** — Mission planning, formation control, sensor fusion, classification, threat assessment, task allocation, DMPC, attitude control, and learning
- **Gymnasium-Compatible Environment** — `ISRGridEnv` with Dict observation spaces, multi-objective rewards, and vectorized training support
- **6-DOF Physics Simulation** — Realistic rigid body dynamics with wind, battery depletion, and collision detection
- **Multi-Objective Reward Shaping** — Five-component reward (coverage, energy, safety, threat engagement, formation)
- **Configurable Mission Scenarios** — Area surveillance, threat response, and search-and-track with YAML-based configuration

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

# Start training
python scripts/train_agent.py --config config/default_config.yaml

# Launch Jupyter notebook
jupyter notebook notebooks/01_system_overview.ipynb
```

## Project Structure

```
isr-rl-dmpc/
├── src/isr_rl_dmpc/           # Main package
│   ├── agents/                # RL agents (DMPCAgent, ActorCriticTrainer)
│   ├── core/                  # Data structures (DroneState, TargetState, MissionState)
│   ├── gym_env/               # Gymnasium environment (ISRGridEnv, simulator, rewards)
│   ├── models/                # Pre-trained models, network definitions, and 3D meshes
│   ├── modules/               # 9 core functional modules
│   ├── utils/                 # Math, logging, visualization, and conversions
│   └── config.py              # Configuration system with dataclass validation
├── scripts/                   # Training, evaluation, and hyperparameter search
│   ├── train_agent.py         # Main training loop
│   ├── foxglove_visualize.py  # Foxglove Studio visualization (live/record)
│   ├── hyperparameter_search.py  # Grid/random hyperparameter search
│   ├── benchmark.py           # Performance benchmarking
│   ├── test_mission.py        # Mission validation
│   ├── calibrate_sensors.py   # Sensor calibration
│   └── visualize_results.py   # Results visualization
├── config/                    # YAML configuration files
│   ├── default_config.yaml    # Default parameters
│   ├── learning_config.yaml   # RL hyperparameters
│   ├── drone_specs.yaml       # Drone physical specifications
│   ├── sensor_specs.yaml      # Sensor specifications
│   └── mission_scenarios.yaml # Pre-defined mission scenarios
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter tutorials
│   ├── 01_system_overview.ipynb
│   ├── 02_module_testing.ipynb
│   ├── 03_learning_curves.ipynb
│   └── 04_mission_analysis.ipynb
├── docs/                      # Documentation
│   ├── TRAINING.md            # Hyperparameter tuning guide
│   ├── ARCHITECTURE.md        # System architecture details
│   ├── MODULE_SPECS.md        # Detailed module specifications
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
| [TRAINING.md](docs/TRAINING.md) | How to tune hyperparameters for optimal model training |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design patterns |
| [MODULE_SPECS.md](docs/MODULE_SPECS.md) | Detailed specifications for all 9 modules |
| [PHASE_GUIDE.md](docs/PHASE_GUIDE.md) | Project phase descriptions and workflows |
| [GYM_DESIGN.md](docs/GYM_DESIGN.md) | Gymnasium environment design and usage |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Function and class API documentation |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [FOXGLOVE_INTEGRATION.md](docs/FOXGLOVE_INTEGRATION.md) | Foxglove Studio visualization setup and usage |
| [WORKFLOW.md](docs/WORKFLOW.md) | Git workflow guide for contributors |

## Technology Stack

| Component | Technology |
|---|---|
| RL Framework | [Gymnasium](https://gymnasium.farama.org/) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Convex Optimization | [CVXPY](https://www.cvxpy.org/) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Visualization | Matplotlib, [Foxglove Studio](https://foxglove.dev/) |
| Configuration | YAML with dataclass validation |

## Training

Train the DMPC agent using the default configuration:

```bash
python scripts/train_agent.py \
    --config config/default_config.yaml \
    --num-episodes 500 \
    --num-steps 1000 \
    --device cuda \
    --output-dir data/training_logs
```

Run hyperparameter search to find the best configuration:

```bash
python scripts/hyperparameter_search.py \
    --config config/default_config.yaml \
    --num-trials 100 \
    --output-dir data/hyperparameter_search
```

See [TRAINING.md](docs/TRAINING.md) for a detailed guide on tuning hyperparameters.

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
