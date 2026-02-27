# System Architecture

This document describes the high-level architecture of the ISR-RL-DMPC system, including its design patterns, data flow, and component interactions.

## Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Layer Structure](#layer-structure)
- [Module System](#module-system)
- [Data Flow](#data-flow)
- [Configuration System](#configuration-system)
- [Design Patterns](#design-patterns)
- [Technology Stack](#technology-stack)

## Overview

ISR-RL-DMPC is a **hybrid learning-control system** that combines Reinforcement Learning with Distributed Model Predictive Control for autonomous multi-drone swarm operations. The architecture follows a layered design where:

1. **Perception Layer** — Sensor fusion and state estimation
2. **Decision Layer** — Mission planning, classification, threat assessment, task allocation
3. **Control Layer** — RL-optimized DMPC and attitude control
4. **Learning Layer** — Value and policy networks for parameter optimization

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ISRGridEnv (Gymnasium)                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Observation (Dict)                         │   │
│  │  swarm: (N,18)  targets: (M,12)  env: (K+4,)  adj: (N,N)   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    DMPCAgent                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │   │
│  │  │ PolicyNetwork  │  │ ValueNetwork   │  │ ExperienceBuffer│ │   │
│  │  │ (Actor)        │  │ (Critic)       │  │ (Prioritized) │  │   │
│  │  └────────────────┘  └────────────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 9 Integrated Modules                          │   │
│  │                                                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ Mission  │ │Formation │ │ Sensor   │ │ Classifi-│       │   │
│  │  │ Planner  │ │Controller│ │ Fusion   │ │ cation   │       │   │
│  │  │  (M1)    │ │  (M2)    │ │  (M3)    │ │  (M4)    │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  │                                                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ Threat   │ │ Task     │ │ DMPC     │ │ Attitude │       │   │
│  │  │ Assessor │ │Allocator │ │Controller│ │Controller│       │   │
│  │  │  (M5)    │ │  (M6)    │ │  (M7)    │ │  (M8)    │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  │                                                               │   │
│  │  ┌──────────┐                                                │   │
│  │  │ Learning │                                                │   │
│  │  │ Module   │                                                │   │
│  │  │  (M9)    │                                                │   │
│  │  └──────────┘                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              EnvironmentSimulator (6-DOF)                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │   │
│  │  │ DronePhysics│ │TargetPhysics│ │ WindModel  │             │   │
│  │  └────────────┘  └────────────┘  └────────────┘             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                 Action: Motor PWM (N, 4)                            │
│                 Reward: Scalar (5 components)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Structure

### Perception Layer

Responsible for sensing and state estimation.

| Component | File | Purpose |
|---|---|---|
| `SensorFusionManager` | `modules/sensor_fusion.py` | Fuses radar, optical, RF, and acoustic sensor data |
| `DroneStateEstimation` | `core/drone_state_estimation.py` | Extended Kalman Filter (EKF) for drone pose estimation |
| `TargetStateEstimation` | `core/target_state_estimation.py` | EKF for target tracking with 11-state model |
| `SensorSimulator` | `gym_env/sensor_simulator.py` | Realistic sensor noise modeling |

### Decision Layer

Mission-level planning and decision making.

| Component | File | Purpose |
|---|---|---|
| `MissionPlanner` | `modules/mission_planner.py` | Grid decomposition and waypoint generation |
| `ClassificationEngine` | `modules/classification_engine.py` | Bayesian target classification |
| `ThreatAssessor` | `modules/threat_assessor.py` | Threat level evaluation |
| `TaskAllocator` | `modules/task_allocator.py` | Hungarian algorithm task assignment |

### Control Layer

Real-time swarm and individual drone control.

| Component | File | Purpose |
|---|---|---|
| `FormationController` | `modules/formation_controller.py` | Consensus-based formation control |
| `DMPC` | `modules/dmpc_controller.py` | CVXPY solver with learned cost weights |
| `AttitudeController` | `modules/attitude_controller.py` | Geometric attitude control |

### Learning Layer

Reinforcement learning for DMPC parameter optimization.

| Component | File | Purpose |
|---|---|---|
| `LearningModule` | `modules/learning_module.py` | Value + Policy networks (PyTorch) |
| `DMPCAgent` | `agents/dmpc_agent.py` | Unified RL agent interface |
| `ActorCriticTrainer` | `agents/actor_critic.py` | GAE-based training with entropy bonus |
| `PrioritizedReplayBuffer` | `agents/experience_buffer.py` | Prioritized experience replay |

## Module System

The system is built around 9 core modules that operate together:

| # | Module | Input | Output |
|---|---|---|---|
| 1 | Mission Planner | Area boundaries, drone count | Grid cells, waypoints |
| 2 | Formation Controller | Drone states, target formation | Formation commands |
| 3 | Sensor Fusion | Raw sensor readings (4 types) | Fused detections |
| 4 | Classification Engine | Sensor features | Target classifications |
| 5 | Threat Assessor | Classifications, dynamics | Threat levels |
| 6 | Task Allocator | Drone states, tasks | Drone–task assignments |
| 7 | DMPC Controller | States, cost weights | Optimal control inputs |
| 8 | Attitude Controller | Desired attitude, current state | Motor torques |
| 9 | Learning Module | States, rewards | Updated cost weights |

## Data Flow

### Training Loop

```
1. env.reset() → initial observation (Dict)
2. agent.act(obs) → action (motor PWM)
3. env.step(action) → (next_obs, reward, terminated, truncated, info)
4. agent.remember(obs, action, reward, next_obs, done)
5. agent.train_on_batch() → (critic_loss, actor_loss)
6. Repeat 2-5 for max_steps
7. Repeat 1-6 for num_episodes
```

### Observation Pipeline

```
DronePhysics (position, velocity, quaternion, battery, health)
    → to_vector() → 18D per drone → swarm observation (N, 18)

TargetPhysics (position, velocity, acceleration, yaw)
    → to_vector() → 12D per target → target observation (M, 12)

CoverageMap (grid_cells) + MissionState (4 scalars)
    → concatenate → environment observation (K+4,)

Inter-drone distances → adjacency matrix (N, N)
```

### Control Pipeline

```
PolicyNetwork(state) → (mean, log_std) → sampled action
    → reshape to (num_drones, 4) → motor PWM commands
    → EnvironmentSimulator.step() → 6-DOF dynamics update
    → collision detection, battery update, wind effects
    → new drone/target states
```

## Configuration System

The configuration system uses Python dataclasses with YAML serialization:

```python
from isr_rl_dmpc.config import load_config

# Load with defaults
config = load_config()

# Load from file with overrides
config = load_config(
    "config/default_config.yaml",
    overrides={"learning": {"batch_size": 64}}
)

# Access parameters
lr = config.learning.learning_rate_critic
horizon = config.dmpc.prediction_horizon

# Validate all parameters
config.validate()
```

### Configuration Hierarchy

```
Config (master)
├── DroneConfig       — Physical and control parameters
├── SensorConfig      — Sensor and frequency parameters
├── MissionConfig     — Coverage and separation parameters
├── LearningConfig    — RL hyperparameters and reward weights
└── DMPCConfig        — Prediction horizon and solver parameters
```

## Design Patterns

| Pattern | Implementation | Purpose |
|---|---|---|
| **Hybrid Control** | CVXPY optimizer + PyTorch networks | Safety through convex constraints, adaptivity through learning |
| **Actor-Critic** | Separate PolicyNetwork and ValueNetwork | Stable policy gradient learning |
| **Prioritized Replay** | PrioritizedReplayBuffer with importance sampling | Efficient experience utilization |
| **State Manager** | Centralized StateManager with thread safety | Consistent state access across modules |
| **Factory Pattern** | `make_env()` and `load_config()` functions | Simplified environment and config creation |
| **Dataclass Configs** | `@dataclass` with `validate()` methods | Type-safe, validated configuration |
| **Modular Design** | 9 independent modules with clear interfaces | Testability and maintainability |

## Technology Stack

```
Application Layer
├── Gymnasium          — RL environment interface
├── PyTorch            — Neural networks and gradient computation
└── CVXPY              — Convex optimization for DMPC

Computation Layer
├── NumPy              — Array operations
├── SciPy              — Scientific algorithms (Hungarian, linalg)
└── scikit-learn       — Random Forest classifier

Infrastructure Layer
├── YAML               — Configuration serialization
├── Matplotlib         — Visualization
├── Jupyter            — Interactive notebooks
└── pytest             — Testing framework
```
