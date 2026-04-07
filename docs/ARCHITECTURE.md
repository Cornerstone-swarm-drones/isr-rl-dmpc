# System Architecture

This document describes the high-level architecture of the ISR-RL-DMPC system,
including its design patterns, data flow, and component interactions.

> **v3.0:** The system implements a **Multi-Agent Reinforcement Learning (MARL)
> framework using MAPPO (Multi-Agent Proximal Policy Optimization) to dynamically
> tune the cost parameters of a Distributed Model Predictive Control (DMPC) layer**.
> Coordination between drones is achieved through **ADMM (Alternating Direction
> Method of Multipliers)** to ensure consensus on shared constraints and objectives.

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

ISR-RL-DMPC is a **MARL-adaptive multi-drone swarm system** that uses MAPPO to
dynamically tune DMPC cost parameters, with ADMM enforcing inter-drone consensus.
The architecture follows a layered design:

1. **Perception Layer** — Sensor fusion and EKF state estimation
2. **Decision Layer** — Mission planning, classification, threat assessment, task allocation
3. **MARL Layer** — MAPPO agents output per-drone Q/R cost scale vectors (14-D)
4. **Consensus Layer** — ADMM synchronises local DMPC sub-problems across drones
5. **Control Layer** — Per-drone DMPC (CVXPY/OSQP) + geometric attitude control
6. **Analytics Layer** — DMPC performance monitoring and parameter diagnostics

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MARLDMPCEnv (Gymnasium)                         │
│  Per-Agent Observation (40-D per drone)                             │
│  own_state(11) + ref(6) + tracking_err(3) + neighbour(7) + misc(13)│
│                              │                                      │
│                              ▼                                      │
│  MAPPOAgent  (Stable-Baselines3 PPO)                                │
│  Actor π_θ(o^i)  →  action (14-D: q_scale(11) + r_scale(3))        │
│  Critic V_φ(o^1,...,o^N)  →  value  [centralised, CTDE]            │
│                              │  q_scale, r_scale                   │
│                              ▼                                      │
│  ADMMConsensus                                                      │
│  z-update → v-update → dual-update  (3–5 iters per step)           │
│                              │  consensus variable v               │
│                              ▼                                      │
│  DMPCAgent (per drone)                                              │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │  DMPC (CVXPY / OSQP)         │  │  AttitudeController (SO(3))│  │
│  │  Q_eff = Q ⊙ diag(q_scale)  │  │  Geometric PD + gyro FF    │  │
│  │  Terminal cost: DARE         │  └────────────────────────────┘  │
│  └──────────────────────────────┘                                  │
│                              │                                      │
│  10 Integrated Modules                                              │
│  M1 MissionPlanner  M2 FormationController  M3 SensorFusion         │
│  M4 Classification  M5 ThreatAssessor       M6 TaskAllocator        │
│  M7 ADMMConsensus   M8 DMPC                 M9 AttitudeController   │
│  M10 DMPCAnalytics                                                  │
│                              │                                      │
│  EnvironmentSimulator (6-DOF rigid-body)                            │
│  DronePhysics  TargetPhysics  WindModel                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Structure

### MARL Layer

| Component | File | Purpose |
|---|---|---|
| `MAPPOAgent` | `agents/mappo_agent.py` | SB3 PPO; shared actor + centralised critic; outputs Q/R scales |
| `MARLDMPCEnv` | `gym_env/marl_env.py` | 40-D obs, 14-D action MARL Gymnasium environment |

### Consensus Layer

| Component | File | Purpose |
|---|---|---|
| `ADMMConsensus` | `modules/admm_consensus.py` | ADMM z/v/dual updates; enforces inter-drone trajectory agreement |

### Control Layer

| Component | File | Purpose |
|---|---|---|
| `DMPC` | `modules/dmpc_controller.py` | CVXPY/OSQP MPC with MARL-adaptive cost and DARE terminal cost |
| `AttitudeController` | `modules/attitude_controller.py` | Geometric SO(3) PD control (fixed gains) |
| `FormationController` | `modules/formation_controller.py` | Consensus-based formation keeping |
| `DMPCAgent` | `agents/dmpc_agent.py` | Unified agent interface |

### Perception & Decision Layers

| Component | File | Purpose |
|---|---|---|
| `SensorFusionManager` | `modules/sensor_fusion.py` | EKF-based multi-sensor fusion |
| `MissionPlanner` | `modules/mission_planner.py` | Grid decomposition and waypoint generation |
| `ClassificationEngine` | `modules/classification_engine.py` | Bayesian target classification |
| `ThreatAssessor` | `modules/threat_assessor.py` | Priority scoring of detected targets |
| `TaskAllocator` | `modules/task_allocator.py` | Hungarian-algorithm task assignment |
| `DMPCStabilityAnalyzer` | `analysis/stability_analysis.py` | Lyapunov, ISS, CBF, feasibility checks |

## Module System

| # | Module | Input | Output |
|---|---|---|---|
| 1 | Mission Planner | Area polygon, num drones | Waypoint sequences |
| 2 | Formation Controller | Waypoints, neighbour states | Formation velocity commands |
| 3 | Sensor Fusion | Raw sensor readings | Filtered drone/target states |
| 4 | Classification Engine | Target observations | Classification labels |
| 5 | Threat Assessor | Classifications, positions | Threat priority scores |
| 6 | Task Allocator | Threats, drone capabilities | Drone → task assignment |
| 7 | ADMM Consensus | Local DMPC solutions (z_i) | Consensus variable v, dual vars μ |
| 8 | DMPC Controller | State, reference, q_scale, r_scale | Optimal accel command (OSQP) |
| 9 | Attitude Controller | State, accel command | Motor thrust commands |
| 10 | DMPC Analytics | Solve results | Performance metrics |

## Data Flow

```
Sensors → M3 SensorFusion → M4 Classification → M5 ThreatAssessment
                                                       ↓
M1 MissionPlanner → M2 FormationController → M6 TaskAllocator
                                                       ↓
                         MAPPO Agent → (q_scale, r_scale per drone)
                                                       ↓
                    M7 ADMMConsensus ←→ M8 DMPC → M9 Attitude → Motor PWM
                                              ↕
                                        M10 Analytics
```

## Configuration System

```python
from isr_rl_dmpc.config import load_config

config = load_config("config/dmpc_config.yaml")
print(config.dmpc.prediction_horizon)  # 20
print(config.dmpc.admm_rho)            # 1.0
```

Key config files:

| File | Purpose |
|---|---|
| `config/dmpc_config.yaml` | MPC horizon, base cost matrices, ADMM rho, solver timeout |
| `config/mappo_config.yaml` | PPO lr, clip, epochs, hidden dims, action bounds |
| `config/drone_specs.yaml` | Physical parameters, sensor specs (hector_quadrotor) |
| `config/mission_scenarios.yaml` | Pre-defined ISR mission scenarios |

## Design Patterns

| Pattern | Implementation | Rationale |
|---|---|---|
| **CTDE (MARL)** | Centralised critic, decentralised actors (MAPPO) | Reduces variance; enables distributed execution |
| **Adaptive Cost** | MAPPO outputs Q/R scales; `Q_eff = Q ⊙ diag(q_s)` | Preserves QP convexity while enabling online adaptation |
| **ADMM Consensus** | ADMMConsensus wraps per-drone DMPC solves | Provable convergence to globally consistent solution |
| **DARE Terminal Cost** | `scipy.linalg.solve_discrete_are` | Guarantees recursive feasibility regardless of MAPPO output |
| **Fixed-Gain Attitude** | Geometric SO(3) with DARE-tuned gains | Deterministic, real-time safe inner loop |
| **Analytics Separation** | DMPCAnalytics independent of control | Non-invasive monitoring |

## Technology Stack

| Component | Technology |
|---|---|
| MARL Training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO (MAPPO) |
| Consensus | ADMM (custom `ADMMConsensus` module) |
| Convex Optimisation | [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/) |
| Terminal Cost | Discrete Algebraic Riccati Equation (SciPy) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Simulation | Gymnasium, 6-DOF rigid-body physics |
| Visualisation | RViz2, TensorBoard, Matplotlib |
| Configuration | YAML with dataclass validation |
