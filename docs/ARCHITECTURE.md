# System Architecture

This document describes the high-level architecture of the ISR-DMPC system,
including its design patterns, data flow, and component interactions.

> **v2.0 note:** All reinforcement-learning and neural-network layers have been
> removed.  The DMPC controller now relies solely on CVXPY/OSQP convex
> optimisation with an LQR-derived terminal cost.  Module numbering (1–6) is
> unchanged; the DMPC (7), Attitude Controller (8), and DMPC Analytics (9)
> replace the former RL-based Modules 7–9.

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

ISR-DMPC is a **purely optimisation-based multi-drone swarm system** that uses
Distributed Model Predictive Control for autonomous ISR missions. The
architecture follows a layered design:

1. **Perception Layer** — Sensor fusion and state estimation
2. **Decision Layer** — Mission planning, classification, threat assessment, task allocation
3. **Control Layer** — Pure DMPC (CVXPY/OSQP) and geometric attitude control
4. **Analytics Layer** — DMPC performance monitoring and parameter diagnostics

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
│  │  ┌────────────────────────────┐  ┌────────────────────────┐  │   │
│  │  │  DMPC (CVXPY/OSQP solver)  │  │  AttitudeController    │  │   │
│  │  │  Terminal cost: DARE       │  │  (Geometric SO(3))     │  │   │
│  │  └────────────────────────────┘  └────────────────────────┘  │   │
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
│  │  │ Threat   │ │ Task     │ │   DMPC   │ │ Attitude │       │   │
│  │  │ Assessor │ │Allocator │ │ (CVXPY)  │ │Controller│       │   │
│  │  │  (M5)    │ │  (M6)    │ │  (M7)    │ │  (M8)    │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  │                                                               │   │
│  │  ┌──────────┐                                                │   │
│  │  │  DMPC    │                                                │   │
│  │  │Analytics │                                                │   │
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

| Component | File | Purpose |
|---|---|---|
| `SensorFusionManager` | `modules/sensor_fusion.py` | Fuses radar, optical, RF, and acoustic sensor data |
| `DroneStateEstimation` | `core/drone_state_estimation.py` | EKF for drone pose estimation |
| `TargetStateEstimation` | `core/target_state_estimation.py` | EKF for target tracking |
| `SensorSimulator` | `gym_env/sensor_simulator.py` | Realistic sensor noise modelling |

### Decision Layer

| Component | File | Purpose |
|---|---|---|
| `MissionPlanner` | `modules/mission_planner.py` | Grid decomposition and waypoint generation |
| `ClassificationEngine` | `modules/classification_engine.py` | Bayesian target classification |
| `ThreatAssessor` | `modules/threat_assessor.py` | Priority scoring of detected targets |
| `TaskAllocator` | `modules/task_allocator.py` | Hungarian-algorithm task assignment |

### Control Layer

| Component | File | Purpose |
|---|---|---|
| `DMPC` | `modules/dmpc_controller.py` | Pure CVXPY/OSQP MPC with DARE terminal cost |
| `MPCSolver` | `modules/dmpc_controller.py` | Low-level QP solver interface |
| `AttitudeController` | `modules/attitude_controller.py` | Geometric SO(3) control (fixed gains) |
| `GeometricController` | `modules/attitude_controller.py` | SO(3) manifold math (NumPy) |
| `FormationController` | `modules/formation_controller.py` | Consensus-based formation keeping |
| `DMPCAgent` | `agents/dmpc_agent.py` | Unified agent interface (no RL) |

### Analytics Layer

| Component | File | Purpose |
|---|---|---|
| `DMPCAnalytics` | `modules/learning_module.py` | Step-level performance metrics |
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
| 7 | DMPC Controller | Current state, reference | Optimal control sequence (CVXPY) |
| 8 | Attitude Controller | State, accel command | Motor thrust commands |
| 9 | DMPC Analytics | Solve results | Performance metrics |

## Data Flow

```
State → M3 (Sensor Fusion) → M4 (Classification) → M5 (Threat Assessment)
                                                    ↓
M1 (Mission Planner) → M2 (Formation Controller) → M6 (Task Allocator)
                                                    ↓
                              M7 (DMPC) → M8 (Attitude) → Motor Commands
                                   ↕
                              M9 (Analytics)
```

## Configuration System

```python
from isr_rl_dmpc.config import load_config

config = load_config("config/dmpc_config.yaml")
print(config.dmpc.prediction_horizon)  # 20
print(config.dmpc.accel_max)           # 10.0
```

The configuration hierarchy is:

```
Config
├── DMPCConfig     — MPC horizon, cost matrices, solver settings
├── DroneConfig    — Physical parameters, sensor specs
├── MissionConfig  — Grid size, coverage targets, scenario
└── SensorConfig   — Update rates, noise models
```

## Design Patterns

| Pattern | Implementation | Rationale |
|---|---|---|
| **Pure Optimisation** | CVXPY/OSQP solver only | Provable stability through convex analysis |
| **Fixed-Gain Attitude Control** | Geometric SO(3) with DARE-tuned gains | Deterministic, real-time safe |
| **DARE Terminal Cost** | scipy.linalg.solve_discrete_are | Guarantees recursive feasibility |
| **Factory Pattern** | `make_env()`, `load_config()` | Simplified environment creation |
| **Analytics Separation** | DMPCAnalytics independent of control | Non-invasive monitoring |

## Technology Stack

| Component | Technology |
|---|---|
| Convex Optimisation | [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/) |
| Terminal Cost | Discrete Algebraic Riccati Equation (SciPy) |
| Attitude Control | Geometric control on SO(3) |
| Task Allocation | Hungarian Algorithm (SciPy) |
| Scientific Computing | NumPy, SciPy |
| Simulation | Gymnasium, 6-DOF rigid-body physics |
| Visualisation | Matplotlib, Foxglove Studio |
| Configuration | YAML with dataclass validation |
