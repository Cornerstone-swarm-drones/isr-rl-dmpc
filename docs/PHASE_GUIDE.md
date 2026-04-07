# Phase Guide

This document describes the mission execution pipeline and the development integration stages for the ISR-DMPC system.

## Table of Contents

- [Mission Execution Phases](#mission-execution-phases)
- [Integration Phases](#integration-phases)

---

## Mission Execution Phases

Each ISR mission progresses through three phases, each handled by a specific subset of the 9 modules.

### Phase 1: Mission Planning

**Objective:** Decompose the mission area and generate coverage paths.

**Active Modules:**
- Module 1 (Mission Planner) — Grid decomposition and waypoint generation

**Process:**

1. **Area Decomposition** — The mission area (defined in `config/mission_scenarios.yaml`) is divided into a grid of cells based on `grid_cell_size` (default: 10 m).
1. **Waypoint Generation** — Coverage waypoints are generated to ensure each grid cell is visited at least once. The planner accounts for sensor `coverage_radius` and drone count.
1. **Path Allocation** — Waypoints are distributed among drones to minimize total travel distance while meeting the `coverage_goal` (default: 0.95).

**Key Parameters:**

| Parameter | Config | Default | Effect |
| :--- | :--- | :--- | :--- |
| `grid_cell_size` | `default_config.yaml` | `10.0 m` | Smaller cells → finer coverage, more waypoints |
| `coverage_radius` | `default_config.yaml` | `5.0 m` | Larger radius → fewer passes needed |
| `coverage_goal` | `default_config.yaml` | `0.95` | Higher goal → longer mission duration |

**Output:** Grid map, waypoint sequences per drone, initial coverage plan.

---

### Phase 2: Swarm Coordination

**Objective:** Coordinate drone formation, fuse sensor data, classify targets, assess threats, and allocate tasks.

**Active Modules:**
- Module 2 (Formation Controller) — Maintains formation geometry
- Module 3 (Sensor Fusion) — Fuses multi-sensor detections
- Module 4 (Classification Engine) — Classifies detected targets
- Module 5 (Threat Assessor) — Evaluates threat levels
- Module 6 (Task Allocator) — Assigns drones to tasks

**Process:**

1. **Formation Control** — Drones maintain a specified formation (grid, wedge, or line) using a distributed consensus protocol. Each drone adjusts its position based on neighbor states within the `communication_radius`.
1. **Sensor Fusion** — Raw readings from radar, optical, RF, and acoustic sensors are fused into unified detections with uncertainty estimates.
1. **Classification** — Detected targets are classified as friendly, hostile, or neutral using Bayesian inference.
1. **Threat Assessment** — Threat scores are computed based on classification confidence, target dynamics, and proximity.
1. **Task Allocation** — The Hungarian algorithm assigns drones to tasks (coverage waypoints, target tracking, threat engagement) to minimize total assignment cost.

**Key Parameters:**

| Parameter | Config | Default | Effect |
| :--- | :--- | :--- | :--- |
| `communication_radius` | `default_config.yaml` | `100.0 m` | Larger radius → more neighbors, better consensus |
| `min_swarm_separation` | `default_config.yaml` | `2.0 m` | Safety distance between drones |
| `formation_type` | `mission_scenarios.yaml` | per-scenario | Grid, wedge, or line |

**Output:** Formation commands, fused detections, target classifications, threat levels, task assignments.

---

### Phase 3: Control and Analytics

**Objective:** Execute optimal control actions and monitor DMPC performance.

**Active Modules:**
- Module 7 (DMPC Controller) — Computes optimal control inputs
- Module 8 (Attitude Controller) — Tracks desired attitudes
- Module 9 (DMPC Analytics) — Records performance metrics

**Process:**

1. **DMPC Optimisation** — The DMPC controller solves a convex QP (via CVXPY/OSQP) over the prediction horizon to compute optimal acceleration commands. Cost weights Q, R, P are fixed (no online adaptation); P is pre-computed from the Discrete Algebraic Riccati Equation (DARE).
1. **Attitude Control** — Desired accelerations are converted to attitude references, and the geometric SO(3) attitude controller computes motor torques to track them.
1. **Analytics** — Per-step statistics (tracking error, solve time, objective value) are accumulated by `DMPCAnalytics` for post-mission diagnostics.

**Key Parameters:**

| Parameter | Config | Default | Effect |
| :--- | :--- | :--- | :--- |
| `prediction_horizon` | `dmpc_config.yaml` | `20` | Longer horizon → better planning, slower solve |
| `accel_max` | `dmpc_config.yaml` | `10.0 m/s²` | Control saturation bound |
| `collision_radius` | `dmpc_config.yaml` | `5.0 m` | Minimum inter-drone separation |
| `solver_timeout` | `dmpc_config.yaml` | `10 ms` | OSQP time budget per drone |

**Output:** Motor PWM commands, per-step DMPC analytics.

---

## Integration Phases

The modules are integrated in a specific order during development:

### Stage 1: Core Infrastructure

- Core data structures (`DroneState`, `TargetState`, `MissionState`)
- Configuration system (`config.py`)
- State manager (`state_manager.py`)
- Utility functions (math, logging, conversions)

### Stage 2: Simulation Environment

- Physics simulator (6-DOF dynamics)
- Sensor simulator (noise modeling)
- Gymnasium environment (`ISRGridEnv`)
- Reward shaper (multi-objective rewards)

### Stage 3: Planning and Perception

- Mission planner (grid decomposition, waypoints)
- Sensor fusion (multi-modal data fusion)
- Classification engine (Bayesian target classification)
- Threat assessor (threat level computation)

### Stage 4: Control

- Formation controller (consensus-based)
- Task allocator (Hungarian algorithm)
- DMPC controller (CVXPY/OSQP, DARE terminal cost)
- Attitude controller (geometric SO(3) control)
- DMPC analytics (performance monitoring)

### Stage 5: Agent and Evaluation

- DMPC agent (unified interface)
- Mission evaluation scripts
- Stability analysis tools
- Benchmarking and visualisation
