# Phase Guide

This document describes the project phases of the ISR-RL-DMPC system, covering the mission execution pipeline and the training lifecycle.

## Table of Contents

- [Mission Execution Phases](#mission-execution-phases)
- [Training Phases](#training-phases)
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
2. **Waypoint Generation** — Coverage waypoints are generated to ensure each grid cell is visited at least once. The planner accounts for sensor `coverage_radius` and drone count.
3. **Path Allocation** — Waypoints are distributed among drones to minimize total travel distance while meeting the `coverage_goal` (default: 0.95).

**Key Parameters:**

| Parameter | Config | Default | Effect |
|---|---|---|---|
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
2. **Sensor Fusion** — Raw readings from radar, optical, RF, and acoustic sensors are fused into unified detections with uncertainty estimates.
3. **Classification** — Detected targets are classified as friendly, hostile, or neutral using Bayesian inference.
4. **Threat Assessment** — Threat scores are computed based on classification confidence, target dynamics, and proximity.
5. **Task Allocation** — The Hungarian algorithm assigns drones to tasks (coverage waypoints, target tracking, threat engagement) to minimize total assignment cost.

**Key Parameters:**

| Parameter | Config | Default | Effect |
|---|---|---|---|
| `communication_radius` | `default_config.yaml` | `100.0 m` | Larger radius → more neighbors, better consensus |
| `min_swarm_separation` | `default_config.yaml` | `2.0 m` | Safety distance between drones |
| `formation_type` | `mission_scenarios.yaml` | per-scenario | Grid, wedge, or line |

**Output:** Formation commands, fused detections, target classifications, threat levels, task assignments.

---

### Phase 3: Control and Learning

**Objective:** Execute optimal control actions and update RL policy parameters.

**Active Modules:**
- Module 7 (DMPC Controller) — Computes optimal control inputs
- Module 8 (Attitude Controller) — Tracks desired attitudes
- Module 9 (Learning Module) — Updates cost function parameters

**Process:**

1. **DMPC Optimization** — The DMPC controller solves a convex optimization problem (via CVXPY) over the prediction horizon to compute optimal acceleration commands. Cost weights (Q, R, P matrices) are adapted in real-time by the `CostWeightNetwork`.
2. **Attitude Control** — Desired accelerations are converted to attitude references, and the geometric attitude controller computes motor torques to track them.
3. **Learning Update** — After each environment step, the transition (s, a, r, s', done) is stored in the replay buffer. When sufficient experience is available, the value and policy networks are updated via gradient descent.

**Key Parameters:**

| Parameter | Config | Default | Effect |
|---|---|---|---|
| `prediction_horizon` | `default_config.yaml` | `10` | Longer horizon → better planning, slower solve |
| `control_horizon` | `default_config.yaml` | `5` | Longer horizon → more control flexibility |
| `discount_factor` | `default_config.yaml` | `0.99` | Higher γ → agent considers longer-term rewards |
| `batch_size` | `default_config.yaml` | `32` | Larger batch → smoother gradients |

**Output:** Motor PWM commands, updated neural network weights.

---

## Training Phases

The training lifecycle consists of four phases:

### Phase A: Warmup (Random Exploration)

**Duration:** First `warmup_steps` (default: 10,000 steps)

During warmup, the agent takes random actions to populate the replay buffer with diverse experiences. No gradient updates occur.

**Purpose:** Ensure the replay buffer has sufficient variety before training begins.

### Phase B: Learning (Policy Optimization)

**Duration:** `num_epochs` × `steps_per_epoch` (default: 500 × 4,000 = 2,000,000 steps)

The core training phase where the agent alternates between:
1. **Data collection** — Executing the current policy in the environment
2. **Gradient updates** — Updating value and policy networks using sampled mini-batches

**Per Episode:**
```
for step in range(max_steps):
    action = agent.act(obs, training=True)
    next_obs, reward, done, info = env.step(action)
    agent.remember(obs, action, reward, next_obs, done)
    if agent.ready_to_train():
        critic_loss, actor_loss = agent.train_on_batch()
```

### Phase C: Evaluation

**Frequency:** Every `eval_interval` epochs (default: every 10 epochs)

The learned policy is evaluated on test episodes using deterministic actions (no exploration noise). Key metrics:

| Metric | Target | Description |
|---|---|---|
| Mean Coverage | ≥ 0.90 | Fraction of grid cells visited |
| Mean Reward | Increasing | Episode total reward |
| Collision Rate | → 0 | Collisions per episode |
| Episode Length | Consistent | Steps per episode |

### Phase D: Checkpointing

**Frequency:** Every `save_interval` epochs (default: every 50 epochs)

Model checkpoints are saved to `data/training_logs/<timestamp>/`:

```
checkpoint_ep50.pt
checkpoint_ep100.pt
...
final_model.pt
training_stats.json
```

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

### Stage 4: Control and Learning

- Formation controller (consensus-based)
- Task allocator (Hungarian algorithm)
- DMPC controller (CVXPY + PyTorch)
- Attitude controller (geometric control)
- Learning module (value + policy networks)

### Stage 5: Agent and Training

- DMPC agent (unified interface)
- Training script
- Hyperparameter search
- Benchmarking and visualization
