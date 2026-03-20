# Module Specifications

Detailed specifications for each of the 9 core modules in the ISR-RL-DMPC system.

## Table of Contents

- [Module 1: Mission Planner](#module-1-mission-planner)
- [Module 2: Formation Controller](#module-2-formation-controller)
- [Module 3: Sensor Fusion](#module-3-sensor-fusion)
- [Module 4: Classification Engine](#module-4-classification-engine)
- [Module 5: Threat Assessor](#module-5-threat-assessor)
- [Module 6: Task Allocator](#module-6-task-allocator)
- [Module 7: DMPC Controller](#module-7-dmpc-controller)
- [Module 8: Attitude Controller](#module-8-attitude-controller)
- [Module 9: Learning Module](#module-9-learning-module)

---

## Module 1: Mission Planner

**File:** `src/isr_rl_dmpc/modules/mission_planner.py`

### Purpose

Decomposes the mission area into a grid, generates waypoints for coverage, and plans paths to maximize area coverage efficiency.

### Key Classes

| Class | Description |
|---|---|
| `GridCell` | Represents a single cell in the mission grid with coverage state |
| `GridDecomposer` | Decomposes the mission area into a grid of cells |
| `WaypointGenerator` | Generates waypoint sequences for coverage path planning |
| `MissionPlanner` | Orchestrates grid decomposition and waypoint assignment |

### Parameters

| Parameter | Source | Default | Description |
|---|---|---|---|
| `grid_cell_size` | `default_config.yaml` | `10.0 m` | Size of each grid cell |
| `coverage_radius` | `default_config.yaml` | `5.0 m` | Sensor coverage radius per drone |
| `coverage_goal` | `default_config.yaml` | `0.95` | Target coverage ratio (0–1) |

### Interface

```python
planner = MissionPlanner(area_boundary, grid_cell_size=10.0)
waypoints = planner.generate_coverage_waypoints(num_drones=4)
coverage = planner.get_coverage_percentage()
```

---

## Module 2: Formation Controller

**File:** `src/isr_rl_dmpc/modules/formation_controller.py`

### Purpose

Maintains swarm formation geometry using consensus-based distributed control. Supports grid, wedge, and line formations.

### Key Classes

| Class | Description |
|---|---|
| `FormationController` | High-level formation management |
| `ConsensusController` | Distributed consensus algorithm for position agreement |
| `FormationGeometry` | Defines formation shapes (grid, wedge, line) |

### Supported Formations

| Formation | Use Case | Drones |
|---|---|---|
| Grid | Area surveillance | 4+ |
| Wedge | Threat response | 3+ |
| Line | Search sweep | 2+ |

### Parameters

| Parameter | Source | Default | Description |
|---|---|---|---|
| `communication_radius` | `default_config.yaml` | `100.0 m` | Maximum inter-drone communication range |
| `min_swarm_separation` | `default_config.yaml` | `2.0 m` | Minimum separation distance |
| `max_swarm_spread` | `default_config.yaml` | `200.0 m` | Maximum spread distance |

---

## Module 3: Sensor Fusion

**File:** `src/isr_rl_dmpc/modules/sensor_fusion.py`

### Purpose

Fuses data from four sensor types (radar, optical, RF, acoustic) into unified target detections with uncertainty estimates.

### Key Classes

| Class | Description |
|---|---|
| `SensorFusionManager` | Coordinates fusion across all sensor modalities |

### Sensor Specifications

| Sensor | Range | Update Rate | Noise σ | FOV |
|---|---|---|---|---|
| Radar | 200 m | 5 Hz | 2.0 m | 360° |
| Optical | 150 m | 30 Hz | 0.5 m | 120° |
| RF | 100 m | 10 Hz | 5.0 m | 360° |
| Acoustic | 50 m | 20 Hz | 3.0 m | 360° |

Full sensor specifications are defined in `config/sensor_specs.yaml`.

---

## Module 4: Classification Engine

**File:** `src/isr_rl_dmpc/modules/classification_engine.py`

### Purpose

Classifies detected targets as friendly, hostile, or neutral using Bayesian inference and feature extraction.

### Key Classes

| Class | Description |
|---|---|
| `ClassificationEngine` | Main classification pipeline |
| `BayesianClassifier` | Bayesian posterior updating for classification |
| `FeatureExtractor` | Extracts features from sensor data for classification |

### Target Types

| Type | Confidence Range | Description |
|---|---|---|
| Hostile | `[-1.0, -0.3]` | Threat requiring engagement |
| Neutral | `(-0.3, 0.3)` | Unknown or non-threat entity |
| Friendly | `[0.3, 1.0]` | Identified friendly asset |

---

## Module 5: Threat Assessor

**File:** `src/isr_rl_dmpc/modules/threat_assessor.py`

### Purpose

Evaluates real-time threat levels for detected targets based on classification, dynamics, and proximity to the swarm.

### Key Classes

| Class | Description |
|---|---|
| `ThreatAssessor` | Computes threat scores from target states and classifications |

### Threat Level Computation

Threat scores combine multiple factors:
- Classification confidence (hostile → higher threat)
- Target velocity and heading toward swarm
- Proximity to swarm drones
- Historical tracking data

---

## Module 6: Task Allocator

**File:** `src/isr_rl_dmpc/modules/task_allocator.py`

### Purpose

Assigns drones to tasks (waypoints, targets, coverage zones) using the Hungarian algorithm for optimal matching.

### Key Classes

| Class | Description |
|---|---|
| `TaskAllocator` | High-level task assignment interface |
| `HungarianAssignment` | Optimal assignment using the Hungarian algorithm |

### Algorithm

The Hungarian algorithm solves the assignment problem by minimizing total cost:

```
minimize  Σ_i Σ_j C[i,j] × X[i,j]
subject to  each drone assigned exactly one task
            each task assigned at most one drone
```

Where `C[i,j]` is the cost of assigning drone `i` to task `j` (typically Euclidean distance or energy cost).

---

## Module 7: DMPC Controller

**File:** `src/isr_rl_dmpc/modules/dmpc_controller.py`

### Purpose

Implements Distributed Model Predictive Control with CVXPY-based convex optimization and PyTorch-learned cost weight adaptation.

### Key Classes

| Class | Description |
|---|---|
| `DMPC` | Main DMPC controller integrating solver and learning |
| `MPCSolver` | CVXPY-based convex optimization solver |
| `CostWeightNetwork` | PyTorch network that learns adaptive Q, R, P cost matrix scales |
| `DynamicsResidualNetwork` | Neural network for learning unmodeled dynamics |

### Configuration

| Parameter | Source | Default | Description |
|---|---|---|---|
| `prediction_horizon` | `default_config.yaml` | `10` | MPC prediction horizon (steps) |
| `control_horizon` | `default_config.yaml` | `5` | MPC control horizon (steps) |
| `constraint_tightening` | `default_config.yaml` | `0.95` | Constraint relaxation factor |
| `solver_max_iterations` | `default_config.yaml` | `100` | Maximum solver iterations |
| `solver_tolerance` | `default_config.yaml` | `0.0001` | Solver convergence tolerance |

### Hybrid Architecture

```
State → CostWeightNetwork → (Q_scale, R_scale, P_scale)
                                     │
                                     ▼
           Q = Q_scale × Q_base,  R = R_scale × R_base,  P = P_scale × P_base
                                     │
                                     ▼
                            MPCSolver (CVXPY)
                                     │
                                     ▼
                          Optimal control input u*
```

The CostWeightNetwork learns to adapt the DMPC cost matrices based on the current state, while the CVXPY solver guarantees constraint satisfaction.

---

## Module 8: Attitude Controller

**File:** `src/isr_rl_dmpc/modules/attitude_controller.py`

### Purpose

Implements geometric attitude control for individual drones with gain adaptation for agile maneuvers.

### Key Classes

| Class | Description |
|---|---|
| `AttitudeController` | High-level attitude control interface |
| `GeometricController` | SE(3) geometric controller for attitude tracking |

### Control Law

The geometric controller computes torques to track desired attitudes on SO(3):

```
τ = -k_R × e_R - k_ω × e_ω + ω × (J × ω)
```

Where:
- `e_R` — Rotation error on SO(3)
- `e_ω` — Angular velocity error
- `k_R, k_ω` — Proportional and derivative gains
- `J` — Inertia matrix

---

## Module 9: Learning Module

**File:** `src/isr_rl_dmpc/modules/learning_module.py`

### Purpose

Implements the RL learning pipeline with value and policy networks for optimizing DMPC cost function parameters.

### Key Classes

| Class | Description |
|---|---|
| `LearningModule` | Orchestrates value and policy network training |
| `ValueNetwork` | Critic network: V(s) ≈ E[Σ γ^k r_k \| s] |
| `PolicyNetwork` | Actor network: π(a\|s) with Gaussian output |
| `ExperienceBuffer` | Replay buffer with priority-based sampling |

### Mathematical Framework

```
State:   s ∈ ℝ^d    (augmented system state)
Action:  a ∈ ℝ^m    (DMPC parameters θ = [Q, R, P, slack])
Reward:  r ∈ ℝ       (multi-objective scalar)

Value function:    V(s) ≈ E[Σ_k γ^k r_k | s]
TD error:          δ_k = r_k + γ V(s_{k+1}) - V(s_k)
Policy gradient:   ∇J(θ) = E[∇_θ log π(a|s) × δ_k]
```

### Network Architecture

**Value Network:**
```
Input(state_dim) → Linear(256) → ReLU → LayerNorm →
Linear(256) → ReLU → LayerNorm →
Linear(128) → ReLU → Linear(1)
```

**Policy Network:**
```
Input(state_dim) → Linear(256) → ReLU →
Linear(256) → ReLU →
Linear(128) → ReLU →
├── Linear(action_dim) → Tanh   [mean]
└── Linear(action_dim)          [log_std]
```
