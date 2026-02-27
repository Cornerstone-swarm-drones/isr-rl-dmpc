# API Reference

Complete function and class documentation for the ISR-RL-DMPC package.

## Table of Contents

- [Configuration](#configuration)
- [Core Data Structures](#core-data-structures)
- [Agents](#agents)
- [Gym Environment](#gym-environment)
- [Modules](#modules)
- [Utilities](#utilities)
- [Scripts](#scripts)

---

## Configuration

**Module:** `isr_rl_dmpc.config`

### `load_config(config_path, overrides)`

Load configuration from YAML with optional overrides.

```python
from isr_rl_dmpc.config import load_config

config = load_config("config/default_config.yaml")
config = load_config(overrides={"learning": {"batch_size": 64}})
```

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str \| None` | `"config/default_config.yaml"` | Path to YAML config file |
| `overrides` | `dict \| None` | `None` | Parameter overrides |

**Returns:** `Config` — Loaded and validated configuration.

---

### `Config`

Master configuration dataclass.

```python
config = Config()
config.validate()
config.to_yaml("output.yaml")
config = Config.from_yaml("config/default_config.yaml")
config.apply_overrides({"learning": {"batch_size": 64}})
```

**Attributes:**

| Name | Type | Description |
|---|---|---|
| `drone` | `DroneConfig` | Drone physical parameters |
| `sensor` | `SensorConfig` | Sensor and frequency parameters |
| `mission` | `MissionConfig` | Mission and coverage parameters |
| `learning` | `LearningConfig` | RL hyperparameters and reward weights |
| `dmpc` | `DMPCConfig` | DMPC solver parameters |

**Methods:**

| Method | Description |
|---|---|
| `validate()` | Validate all configuration sections |
| `to_dict()` | Convert to nested dictionary |
| `from_dict(data)` | Class method: load from dictionary |
| `to_yaml(filepath)` | Save to YAML file |
| `from_yaml(filepath)` | Class method: load from YAML file |
| `apply_overrides(overrides)` | Apply parameter overrides |

---

### `DroneConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `mass` | `float` | `1.0` | Mass (kg) |
| `inertia` | `List[float]` | `[0.05, 0.05, 0.1]` | Inertia (kg·m²) |
| `max_acceleration` | `float` | `10.0` | Max acceleration (m/s²) |
| `max_velocity` | `float` | `20.0` | Max velocity (m/s) |
| `max_angular_velocity` | `float` | `2.0` | Max angular velocity (rad/s) |
| `max_yaw_rate` | `float` | `2.0` | Max yaw rate (rad/s) |
| `battery_capacity` | `float` | `5000.0` | Battery capacity (Wh) |

### `SensorConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `control_frequency` | `float` | `50.0` | Control loop frequency (Hz) |
| `radar_range` | `float` | `200.0` | Radar detection range (m) |
| `radar_update_rate` | `float` | `5.0` | Radar update rate (Hz) |
| `optical_fov` | `float` | `120.0` | Optical field of view (degrees) |
| `optical_update_rate` | `float` | `30.0` | Optical update rate (Hz) |
| `rf_range` | `float` | `100.0` | RF detection range (m) |
| `rf_update_rate` | `float` | `10.0` | RF update rate (Hz) |
| `max_radar_targets` | `int` | `50` | Max radar targets |
| `max_optical_targets` | `int` | `20` | Max optical targets |

### `MissionConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `grid_cell_size` | `float` | `10.0` | Grid cell size (m) |
| `coverage_radius` | `float` | `5.0` | Sensor coverage radius (m) |
| `communication_radius` | `float` | `100.0` | Communication range (m) |
| `coverage_goal` | `float` | `0.95` | Target coverage ratio [0, 1] |
| `min_swarm_separation` | `float` | `2.0` | Min drone separation (m) |
| `max_swarm_spread` | `float` | `200.0` | Max swarm spread (m) |

### `LearningConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `discount_factor` | `float` | `0.99` | Discount factor γ |
| `learning_rate_critic` | `float` | `1e-3` | Critic learning rate |
| `learning_rate_actor` | `float` | `1e-4` | Actor learning rate |
| `batch_size` | `int` | `32` | Mini-batch size |
| `buffer_size` | `int` | `1e5` | Replay buffer capacity |
| `target_update_frequency` | `int` | `1000` | Target network update frequency |
| `weight_coverage` | `float` | `10.0` | Coverage reward weight |
| `weight_energy` | `float` | `5.0` | Energy penalty weight |
| `weight_collision` | `float` | `-100.0` | Collision penalty weight |
| `weight_target_engagement` | `float` | `20.0` | Target engagement reward weight |
| `weight_formation` | `float` | `2.0` | Formation reward weight |

### `DMPCConfig`

| Attribute | Type | Default | Description |
|---|---|---|---|
| `prediction_horizon` | `int` | `10` | MPC prediction horizon (steps) |
| `control_horizon` | `int` | `5` | MPC control horizon (steps) |
| `receding_horizon_step` | `int` | `1` | Receding horizon step size |
| `constraint_tightening` | `float` | `0.95` | Constraint relaxation factor |
| `solver_max_iterations` | `int` | `100` | Max solver iterations |
| `solver_tolerance` | `float` | `1e-4` | Solver convergence tolerance |

---

## Core Data Structures

**Module:** `isr_rl_dmpc.core`

### `DroneState`

State representation of a single drone in the swarm.

```python
from isr_rl_dmpc.core import DroneState

state = DroneState(
    position=np.array([100.0, 200.0, 50.0]),
    velocity=np.zeros(3),
    battery_energy=5000.0,
    health=1.0,
)

vector = state.to_vector()          # (18,) numpy array
vector_norm = state.to_vector(normalize=True)  # normalized to [-1, 1]
state_dict = state.to_dict()        # dictionary
state_copy = state.copy()           # deep copy
```

**Attributes:**

| Name | Type | Shape | Description |
|---|---|---|---|
| `position` | `np.ndarray` | `(3,)` | 3D position [x, y, z] (m) |
| `velocity` | `np.ndarray` | `(3,)` | 3D velocity (m/s) |
| `acceleration` | `np.ndarray` | `(3,)` | 3D acceleration (m/s²) |
| `quaternion` | `np.ndarray` | `(4,)` | Unit quaternion [qw, qx, qy, qz] |
| `angular_velocity` | `np.ndarray` | `(3,)` | Angular velocity (rad/s) |
| `battery_energy` | `float` | — | Available energy (Wh) |
| `health` | `float` | — | Structural health [0, 1] |
| `last_update` | `float` | — | Last update timestamp (s) |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `to_vector(normalize=False)` | `np.ndarray (18,)` | Flatten state to 1D array |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(data)` | `DroneState` | Reconstruct from dictionary |
| `copy()` | `DroneState` | Deep copy |

---

### `TargetState`

State representation of a detected target.

```python
from isr_rl_dmpc.core import TargetState

target = TargetState(
    position=np.array([300.0, 400.0, 100.0]),
    velocity=np.array([5.0, 0.0, 0.0]),
    classification_confidence=-0.8,  # hostile
    target_id="hostile",
    threat_score=0.9,
)

vector = target.to_vector()  # (12,) numpy array
```

**Attributes:**

| Name | Type | Shape | Description |
|---|---|---|---|
| `position` | `np.ndarray` | `(3,)` | 3D position (m) |
| `velocity` | `np.ndarray` | `(3,)` | 3D velocity (m/s) |
| `acceleration` | `np.ndarray` | `(3,)` | 3D acceleration (m/s²) |
| `yaw_angle` | `float` | — | Heading angle (rad) [0, 2π) |
| `yaw_rate` | `float` | — | Heading rate (rad/s) |
| `classification_confidence` | `float` | — | Classification [-1, 1] |
| `target_id` | `str` | — | Target identifier |
| `threat_score` | `float` | — | Threat level |
| `covariance` | `np.ndarray` | `(11, 11)` | EKF covariance matrix |
| `last_update` | `float` | — | Last update timestamp (s) |
| `tracked_duration` | `float` | — | Total tracking time (s) |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `to_vector()` | `np.ndarray (12,)` | Flatten state (excl. covariance) |
| `from_vector(state)` | `TargetState` | Reconstruct from 11D EKF vector |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(data)` | `TargetState` | Reconstruct from dictionary |
| `copy()` | `TargetState` | Deep copy |

---

### `MissionState`

Global mission state including area coverage and objectives.

```python
from isr_rl_dmpc.core import MissionState

mission = MissionState(
    area_boundary=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]),
    coverage_matrix=np.zeros(2500, dtype=bool),
    mission_duration=1800.0,
)

coverage_pct = mission.get_coverage_percentage()  # 0.0 to 100.0
progress = mission.get_mission_progress()          # 0.0 to 1.0
mission.mark_cell_covered(42)
mission.add_waypoint(np.array([100.0, 200.0]))
```

**Attributes:**

| Name | Type | Description |
|---|---|---|
| `area_boundary` | `np.ndarray (N, 2)` | Polygon vertices of search area |
| `waypoints` | `List[np.ndarray]` | Navigation waypoints |
| `coverage_matrix` | `np.ndarray (bool)` | Grid cell coverage status |
| `elapsed_time` | `float` | Time since mission start (s) |
| `mission_duration` | `float` | Maximum mission time (s) |
| `coverage_efficiency` | `float` | Coverage effectiveness [0, 1] |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `get_coverage_percentage()` | `float` | Coverage percentage (0–100) |
| `get_mission_progress()` | `float` | Time progress ratio (0–1) |
| `add_waypoint(waypoint)` | `None` | Add a navigation waypoint |
| `mark_cell_covered(cell_idx)` | `None` | Mark a grid cell as covered |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(data)` | `MissionState` | Reconstruct from dictionary |
| `copy()` | `MissionState` | Deep copy |

---

## Agents

**Module:** `isr_rl_dmpc.agents`

### `DMPCAgent`

RL agent for DMPC parameter optimization with a unified act/remember/train interface.

```python
from isr_rl_dmpc.agents import DMPCAgent

agent = DMPCAgent(
    state_dim=818,        # Flattened observation dimension
    action_dim=40,        # num_drones × 4 motors
    learning_rate_critic=1e-3,
    learning_rate_actor=1e-4,
    gamma=0.99,
    batch_size=32,
    buffer_size=100000,
    exploration_noise=0.1,
    device="cpu",
)
```

**Methods:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `act` | `(observation, training=True, deterministic=False)` | `np.ndarray` | Select action from policy |
| `remember` | `(observation, action, reward, next_observation, done)` | `None` | Store transition in replay buffer |
| `ready_to_train` | `()` | `bool` | Check if enough experience for training |
| `train_on_batch` | `()` | `(float, float)` | One gradient update; returns (critic_loss, actor_loss) |
| `save` | `(filepath)` | `None` | Save checkpoint |
| `load` | `(filepath)` | `None` | Load checkpoint |
| `flatten_obs` | `(obs)` | `np.ndarray` | Static method: flatten dict observation |

---

## Gym Environment

**Module:** `isr_rl_dmpc.gym_env`

### `ISRGridEnv`

Gymnasium-compatible environment for ISR swarm control.

```python
from isr_rl_dmpc.gym_env import ISRGridEnv

env = ISRGridEnv(num_drones=10, max_targets=5)
```

**Methods:**

| Method | Signature | Returns | Description |
|---|---|---|---|
| `reset` | `(seed=None, options=None)` | `(obs, info)` | Reset to initial state |
| `step` | `(action)` | `(obs, reward, terminated, truncated, info)` | Execute one step |
| `render` | `()` | `np.ndarray \| None` | Render environment |
| `close` | `()` | `None` | Clean up resources |
| `seed` | `(seed=None)` | `List[int]` | Set random seed |
| `set_mission_config` | `(**kwargs)` | `None` | Update mission configuration |

See [GYM_DESIGN.md](GYM_DESIGN.md) for detailed observation/action space documentation.

### `VectorEnv`

Vectorized environment for parallel training.

```python
from isr_rl_dmpc.gym_env.isr_env import VectorEnv

envs = [ISRGridEnv(num_drones=10) for _ in range(4)]
vec_env = VectorEnv(envs)
```

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `reset()` | `dict` | Reset all environments |
| `step(actions)` | `(obs, rewards, dones, infos)` | Step all environments |
| `close()` | `None` | Close all environments |
| `render(mode)` | varies | Render first environment |

### `make_env(env_id, num_envs, **kwargs)`

Factory function to create single or vectorized environments.

```python
from isr_rl_dmpc.gym_env import make_env

env = make_env('ISRGridEnv-v0', num_drones=10)
vec_env = make_env('ISRGridEnv-v0', num_envs=4, num_drones=10)
```

---

### `RewardShaper`

**Module:** `isr_rl_dmpc.gym_env.reward_shaper`

Multi-objective reward function with 5 components.

```python
from isr_rl_dmpc.gym_env.reward_shaper import RewardShaper, RewardWeights

shaper = RewardShaper(
    num_drones=10,
    max_targets=5,
    grid_cells=400,
    weights=RewardWeights(w_coverage=1.0, w_energy=0.5, w_safety=10.0),
)

reward = shaper.compute_reward(
    drone_states=drone_states,
    target_states=target_states,
    mission_state=mission_state,
    motor_commands=action,
    coverage_map=coverage_map,
    step=step,
)
```

---

## Modules

All modules are located under `isr_rl_dmpc.modules`.

### Module 7: DMPC Controller

**Module:** `isr_rl_dmpc.modules.dmpc_controller`

| Class | Description |
|---|---|
| `DMPC` | Main DMPC controller |
| `MPCSolver` | CVXPY-based solver |
| `CostWeightNetwork` | PyTorch network for adaptive cost weights |
| `DynamicsResidualNetwork` | Neural network for unmodeled dynamics |
| `DMPCConfig` | Configuration dataclass |

### Module 9: Learning Module

**Module:** `isr_rl_dmpc.modules.learning_module`

| Class | Description |
|---|---|
| `LearningModule` | Orchestrates value and policy training |
| `ValueNetwork` | Critic: V(s) approximation |
| `PolicyNetwork` | Actor: π(a\|s) with Gaussian output |
| `ExperienceBuffer` | Replay buffer with prioritized sampling |
| `Transition` | Single (s, a, r, s', done) tuple |

---

## Utilities

**Module:** `isr_rl_dmpc.utils`

### `math_utils`

Quaternion operations, matrix utilities, and coordinate transformations.

### `logging_utils`

| Function/Class | Description |
|---|---|
| `setup_logger(name, log_file)` | Create a configured logger |
| `MetricsLogger(name, log_dir)` | Metrics tracking with file output |

### `visualization`

Six visualization classes for mission analysis and training monitoring.

### `conversions`

Unit and attitude conversion utilities.

---

## Scripts

### `train_agent.py`

```bash
python scripts/train_agent.py [options]
```

See [TRAINING.md](TRAINING.md) for detailed usage.

**Key Functions:**

| Function | Description |
|---|---|
| `train(...)` | Main training loop |
| `evaluate_policy(agent, env, num_episodes)` | Evaluate learned policy |
| `parse_args()` | Parse CLI arguments |

### `hyperparameter_search.py`

```bash
python scripts/hyperparameter_search.py [options]
```

**Key Functions:**

| Function | Description |
|---|---|
| `grid_search(...)` | Random search over hyperparameter space |
| `evaluate_config(config, ...)` | Evaluate a single configuration |
| `get_search_space()` | Define search space |
| `compute_convergence_speed(rewards)` | Compute convergence metric |

### `benchmark.py`

Performance benchmarking for GPU/CPU comparison.

### `test_mission.py`

End-to-end mission validation.

### `calibrate_sensors.py`

Sensor calibration utilities.

### `visualize_results.py`

Training results visualization.
