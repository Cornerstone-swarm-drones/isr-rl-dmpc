# Gym Environment Design

This document describes the design of the ISR-RL-DMPC Gymnasium environment, including observation and action spaces, reward structure, and simulation details.

## Table of Contents

- [Overview](#overview)
- [ISRGridEnv](#isrgridenv)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Structure](#reward-structure)
- [Episode Dynamics](#episode-dynamics)
- [Physics Simulator](#physics-simulator)
- [Vectorized Environment](#vectorized-environment)
- [Mission Scenarios](#mission-scenarios)
- [Usage Examples](#usage-examples)

## Overview

The `ISRGridEnv` class implements a Gymnasium-compatible environment for training RL agents to control multi-drone swarms in ISR missions. It wraps a 6-DOF physics simulator and provides a structured observation-action interface for RL algorithms.

**File:** `src/isr_rl_dmpc/gym_env/isr_env.py`

## ISRGridEnv

### Constructor

```python
env = ISRGridEnv(
    num_drones=10,       # Number of drones in swarm
    max_targets=5,       # Maximum number of targets
    grid_size=(20, 20),  # Mission grid dimensions (cells)
    mission_duration=500, # Episode length (steps = 10s at 50Hz)
    render_mode=None,    # 'human', 'rgb_array', or None
)
```

### Registration

The environment can also be created via the factory function:

```python
from isr_rl_dmpc.gym_env import make_env

# Single environment
env = make_env('ISRGridEnv-v0', num_drones=10, max_targets=5)

# Vectorized environment (parallel training)
vec_env = make_env('ISRGridEnv-v0', num_envs=4, num_drones=10)
```

## Observation Space

The observation is a `Dict` space with four components:

### `swarm` — Drone States

```
Shape: (num_drones, 18)
Type:  float32
Range: (-inf, inf)
```

Each drone's 18-dimensional state vector:

| Index | Field | Dimensions | Units | Description |
|---|---|---|---|---|
| 0–2 | Position | 3 | m | [x, y, z] world coordinates |
| 3–5 | Velocity | 3 | m/s | [vx, vy, vz] linear velocity |
| 6–8 | Acceleration | 3 | m/s² | [ax, ay, az] linear acceleration |
| 9–12 | Quaternion | 4 | — | [qw, qx, qy, qz] unit quaternion |
| 13–15 | Angular Velocity | 3 | rad/s | [ωx, ωy, ωz] body rates |
| 16 | Battery Energy | 1 | Wh | Available energy |
| 17 | Health | 1 | — | Structural health [0, 1] |

### `targets` — Target States

```
Shape: (max_targets, 12)
Type:  float32
Range: (-inf, inf)
```

Each target's 12-dimensional state vector:

| Index | Field | Dimensions | Units | Description |
|---|---|---|---|---|
| 0–2 | Position | 3 | m | [x, y, z] world coordinates |
| 3–5 | Velocity | 3 | m/s | [vx, vy, vz] linear velocity |
| 6–8 | Acceleration | 3 | m/s² | [ax, ay, az] linear acceleration |
| 9 | Yaw Angle | 1 | rad | Heading [0, 2π) |
| 10 | Yaw Rate | 1 | rad/s | Heading rate of change |
| 11 | (Reserved) | 1 | — | Padding for alignment |

### `environment` — Grid Coverage and Mission State

```
Shape: (grid_cells + 4,)
Type:  float32
Range: [0, 1]
```

| Index | Field | Description |
|---|---|---|
| 0 to K-1 | Coverage Map | Binary (0/1) coverage status per grid cell |
| K | Mission Progress | `step / mission_duration` |
| K+1 | Coverage Ratio | Mean of coverage map |
| K+2 | Threat Level | Aggregate threat (placeholder) |
| K+3 | Energy Efficiency | Aggregate efficiency (placeholder) |

Where `K = grid_size[0] × grid_size[1]` (default: 400 cells for 20×20 grid).

### `adjacency` — Formation Adjacency Matrix

```
Shape: (num_drones, num_drones)
Type:  float32
Range: [0, 1]
```

Weighted adjacency matrix where `adjacency[i, j] = 1/distance(i, j)` if drones i and j are within 500 m, else 0. Diagonal entries are 0.

## Action Space

```
Shape: (num_drones, 4)
Type:  float32
Range: [0, 1]
```

Motor PWM commands for each drone's 4 rotors. Values are clipped to [0, 1]:

| Index | Description |
|---|---|
| 0 | Motor 1 PWM (front-left) |
| 1 | Motor 2 PWM (front-right) |
| 2 | Motor 3 PWM (rear-left) |
| 3 | Motor 4 PWM (rear-right) |

PWM values of 0.5 correspond approximately to hover thrust.

## Reward Structure

The reward is a scalar value computed by the `RewardShaper` class, combining 5 weighted components:

### Components

| Component | Weight | Range | Description |
|---|---|---|---|
| Coverage (`r_cov`) | `w_coverage = 1.0` | [0, ∞) | Reward for newly covered grid cells |
| Energy (`r_eng`) | `w_energy = 0.5` | (-∞, 0] | Penalty for energy consumption |
| Safety (`r_safe`) | `w_safety = 10.0` | (-∞, 0] | Penalty for collisions and geofence violations |
| Threat (`r_threat`) | `w_threat = 2.0` | [0, ∞) | Reward for successful threat engagement |
| Learning (`r_learn`) | `w_learning = 0.1` | ℝ | TD-error gradient signal |

### Reward Formula

```
r_total = w_cov × r_cov + w_eng × r_eng + w_safe × r_safe + w_threat × r_threat + w_learn × r_learn
```

### Configurable Weights

Reward weights can also be set via `config/default_config.yaml`:

```yaml
learning:
  weight_coverage: 10.0
  weight_energy: 5.0
  weight_collision: -100.0
  weight_target_engagement: 20.0
  weight_formation: 2.0
```

See [TRAINING.md](TRAINING.md) for guidance on tuning these weights.

## Episode Dynamics

### Reset

On `env.reset()`:
1. Step counter resets to 0
2. Coverage map resets to all zeros
3. Drones are placed in a grid formation across the mission area
4. Targets are spawned at random positions with random types (friendly, hostile, neutral)
5. Simulator state is reinitialized

### Step

On `env.step(action)`:
1. Actions are clipped to [0, 1]
2. Simulator advances one physics step (6-DOF dynamics)
3. Coverage map is updated based on drone positions and sensor footprints
4. Reward is computed from all 5 components
5. Termination is checked (max steps or all drones inactive)

### Termination Conditions

| Condition | Type |
|---|---|
| `step_count >= mission_duration` | Terminated |
| All drones inactive (battery depleted or destroyed) | Terminated |

### Info Dictionary

The `info` dict returned by `step()` contains:

| Key | Type | Description |
|---|---|---|
| `step` | int | Current step number |
| `coverage` | float | Coverage ratio [0, 1] |
| `active_drones` | int | Number of active drones |
| `total_drones` | int | Total drones in swarm |
| `collisions` | int | Collision count |
| `geofence_violations` | int | Geofence violation count |
| `avg_battery` | float | Average battery energy |
| `wind` | array | Current wind vector |

## Physics Simulator

**File:** `src/isr_rl_dmpc/gym_env/simulator.py`

The `EnvironmentSimulator` provides realistic 6-DOF rigid body dynamics.

### Drone Physics

- 4-motor quadrotor with individual thrust control
- Gravity, drag, and motor dynamics
- Battery depletion model based on thrust and time
- Collision detection (drone-drone and drone-geofence)
- Dryden wind turbulence model

### Drone Specifications

Two drone types are defined in `config/drone_specs.yaml`:

| Spec | Reconnaissance | Interceptor |
|---|---|---|
| Mass | 1.2 kg | 2.5 kg |
| Max Velocity | 15 m/s | 30 m/s |
| Max Acceleration | 8 m/s² | 15 m/s² |
| Battery | 6000 mAh | 8000 mAh |
| Endurance | 35 min | 20 min |
| Rotors | 4 | 4 |

### Target Physics

- Constant velocity or constant acceleration motion models
- Target types: unknown, friendly, hostile, neutral
- Configurable velocity and heading

### Wind Model

Dryden wind turbulence with configurable intensity. Wind affects drone dynamics as external disturbances.

## Vectorized Environment

The `VectorEnv` class wraps multiple `ISRGridEnv` instances for parallel training:

```python
from isr_rl_dmpc.gym_env import make_env

vec_env = make_env('ISRGridEnv-v0', num_envs=4, num_drones=10)

# Reset all environments
observations = vec_env.reset()

# Step all environments
actions = np.random.uniform(0, 1, size=(4, 10, 4))
observations, rewards, dones, infos = vec_env.step(actions)
```

`VectorEnv` provides:
- Batch `reset()` and `step()` across all environments
- Stacked dict observations
- Array rewards and done flags
- Rendering of the first environment

## Mission Scenarios

Three pre-defined scenarios are available in `config/mission_scenarios.yaml`:

### Area Surveillance

```yaml
area_surveillance:
  area_size: [500.0, 500.0]
  num_drones: 4
  num_targets: 0
  coverage_goal: 0.95
  max_duration: 1800.0
  formation_type: "grid"
  revisit_interval: 60.0
```

### Threat Response

```yaml
threat_response:
  area_size: [300.0, 300.0]
  num_drones: 6
  num_targets: 3
  coverage_goal: 0.80
  max_duration: 600.0
  formation_type: "wedge"
  threat_level: "high"
```

### Search and Track

```yaml
search_and_track:
  area_size: [800.0, 800.0]
  num_drones: 5
  num_targets: 4
  coverage_goal: 0.90
  max_duration: 1200.0
  formation_type: "line"
  search_pattern: "expanding_square"
```

## Usage Examples

### Basic Training Loop

```python
from isr_rl_dmpc.gym_env import ISRGridEnv
from isr_rl_dmpc.agents import DMPCAgent
import numpy as np

env = ISRGridEnv(num_drones=10, max_targets=5)
obs, info = env.reset()

state_dim = DMPCAgent.flatten_obs(obs).shape[0]
action_dim = int(np.prod(env.action_space.shape))

agent = DMPCAgent(state_dim=state_dim, action_dim=action_dim)

for episode in range(100):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(obs, training=True)
        action_env = np.clip(action.reshape(env.action_space.shape), 0, 1)
        next_obs, reward, terminated, truncated, info = env.step(action_env)
        done = terminated or truncated

        agent.remember(obs, action, reward, next_obs, done)
        if agent.ready_to_train():
            agent.train_on_batch()

        obs = next_obs
        total_reward += reward

    print(f"Episode {episode}: reward={total_reward:.2f}, coverage={info['coverage']:.2%}")
```

### Evaluating a Trained Agent

```python
agent.load("data/training_logs/<timestamp>/final_model.pt")

obs, info = env.reset()
done = False

while not done:
    action = agent.act(obs, training=False, deterministic=True)
    action_env = np.clip(action.reshape(env.action_space.shape), 0, 1)
    obs, reward, terminated, truncated, info = env.step(action_env)
    done = terminated or truncated

    env.render()  # Render in human mode

print(f"Final coverage: {info['coverage']:.2%}")
```

### Rendering

```python
# Text output (human mode)
env = ISRGridEnv(render_mode='human')
env.reset()
env.step(action)
env.render()
# Output: Step:    1 | Drones: 10/10 | Coverage: 25.00% | Collisions:   0 | Battery: 4990 J

# RGB array (for video recording)
env = ISRGridEnv(render_mode='rgb_array')
env.reset()
img = env.render()  # Returns (100, 100, 3) uint8 array
```
