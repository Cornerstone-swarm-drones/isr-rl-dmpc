# MARL Gymnasium Environment Design

This document describes the design of the **MARLDMPCEnv** Gymnasium environment
used to train and evaluate the MAPPO policy that adaptively tunes DMPC cost
parameters for multi-drone ISR missions.

## Table of Contents

- [Overview](#overview)
- [MARLDMPCEnv](#marldmpcenv)
- [Observation Space (40-D per agent)](#observation-space-40-d-per-agent)
- [Action Space (14-D per agent)](#action-space-14-d-per-agent)
- [Reward Structure](#reward-structure)
- [Episode Dynamics](#episode-dynamics)
- [Physics Simulator](#physics-simulator)
- [Training with MAPPO](#training-with-mappo)
- [Usage Examples](#usage-examples)

## Overview

`MARLDMPCEnv` is a Gymnasium-compatible **multi-agent environment** for training
MAPPO to dynamically tune the cost parameters of a DMPC swarm controller.  Each
agent (drone) receives a 40-dimensional local observation and outputs a 14-dimensional
action (Q and R scale vectors).  The DMPC then solves the resulting QP at 50 Hz,
and ADMM enforces consensus across drones.

**File:** `src/isr_rl_dmpc/gym_env/marl_env.py`

The environment follows the **Centralised Training / Decentralised Execution (CTDE)**
paradigm: during training a centralised critic observes the joint state of all
drones; during execution each drone's actor conditions only on its local observation.

## MARLDMPCEnv

### Constructor

```python
from isr_rl_dmpc.gym_env import MARLDMPCEnv

env = MARLDMPCEnv(
    num_drones=4,            # Number of drones (agents)
    max_targets=3,           # Maximum number of tracked targets
    mission_duration=1000,   # Episode length (steps at 50 Hz)
    admm_iters=5,            # ADMM iterations per control step
    render_mode=None,        # 'human', 'rgb_array', or None
)
```

### Registration

```python
from isr_rl_dmpc.gym_env import make_env

env = make_env('MARLDMPCEnv-v0', num_drones=4)
```

## Observation Space (40-D per agent)

Each drone receives its own **40-dimensional local observation** vector:

```
obs^(i) ∈ R^40
```

| Indices | Component | Dim | Description |
| :--- | :--- | :--- | :--- |
| 0–10 | Own DMPC state | 11 | [p(3), v(3), a(3), ψ, ψ̇] |
| 11–13 | Reference position | 3 | Current waypoint p_ref |
| 14–16 | Reference velocity | 3 | v_ref |
| 17–19 | Tracking error | 3 | e_p = p − p_ref |
| 20–25 | Nearest neighbour relative state | 6 | [Δp(3), Δv(3)] to closest drone |
| 26–28 | Mean swarm offset | 3 | p̄_neighbours − p^(i) |
| 29 | Battery level | 1 | Normalised [0, 1] |
| 30 | Health | 1 | Structural health [0, 1] |
| 31–33 | Last applied control | 3 | Previous u^(i) |
| 34–36 | ADMM primal residual | 3 | ‖z_i − v‖ per axis |
| 37 | DMPC solve time | 1 | Normalised last QP solve time |
| 38 | Collision margin | 1 | min_j‖p^(i)−p^(j)‖ − r_min (norm.) |
| 39 | Mission progress | 1 | t / T_max |

**Gymnasium space:**

```python
observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(40,), dtype=np.float32
)
```

## Action Space (14-D per agent)

Each drone's action is a **14-dimensional vector of multiplicative cost scale factors**:

```
a^(i) = [q_s(0..10), r_s(0..2)]  ∈ [0.1, 10.0]^14
```

These are applied to the DMPC base cost matrices:

```
Q_eff^(i) = Q ⊙ diag(q_s^(i))
R_eff^(i) = R ⊙ diag(r_s^(i))
```

**Gymnasium space:**

```python
action_space = spaces.Box(
    low=0.1, high=10.0,
    shape=(14,), dtype=np.float32
)
```

## Reward Structure

Each agent receives a scalar reward per step:

```
r^(i) = w_track * r_track + w_form * r_form + w_safe * r_safe + w_eff * r_eff
```

| Component | Formula | Weight |
| :--- | :--- | :--- |
| Tracking | `exp(-0.1 * ‖e_p‖²) − 1` | 5.0 |
| Formation | `−mean(‖Δp_ij − d_ij‖)` over neighbours | 2.0 |
| Safety | `sum(min(0, ‖p_i−p_j‖ − r_min))` | 10.0 |
| Efficiency | `−‖u^(i)‖²` | 0.1 |

The centralised critic during training uses the sum of all agents' rewards to
compute a global value estimate.

## Episode Dynamics

1. **Reset:** Drones initialised in a random formation within the mission area.
   MAPPO action initialised to all-ones (identity scaling of Q and R).
2. **Step:** MAPPO outputs `(q_scale, r_scale)` per drone → ADMM consensus runs
   for `admm_iters` iterations → each drone's DMPC solves its QP → physics
   simulator advances by `dt = 0.02 s`.
3. **Termination:** Episode ends after `mission_duration` steps, or if any
   drone collision occurs (`‖p_i − p_j‖ < 0.5 * r_min`).

## Physics Simulator

The environment wraps the same 6-DOF `EnvironmentSimulator` used by the ROS2
node, ensuring sim-to-real consistency:

- **Drone:** hector\_quadrotor airframe (mass 1.477 kg, J = diag(0.01152, 0.01152, 0.02180) kg·m²)
- **Aerodynamics:** Linear drag (c_d = 0.22), hover thrust compensation
- **Wind:** Dryden turbulence model (configurable intensity)
- **Battery:** First-order discharge model
- **Collisions:** Elastic rebound with penalty reward signal

## Training with MAPPO

```python
from stable_baselines3 import PPO
from isr_rl_dmpc.gym_env import MARLDMPCEnv
from isr_rl_dmpc.agents import MAPPOAgent

env = MARLDMPCEnv(num_drones=4)

agent = MAPPOAgent(
    env=env,
    policy='MlpPolicy',
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log='logs/mappo_dmpc',
)

agent.learn(total_timesteps=1_000_000)
agent.save('models/mappo_dmpc_v1')
```

Or using the training script:

```bash
python scripts/train_mappo.py --config config/mappo_config.yaml
```

## Usage Examples

```python
# Evaluate a trained MAPPO policy
from isr_rl_dmpc.gym_env import MARLDMPCEnv
from isr_rl_dmpc.agents import MAPPOAgent

env = MARLDMPCEnv(num_drones=4, render_mode='human')
agent = MAPPOAgent.load('models/mappo_dmpc_v1', env=env)

obs, _ = env.reset()
for _ in range(1000):
    actions, _ = agent.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        obs, _ = env.reset()
```
