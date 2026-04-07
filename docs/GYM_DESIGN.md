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

$$
\boldsymbol{obs}^{(i)} \in \mathbb{R}^{40}
$$

| Indices | Component | Dim | Description |
| :--- | :--- | :--- | :--- |
| 0–10 | Own DMPC state | 11 | $[\boldsymbol{p}(3), \boldsymbol{v}(3), \boldsymbol{a}(3), \psi, \dot{\psi}]$ |
| 11–13 | Reference position | 3 | Current waypoint $p_\text{ref}$ |
| 14–16 | Reference velocity | 3 | $v_\text{ref}$ |
| 17–19 | Tracking error | 3 | $e_p = p - p_\text{ref}$ |
| 20–25 | Nearest neighbour relative state | 6 | $[\Delta p(3), \Delta v(3)]$ to closest drone |
| 26–28 | Mean swarm offset | 3 | $\bar{p}_\text{neighbours} - p^{(i)}$ |
| 29 | Battery level | 1 | Normalised [0, 1] |
| 30 | Health | 1 | Structural health [0, 1] |
| 31–33 | Last applied control | 3 | Previous $u^{(i)}$ |
| 34–36 | ADMM primal residual | 3 | $\lVert z_i - v\rVert$ per axis |
| 37 | DMPC solve time | 1 | Normalised last QP solve time |
| 38 | Collision margin | 1 | $\min_j\lVert p^{(i)}-p^{(j)}\rVert - r_\min$ (norm.) |
| 39 | Mission progress | 1 | $t / T_\max$ |

**Gymnasium space:**

```python
observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(40,), dtype=np.float32
)
```

## Action Space (14-D per agent)

Each drone's action is a **14-dimensional vector of multiplicative cost scale factors**:

$$
\boldsymbol{a}^{(i)} = [\boldsymbol{q}_s^\top,\; \boldsymbol{r}_s^\top]^\top \in [0.1,\; 10.0]^{14}
$$

These are applied to the DMPC base cost matrices:

$$
Q_\text{eff}^{(i)} = Q \odot \mathrm{diag}(\boldsymbol{q}_s^{(i)}), \qquad
R_\text{eff}^{(i)} = R \odot \mathrm{diag}(\boldsymbol{r}_s^{(i)})
$$

**Gymnasium space:**

```python
action_space = spaces.Box(
    low=0.1, high=10.0,
    shape=(14,), dtype=np.float32
)
```

## Reward Structure

Each agent receives a scalar reward per step:

$$
r^{(i)} = w_\text{track}\,r_\text{track} + w_\text{form}\,r_\text{form} + w_\text{safe}\,r_\text{safe} + w_\text{eff}\,r_\text{eff}
$$

| Component | Formula | Weight |
| :--- | :--- | :--- |
| Tracking | $\exp(-0.1\lVert\boldsymbol{e}_p\rVert^2) - 1$ | 5.0 |
| Formation | $-\text{mean}(\lVert\Delta\boldsymbol{p}_{ij} - \boldsymbol{d}_{ij}\rVert)$ over neighbours | 2.0 |
| Safety | $\sum_j\min(0, \lVert\boldsymbol{p}_i-\boldsymbol{p}_j\rVert - r_{\min})$ | 10.0 |
| Efficiency | $-\lVert\boldsymbol{u}^{(i)}\rVert^2$ | 0.1 |

The centralised critic during training uses the sum of all agents' rewards to
compute a global value estimate.

## Episode Dynamics

1. **Reset:** Drones initialised in a random formation within the mission area.
   MAPPO action initialised to all-ones (identity scaling of Q and R).
1. **Step:** MAPPO outputs `(q_scale, r_scale)` per drone → ADMM consensus runs
   for `admm_iters` iterations → each drone's DMPC solves its QP → physics
   simulator advances by `dt = 0.02 s`.
1. **Termination:** Episode ends after `mission_duration` steps, or if any
   drone collision occurs ($\|\boldsymbol{p}_i - \boldsymbol{p}_j\| < 0.5\,r_{\min}$).

## Physics Simulator

The environment wraps the same 6-DOF `EnvironmentSimulator` used by the ROS2
node, ensuring sim-to-real consistency:

- **Drone:** hector\_quadrotor airframe (mass 1.477 kg, $J = \mathrm{diag}(0.01152, 0.01152, 0.02180)\;\text{kg·m}^2$)
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
