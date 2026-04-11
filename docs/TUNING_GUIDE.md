# ISR-RL-DMPC Tuning Guide

This guide walks through the complete workflow for tuning the **pure DMPC**
controller, training the **RL-DMPC (MAPPO)** adaptive policy, and running
the final models in the **PyBullet simulation**.

> **Prerequisites** — Install the project:
> ```bash
> conda env create -f environment.yml && conda activate isr-rl-dmpc
> pip install -e .
> ```

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1 — Tune Pure DMPC](#2-phase-1--tune-pure-dmpc)
3. [Phase 2 — Train RL-DMPC (MAPPO)](#3-phase-2--train-rl-dmpc-mappo)
4. [Phase 3 — Run in PyBullet](#4-phase-3--run-in-pybullet)
5. [Parameter Reference](#5-parameter-reference)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Architecture Overview

The system has two operating modes:

| Mode | Controller | Q / R Matrices | When to Use |
|------|-----------|---------------|-------------|
| **Pure DMPC** | CVXPY / OSQP convex QP | Fixed (from config) | Baseline, hardware, no GPU |
| **RL-DMPC** | MAPPO policy → Q/R scales → DMPC | Adapted per step | Adaptive missions, training available |

The DMPC solves at each time step:

```
min  Σ_{k=0}^{N-1} [‖e_k‖²_Q + ‖u_k‖²_R] + ‖e_N‖²_P
s.t. x_{k+1} = A x_k + B u_k
     |u_{k,l}| ≤ u_max          (per-axis box saturation)
     CBF(p_k, p_j) ≥ 0          (collision avoidance)
```

In RL-DMPC mode, the MAPPO policy outputs per-drone `q_scale` (11-D) and
`r_scale` (3-D) vectors that multiplicatively adjust Q and R:

```
Q_eff = Q ⊙ diag(q_scale)     R_eff = R ⊙ diag(r_scale)
```

---

## 2. Phase 1 — Tune Pure DMPC

### 2.1 Configuration File

All DMPC parameters live in **`config/dmpc_config.yaml`**:

```yaml
dmpc:
  prediction_horizon: 20       # N — number of look-ahead steps
  dt: 0.02                     # Discretisation step [s] (50 Hz)
  accel_max: 8.0               # Max acceleration [m/s²]
  collision_radius: 3.0        # Min inter-drone separation [m]
  solver_timeout: 0.015        # OSQP time limit per solve [s]

cost:
  Q_diag: 1.0                  # Scalar × identity for Q
  R_diag: 0.15                 # Scalar × identity for R
```

### 2.2 Tuning Q (State Tracking Cost)

Q penalises tracking error `e_k = x_k − x_ref_k` across the 11-D state:

| Index | State    | Description         | Tuning Effect |
|-------|----------|---------------------|---------------|
| 0–2   | p (3)    | Position            | ↑ = tighter position tracking |
| 3–5   | v (3)    | Velocity            | ↑ = faster velocity convergence |
| 6–8   | a (3)    | Acceleration        | ↑ = smoother accel profile |
| 9     | ψ (yaw)  | Yaw angle           | ↑ = tighter heading hold |
| 10    | ψ̇ (yaw rate) | Yaw rate      | ↑ = less yaw overshoot |

**Recommended starting point:**

```python
Q = np.diag([
    5.0, 5.0, 5.0,     # Position (dominant)
    1.0, 1.0, 1.0,     # Velocity
    0.1, 0.1, 0.1,     # Acceleration
    0.5,                # Yaw
    0.1,                # Yaw rate
])
```

**Guidelines:**
- Position weights should be the largest (primary tracking objective)
- Keep velocity weights moderate to avoid overshoot
- Small acceleration weights give smoother control but slower response
- Increase yaw weights if heading accuracy matters for your ISR sensors

### 2.3 Tuning R (Control Effort Cost)

R penalises control magnitude `u_k = [a_x, a_y, a_z]`:

```python
R = np.diag([0.1, 0.1, 0.1])
```

**Guidelines:**
- Larger R → gentler manoeuvres, less battery drain, slower response
- Smaller R → aggressive tracking, higher energy, faster convergence
- For **area_surveillance** (long endurance): `R_diag: 0.2–0.5`
- For **threat_response** (fast intercept): `R_diag: 0.05–0.1`
- The terminal cost P is auto-computed from the DARE (no manual tuning needed)

### 2.4 Tuning the Prediction Horizon

| Horizon N | Behaviour | Solver Time |
|-----------|-----------|-------------|
| 10 | Reactive, may miss obstacles | ~1–2 ms |
| 20 (default) | Good balance | ~2–5 ms |
| 30 | Far-sighted, smoother | ~5–15 ms |
| 40+ | Diminishing returns | >15 ms (risk of timeout) |

The solver timeout (`solver_timeout: 0.015`) must exceed the typical solve
time.  Monitor `mean_solve_ms` in the PyBullet sim output.

### 2.5 Collision Radius

`collision_radius` sets the CBF safety distance between drones:

- **3.0 m** (default) — ~5× the hector_quadrotor rotor span (0.55 m)
- Decrease to **2.0 m** for tight formations (only if GPS accuracy < 0.5 m)
- Increase to **5.0 m** for windy conditions or GPS-denied environments

### 2.6 Running Pure DMPC

```bash
# Quick test — 500 steps, area_surveillance scenario
python scripts/run_dmpc.py --scenario area_surveillance --max-steps 500

# Sweep R to find optimal control effort
for r in 0.05 0.1 0.2 0.5; do
    python scripts/run_dmpc.py --scenario area_surveillance \
        --max-steps 1000 --output data/results/dmpc_sweep
done
```

**Metrics to monitor:**
- `total_reward` — higher is better
- `mean_solve_time_ms` — should stay below `solver_timeout × 1000`
- `final_battery_mean` — should remain > 0.2 for mission completion
- `terminated_early` — `false` means no collisions or battery death

### 2.7 Tuning the Attitude Controller

Gains in `config/dmpc_config.yaml` under `attitude:`:

| Gain | Default | Effect |
|------|---------|--------|
| `Kp_position` | 2.0 | Position PD — proportional |
| `Kd_position` | 1.5 | Position PD — derivative (damping) |
| `Kp_attitude` | 4.5 | SO(3) attitude — proportional |
| `Kd_attitude` | 1.5 | SO(3) attitude — derivative |

**Tuning tips:**
- If drones oscillate around waypoints: increase `Kd_position`
- If drones are sluggish: increase `Kp_position`
- If attitude response is jittery: decrease `Kp_attitude`, increase `Kd_attitude`
- Always maintain `Kd / Kp ≈ 0.3–0.5` for stable critically-damped response

---

## 3. Phase 2 — Train RL-DMPC (MAPPO)

### 3.1 When to Use RL-DMPC

Use RL-DMPC when:
- The mission involves changing conditions (e.g., emerging threats)
- Fixed Q/R do not yield satisfactory performance across scenarios
- You have GPU access for training (CPU training is 10–50× slower)

The MAPPO policy learns to output `q_scale` (11-D) and `r_scale` (3-D)
that adapt the DMPC cost matrices in real time.

### 3.2 Pre-Training Checklist

1. **Tune pure DMPC first** (Phase 1) — the RL policy adapts *around* the
   baseline Q/R.  If baseline Q/R are poor, the policy must learn larger
   corrections, which is harder.
2. **Verify the environment** runs without errors:
   ```bash
   python -c "
   from isr_rl_dmpc.gym_env.marl_env import MARLDMPCEnv
   env = MARLDMPCEnv(num_drones=4)
   obs, _ = env.reset()
   print(f'Obs shape: {obs.shape}')  # Expected: (160,) = 4 drones × 40
   import numpy as np
   action = np.ones(env.action_space.shape)
   obs, reward, term, trunc, info = env.step(action)
   print(f'Reward: {reward:.4f}, Terminated: {term}')
   env.close()
   "
   ```
3. **Choose hardware**: GPU with ≥4 GB VRAM (RTX 3060+) recommended.

### 3.3 Configuration File

Edit **`config/mappo_config.yaml`**:

```yaml
environment:
  num_drones: 4
  max_targets: 3
  mission_duration: 1000
  horizon: 20
  dt: 0.02
  admm_iters: 5
  accel_max: 8.0
  collision_radius: 3.0

training:
  total_timesteps: 1_000_000
  eval_freq: 10_000
  checkpoint_freq: 50_000
  device: auto
  seed: 42

ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
```

### 3.4 Key Hyperparameters

| Parameter | Default | Tuning Notes |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Decrease to 1e-4 if training is unstable; increase to 5e-4 for faster early progress |
| `n_steps` | 2048 | Increase to 4096 for more stable gradients (needs more RAM) |
| `batch_size` | 256 | Must divide `n_steps × num_envs` evenly |
| `n_epochs` | 10 | Decrease to 5 if policy updates are too aggressive |
| `gamma` | 0.99 | Decrease to 0.95 for shorter-horizon tasks |
| `ent_coef` | 0.01 | Increase to 0.05 for more exploration early on; decrease to 0.001 after convergence |
| `clip_range` | 0.2 | Standard PPO value; decrease to 0.1 for more conservative updates |
| `total_timesteps` | 1M | 1–2M steps for 4 drones; scale proportionally for more drones |

### 3.5 Training

```bash
# Basic training
python scripts/train_mappo.py

# With evaluation environment and custom timesteps
python scripts/train_mappo.py --eval --timesteps 2000000 --device cuda

# Monitor with TensorBoard
tensorboard --logdir logs/mappo_dmpc
```

**TensorBoard metrics to watch:**

| Metric | Healthy Range | Problem If |
|--------|--------------|------------|
| `rollout/ep_rew_mean` | Increasing over time | Flat or decreasing after 200K steps |
| `train/policy_gradient_loss` | Small magnitude, stable | Large oscillations |
| `train/value_loss` | Decreasing | Increasing or exploding |
| `train/entropy_loss` | Slowly decreasing | Collapses to 0 (policy collapsed) |
| `train/clip_fraction` | 0.05–0.20 | >0.30 (updates too large) or <0.01 (not learning) |
| `train/approx_kl` | <0.02 | >0.05 (unstable updates) |

### 3.6 Reward Tuning

The environment reward (in `marl_env.py`) has four components:

```python
W_TRACK = 5.0    # Tracking accuracy
W_FORM  = 2.0    # Formation maintenance
W_SAFE  = 10.0   # Collision avoidance
W_EFF   = 0.1    # Control efficiency
```

**Tuning guidelines:**
- If drones collide during training: increase `W_SAFE` (15–20)
- If drones don't track targets well: increase `W_TRACK` (8–10)
- If formation breaks apart: increase `W_FORM` (4–5)
- If drones are too aggressive: increase `W_EFF` (0.5–1.0)
- Always keep `W_SAFE` as the largest weight

### 3.7 Training Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Reward doesn't improve | Learning rate too low / high | Try 1e-4 or 5e-4 |
| Reward oscillates wildly | Batch size too small | Increase to 512 or 1024 |
| NaN in loss | Exploding gradients | Decrease `max_grad_norm` to 0.3; decrease lr |
| Policy collapses (entropy → 0) | Exploration died | Increase `ent_coef` to 0.05; reset training |
| Slow training | CPU bottleneck on DMPC solves | Decrease horizon to 10 for training |

### 3.8 Evaluating Trained Models

```bash
# Run RL-DMPC with the trained model
python scripts/run_dmpc_rl.py \
    --scenario area_surveillance \
    --model models/mappo_dmpc/final \
    --episodes 5 \
    --deterministic

# Compare with pure DMPC baseline
python scripts/run_dmpc.py \
    --scenario area_surveillance \
    --episodes 5
```

**Compare these metrics:**
- `total_reward`: RL-DMPC should exceed pure DMPC by 10–30%
- `mean_solve_time_ms`: should be similar (DMPC solver is the bottleneck)
- `mean_q_scale` / `std_q_scale`: non-trivial std means the policy is adapting

If RL-DMPC is worse than pure DMPC, the policy needs more training or the
reward weights need adjustment.

---

## 4. Phase 3 — Run in PyBullet

### 4.1 Quick Start

```bash
# Interactive 3D window (default: 4 drones, 2 targets)
python pybullet_sim/swarm_pybullet_sim.py

# Headless (CI / server)
python pybullet_sim/swarm_pybullet_sim.py --no-gui --max-steps 2000

# Custom configuration
python pybullet_sim/swarm_pybullet_sim.py \
    --n-drones 6 \
    --n-targets 3 \
    --horizon 20 \
    --dt 0.02 \
    --accel-max 8.0 \
    --collision-radius 3.0 \
    --realtime
```

### 4.2 PyBullet Sim Architecture

The PyBullet simulation uses **pure DMPC** (no RL) via `DMPCAgent`:

1. `DMPCAgent.act()` runs the DMPC optimiser → acceleration command
2. The geometric attitude controller converts acceleration → motor thrusts
3. Motor thrusts drive the 6-DOF `DronePhysics` rigid-body engine
4. PyBullet syncs visual bodies to the physics engine state

The drone URDF is loaded from `src/isr_rl_dmpc/models/hector_quadrotor/drone.urdf`,
which references the hector_quadrotor STL mesh.

### 4.3 CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-drones` | 4 | Number of drones |
| `--n-targets` | 2 | Number of ground targets |
| `--dt` | 0.02 | Simulation timestep [s] (50 Hz) |
| `--horizon` | 20 | DMPC prediction horizon |
| `--accel-max` | 8.0 | Max acceleration [m/s²] |
| `--collision-radius` | 3.0 | Min inter-drone distance [m] |
| `--seed` | 42 | Random seed |
| `--traj-length` | 200 | Trajectory trail length |
| `--max-steps` | 0 | Max steps (0 = unlimited) |
| `--no-gui` | false | Headless mode |
| `--realtime` | false | Pace to real-time |
| `--no-auto-camera` | false | Disable auto-follow camera |

### 4.4 Tuning for PyBullet Visualisation

The PyBullet sim uses the same DMPC parameters as the analytical scripts.
To tune behaviour in the 3D view:

1. **Start with defaults** — verify drones hover stably
2. **Adjust `--accel-max`** — lower values (4–6) give smoother flight
3. **Adjust `--collision-radius`** — increase if drones fly too close
4. **Change `--dt`** — smaller values (0.01) improve accuracy but slow the sim
5. **Increase `--horizon`** — longer horizons give smoother trajectories

### 4.5 Using RL-DMPC Models in PyBullet

The current PyBullet sim runs pure DMPC.  To use a trained MAPPO model
in PyBullet, modify the step loop in `swarm_pybullet_sim.py`:

```python
from isr_rl_dmpc.agents.mappo_agent import MAPPOAgent
from isr_rl_dmpc.gym_env.marl_env import MARLDMPCEnv

# After creating the sim, load the trained policy:
marl_env = MARLDMPCEnv(num_drones=n_drones, ...)
agent = MAPPOAgent.load("models/mappo_dmpc/final", env=marl_env)

# In the step loop, get adaptive Q/R scales from the policy:
obs = marl_env._build_observation()
action, _ = agent.predict(obs, deterministic=True)
# action contains per-drone [q_scale(11), r_scale(3)]
# Pass these to the DMPC via dmpc(x, x_ref, neighbors, q_scale=..., r_scale=...)
```

---

## 5. Parameter Reference

### 5.1 Drone Physical Parameters

From `config/drone_specs.yaml` (hector_quadrotor):

| Parameter | Value | Unit |
|-----------|-------|------|
| Mass | 1.477 | kg |
| Inertia (Ixx, Iyy, Izz) | 0.01152, 0.01152, 0.02180 | kg·m² |
| Arm length | 0.215 | m |
| Max thrust per motor | 9.5 | N |
| Max velocity | 15.0 | m/s |
| Max acceleration | 8.0 | m/s² |
| Hover power | 64.0 | W |
| Battery endurance | ~22 | min |

### 5.2 DMPC Parameters Summary

| Parameter | Config Key | Default | Range |
|-----------|-----------|---------|-------|
| Horizon | `prediction_horizon` | 20 | 10–40 |
| Timestep | `dt` | 0.02 | 0.01–0.05 |
| Max accel | `accel_max` | 8.0 | 4.0–12.0 |
| Collision radius | `collision_radius` | 3.0 | 2.0–5.0 |
| Q diagonal | `Q_diag` | 1.0 | 0.1–10.0 |
| R diagonal | `R_diag` | 0.15 | 0.05–1.0 |
| Solver timeout | `solver_timeout` | 0.015 | 0.01–0.05 |

### 5.3 MAPPO Hyperparameters Summary

| Parameter | Config Key | Default | Range |
|-----------|-----------|---------|-------|
| Learning rate | `learning_rate` | 3e-4 | 1e-4 – 1e-3 |
| Rollout steps | `n_steps` | 2048 | 1024–8192 |
| Batch size | `batch_size` | 256 | 64–1024 |
| PPO epochs | `n_epochs` | 10 | 3–15 |
| Discount | `gamma` | 0.99 | 0.95–0.999 |
| GAE lambda | `gae_lambda` | 0.95 | 0.9–0.99 |
| Clip range | `clip_range` | 0.2 | 0.1–0.3 |
| Entropy coeff | `ent_coef` | 0.01 | 0.001–0.1 |

---

## 6. Troubleshooting

### Pure DMPC

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Solver returns "infeasible" | Conflicting CBF constraints | Increase `collision_radius` or decrease `accel_max` |
| Solver timeout exceeded | Horizon too large | Decrease `prediction_horizon` or increase `solver_timeout` |
| Drones drift slowly | Q position weights too low | Increase Q[0:3] weights |
| Oscillation around waypoints | R too small / Q too large | Increase R or decrease Q velocity weights |
| Battery runs out quickly | Control effort too high | Increase R weights |

### RL-DMPC Training

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Import error for `stable-baselines3` | Missing dependency | `pip install stable-baselines3[extra]` |
| CUDA out of memory | Batch too large | Decrease `batch_size` |
| Training crashes with NaN | Gradient explosion | Decrease `learning_rate` and `max_grad_norm` |
| Policy worse than baseline | Insufficient training | Increase `total_timesteps` to 2–5M |

### PyBullet Simulation

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| "pybullet not installed" | Missing package | `pip install pybullet` |
| URDF not found | Path resolution | Ensure `pip install -e .` was run |
| Drones fall through ground | Physics conflict | Gravity is disabled in PyBullet (handled by custom engine) |
| No mesh visible | STL not found | Check that `src/isr_rl_dmpc/models/meshes/hector_quadrotor/quadrotor_base.stl` exists |
| Very slow rendering | Too many drones + debug lines | Decrease `--traj-length` or increase `--dt` |
