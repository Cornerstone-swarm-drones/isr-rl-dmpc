# Training Guide

This guide covers how to train the ISR-RL-DMPC agent and tune hyperparameters to achieve the best model performance.

## Table of Contents

- [Overview](#overview)
- [Training Script](#training-script)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Tuning Strategy](#tuning-strategy)
- [Reward Weight Tuning](#reward-weight-tuning)
- [Network Architecture Tuning](#network-architecture-tuning)
- [Automated Hyperparameter Search](#automated-hyperparameter-search)
- [Monitoring Training](#monitoring-training)
- [Checkpointing and Resuming](#checkpointing-and-resuming)
- [Recommended Configurations](#recommended-configurations)

## Overview

The training pipeline uses a Soft Actor-Critic (SAC) algorithm to optimize DMPC cost function parameters. The agent learns to adapt the DMPC's Q, R, and P cost matrices based on the current swarm state, improving coverage efficiency, energy usage, and safety.

**Training Flow:**

```
ISRGridEnv (observation) → DMPCAgent.act() → environment.step() → reward
    ↓
DMPCAgent.remember(transition)
    ↓
DMPCAgent.train_on_batch() → gradient update (critic + actor)
    ↓
Repeat for N episodes
```

## Training Script

### Basic Usage

```bash
python scripts/train_agent.py --config config/default_config.yaml --task recon
```

### Task-Based Training

Models are trained and saved per task type. Each task applies different reward weight
presets so the agent learns behaviour appropriate for the mission:

| Task | Description | Primary Reward Focus |
|---|---|---|
| `recon` | Area reconnaissance — maximise area coverage | Coverage (w=50) |
| `intel` | Intelligence gathering — search and classify targets | Coverage + Threat (w=20/5) |
| `target_pursuit` | Target pursuit — track and engage hostile targets | Threat (w=20) |

```bash
# Train a recon model with 6 drones and 2 targets
python scripts/train_agent.py --task recon --num-drones 6 --max-targets 2

# Train a target pursuit model
python scripts/train_agent.py --task target_pursuit --num-drones 8 --max-targets 5

# Train an intel model
python scripts/train_agent.py --task intel --num-drones 4 --max-targets 3
```

Models are saved to both `data/training_logs/<task>_d<N>_t<M>_<timestamp>/` and
`data/trained_models/<task>_d<N>_t<M>.pt` for easy loading in production.

### Training with Live Foxglove Visualization

Enable real-time visualization of drone positions, targets, and metrics during training:

```bash
python scripts/train_agent.py --task recon --foxglove --foxglove-port 8765
```

Then open Foxglove Studio and connect to `ws://localhost:8765` to watch the training in real-time.

### Full Options

```bash
python scripts/train_agent.py \
    --config config/default_config.yaml \
    --task recon \
    --num-drones 10 \
    --max-targets 5 \
    --num-episodes 500 \
    --num-steps 1000 \
    --save-freq 50 \
    --eval-freq 100 \
    --output-dir data/training_logs \
    --device cuda \
    --seed 42 \
    --foxglove
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/default_config.yaml` | Path to configuration file |
| `--task` | `recon` | Task type: `recon`, `intel`, or `target_pursuit` |
| `--num-drones` | `10` | Number of drones (overrides config) |
| `--max-targets` | `5` | Number of targets (overrides config) |
| `--num-episodes` | `500` | Total training episodes |
| `--num-steps` | `1000` | Maximum steps per episode |
| `--save-freq` | `50` | Checkpoint save frequency (episodes) |
| `--eval-freq` | `100` | Evaluation frequency (episodes) |
| `--output-dir` | `data/training_logs` | Output directory |
| `--device` | `cuda` (if available) | `cuda` or `cpu` |
| `--seed` | `42` | Random seed for reproducibility |
| `--foxglove` | disabled | Enable live Foxglove Studio visualization |
| `--foxglove-port` | `8765` | WebSocket server port for Foxglove |

## Hyperparameter Reference

All hyperparameters are defined in two configuration files:

### Core Parameters (`config/default_config.yaml`)

| Section | Parameter | Default | Range | Description |
|---|---|---|---|---|
| `learning` | `discount_factor` | `0.99` | `[0.9, 0.999]` | Discount factor (γ) for future rewards |
| `learning` | `learning_rate_critic` | `0.001` | `[1e-4, 1e-2]` | Learning rate for value network |
| `learning` | `learning_rate_actor` | `0.0001` | `[1e-5, 1e-3]` | Learning rate for policy network |
| `learning` | `batch_size` | `32` | `[16, 256]` | Mini-batch size for training |
| `learning` | `buffer_size` | `100000` | `[10000, 1000000]` | Replay buffer capacity |
| `learning` | `target_update_frequency` | `1000` | `[100, 5000]` | Steps between target network updates |

### Extended Parameters (`config/learning_config.yaml`)

| Section | Parameter | Default | Description |
|---|---|---|---|
| `training` | `algorithm` | `SAC` | RL algorithm (Soft Actor-Critic) |
| `training` | `tau` | `0.005` | Soft target update rate |
| `training` | `num_epochs` | `500` | Total training epochs |
| `training` | `steps_per_epoch` | `4000` | Environment steps per epoch |
| `training` | `warmup_steps` | `10000` | Random exploration steps before training |
| `training` | `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `training` | `entropy_coefficient` | `0.2` | Entropy regularization weight |
| `training` | `auto_entropy_tuning` | `true` | Automatically adjust entropy coefficient |
| `experience_buffer` | `type` | `prioritized` | `uniform` or `prioritized` replay |
| `experience_buffer` | `alpha` | `0.6` | Prioritization exponent |
| `experience_buffer` | `beta_start` | `0.4` | Importance sampling start |
| `experience_buffer` | `beta_end` | `1.0` | Importance sampling end |
| `exploration` | `noise_type` | `gaussian` | `gaussian` or `ou` (Ornstein-Uhlenbeck) |
| `exploration` | `noise_std` | `0.1` | Exploration noise standard deviation |
| `exploration` | `noise_decay` | `0.995` | Noise decay per episode |

### Network Architecture (`config/learning_config.yaml`)

| Network | Parameter | Default | Description |
|---|---|---|---|
| `value_network` | `hidden_layers` | `[256, 256, 128]` | Hidden layer sizes |
| `value_network` | `activation` | `relu` | Activation function |
| `value_network` | `dropout_rate` | `0.1` | Dropout rate for regularization |
| `value_network` | `use_layer_norm` | `true` | Enable layer normalization |
| `policy_network` | `hidden_layers` | `[256, 256, 128]` | Hidden layer sizes |
| `policy_network` | `activation` | `relu` | Activation function |
| `policy_network` | `output_activation` | `tanh` | Output squashing function |
| `policy_network` | `log_std_min` | `-5.0` | Minimum log standard deviation |
| `policy_network` | `log_std_max` | `2.0` | Maximum log standard deviation |

## Tuning Strategy

### Step 1: Start with Default Configuration

Run a baseline training with the default settings to establish performance metrics:

```bash
python scripts/train_agent.py --num-episodes 200 --seed 42
```

### Step 2: Tune Learning Rates

The learning rates have the most significant impact on training stability and convergence:

- **Critic LR too high** → Value estimates oscillate, training diverges
- **Critic LR too low** → Slow convergence, poor value estimates
- **Actor LR too high** → Policy collapses, unstable behavior
- **Actor LR too low** → Policy does not improve

**Recommended approach:** Keep the critic learning rate 5–10× higher than the actor learning rate.

| Configuration | Critic LR | Actor LR | Use Case |
|---|---|---|---|
| Conservative | `5e-4` | `5e-5` | Stable training, slower convergence |
| Default | `1e-3` | `1e-4` | Balanced performance |
| Aggressive | `5e-3` | `5e-4` | Fast convergence, risk of instability |

### Step 3: Tune Discount Factor

The discount factor (γ) controls how much the agent values future rewards:

| Value | Effect |
|---|---|
| `0.95` | Short-sighted; good for tasks with immediate feedback |
| `0.99` | Balanced; recommended for most ISR missions |
| `0.995` | Far-sighted; better for long-horizon missions (30+ minutes) |

### Step 4: Tune Batch Size and Buffer Size

| Batch Size | Effect |
|---|---|
| `16–32` | More frequent updates, noisier gradients |
| `64–128` | Balanced update frequency and gradient quality |
| `256` | Smoother gradients, fewer updates per step |

Increase buffer size proportionally with batch size. A buffer at least 100× the batch size is recommended.

### Step 5: Tune Exploration

Start with higher exploration noise and decay it over training:

```yaml
exploration:
  noise_std: 0.2        # Start with higher noise
  noise_decay: 0.995    # Decay per episode
  noise_min: 0.01       # Minimum noise floor
```

## Reward Weight Tuning

Reward weights directly shape agent behavior. The training script supports three
built-in task presets that apply optimised weights automatically via `--task`:

### Built-in Task Presets (defined in `reward_shaper.py` and `learning_config.yaml`)

| Task | `w_coverage` | `w_energy` | `w_safety` | `w_threat` | `w_learning` |
|---|---|---|---|---|---|
| `recon` | 50.0 | 1.0 | 10.0 | 0.5 | 0.1 |
| `intel` | 20.0 | 2.0 | 10.0 | 5.0 | 0.1 |
| `target_pursuit` | 5.0 | 0.5 | 10.0 | 20.0 | 0.1 |

### Reward Component Ranges

Each component is normalised so that weights directly control relative importance:

| Component | Per-Step Range | Description |
|---|---|---|
| Coverage (`r_cov`) | `[-0.1, 1.0]` | Incremental + absolute coverage level |
| Energy (`r_eng`) | `[-0.01, 0]` | Penalises energy consumption |
| Safety (`r_safe`) | `[-10.0, 0.1]` | Collision/geofence penalty, +0.1 when safe |
| Threat (`r_threat`) | `[-1.0, 1.0]` | Normalised per-detection reward |
| Learning (`r_learn`) | `[-0.1, 0]` | TD-error gradient signal |

### Manual Weight Tuning

You can also adjust weights in `config/default_config.yaml` or `config/learning_config.yaml`:

| Weight | Effect of Increasing | Effect of Decreasing |
|---|---|---|
| `coverage` | Agent prioritizes area coverage | Agent may ignore uncovered areas |
| `energy` | Stronger energy conservation | Agent uses energy freely |
| `safety` | More conservative trajectories | Agent takes riskier paths |
| `threat` | Agent prioritizes threat engagement | Agent may ignore threats |

**Tips:**

- Keep the safety weight high enough to discourage collisions (≥ 10)
- For surveillance-only missions, set `w_threat` low and `w_coverage` high
- If the agent does not move, reduce `w_energy`

## Network Architecture Tuning

### When to Change Architecture

- **Underfitting** (low reward, poor coverage): Increase network capacity (`[512, 256, 128]`)
- **Overfitting** (training improves but evaluation does not): Add dropout (`0.2`), reduce network size
- **Instability** (oscillating loss): Enable layer normalization, reduce learning rate

### Configuration Example

```yaml
value_network:
  hidden_layers: [512, 256, 128]
  activation: "relu"
  dropout_rate: 0.2
  use_layer_norm: true

policy_network:
  hidden_layers: [256, 256]
  activation: "relu"
  output_activation: "tanh"
  dropout_rate: 0.0
```

## Automated Hyperparameter Search

Use the hyperparameter search script to explore configurations automatically:

```bash
python scripts/hyperparameter_search.py \
    --config config/default_config.yaml \
    --num-trials 100 \
    --output-dir data/hyperparameter_search \
    --device cuda
```

The search explores:

| Parameter | Search Values |
|---|---|
| `learning_rate_critic` | `[0.0001, 0.0005, 0.001, 0.005]` |
| `learning_rate_actor` | `[0.0001, 0.0005, 0.001]` |
| `gamma` | `[0.95, 0.99, 0.995]` |
| `batch_size` | `[16, 32, 64]` |
| `buffer_size` | `[10000, 50000, 100000]` |
| `exploration_noise` | `[0.1, 0.2, 0.3]` |

Results are scored using a composite metric:

```
score = mean_reward × 0.4 + mean_coverage × 100 × 0.4 + (1 - collision_rate) × 100 × 0.1 + time_bonus × 0.1
```

Best configuration is saved to `data/hyperparameter_search/<timestamp>/best_config.yaml`.

## Monitoring Training

Training logs are written to `data/training_logs/<timestamp>/`:

| File | Content |
|---|---|
| `train.log` | Text log with episode summaries |
| `training_stats.json` | Episode rewards, coverage, losses |
| `checkpoint_ep*.pt` | Model checkpoints |
| `final_model.pt` | Final trained model |

Key metrics to monitor:

- **Episode reward** — Should increase over training
- **Coverage efficiency** — Target ≥ 0.90 for surveillance missions
- **Collision count** — Should decrease and stabilize near zero
- **Critic loss** — Should decrease and stabilize
- **Actor loss** — May fluctuate; watch for divergence

## Checkpointing and Resuming

Checkpoints are saved every `--save-freq` episodes. To resume from a checkpoint, load the model into the agent:

```python
from isr_rl_dmpc.agents import DMPCAgent

agent = DMPCAgent(state_dim=..., action_dim=...)
agent.load("data/training_logs/<timestamp>/checkpoint_ep200.pt")
```

## Recommended Configurations

### Recon Mission (Long Duration, High Coverage)

```bash
python scripts/train_agent.py --task recon --num-drones 4 --max-targets 0 \
    --num-episodes 500 --num-steps 1000
```

```yaml
learning:
  discount_factor: 0.995
  learning_rate_critic: 0.0005
  learning_rate_actor: 0.00005
  batch_size: 64
  buffer_size: 500000
```

### Intel Mission (Search and Classify)

```bash
python scripts/train_agent.py --task intel --num-drones 5 --max-targets 4 \
    --num-episodes 500 --num-steps 1000
```

```yaml
learning:
  discount_factor: 0.99
  learning_rate_critic: 0.001
  learning_rate_actor: 0.0001
  batch_size: 32
  buffer_size: 100000
```

### Target Pursuit (Fast Reaction, Engage Threats)

```bash
python scripts/train_agent.py --task target_pursuit --num-drones 6 --max-targets 3 \
    --num-episodes 500 --num-steps 500
```

```yaml
learning:
  discount_factor: 0.99
  learning_rate_critic: 0.001
  learning_rate_actor: 0.0001
  batch_size: 32
  buffer_size: 100000
```
