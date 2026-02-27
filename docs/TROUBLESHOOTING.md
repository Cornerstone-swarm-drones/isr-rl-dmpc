# Troubleshooting

Common issues and solutions when working with ISR-RL-DMPC.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Issues](#training-issues)
- [Environment Issues](#environment-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [Testing Issues](#testing-issues)

---

## Installation Issues

### CVXPY Installation Fails

**Symptom:** `pip install cvxpy` fails with compilation errors.

**Solution:**

```bash
# Install system dependencies first
sudo apt-get install -y build-essential libopenblas-dev

# Then install CVXPY
pip install cvxpy
```

On macOS:

```bash
brew install openblas
pip install cvxpy
```

---

### PyTorch CUDA Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False` even with a GPU.

**Solution:**

1. Verify your NVIDIA driver: `nvidia-smi`
2. Install the correct PyTorch version for your CUDA:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

---

### Import Errors After Installation

**Symptom:** `ModuleNotFoundError: No module named 'isr_rl_dmpc'`

**Solution:**

```bash
# Install in development mode
pip install -e .

# Verify
python -c "import isr_rl_dmpc; print('OK')"
```

Make sure you are running commands from the repository root directory.

---

## Training Issues

### Training Diverges (Reward Decreases or NaN)

**Symptoms:**
- Episode reward decreasing over time
- NaN values in loss or reward
- Critic loss exploding

**Solutions:**

1. **Reduce learning rates:**

   ```yaml
   learning:
     learning_rate_critic: 0.0005  # from 0.001
     learning_rate_actor: 0.00005  # from 0.0001
   ```

2. **Reduce gradient clipping threshold:**

   ```yaml
   training:
     max_grad_norm: 0.5  # from 1.0
   ```

3. **Increase warmup steps** to populate the buffer with diverse experience:

   ```yaml
   training:
     warmup_steps: 20000  # from 10000
   ```

4. **Check reward weights** — an extremely large collision penalty can cause value function instability. Try:

   ```yaml
   learning:
     weight_collision: -50.0  # from -100.0
   ```

---

### Agent Does Not Improve (Flat Reward Curve)

**Symptoms:**
- Episode reward stays constant
- Coverage does not improve

**Solutions:**

1. **Increase exploration noise:**

   ```yaml
   exploration:
     noise_std: 0.3   # from 0.1
     noise_decay: 0.999  # slower decay
   ```

2. **Increase batch size** for more stable gradient estimates:

   ```yaml
   learning:
     batch_size: 128  # from 32
   ```

3. **Check that the replay buffer is filling up.** If `agent.ready_to_train()` is never `True`, the buffer is too small or the batch size is too large relative to the number of steps.

4. **Increase the number of episodes.** Some configurations need 300+ episodes to show improvement.

---

### Agent Always Goes to the Same Location

**Symptom:** All drones converge to a single point.

**Solutions:**

1. **Increase formation reward weight** to encourage separation:

   ```yaml
   learning:
     weight_formation: 10.0  # from 2.0
   ```

2. **Increase `min_swarm_separation`:**

   ```yaml
   mission:
     min_swarm_separation: 5.0  # from 2.0
   ```

3. **Ensure coverage reward incentivizes new areas.** The reward for newly covered cells should be significantly positive.

---

### Out of Memory (OOM) During Training

**Symptom:** `RuntimeError: CUDA out of memory` or system memory exhaustion.

**Solutions:**

1. **Reduce batch size:**

   ```yaml
   learning:
     batch_size: 16  # from 32 or 64
   ```

2. **Reduce buffer size:**

   ```yaml
   learning:
     buffer_size: 50000  # from 100000
   ```

3. **Reduce number of drones:**

   ```python
   env = ISRGridEnv(num_drones=4)  # instead of 10
   ```

4. **Use CPU instead of GPU** for small experiments:

   ```bash
   python scripts/train_agent.py --device cpu
   ```

---

## Environment Issues

### Simulator Running in Stub Mode

**Symptom:** Log message: `[ISRGridEnv] Simulator import failed - running in stub mode`

**Cause:** The `EnvironmentSimulator` or its dependencies failed to import.

**Solution:**

```bash
# Ensure all dependencies are installed
pip install -r requirements/base.txt

# Reinstall the package
pip install -e .
```

If the issue persists, check for circular import errors:

```bash
python -c "from isr_rl_dmpc.gym_env.simulator import EnvironmentSimulator"
```

---

### Observation Shape Mismatch

**Symptom:** `ValueError` or `AssertionError` about array shapes.

**Cause:** Agent `state_dim` does not match the flattened observation size.

**Solution:**

Compute the state dimension dynamically from the environment:

```python
obs, _ = env.reset()
state_dim = DMPCAgent.flatten_obs(obs).shape[0]
agent = DMPCAgent(state_dim=state_dim, ...)
```

---

### Environment Rendering Does Not Display

**Symptom:** `env.render()` returns `None` or does not show anything.

**Solution:**

Make sure you set `render_mode` when creating the environment:

```python
# For text output
env = ISRGridEnv(render_mode='human')

# For image output
env = ISRGridEnv(render_mode='rgb_array')
img = env.render()  # Returns numpy array
```

---

## Performance Issues

### Training is Slow

**Solutions:**

1. **Use GPU:**

   ```bash
   python scripts/train_agent.py --device cuda
   ```

2. **Use vectorized environments** for parallel data collection:

   ```python
   vec_env = make_env('ISRGridEnv-v0', num_envs=4, num_drones=10)
   ```

3. **Reduce DMPC solver iterations** if the solver is the bottleneck:

   ```yaml
   dmpc:
     solver_max_iterations: 50   # from 100
     solver_tolerance: 0.001     # from 0.0001
   ```

4. **Reduce grid resolution** for faster simulation:

   ```python
   env = ISRGridEnv(grid_size=(10, 10))  # instead of (20, 20)
   ```

---

### CVXPY Solver Timeout

**Symptom:** DMPC controller takes too long, solver returns suboptimal solutions.

**Solutions:**

1. **Reduce prediction horizon:**

   ```yaml
   dmpc:
     prediction_horizon: 5   # from 10
     control_horizon: 3      # from 5
   ```

2. **Increase solver tolerance:**

   ```yaml
   dmpc:
     solver_tolerance: 0.001  # from 0.0001
   ```

3. **Reduce the number of neighbor constraints:**

   ```python
   DMPCConfig(n_neighbors=2)  # from 4
   ```

---

## Configuration Issues

### Configuration Validation Fails

**Symptom:** `AssertionError` when loading configuration.

**Solution:**

Check that all parameter values are within valid ranges:

```python
from isr_rl_dmpc.config import load_config

try:
    config = load_config("config/default_config.yaml")
except AssertionError as e:
    print(f"Invalid config: {e}")
```

Common validation rules:
- `mass > 0`, `max_velocity > 0`, `battery_capacity > 0`
- `0 < discount_factor < 1`
- `control_horizon <= prediction_horizon`
- `0 < constraint_tightening <= 1`
- `0 <= coverage_goal <= 1`

---

### YAML Parsing Error

**Symptom:** `yaml.scanner.ScannerError` when loading config.

**Solution:**

1. Verify YAML syntax (indentation must use spaces, not tabs)
2. Validate with an online YAML validator
3. Check for special characters that need quoting

---

## Testing Issues

### Tests Fail After Installation

**Symptom:** `pytest tests/` shows import errors or failures.

**Solution:**

```bash
# Install dev dependencies
pip install -r requirements/dev.txt

# Install package in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

---

### Flaky Tests (Intermittent Failures)

**Cause:** Random seed not set, or timing-dependent behavior.

**Solution:**

Set a fixed seed before running tests:

```bash
pytest tests/ --seed 42
```

Or in code:

```python
import numpy as np
np.random.seed(42)
```

---

### Test Coverage Below Threshold

**Solution:**

```bash
# Run with coverage reporting
pytest tests/ --cov=src/isr_rl_dmpc --cov-report=html

# View report
open htmlcov/index.html
```

Focus on adding tests for uncovered modules in `tests/unit/`.
