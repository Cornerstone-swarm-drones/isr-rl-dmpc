# Troubleshooting

Common issues and solutions when working with ISR-DMPC.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Environment Issues](#environment-issues)
- [DMPC Solver Issues](#dmpc-solver-issues)
  - [CVXPY DPP Warning — Simulation Running 100× Too Slowly](#cvxpy-dpp-warning--simulation-running-100-too-slowly)
  - [DMPC Acceleration Not Applied to Drones](#dmpc-acceleration-not-applied-to-drones)
  - [CVXPY Solver Timeout](#cvxpy-solver-timeout)
  - [DMPC Returns Zero Control (Infeasible)](#dmpc-returns-zero-control-infeasible)
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

## DMPC Solver Issues

### CVXPY DPP Warning — Simulation Running 100× Too Slowly

**Symptom:**

```
PP. Because the problem is not DPP, subsequent solves will not be faster
than the first one. For more information, see the documentation on
Disciplined Parametrized Programming…
```

Wall time is 20–50× the simulated time even though CVXPY/OSQP reports
solve times of only 1–5 ms.

**Cause:**

CVXPY's DPP (Disciplined Parametrized Programming) cache stores the
compiled KKT system so that subsequent solves only update parameter values
rather than re-canonicalising the full problem from scratch.  The cache
is only active when **all** matrix–variable products use plain NumPy
constants, not ``cp.Parameter`` objects.

The DPP violation was:

1. ``A_param @ x_var`` and ``B_param @ u_var`` — matrix parameters
   multiplied by decision variables.
2. ``cp.quad_form(x_err, Q_param)`` — matrix parameter inside a quadratic
   form.
3. A fresh ``cp.Problem(...)`` was constructed inside ``solve()`` on every
   call, forcing full re-canonicalization regardless of DPP.

**Fix (already applied):**

``MPCSolver`` now embeds A, B, Q, R, P as NumPy constants at construction
time and builds exactly **one** ``cp.Problem``.  Only ``x0_param`` and
``x_ref_param`` are ``cp.Parameter`` objects; CBF constraints use
pre-allocated vector-parameter slots (``cp.Parameter(3)`` inner products
are DPP-compliant).  OSQP warm-starting is enabled.

After the fix the first solve pays the canonicalization cost once (≈20 ms);
all subsequent solves take ≈1–5 ms, giving near real-time performance.

---

### DMPC Acceleration Not Applied to Drones

**Symptom:** Drones hover in place (zero commanded acceleration) even
though the DMPC solver reports ``optimal`` status and non-zero control.

**Cause:**

``swarm_pybullet_sim.py`` reads:

```python
dmpc_u = info.get("u0", np.zeros(3))
```

But the ``info`` dict returned by ``DMPCAgent.act()`` did not contain the
``"u0"`` key, so the fallback ``np.zeros(3)`` was silently used every step.

**Fix (already applied):**

``DMPCAgent.act()`` now explicitly inserts ``"u0": u0`` into the returned
info dictionary before merging with the attitude-controller output.

---

### CVXPY Solver Timeout

**Symptom:** DMPC controller takes too long, solver returns suboptimal solutions.

**Solutions:**

1. **Reduce prediction horizon:**

   ```yaml
   dmpc:
     prediction_horizon: 12   # from 20 (safe minimum for 50 Hz)
   ```

1. **Enable OSQP warm-starting** for 3–8× speedup on subsequent solves:

   ```python
   # In MPCSolver.solve():
   self.problem.solve(solver=cp.OSQP, warm_start=True, …)
   ```

1. **Increase solver tolerance** (reduces iterations at slight accuracy cost):

   ```yaml
   dmpc:
     solver_tolerance: 0.001  # from 0.0001
   ```

1. **Reduce neighbour constraints:**

   ```python
   DMPCConfig(n_neighbors=2)  # from 4
   ```

---

### DMPC Returns Zero Control (Infeasible)

**Symptom:** Drone hovers in place; DMPC logs `solver_status != optimal`.

**Cause:** A neighbour drone has entered the collision-avoidance radius,
making the DMPC QP infeasible.

**Solutions:**

1. Increase the collision radius in `dmpc_config.yaml`:

   ```yaml
   dmpc:
     collision_radius: 3.0   # reduce if spacing allows
   ```

1. Ensure the formation controller (Module 2) maintains `communication_radius`
   and `min_swarm_separation` so drones never start inside each other's
   exclusion zone.

---

## Performance Issues

### Simulation Running Slowly

**Solutions:**

1. **Reduce grid resolution** for faster simulation:

   ```python
   env = ISRGridEnv(grid_size=(10, 10))  # instead of (20, 20)
   ```

1. **Reduce DMPC solver iterations** if the solver is the bottleneck:

   ```yaml
   dmpc:
     solver_max_iterations: 50   # from 100
     solver_tolerance: 0.001     # from 0.0001
   ```

1. **Use vectorized environments** for parallel data collection:

   ```python
   vec_env = make_env('ISRGridEnv-v0', num_envs=4, num_drones=10)
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
1. Validate with an online YAML validator
1. Check for special characters that need quoting

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
