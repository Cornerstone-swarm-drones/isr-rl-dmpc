# DMPC Optimisation Problem with MARL-Adaptive Cost

**Source files:**
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `DMPC`, `MPCSolver`, `DMPCConfig`, `compute_lqr_terminal_cost`
- `src/isr_rl_dmpc/agents/mappo_agent.py` — `MAPPOAgent` (provides $\mathbf{q}_s, \mathbf{r}_s$)
- `src/isr_rl_dmpc/modules/admm_consensus.py` — `ADMMConsensus`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [MARL-Adaptive Cost Parameters](#2-marl-adaptive-cost-parameters)
3. [Decision Variables](#3-decision-variables)
4. [Cost Function](#4-cost-function)
5. [Constraints](#5-constraints)
6. [Discrete Algebraic Riccati Equation (DARE)](#6-discrete-algebraic-riccati-equation-dare)
7. [LQR Terminal Cost](#7-lqr-terminal-cost)
8. [QP Standard Form](#8-qp-standard-form)
9. [ADMM Augmented Lagrangian](#9-admm-augmented-lagrangian)
10. [OSQP Solver Settings](#10-osqp-solver-settings)
11. [Collision Avoidance Constraints](#11-collision-avoidance-constraints)
12. [Default Parameter Values](#12-default-parameter-values)
13. [References](#13-references)

---

## 1  Problem Statement

At each control time step $t$, drone $i$ solves a finite-horizon **Quadratic
Program (QP)** whose cost matrices are *dynamically scaled* by the MAPPO policy:

$$
\min_{\mathbf{x}, \mathbf{u}} \; \sum_{k=0}^{N-1} \Bigl[ \mathbf{e}_k^\top Q_{\text{eff}}^{(i)} \mathbf{e}_k + \mathbf{u}_k^\top R_{\text{eff}}^{(i)} \mathbf{u}_k \Bigr] + \mathbf{e}_N^\top P\, \mathbf{e}_N
$$

subject to:

$$
\mathbf{x}_{k+1} = A\,\mathbf{x}_k + B\,\mathbf{u}_k
  \quad (\text{linearised dynamics})
$$

$$
\|\mathbf{u}_k\|_2 \le u_{\max}
  \quad (\text{acceleration saturation})
$$

$$
\|\mathbf{p}_k - \mathbf{p}_j\|_2 \ge r_{\min}
  \quad \forall\, j \in \mathcal{N}(i)
  \quad (\text{collision avoidance})
$$

$$
\mathbf{x}_0 = \mathbf{x}(t) \quad (\text{initial condition})
$$

where:
- $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^{\text{ref}}_k$ — tracking error at prediction step $k$
- $N$ — prediction horizon (default: 20 steps = 0.4 s at 50 Hz)
- $Q_{\text{eff}}^{(i)} = Q \odot \mathrm{diag}(\mathbf{q}_s^{(i)}) \succ 0$ — effective state-error cost ($11 \times 11$)
- $R_{\text{eff}}^{(i)} = R \odot \mathrm{diag}(\mathbf{r}_s^{(i)}) \succ 0$ — effective control-effort cost ($3 \times 3$)
- $P \succ 0$ — terminal cost matrix ($11 \times 11$), computed from DARE
- $\mathbf{q}_s^{(i)}, \mathbf{r}_s^{(i)}$ — positive scaling vectors output by the MAPPO agent

This is a **Distributed MPC (DMPC)**: each drone solves its own QP independently,
using the most recent communicated positions of neighbours as fixed obstacle positions
in the collision constraints.  The MAPPO policy adapts the cost weights online so the
controller can balance tracking accuracy, energy, and formation keeping dynamically.

---

## 2  MARL-Adaptive Cost Parameters

The MAPPO agent (see [09_MAPPO_AGENT.md](09_MAPPO_AGENT.md)) outputs a 14-dimensional
action vector per drone:

$$
\mathbf{a}^{(i)} = \bigl[\underbrace{q_{s,0}, \ldots, q_{s,10}}_{\mathbf{q}_s \in \mathbb{R}^{11}},\;
  \underbrace{r_{s,0}, r_{s,1}, r_{s,2}}_{\mathbf{r}_s \in \mathbb{R}^{3}}\bigr]
$$

These are element-wise scale factors applied to the base cost matrices:

$$
Q_{\text{eff}}^{(i)} = Q \odot \mathrm{diag}(\mathbf{q}_s^{(i)}), \qquad
R_{\text{eff}}^{(i)} = R \odot \mathrm{diag}(\mathbf{r}_s^{(i)})
$$

where $\odot$ denotes element-wise (Hadamard) product.  All scale values are
clipped to $[0.1, 10.0]$ to prevent ill-conditioning of the QP.

**Why scale instead of replace?**  Scaling preserves the positive-definiteness
guaranteed by the base matrices $Q \succ 0$, $R \succ 0$, so the DARE terminal
cost $P$ remains valid regardless of the MAPPO output.

---

## 3  Decision Variables

| Variable | Shape | Description |
| :--- | :--- | :--- |
| `x_var` | $(11,\; N{+}1)$ | Predicted state trajectory |
| `u_var` | $(3,\; N)$ | Predicted control sequence |

The first control in the sequence, `u_var[:, 0]`, is applied to the drone.
The remaining $N{-}1$ controls are discarded (receding-horizon principle).

---

## 4  Cost Function

### Stage Cost

$$
\ell(\mathbf{e}_k, \mathbf{u}_k) = \mathbf{e}_k^\top Q_{\text{eff}} \mathbf{e}_k + \mathbf{u}_k^\top R_{\text{eff}} \mathbf{u}_k
$$

- $\|\mathbf{e}_k\|^2_{Q_{\text{eff}}}$: penalises deviation from the reference trajectory
  in each state channel, weighted by the MAPPO-adapted $Q_{\text{eff}}$.
- $\|\mathbf{u}_k\|^2_{R_{\text{eff}}}$: penalises large control inputs, promoting smooth
  trajectories and limiting actuator wear.

### Terminal Cost

$$
V_f(\mathbf{e}_N) = \mathbf{e}_N^\top P\, \mathbf{e}_N
$$

$P$ is the **LQR optimal cost-to-go** matrix solved from the DARE.  Using the
LQR cost-to-go as the terminal cost provides:

1. **Recursive feasibility** — if the problem is feasible at step $t$, it is
   feasible at step $t{+}1$.
2. **Asymptotic stability** — the DMPC inherits the LQR's stability guarantee
   within a neighbourhood of the terminal set.

### Total Cost

$$
J = \sum_{k=0}^{N-1} \bigl(\mathbf{e}_k^\top Q_{\text{eff}} \mathbf{e}_k + \mathbf{u}_k^\top R_{\text{eff}} \mathbf{u}_k\bigr) + \mathbf{e}_N^\top P\, \mathbf{e}_N
$$

This is convex (sum of convex quadratics) and can be solved globally by a
convex QP solver.

---

## 5  Constraints

### 5.1  Dynamics Constraints

$$
\mathbf{x}_{k+1} = A\,\mathbf{x}_k + B\,\mathbf{u}_k, \quad k = 0, 1, \ldots, N{-}1
$$

$$
\mathbf{x}_0 = \mathbf{x}(t)
$$

These are **equality constraints** in the QP.  $A$ and $B$ are constant
(time-invariant linearisation); see [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md)
for their structure.

### 5.2  Control Saturation

$$
\|\mathbf{u}_k\|_2 \le u_{\max}, \quad k = 0, \ldots, N{-}1
$$

$u_{\max} = 10.0\;\text{m/s}^2$ (default).  CVXPY/OSQP represents this as a
**second-order cone (SOC) constraint**.

### 5.3  Collision Avoidance Constraints

For each neighbour $j$ at (fixed) position $\mathbf{p}_j$ and each prediction step $k$:

$$
\|\mathbf{p}_k - \mathbf{p}_j\|_2 \ge r_{\min}
$$

where $r_{\min} = 0.9 \times r_{\text{collision}}$ (10% safety margin).

---

## 6  Discrete Algebraic Riccati Equation (DARE)

The DARE is the fixed-point equation for the infinite-horizon discrete-time LQR cost:

$$
P = Q + A^\top P A - A^\top P B \,(R + B^\top P B)^{-1} B^\top P A
$$

**Solution:** The unique symmetric positive-definite solution $P$ is found
numerically via `scipy.linalg.solve_discrete_are(A, B, Q, R)`.

The implementation uses the **9-state translational sub-system** $A_9, B_9$
(rows/cols 0–8) since the yaw sub-system is not driven by the translational input:

```python
A = np.eye(state_dim)
A[0:3, 3:6] = dt * np.eye(3)   # dp/dv
A[3:6, 6:9] = dt * np.eye(3)   # dv/da

B = np.zeros((state_dim, control_dim))
B[6:9, 0:3] = dt * np.eye(3)   # da/du

P = solve_discrete_are(A, B, Q, R)
P = (P + P.T) / 2               # symmetrise
```

The resulting $P$ is expanded to the full $11 \times 11$ state space (the yaw
rows/cols retain a $Q$-proportional diagonal).

---

## 7  LQR Terminal Cost

Given the DARE solution $P$, the **optimal LQR gain** is:

$$
K_{\text{LQR}} = (R + B^\top P B)^{-1} B^\top P A
$$

The **closed-loop matrix** under the LQR controller is:

$$
A_{\text{cl}} = A - B K_{\text{LQR}}
$$

For the default $Q = I_{11}$, $R = 0.1\,I_3$, $\Delta t = 0.02\;\text{s}$, the spectral radius
$\rho(A_{\text{cl}}) \approx 0.983$, confirming asymptotic stability ($|\lambda_i| < 1$).

The **LQR value function**:

$$
V(\mathbf{e}) = \mathbf{e}^\top P\, \mathbf{e}
$$

is a valid Lyapunov function because $P \succ 0$ and $\Delta V < 0$ under the LQR
law (see [04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) for the
full proof).

---

## 8  QP Standard Form

CVXPY canonicalises the DMPC problem into the standard OSQP form:

$$
\min_{\mathbf{y}} \; \tfrac{1}{2}\,\mathbf{y}^\top H_{\text{qp}}\,\mathbf{y} + \mathbf{c}^\top \mathbf{y} \quad \text{s.t.} \quad \mathbf{l} \le A_{\text{qp}}\,\mathbf{y} \le \mathbf{u}
$$

where $\mathbf{y} = [\mathrm{vec}(\mathtt{x\_var});\; \mathrm{vec}(\mathtt{u\_var})]$
is the stacked decision variable vector.

| Problem size (default, 1 drone, $N{=}20$) | Value |
| :--- | :--- |
| Decision variables | $11 \times 21 + 3 \times 20 = 291$ |
| Equality constraints (dynamics) | $11 \times 20 = 220$ |
| Inequality constraints (saturation) | 20 |
| Collision constraints (4 neighbours) | $4 \times 20 = 80$ |
| **Total constraints** | **320** |

---

## 9  ADMM Augmented Lagrangian

The ADMM consensus layer (see [10_ADMM_CONSENSUS.md](10_ADMM_CONSENSUS.md)) couples
the local DMPC sub-problems across drones.  The **augmented Lagrangian** for the
global consensus problem is:

$$
\mathcal{L}_\rho(\mathbf{z}_1,\ldots,\mathbf{z}_N, \mathbf{v}, \boldsymbol{\mu}) = \sum_{i=1}^{N} J_i(\mathbf{z}_i) + \sum_{i=1}^{N} \boldsymbol{\mu}_i^\top (\mathbf{z}_i - \mathbf{v}) + \frac{\rho}{2} \sum_{i=1}^{N} \|\mathbf{z}_i - \mathbf{v}\|^2
$$

where:
- $\mathbf{z}_i$ — drone $i$'s local copy of the shared variable (reference trajectory)
- $\mathbf{v}$ — global consensus variable (average trajectory)
- $\boldsymbol{\mu}_i$ — dual variable (Lagrange multiplier) for drone $i$
- $\rho > 0$ — ADMM penalty parameter (default: 1.0)

ADMM iterates three steps per DMPC solve cycle:

**x-update** (local QP solve per drone, parallelisable):

$$
\mathbf{z}_i^{k+1} \leftarrow \arg\min_{\mathbf{z}_i} \Bigl[ J_i(\mathbf{z}_i) + (\boldsymbol{\mu}_i^k)^\top(\mathbf{z}_i - \mathbf{v}^k) + \tfrac{\rho}{2}\|\mathbf{z}_i - \mathbf{v}^k\|^2 \Bigr]
$$

**v-update** (global average, closed form):

$$
\mathbf{v}^{k+1} \leftarrow \frac{1}{N} \sum_{i=1}^{N}
  \bigl(\mathbf{z}_i^{k+1} + \boldsymbol{\mu}_i^k / \rho\bigr)
$$

**Dual update**:

$$
\boldsymbol{\mu}_i^{k+1} \leftarrow \boldsymbol{\mu}_i^k + \rho\,(\mathbf{z}_i^{k+1} - \mathbf{v}^{k+1})
$$

---

## 10  OSQP Solver Settings

```python
problem.solve(
    solver=cp.OSQP,
    max_iter=3000,
    eps_abs=1e-3,
    eps_rel=1e-3,
    time_limit=0.010,   # 10 ms budget per drone
)
```

| Setting | Value | Rationale |
| :--- | :--- | :--- |
| `max_iter` | 3000 | Sufficient for well-conditioned swarm QPs |
| `eps_abs` | 1e-3 | Adequate accuracy for metre-scale control |
| `eps_rel` | 1e-3 | Same as absolute tolerance |
| `time_limit` | 10 ms | Leaves 10 ms margin in the 50 Hz loop |

**Solver fallback:** If OSQP returns a non-optimal status (infeasible, time
exceeded), the controller outputs a zero control command and logs a warning.

---

## 11  Collision Avoidance Constraints

### Current Implementation

The collision constraints are rebuilt each solve to include the current
neighbour positions:

```python
for neighbor_pos in neighbor_positions:
    for k in range(horizon):
        pos_k = x_var[:3, k]
        dist_k = cp.norm(pos_k - neighbor_pos)
        constraints.append(dist_k >= collision_radius * 0.9)
```

### Control Barrier Function Alternative

A formally safe alternative replaces the distance constraint with a
**Control Barrier Function (CBF)** inequality:

$$
h(\mathbf{p},\, \mathbf{p}_j) = \|\mathbf{p} - \mathbf{p}_j\|^2 - r_{\min}^2 \ge 0
$$

$$
\text{CBF condition:} \quad \Delta h + \alpha\, h \ge 0
$$

This keeps the feasibility set forward-invariant even when OSQP cannot
achieve a strictly feasible point within the time budget.  See
[04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) for CBF theory.

---

## 12  Default Parameter Values

| Parameter | Symbol | Value | Source |
| :--- | :--- | :--- | :--- |
| Prediction horizon | $N$ | 20 | `DMPCConfig.horizon` |
| Time step | $\Delta t$ | 0.02 s | `DMPCConfig.dt` |
| State dimension | $n$ | 11 | `DMPCConfig.state_dim` |
| Control dimension | $m$ | 3 | `DMPCConfig.control_dim` |
| Max acceleration | $u_{\max}$ | 10.0 m/s² | `DMPCConfig.accel_max` |
| Collision radius | $r_{\min}$ | 5.0 m | `DMPCConfig.collision_radius` |
| Safety margin | — | $0.9 \times r_{\min}$ | `MPCSolver.solve()` |
| Base state cost | $Q$ | $I_{11}$ | `DMPCConfig.Q_base` |
| Base control cost | $R$ | $0.1 \cdot I_3$ | `DMPCConfig.R_base` |
| Terminal cost | $P$ | DARE solution | `compute_lqr_terminal_cost()` |
| ADMM penalty | $\rho$ | 1.0 | `ADMMConsensus.rho` |
| Solver timeout | — | 10 ms | `DMPCConfig.solver_timeout` |

---

## 13  References

1. J. B. Rawlings, D. Q. Mayne, and M. Diehl, *Model Predictive Control:
   Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2019.
2. B. Stellato et al., "OSQP: An Operator Splitting Solver for Quadratic
   Programs," *Math. Prog. Comp.*, 12:637–672, 2020.
3. S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed
   Optimization and Statistical Learning via the Alternating Direction Method
   of Multipliers," *Found. Trends Mach. Learn.*, 3(1):1–122, 2011.
4. J. Yu, M. Dong, and X. Li, "Multi-Agent PPO for Cooperative UAV Control,"
   *IEEE Trans. Aerosp. Electron. Syst.*, 2023.
