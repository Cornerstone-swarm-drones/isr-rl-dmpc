# DMPC Optimisation Problem

**Source files:**
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `DMPC`, `MPCSolver`, `DMPCConfig`, `compute_lqr_terminal_cost`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Decision Variables](#2-decision-variables)
3. [Cost Function](#3-cost-function)
4. [Constraints](#4-constraints)
5. [Discrete Algebraic Riccati Equation (DARE)](#5-discrete-algebraic-riccati-equation-dare)
6. [LQR Terminal Cost](#6-lqr-terminal-cost)
7. [QP Standard Form](#7-qp-standard-form)
8. [OSQP Solver Settings](#8-osqp-solver-settings)
9. [Collision Avoidance Constraints](#9-collision-avoidance-constraints)
10. [Default Parameter Values](#10-default-parameter-values)
11. [References](#11-references)

---

## 1  Problem Statement

At each control time step **t**, every drone **i** in the swarm independently
solves the following finite-horizon **Quadratic Program (QP)**:

```
min   Σ_{k=0}^{N-1} [ ‖e_k‖²_Q  +  ‖u_k‖²_R ]  +  ‖e_N‖²_P
x,u

s.t.  x_{k+1} = A x_k + B u_k              (linearised dynamics)
      ‖u_k‖₂  ≤ u_max                       (acceleration saturation)
      ‖p_k − p_j‖₂ ≥ r_min  ∀ j ∈ 𝒩(i)    (collision avoidance)
      x_0 = x(t)                             (initial condition)
```

where:
- `e_k = x_k − x_ref_k` — tracking error at prediction step **k**
- **N** — prediction horizon (default: 20 steps = 0.4 s at 50 Hz)
- **Q ≻ 0** — state-error cost matrix (11×11)
- **R ≻ 0** — control-effort cost matrix (3×3)
- **P ≻ 0** — terminal cost matrix (11×11), computed from DARE

This is a **Distributed MPC (DMPC)**: each drone solves its own QP
independently, using the most recent communicated positions of neighbours as
fixed obstacle positions in the collision constraints.

---

## 2  Decision Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `x_var` | (11, N+1) | Predicted state trajectory |
| `u_var` | (3, N) | Predicted control sequence |

The first control in the sequence, `u_var[:, 0]`, is applied to the drone.
The remaining N−1 controls are discarded (receding-horizon principle).

---

## 3  Cost Function

### Stage Cost

```
ℓ(e_k, u_k) = e_kᵀ Q e_k  +  u_kᵀ R u_k
```

- **‖e_k‖²_Q**: penalises deviation from the reference trajectory in each
  state channel, weighted by Q.
- **‖u_k‖²_R**: penalises large control inputs, promoting smooth trajectories
  and limiting actuator wear.

### Terminal Cost

```
V_f(e_N) = e_Nᵀ P e_N
```

P is the **LQR optimal cost-to-go** matrix solved from the Discrete Algebraic
Riccati Equation (DARE).  Using the LQR cost-to-go as the terminal cost is the
standard mechanism that provides:
1. **Recursive feasibility** — if the problem is feasible at step t, it is
   feasible at step t+1.
2. **Asymptotic stability** — the DMPC inherits the LQR's stability guarantee
   within a neighbourhood of the terminal set.

### Total Cost

```
J = Σ_{k=0}^{N-1} (e_kᵀ Q e_k + u_kᵀ R u_k) + e_Nᵀ P e_N
```

This is convex (sum of convex quadratics) and can be solved globally by a
convex QP solver.

---

## 4  Constraints

### 4.1  Dynamics Constraints

```
x_{k+1} = A x_k + B u_k,   k = 0, 1, …, N−1
x_0 = x(t)
```

These are **equality constraints** in the QP.  A and B are constant (time-
invariant linearisation); see [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md)
for their structure.

### 4.2  Control Saturation (Second-Order Cone Constraint)

```
‖u_k‖₂ ≤ u_max,   k = 0, …, N−1
```

`u_max` = 10.0 m/s² (default).  This is equivalent to constraining each
control vector to lie inside a sphere.  CVXPY/OSQP represents this as a
**second-order cone (SOC) constraint**.

### 4.3  Collision Avoidance Constraints

For each neighbour j at (fixed) position **p_j** and each prediction step k:

```
‖p_k − p_j‖₂ ≥ r_min
```

where `r_min = 0.9 × r_collision = 0.9 × 5.0 = 4.5 m` (10 % safety margin).

These are **non-convex distance constraints**; they are reformulated inside
CVXPY/OSQP using the `cp.norm(pos_k − p_j) >= threshold` form, which OSQP
handles as a second-order cone constraint after linearisation.

---

## 5  Discrete Algebraic Riccati Equation (DARE)

The DARE is the fixed-point equation for the infinite-horizon discrete-time LQR
cost:

```
P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
```

**Solution:** The unique symmetric positive-definite solution P is found
numerically via `scipy.linalg.solve_discrete_are(A, B, Q, R)`.

The implementation in `compute_lqr_terminal_cost()` uses the **9-state
translational sub-system** A₉, B₉ (rows/cols 0–8) since the yaw sub-system
is not driven by the translational input:

```python
A = np.eye(state_dim)
A[0:3, 3:6] = dt * np.eye(3)   # dp/dv
A[3:6, 6:9] = dt * np.eye(3)   # dv/da

B = np.zeros((state_dim, control_dim))
B[6:9, 0:3] = dt * np.eye(3)   # da/du

P = solve_discrete_are(A, B, Q, R)
P = (P + P.T) / 2               # symmetrise
```

The resulting P is expanded to the full 11×11 state space (the yaw rows/cols
retain a Q-proportional diagonal).

---

## 6  LQR Terminal Cost

Given the DARE solution P, the **optimal LQR gain** is:

```
K_LQR = (R + Bᵀ P B)⁻¹ Bᵀ P A
```

The **closed-loop matrix** under the LQR controller is:

```
A_cl = A − B K_LQR
```

For the default Q = I₁₁, R = 0.1 I₃, Δt = 0.02 s, the spectral radius
ρ(A_cl) ≈ 0.983, confirming asymptotic stability (all |λᵢ| < 1).

The **LQR value function**:

```
V(e) = eᵀ P e
```

is a valid Lyapunov function because P ≻ 0 and ΔV < 0 under the LQR law
(see [04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) for the
full proof).

---

## 7  QP Standard Form

CVXPY canonicalises the DMPC problem into the standard OSQP form:

```
min   ½ yᵀ H_qp y  +  cᵀ y
y

s.t.  l ≤ A_qp y ≤ u
```

where `y = [vec(x_var); vec(u_var)]` is the stacked decision variable vector.

| Problem size (default, 1 drone, N=20) | Value |
|---------------------------------------|-------|
| Decision variables | 11×21 + 3×20 = 291 |
| Equality constraints (dynamics) | 11×20 = 220 |
| Inequality constraints (saturation) | 20 |
| Collision constraints (4 neighbours) | 4×20 = 80 |
| **Total constraints** | **320** |

---

## 8  OSQP Solver Settings

The DMPC uses CVXPY's OSQP backend with the following settings:

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
|---------|-------|-----------|
| `max_iter` | 3000 | Sufficient for well-conditioned swarm QPs |
| `eps_abs` | 1e-3 | Adequate accuracy for metre-scale control |
| `eps_rel` | 1e-3 | Same as absolute tolerance |
| `time_limit` | 10 ms | Leaves 10 ms margin in the 50 Hz loop |

**Solver fallback:** If OSQP returns a non-optimal status (infeasible, time
exceeded), the controller outputs a zero control command and logs a warning.

### Warm-Starting (Recommended Enhancement)

OSQP supports warm-starting from a previous solution:

```python
problem.solve(solver=cp.OSQP, warm_start=True, …)
```

This typically reduces solve time by 3–8× (8 ms → 1–2 ms on x86-64) for
adjacent time steps.  See `docs/MATH_OPTIMIZATION.md` for implementation
details.

---

## 9  Collision Avoidance Constraints

### Current Implementation

The collision constraints are **rebuilt each solve** to include the current
neighbour positions:

```python
for neighbor_pos in neighbor_positions:
    for k in range(horizon):
        pos_k = x_var[:3, k]
        dist_k = cp.norm(pos_k - neighbor_pos)
        constraints.append(dist_k >= collision_radius * 0.9)
```

**Limitation:** Rebuilding the CVXPY problem graph each step prevents
warm-starting.

### Control Barrier Function Alternative

A formally safe alternative replaces the distance constraint with a
**Control Barrier Function (CBF)** inequality:

```
h(p, p_j) = ‖p − p_j‖² − r_min²   ≥ 0

CBF condition:  Δh + α h ≥ 0
```

This keeps the feasibility set forward-invariant even when OSQP cannot
achieve a strictly feasible point within the time budget.  See
[04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) for the
CBF theory.

---

## 10  Default Parameter Values

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Prediction horizon | N | 20 | `DMPCConfig.horizon` |
| Time step | Δt | 0.02 s | `DMPCConfig.dt` |
| State dimension | n | 11 | `DMPCConfig.state_dim` |
| Control dimension | m | 3 | `DMPCConfig.control_dim` |
| Max acceleration | u_max | 10.0 m/s² | `DMPCConfig.accel_max` |
| Collision radius | r_min | 5.0 m | `DMPCConfig.collision_radius` |
| Safety margin | — | 0.9 × r_min | `MPCSolver.solve()` |
| State cost | Q | I₁₁ | `DMPCConfig.Q_base` |
| Control cost | R | 0.1 · I₃ | `DMPCConfig.R_base` |
| Terminal cost | P | DARE solution | `compute_lqr_terminal_cost()` |
| Solver timeout | — | 10 ms | `DMPCConfig.solver_timeout` |

---

## 11  References

1. J. B. Rawlings, D. Q. Mayne, and M. Diehl, *Model Predictive Control:
   Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2019.
2. B. Stellato et al., "OSQP: An Operator Splitting Solver for Quadratic
   Programs," *Math. Prog. Comp.*, 12:637–672, 2020.
3. D. Q. Mayne, "Model predictive control: Recent developments and future
   promise," *Automatica*, 50(12):2967–2986, 2014.
