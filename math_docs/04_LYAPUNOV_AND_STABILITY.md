# Lyapunov Stability, ISS, CBF and Recursive Feasibility

**Source files:**
- `src/isr_rl_dmpc/analysis/stability_analysis.py` — `DMPCStabilityAnalyzer`
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `compute_lqr_terminal_cost`

---

## Table of Contents

1. [System Model for Stability Analysis](#1-system-model-for-stability-analysis)
1. [Lyapunov Stability](#2-lyapunov-stability)
1. [Eigenvalue Analysis](#3-eigenvalue-analysis)
1. [Input-to-State Stability (ISS)](#4-input-to-state-stability-iss)
1. [Control Barrier Functions (CBF)](#5-control-barrier-functions-cbf)
1. [Recursive Feasibility](#6-recursive-feasibility)
1. [Swarm-Level Stability](#7-swarm-level-stability)
1. [Running the Analysis](#8-running-the-analysis)
1. [References](#9-references)

---

## 1. System Model for Stability Analysis

### Translational Controllable Subsystem (9-D)

The stability analysis focuses on the controllable part of the state:

$$
\boldsymbol{x}_c = [\boldsymbol{p}(3),\; \boldsymbol{v}(3),\; \boldsymbol{a}(3)]^\top \in \mathbb{R}^9
$$

Discrete-time dynamics:

$$
\boldsymbol{x}_c[k{+}1] = A_9\,\boldsymbol{x}_c[k] + B_9\,\boldsymbol{u}[k]
$$

where $A_9$ and $B_9$ are the $9 \times 9$ and $9 \times 3$ sub-matrices of the full
11-D $A$, $B$ matrices (see [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md)).

### Yaw Decoupling

The yaw sub-state $[\psi, \dot\psi]$ (indices 9–10) is uncontrollable from the
translational input $\boldsymbol{u}$.  Yaw stability is analysed separately as it is
regulated by the geometric attitude controller (Module 8).  The two sub-systems
are therefore treated as decoupled.

### Error Dynamics

Let $\boldsymbol{e}_k = \boldsymbol{x}_c[k] - \boldsymbol{x}_{\text{ref}}[k]$ be the tracking error.
Under the optimal DMPC control the error satisfies approximately:

$$
\boldsymbol{e}[k{+}1] \approx A_{\text{cl}}\,\boldsymbol{e}[k], \quad A_{\text{cl}} = A_9 - B_9\,K_{\text{LQR}}
$$

near the terminal set.

---

## 2. Lyapunov Stability

### Candidate Lyapunov Function

Use the LQR optimal value function:

$$
V(\boldsymbol{e}) = \boldsymbol{e}^\top P\, \boldsymbol{e}
$$

where $P \succ 0$ is the unique symmetric positive-definite solution to the DARE:

$$
P = Q + A^\top P A - A^\top P B\,(R + B^\top P B)^{-1} B^\top P A
$$

### Conditions for Asymptotic Stability

**Condition 1 — Positive Definiteness:**

$$
V(\boldsymbol{e}) > 0 \quad \forall\, \boldsymbol{e} \ne \boldsymbol{0}, \qquad V(\boldsymbol{0}) = 0
$$

This holds since $P \succ 0$.

**Condition 2 — Monotone Decrease:**

Under the LQR gain $K = (R + B^\top P B)^{-1} B^\top P A$:

$$
\Delta V = V(A_{\text{cl}}\,\boldsymbol{e}) - V(\boldsymbol{e})
= \boldsymbol{e}^\top (A_{\text{cl}}^\top P A_{\text{cl}} - P)\,\boldsymbol{e}
= -\boldsymbol{e}^\top (Q + K^\top R K)\,\boldsymbol{e}
< 0 \quad \forall\, \boldsymbol{e} \ne \boldsymbol{0}
$$

The last inequality follows because $Q \succ 0$ and $K^\top R K \succeq 0$.

**Conclusion:** The LQR-closed-loop translational dynamics (under the constant
feedback $\boldsymbol{u} = -K\boldsymbol{e}$ with fixed cost matrices $Q, R$) are **globally
asymptotically stable**.  The DMPC *may* inherit this property through the
terminal cost $P$, provided that (i) the terminal set $\Omega_f$ is explicitly
enforced as a constraint in the QP, (ii) the stage-cost matrices are fixed and
equal to the $(Q, R)$ used to compute $P$, and (iii) the standard MPC
terminal-set assumptions (Rawlings & Mayne, 2019) are satisfied.  When
MAPPO-scaled cost matrices are used or the terminal set is not explicitly
enforced, these guarantees do not apply rigorously (see
[Section 6](#6-recursive-feasibility)).

### Stability Margin

Define the stability margin as:

$$
\text{margin} = 1 - \max\!\left\{
  \frac{\boldsymbol{e}^\top (A_{\text{cl}}^\top P A_{\text{cl}} - P)\,\boldsymbol{e}}
    {\boldsymbol{e}^\top P\,\boldsymbol{e}} : \boldsymbol{e} \ne \boldsymbol{0}
\right\}
$$

Typical value with default parameters: **margin ≈ 0.017**.

---

## 3. Eigenvalue Analysis

### Closed-Loop Eigenvalues

The DMPC is asymptotically stable if and only if all eigenvalues of $A_{\text{cl}}$
satisfy $|\lambda_i| < 1$ (Schur stability).

**Computation:**

```python
K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
A_cl  = A - B @ K_lqr
eigs  = np.linalg.eigvals(A_cl)
rho   = np.max(np.abs(eigs))   # spectral radius
```

**Typical result** (default $Q = I_9$, $R = 0.1 I_3$, $\Delta t = 0.02\;\text{s}$):

| Mode | $\lvert\lambda_i\rvert$ |
| :--- | :--- |
| Position convergence | ≈ 0.942 |
| Velocity convergence | ≈ 0.983 |
| Acceleration convergence | ≈ 0.942 |

All eigenvalues satisfy $|\lambda_i| < 1$; the spectral radius $\rho \approx 0.983$.

### Convergence Rate

The tracking error norm satisfies the bound:

$$
\lVert\boldsymbol{e}[k]\rVert \le C\,\rho^k\,\lVert\boldsymbol{e}[0]\rVert
$$

For $\rho \approx 0.983$, errors halve every:

$$
k_{1/2} = \frac{\ln 2}{-\ln \rho} \approx 40\;\text{steps} = 0.8\;\text{s at 50 Hz}
$$

---

## 4. Input-to-State Stability (ISS)

### Definition

The closed-loop system with disturbance $\boldsymbol{w}[k]$:

$$
\boldsymbol{e}[k{+}1] = A_{\text{cl}}\,\boldsymbol{e}[k] + \boldsymbol{w}[k]
$$

is **Input-to-State Stable (ISS)** if there exist class-$\mathcal{KL}$ function
$\beta$ and class-$\mathcal{K}$ function $\gamma$ such that:

$$
\lVert\boldsymbol{e}[t]\rVert \le \beta(\lVert\boldsymbol{e}[0]\rVert,\, t) + \gamma\!\left(\sup_{0 \le s \le t} \lVert\boldsymbol{w}[s]\rVert\right)
$$

### ISS Gain Bound

For the quadratic Lyapunov function $V(\boldsymbol{e}) = \boldsymbol{e}^\top P\,\boldsymbol{e}$:

$$
\gamma_{\text{iss}} = \sqrt{\frac{\lambda_{\max}(P)}{\lambda_{\min}(P)}}
  \cdot \frac{\lVert A_{\text{cl}}\rVert_2}{1 - \rho}
$$

A finite ISS gain exists because $\rho < 1$ (Schur stability confirmed above).

### Maximum Tolerable Disturbance

For steady-state tracking error below bound $\varepsilon$:

$$
\lVert\boldsymbol{w}\rVert_{\max} = \varepsilon \cdot
  \frac{\lambda_{\min}(Q + K^\top R K)}{\lambda_{\max}(P)}
$$

**Typical result:** With $\varepsilon = 1\;\text{m}$ the system tolerates disturbances up to
≈ 1.3 mm/s² acceleration noise — consistent with typical IMU noise levels.

### Disturbance Sources

| Source | Typical magnitude | DMPC tolerance |
| :--- | :--- | :--- |
| Moderate wind gust | 2–5 m/s velocity | Handled by prediction horizon |
| IMU noise | < 0.01 m/s² | Within ISS bound |
| Sensor fusion lag | 20–50 ms (1–2 steps) | Within ISS bound |
| Formation perturbation | < 2 m | Within collision margin |

---

## 5. Control Barrier Functions (CBF)

### Safety Function

For drones $i$ and $j$, define the safety function:

$$
h_{ij}(\boldsymbol{x}) = \lVert\boldsymbol{p}_i - \boldsymbol{p}_j\rVert^2 - r_{\min}^2
$$

The **safe set** is $\mathcal{C} = \{\boldsymbol{x} : h_{ij}(\boldsymbol{x}) \ge 0\;\forall\,(i,j)\}$.

### Discrete-Time CBF Condition

A discrete-time CBF requires:

$$
h_{ij}(\boldsymbol{x}[k{+}1]) - h_{ij}(\boldsymbol{x}[k]) \ge -\alpha\,h_{ij}(\boldsymbol{x}[k])
$$

for some $\alpha \in (0, 1]$.  This is equivalent to:

$$
h_{ij}(\boldsymbol{x}[k{+}1]) \ge (1 - \alpha)\,h_{ij}(\boldsymbol{x}[k])
$$

$\alpha = 1$ gives the strongest condition: $h_{ij}(\boldsymbol{x}[k{+}1]) \ge 0$, i.e. the
constraint must be satisfied at every time step.

### CBF as an Affine Control Constraint

Linearising $h_{ij}$ around the current position yields an affine constraint
in the control $\boldsymbol{u}$:

Let $\boldsymbol{\delta} = \boldsymbol{p}_i - \boldsymbol{p}_j$,
$h = \lVert\boldsymbol{\delta}\rVert^2 - r_{\min}^2$.

$$
\text{Linearised CBF:} \quad
2\,\boldsymbol{\delta}^\top (\boldsymbol{v} + B_{\text{pos}}\,\boldsymbol{u}) + \alpha\,h \ge 0
$$

$$
\Longrightarrow \quad -2\,\boldsymbol{\delta}^\top B_{\text{pos}}\,\boldsymbol{u}
\le 2\,\boldsymbol{\delta}^\top \boldsymbol{v} + \alpha\,h
$$

where $B_{\text{pos}} = B[0{:}3,\;:]$ extracts the position rows of $B$.

### Forward Invariance Theorem

If $h_{ij}(\boldsymbol{x}_0) \ge 0$ and the **affine CBF control constraint** (the
linearised inequality derived above) is enforced at every step, then by
induction $h_{ij}(\boldsymbol{x}_k) \ge 0$ for all $k \ge 0$ — the swarm remains in
the safe set.

> **Important:** This forward-invariance result applies only when the CBF
> inequality is imposed as an affine constraint on the **control input** $\boldsymbol{u}$.
> It does not follow from imposing a state constraint
> $\lVert\boldsymbol{p}_k - \boldsymbol{p}_j\rVert \ge r_{\min}$ at sampled prediction times, which
> constrains predicted positions but does not directly regulate the control
> input and does not address what happens when the QP is infeasible.  The
> linearised CBF formulation above is the primary mechanism that provides the
> theoretical forward-invariance guarantee; the hard distance state constraint
> in the main DMPC is a complementary heuristic safety mechanism.

### Parameter `cbf_alpha`

`config/dmpc_config.yaml` exposes `stability.cbf_alpha = 0.3`.  Values closer
to 1.0 enforce the barrier more aggressively (better safety, less manoeuvrability).

---

## 6. Recursive Feasibility

### Standard MPC Result

**Theorem (Rawlings & Mayne, 2019):** Let the DMPC be equipped with:

1. Terminal cost $V_f(\boldsymbol{e}) = \boldsymbol{e}^\top P\,\boldsymbol{e}$ (DARE solution).
1. Terminal set $\Omega_f = \{\boldsymbol{e} : \boldsymbol{e}^\top P\,\boldsymbol{e} \le c\}$ (LQR-invariant ellipsoid).
1. Terminal control law $\boldsymbol{u}_f = -K\,\boldsymbol{e}$.

If the problem is **feasible at time $t = 0$**, it remains feasible for all $t > 0$.

*Proof sketch:*  
At time $t{+}1$, the shifted trajectory $\{\tilde{\boldsymbol{x}}_k\} = \{\boldsymbol{x}_{k+1}, \ldots, \boldsymbol{x}_N, A_{\text{cl}}\boldsymbol{x}_N\}$
is a feasible candidate solution (it satisfies all constraints, and
$A_{\text{cl}}\boldsymbol{x}_N \in \Omega_f$ by invariance of $\Omega_f$ under $A_{\text{cl}}$).

> **Implementation caveat:** The theorem requires all three components—terminal
> cost, terminal set constraint, and terminal control law—to be active in the
> optimisation problem, and relies on the cost matrices being fixed and equal to
> the $(Q, R)$ used to compute $P$.  In the current DMPC implementation the
> terminal set $\Omega_f$ is **not** enforced as an explicit QP constraint; $P$
> is used only as a soft terminal penalty.  Additionally, the deployed controller
> uses MAPPO-scaled cost matrices $Q_{\text{eff}}, R_{\text{eff}}$ that vary
> online, so the Bellman identity $A_{\text{cl}}^\top P A_{\text{cl}} \prec P$
> does not hold for the modified stage cost.  Recursive feasibility and
> asymptotic stability are therefore design objectives rather than strict
> mathematical guarantees for the deployed controller.

### Invariance of Terminal Set

The LQR-invariant ellipsoid satisfies:

$$
A_{\text{cl}}^\top P A_{\text{cl}} \prec P
\quad \Longleftrightarrow \quad A_{\text{cl}} \text{ maps } \Omega_f \text{ into } \Omega_f
$$

Numerically verified by checking:

$$
\lambda_{\max}\!\bigl(P^{-1} A_{\text{cl}}^\top P A_{\text{cl}}\bigr) < 1
$$

### Feasibility and Collision Constraints

Collision constraints can break feasibility if a neighbour enters the minimum
separation radius at the start of a time step.  This is prevented by:

1. The formation controller (Module 2) maintaining a minimum gap $> r_{\min}$.
1. The DMPC constraint tightening factor $0.9 \times r_{\min}$ providing a buffer zone.

---

## 7. Swarm-Level Stability

### Distributed Stability

Each drone's DMPC is solved **independently**; neighbours' positions are
treated as fixed obstacles.  This introduces a coupling error due to the
one-step communication delay $\tau$:

$$
\lVert\boldsymbol{p}_j(t) - \hat{\boldsymbol{p}}_j(t)\rVert \le v_{\max} \cdot \tau
$$

For $v_{\max} = 20\;\text{m/s}$ and $\tau = 0.02\;\text{s}$ (50 Hz ROS2 loop), the
coupling error is bounded by **0.4 m** — well inside the 5 m collision radius.

### ADMM Consensus Stability

The ADMM layer (see [10_ADMM_CONSENSUS.md](10_ADMM_CONSENSUS.md)) ensures that
all local DMPC solutions converge to a globally consistent reference.  The
ADMM primal residual decays geometrically:

$$
r_{\text{prim}}^k \le C \cdot \beta^k, \quad \beta \in (0, 1)
$$

providing a quantitative bound on inter-drone trajectory disagreement at each
control step.

### Formation Consensus Stability

The formation controller uses the consensus protocol:

$$
\boldsymbol{v}_i \leftarrow \boldsymbol{v}_i + \varepsilon \sum_{j \in \mathcal{N}(i)} (\boldsymbol{x}_j - \boldsymbol{x}_i - \boldsymbol{d}_{ij})
$$

where $\boldsymbol{d}_{ij}$ is the desired relative position.  Convergence is governed
by the **graph Laplacian** $L$ (see [05_FORMATION_CONSENSUS.md](05_FORMATION_CONSENSUS.md)).
For any connected graph, all non-zero eigenvalues of $L$ are positive, guaranteeing
convergence of the formation error to zero.

---

## 8. Running the Analysis

```python
from isr_rl_dmpc.analysis import DMPCStabilityAnalyzer

analyzer = DMPCStabilityAnalyzer(
    state_dim=9,          # translational sub-system
    control_dim=3,
    dt=0.02,
    collision_radius=5.0,
)

# Full report
report = analyzer.full_stability_report(horizon=20)
print(report.summary)

# Individual checks
lyap = analyzer.check_lyapunov_stability()
eig  = analyzer.check_eigenvalue_stability()
iss  = analyzer.check_iss(error_bound=1.0)
cbf  = analyzer.check_collision_barrier(drone_positions=positions)
rf   = analyzer.check_recursive_feasibility(horizon=20)
```

Expected output for default parameters:

```
============================================================
DMPC Swarm Stability Report
============================================================
  Lyapunov stable     : True  (ΔV_max=-0.917,  margin=0.00274)
  Eigenvalue stable   : True  (ρ=0.982567,  margin=0.017433)
  ISS                 : True  (gain=13.93,   max_dist=0.00130)
  CBF valid           : True  (min_sep=20.00 m, margin=15.00 m)
  Recursive feasible  : True  (horizon_ok=True)
------------------------------------------------------------
  OVERALL STABLE      : True
============================================================
```

---

## 9. References

1. J. B. Rawlings, D. Q. Mayne, and M. Diehl, *Model Predictive Control:
   Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2019.
1. D. Q. Mayne, "Model predictive control: Recent developments and future
   promise," *Automatica*, 50(12):2967–2986, 2014.
1. I. Kolmanovsky and E. G. Gilbert, "Theory and computation of disturbance
   invariant sets for discrete-time linear systems," *Math. Probl. Eng.*,
   4(4):317–367, 1998.
1. A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and
   P. Tabuada, "Control barrier functions: Theory and applications,"
   *Proc. European Control Conference*, 2019.
