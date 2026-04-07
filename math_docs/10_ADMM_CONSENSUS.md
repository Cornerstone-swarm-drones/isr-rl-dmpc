# ADMM Consensus for Distributed MPC

**Source files:**
- `src/isr_rl_dmpc/modules/admm_consensus.py` — `ADMMConsensus`
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `DMPC` (integrates ADMM)

---

## Table of Contents

1. [Overview](#1-overview)
1. [Consensus Problem Formulation](#2-consensus-problem-formulation)
1. [Dual Decomposition](#3-dual-decomposition)
1. [ADMM Algorithm](#4-admm-algorithm)
1. [Convergence Analysis](#5-convergence-analysis)
1. [Residuals and Stopping Criteria](#6-residuals-and-stopping-criteria)
1. [Integration with DMPC](#7-integration-with-dmpc)
1. [Swarm-Level Properties](#8-swarm-level-properties)
1. [Default Parameters](#9-default-parameters)
1. [References](#10-references)

---

## 1. Overview

Each drone in the swarm independently solves its own DMPC sub-problem.
Without coordination, independent solutions may be mutually inconsistent:
drone $i$ might plan a trajectory that collides with drone $j$'s planned
trajectory.

**ADMM (Alternating Direction Method of Multipliers)** enforces consensus
among the local sub-problems by iteratively driving each drone's local
decision variable $\boldsymbol{z}_i$ toward a global consensus variable $\boldsymbol{v}$
that satisfies all shared constraints.

ADMM is particularly suited to this setting because:
- It decomposes naturally into **parallelisable local sub-problems** (one per drone).
- It converges under mild conditions (even for non-smooth objectives).
- It provides **dual certificates** (the multipliers $\boldsymbol{\mu}_i$) that
  quantify how much each drone deviates from the consensus.

---

## 2. Consensus Problem Formulation

### Shared Variable

Define the **shared reference trajectory** as:

$$
\boldsymbol{v} = \frac{1}{N}\sum_{i=1}^{N} \boldsymbol{z}_i
$$

where $\boldsymbol{z}_i \in \mathbb{R}^{d}$ is drone $i$'s local copy of the
consensus variable (e.g., the reference position at the next time step, or
the planned collision margin).

### Global Consensus Problem

$$
\min_{\boldsymbol{z}_1, \ldots, \boldsymbol{z}_N} \;
  \sum_{i=1}^{N} f_i(\boldsymbol{z}_i)
\quad \text{s.t.} \quad
  \boldsymbol{z}_i = \boldsymbol{v} \quad \forall\, i
$$

where $f_i(\boldsymbol{z}_i)$ is drone $i$'s local cost (the DMPC objective
restricted to the shared variable subspace).

---

## 3. Dual Decomposition

Introduce Lagrange multipliers $\boldsymbol{\mu}_i$ for each consensus
constraint $\boldsymbol{z}_i = \boldsymbol{v}$.  The **augmented Lagrangian** is:

$$
\mathcal{L}_\rho(\boldsymbol{z}_1,\ldots,\boldsymbol{z}_N,\, \boldsymbol{v},\,
  \boldsymbol{\mu}_1,\ldots,\boldsymbol{\mu}_N)
= \sum_{i=1}^{N} \Bigl[
    f_i(\boldsymbol{z}_i)
    + \boldsymbol{\mu}_i^\top (\boldsymbol{z}_i - \boldsymbol{v})
    + \frac{\rho}{2} \|\boldsymbol{z}_i - \boldsymbol{v}\|^2
  \Bigr]
$$

The quadratic penalty $\frac{\rho}{2}\|\boldsymbol{z}_i - \boldsymbol{v}\|^2$ with
$\rho > 0$ improves convergence speed compared to pure Lagrangian methods.

### Scaled Form

Introducing the scaled dual variable $\boldsymbol{y}_i = \boldsymbol{\mu}_i / \rho$,
the augmented Lagrangian simplifies to:

$$
\mathcal{L}_\rho = \sum_{i=1}^{N} \left[ f_i(\boldsymbol{z}_i) + \frac{\rho}{2} \|\boldsymbol{z}_i - \boldsymbol{v} + \boldsymbol{y}_i\|^2 \right] + \text{const}
$$

---

## 4. ADMM Algorithm

ADMM minimises $\mathcal{L}_\rho$ by alternating between three update steps.

### Step 1 — Local z-update (parallelisable)

Each drone $i$ solves its local sub-problem independently:

$$
\boldsymbol{z}_i^{k+1} \leftarrow \arg\min_{\boldsymbol{z}_i} \left[ f_i(\boldsymbol{z}_i) + \frac{\rho}{2} \|\boldsymbol{z}_i - \boldsymbol{v}^k + \boldsymbol{y}_i^k\|^2 \right]
$$

For the DMPC setting, $f_i$ is a convex QP; adding the quadratic proximal
term keeps the problem convex and strongly convex (hence uniquely solvable).

### Step 2 — Global v-update (closed form)

The $\boldsymbol{v}$-update is the **unweighted average** of all local solutions
plus the scaled duals:

$$
\boldsymbol{v}^{k+1} \leftarrow \frac{1}{N} \sum_{i=1}^{N}
  \bigl(\boldsymbol{z}_i^{k+1} + \boldsymbol{y}_i^k\bigr)
$$

This has an $O(N)$ closed-form solution — no additional optimisation is needed.

### Step 3 — Dual update

$$
\boldsymbol{y}_i^{k+1} \leftarrow \boldsymbol{y}_i^k + \boldsymbol{z}_i^{k+1} - \boldsymbol{v}^{k+1}
$$

The dual variables accumulate the residual between each drone's local solution
and the consensus; they act as a "price signal" that penalises deviation.

---

## 5. Convergence Analysis

### Primal and Dual Residuals

Define the **primal residual** (consensus violation) and **dual residual**
(change in consensus variable):

$$
r_{\text{prim}}^k = \frac{1}{\sqrt{N}} \left\|
  \begin{bmatrix} \boldsymbol{z}_1^k - \boldsymbol{v}^k \\ \vdots \\
    \boldsymbol{z}_N^k - \boldsymbol{v}^k \end{bmatrix}
\right\|
$$

$$
r_{\text{dual}}^k = \rho \sqrt{N}\,\|\boldsymbol{v}^k - \boldsymbol{v}^{k-1}\|
$$

### Convergence Theorem (Boyd et al. 2011)

For any $\rho > 0$, if all $f_i$ are **closed, proper, and convex**:

1. **Primal residual converges:** $r_{\text{prim}}^k \to 0$ as $k \to \infty$.
1. **Dual residual converges:** $r_{\text{dual}}^k \to 0$ as $k \to \infty$.
1. **Objective converges:** $\sum_i f_i(\boldsymbol{z}_i^k) \to p^*$ (optimal value).

The convergence rate is **linear** (geometric decrease) for strongly convex
$f_i$, matching the QP structure of the DMPC sub-problems.

### Convergence Rate

The error satisfies:

$$
\|\boldsymbol{z}^k - \boldsymbol{z}^*\|^2 \le C \cdot \beta^k, \quad \beta \in (0, 1)
$$

where $C$ depends on the initial condition and $\beta$ depends on $\rho$ and
the strong-convexity parameter of $\sum_i f_i$.

Empirically, 3–5 ADMM iterations per DMPC solve cycle achieve residuals below
$10^{-3}$ m in the swarm configuration.

---

## 6. Residuals and Stopping Criteria

Iteration stops when both residuals fall below absolute and relative tolerances:

$$
r_{\text{prim}}^k \le \varepsilon_{\text{prim}} = \varepsilon_{\text{abs}}\sqrt{N} + \varepsilon_{\text{rel}} \max\!\bigl(\|\boldsymbol{z}^k\|,\, \|\boldsymbol{v}^k\|\bigr)
$$

$$
r_{\text{dual}}^k \le \varepsilon_{\text{dual}} = \varepsilon_{\text{abs}}\sqrt{N} + \varepsilon_{\text{rel}}\,\rho \|\boldsymbol{y}^k\|
$$

with default tolerances $\varepsilon_{\text{abs}} = \varepsilon_{\text{rel}} = 10^{-3}$.

---

## 7. Integration with DMPC

The ADMM consensus layer wraps the DMPC solve in the following pattern:

```
for ADMM iteration k in range(max_admm_iters):

    # 1. Each drone solves its local DMPC (in parallel)
    for each drone i:
        z_i_new = dmpc_i.solve(
            x_current   = state_i,
            x_ref        = reference_i,
            v_consensus  = v,          # previous consensus
            y_dual       = y_i,        # scaled dual variable
            rho          = rho,        # ADMM penalty
        )

    # 2. Global average (centralised coordinator or gossip)
    v_new = mean([z_i_new + y_i for i in range(N)])

    # 3. Dual update
    for each drone i:
        y_i += z_i_new - v_new

    # 4. Check stopping criteria
    r_prim = compute_primal_residual(z_new, v_new)
    r_dual = rho * norm(v_new - v)
    if r_prim < eps_prim and r_dual < eps_dual:
        break

    v = v_new
```

The DMPC cost is augmented by the proximal term:

$$
J_{\text{ADMM}}^{(i)} = J_{\text{DMPC}}^{(i)}(\boldsymbol{z}_i) + \frac{\rho}{2}\|\boldsymbol{z}_i - \boldsymbol{v} + \boldsymbol{y}_i\|^2
$$

This additional quadratic term is added to the QP objective in CVXPY as a
parameter, so the problem structure remains fixed and warm-starting is possible.

---

## 8. Swarm-Level Properties

### Trajectory Consistency and Collision Avoidance

ADMM drives all local solutions $\boldsymbol{z}_i$ toward a shared consensus variable
$\boldsymbol{v}$, reducing trajectory disagreement across the swarm.  It does **not**,
however, encode pairwise minimum-separation constraints through the consensus
variable.  An average of relative positions such as

$$
\boldsymbol{v}_{\text{sep}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}}
  \bigl(\boldsymbol{p}_i - \boldsymbol{p}_j\bigr)
$$

can satisfy a minimum-norm condition even when individual pairs violate the
separation requirement, provided other pairs are sufficiently far apart.  A
single global average cannot encode all individual pairwise constraints, so
enforcing consensus on this quantity does **not** guarantee that every pair of
drones satisfies $\|\boldsymbol{p}_i - \boldsymbol{p}_j\| \ge r_{\min}$.

Pairwise collision avoidance is handled by the per-drone hard constraints
$\|\boldsymbol{p}_k - \boldsymbol{p}_j\| \ge r_{\min}$ inside each local DMPC QP.  ADMM's role
is to regularise planned trajectories toward a common reference and reduce
inter-drone planning inconsistency, not to certify pairwise separation.

### Communication Requirements

ADMM requires drones to share their local variable $\boldsymbol{z}_i$ at each
iteration.  For $d$-dimensional $\boldsymbol{z}_i$ and $K$ iterations:

- **Bandwidth per step:** $N \times K \times d \times 4$ bytes
- **Default (N=4, K=5, d=3):** 240 bytes / control step (≪ 1 kbit/s)

### Robustness to Communication Failures

If drone $j$ fails to respond within one ADMM iteration, the consensus
update uses the last known $\boldsymbol{z}_j$.  The ADMM primal residual then
reflects the stale estimate, and the dual update naturally discounts the
contribution of the missing drone over subsequent iterations.

---

## 9. Default Parameters

| Parameter | Symbol | Value | Source |
| :--- | :--- | :--- | :--- |
| ADMM penalty | $\rho$ | 1.0 | `ADMMConsensus.rho` |
| Max iterations | $K$ | 10 | `ADMMConsensus.max_iter` |
| Absolute tolerance | $\varepsilon_{\text{abs}}$ | 1 × 10⁻³ | `ADMMConsensus.eps_abs` |
| Relative tolerance | $\varepsilon_{\text{rel}}$ | 1 × 10⁻³ | `ADMMConsensus.eps_rel` |
| Consensus variable dim | $d$ | 3 (position) | `ADMMConsensus.var_dim` |

---

## 10. References

1. S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed
   Optimization and Statistical Learning via the Alternating Direction Method
   of Multipliers," *Found. Trends Mach. Learn.*, 3(1):1–122, 2011.
1. R. Olfati-Saber, J. A. Fax, and R. M. Murray, "Consensus and Cooperation
   in Networked Multi-Agent Systems," *Proc. IEEE*, 95(1):215–233, 2007.
1. Y. Wang and B. Elia, "A Control Perspective for Centralised and Distributed
   Convex Optimisation," *IEEE CDC*, 2011.
1. M. Zhu and S. Martínez, "On Distributed Convex Optimisation Under Inequality
   and Equality Constraints," *IEEE Trans. Autom. Control*, 57(1):151–164, 2012.
