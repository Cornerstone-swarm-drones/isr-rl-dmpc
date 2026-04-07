# Lyapunov Stability, ISS, CBF and Recursive Feasibility

**Source files:**
- `src/isr_rl_dmpc/analysis/stability_analysis.py` — `DMPCStabilityAnalyzer`
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `compute_lqr_terminal_cost`

---

## Table of Contents

1. [System Model for Stability Analysis](#1-system-model-for-stability-analysis)
2. [Lyapunov Stability](#2-lyapunov-stability)
3. [Eigenvalue Analysis](#3-eigenvalue-analysis)
4. [Input-to-State Stability (ISS)](#4-input-to-state-stability-iss)
5. [Control Barrier Functions (CBF)](#5-control-barrier-functions-cbf)
6. [Recursive Feasibility](#6-recursive-feasibility)
7. [Swarm-Level Stability](#7-swarm-level-stability)
8. [Running the Analysis](#8-running-the-analysis)
9. [References](#9-references)

---

## 1  System Model for Stability Analysis

### Translational Controllable Subsystem (9-D)

The stability analysis focuses on the controllable part of the state:

```
x_c = [p(3), v(3), a(3)] ∈ ℝ⁹
```

Discrete-time dynamics:

```
x_c[k+1] = A₉ x_c[k] + B₉ u[k]
```

where A₉ and B₉ are the 9×9 and 9×3 sub-matrices of the full 11-D A, B
matrices (see [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md)).

### Yaw Decoupling

The yaw sub-state `[ψ, ψ̇]` (indices 9–10) is uncontrollable from the
translational input **u**.  Yaw stability is analysed separately as it is
regulated by the geometric attitude controller (Module 8).  The two
sub-systems are therefore treated as decoupled.

### Error Dynamics

Let `e_k = x_c[k] − x_ref[k]` be the tracking error.  Under the optimal
DMPC control the error satisfies approximately:

```
e[k+1] ≈ A_cl e[k]   where  A_cl = A₉ − B₉ K_LQR
```

near the terminal set.

---

## 2  Lyapunov Stability

### Candidate Lyapunov Function

Use the LQR optimal value function:

```
V(e) = eᵀ P e
```

where **P ≻ 0** is the unique symmetric positive-definite solution to the
DARE:

```
P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
```

### Conditions for Asymptotic Stability

**Condition 1 — Positive Definiteness:**

```
V(e) > 0  ∀ e ≠ 0  (holds since P ≻ 0)
V(0) = 0
```

**Condition 2 — Monotone Decrease:**

Under the LQR gain `K = (R + BᵀPB)⁻¹ BᵀPA`:

```
ΔV = V(A_cl e) − V(e)
   = eᵀ (A_clᵀ P A_cl − P) e
   = −eᵀ (Q + Kᵀ R K) e
   < 0   ∀ e ≠ 0
```

The last inequality follows because `Q ≻ 0` and `Kᵀ R K ≽ 0`.

**Conclusion:** The LQR-closed-loop translational dynamics are **globally
asymptotically stable**.  The DMPC controller inherits this stability guarantee
within the terminal set Ω_f via the terminal cost (see
[Section 6](#6-recursive-feasibility)).

### Stability Margin

Define the stability margin as:

```
margin = 1 − max{eᵀ (A_clᵀ P A_cl − P) e / (eᵀ P e) : e ≠ 0}
```

Typical value with default parameters: **margin ≈ 0.017**.

---

## 3  Eigenvalue Analysis

### Closed-Loop Eigenvalues

The DMPC is asymptotically stable if and only if all eigenvalues of A_cl
satisfy `|λᵢ| < 1` (Schur stability).

**Computation:**

```python
K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
A_cl  = A - B @ K_lqr
eigs  = np.linalg.eigvals(A_cl)
rho   = np.max(np.abs(eigs))   # spectral radius
```

**Typical result** (default Q = I₉, R = 0.1·I₃, Δt = 0.02 s):

| Mode | |λᵢ| |
|------|---------|
| Position convergence | ≈ 0.942 |
| Velocity convergence | ≈ 0.983 |
| Acceleration convergence | ≈ 0.942 |

All eigenvalues satisfy `|λᵢ| < 1`; the spectral radius ρ ≈ 0.983.

### Convergence Rate

The tracking error norm satisfies the bound:

```
‖e[k]‖ ≤ C ρᵏ ‖e[0]‖
```

For ρ ≈ 0.983, errors halve every:

```
k_{1/2} = ln(2) / (−ln ρ) ≈ 40 steps = 0.8 s  at 50 Hz
```

---

## 4  Input-to-State Stability (ISS)

### Definition

The closed-loop system with disturbance `d[k]`:

```
e[k+1] = A_cl e[k] + w[k],   w = disturbance
```

is **Input-to-State Stable (ISS)** if there exist class-KL function β and
class-K function γ such that:

```
‖e[t]‖ ≤ β(‖e[0]‖, t) + γ(sup_{0≤s≤t} ‖w[s]‖)
```

### ISS Gain Bound

For the quadratic Lyapunov function `V(e) = eᵀ P e`:

```
γ_iss = sqrt(λ_max(P) / λ_min(P))  ×  ‖A_cl‖₂ / (1 − ρ)
```

A finite ISS gain exists because ρ < 1 (Schur stability confirmed above).

### Maximum Tolerable Disturbance

For steady-state tracking error below bound ε:

```
‖w‖_max = ε · λ_min(Q + KᵀRK) / λ_max(P)
```

**Typical result:** With ε = 1 m the system tolerates disturbances up to
≈ 1.3 mm/s² acceleration noise — consistent with typical IMU noise levels.

### Disturbance Sources

| Source | Typical magnitude | DMPC tolerance |
|--------|------------------|----------------|
| Moderate wind gust | 2–5 m/s velocity | Handled by prediction horizon |
| IMU noise | < 0.01 m/s² | Within ISS bound |
| Sensor fusion lag | 20–50 ms (1–2 steps) | Within ISS bound |
| Formation perturbation | < 2 m | Within collision margin |

---

## 5  Control Barrier Functions (CBF)

### Safety Function

For drones **i** and **j**, define the safety function:

```
h_{ij}(x) = ‖p_i − p_j‖² − r_min²
```

The **safe set** is `C = { x : h_{ij}(x) ≥ 0 ∀ (i,j) }`.

### Discrete-Time CBF Condition

A discrete-time CBF requires:

```
h_{ij}(x[k+1]) − h_{ij}(x[k]) ≥ −α h_{ij}(x[k])
```

for some `α ∈ (0, 1]`.  This is equivalent to:

```
h_{ij}(x[k+1]) ≥ (1 − α) h_{ij}(x[k])
```

**α = 1** gives the strongest condition: `h_{ij}(x[k+1]) ≥ 0`, i.e. the
constraint must be satisfied at every time step.

### CBF as an Affine Control Constraint

Linearising `h_{ij}` around the current position yields an affine constraint
in the control `u`:

```
Let δ = p_i − p_j,  h = ‖δ‖² − r_min²

Linearised CBF: 2 δᵀ (v + B_pos u) + α h ≥ 0

→  −2 δᵀ B_pos u ≤ 2 δᵀ v + α h
```

where `B_pos = B[0:3, :]` extracts the position rows of B.

### Forward Invariance Theorem

If `h_{ij}(x_0) ≥ 0` and the CBF constraint is enforced at every step, then
by induction `h_{ij}(x_k) ≥ 0` for all `k ≥ 0` — the swarm remains in the
safe set.

### Parameter `cbf_alpha`

`config/dmpc_config.yaml` exposes `stability.cbf_alpha = 0.3`.  Values closer
to 1.0 enforce the barrier more aggressively (better safety, less
manoeuvrability).

---

## 6  Recursive Feasibility

### Standard MPC Result

**Theorem (Rawlings & Mayne, 2019):** Let the DMPC be equipped with:

1. Terminal cost `V_f(e) = eᵀ P e` (DARE solution).
2. Terminal set `Ω_f = { e : eᵀ P e ≤ c }` (LQR-invariant ellipsoid).
3. Terminal control law `u_f = −K e`.

If the problem is **feasible at time t = 0**, it remains feasible for all
t > 0.

*Proof sketch:*  
At time **t+1**, the shifted trajectory `{x̃_k} = {x_k+1, …, x_N, A_cl x_N}`
is a feasible candidate solution (it satisfies all constraints, and
`A_cl x_N ∈ Ω_f` by invariance of Ω_f under A_cl).

### Invariance of Terminal Set

The LQR-invariant ellipsoid satisfies:

```
A_cl^T P A_cl ≺ P   ⟺   A_cl maps Ω_f into Ω_f
```

Numerically verified by checking:

```
λ_max( P⁻¹ A_clᵀ P A_cl ) < 1
```

### Feasibility and Collision Constraints

Collision constraints can break feasibility if a neighbour enters the minimum
separation radius at the start of a time step.  This is prevented by:

1. The formation controller (Module 2) maintaining a minimum gap > r_min.
2. The DMPC constraint tightening factor 0.9 × r_min providing a buffer zone.

---

## 7  Swarm-Level Stability

### Distributed Stability

Each drone's DMPC is solved **independently**; neighbours' positions are
treated as fixed obstacles.  This introduces a coupling error due to the
one-step communication delay τ:

```
‖p_j(t) − p̂_j(t)‖ ≤ v_max · τ
```

For `v_max = 20 m/s` and `τ = 0.02 s` (50 Hz ROS2 loop), the coupling error
is bounded by **0.4 m** — well inside the 5 m collision radius.

### Formation Consensus Stability

The formation controller uses the consensus protocol:

```
v_i ← v_i + ε Σ_{j ∈ 𝒩(i)} (x_j − x_i − d_{ij})
```

where `d_{ij}` is the desired relative position.  The convergence of this
protocol is governed by the **graph Laplacian L** of the communication graph
(see [05_FORMATION_CONSENSUS.md](05_FORMATION_CONSENSUS.md)).  For any
connected graph, all non-zero eigenvalues of L are positive, guaranteeing
convergence of the formation error to zero.

---

## 8  Running the Analysis

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

## 9  References

1. J. B. Rawlings, D. Q. Mayne, and M. Diehl, *Model Predictive Control:
   Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2019.
2. D. Q. Mayne, "Model predictive control: Recent developments and future
   promise," *Automatica*, 50(12):2967–2986, 2014.
3. I. Kolmanovsky and E. G. Gilbert, "Theory and computation of disturbance
   invariant sets for discrete-time linear systems," *Math. Probl. Eng.*,
   4(4):317–367, 1998.
4. A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and
   P. Tabuada, "Control barrier functions: Theory and applications,"
   *Proc. European Control Conference*, 2019.
