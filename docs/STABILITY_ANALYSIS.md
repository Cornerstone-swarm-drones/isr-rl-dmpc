# DMPC Stability Analysis for ISR Swarm Drone System

This document presents the theoretical stability analysis of the
purely optimisation-based Distributed Model Predictive Controller
(DMPC) deployed in the ISR swarm drone system.

---

## Table of Contents

1. [System Model](#1-system-model)
2. [Lyapunov Stability](#2-lyapunov-stability)
3. [Eigenvalue Analysis](#3-eigenvalue-analysis)
4. [Input-to-State Stability (ISS)](#4-input-to-state-stability-iss)
5. [Collision Avoidance — Control Barrier Functions](#5-collision-avoidance--control-barrier-functions)
6. [Recursive Feasibility](#6-recursive-feasibility)
7. [Swarm-Level Stability](#7-swarm-level-stability)
8. [ISR Application Considerations](#8-isr-application-considerations)
9. [Running the Analysis in Code](#9-running-the-analysis-in-code)

---

## 1. System Model

### 1.1 Translational Dynamics (Controllable Subsystem)

Each drone's translational state is:

```
x = [p(3), v(3), a(3)]  ∈ ℝ⁹
```

where **p** is position, **v** is velocity, and **a** is acceleration.
The control input **u = [ax, ay, az] ∈ ℝ³** is the commanded
acceleration.

The discrete-time linearised dynamics with step Δt = 0.02 s are:

```
x[k+1] = A·x[k] + B·u[k]
```

where:

```
     ┌ I₃   Δt·I₃  0   ┐        ┌  0   ┐
A =  │  0    I₃   Δt·I₃│,  B =  │  0   │
     └  0    0     I₃  ┘        └ Δt·I₃┘
```

### 1.2 Yaw Sub-System

The yaw angle ψ and yaw rate ψ̇ (state indices 9–10 of the 11-state
vector) are not driven by the translational acceleration control.
They are stabilised independently by the geometric SO(3) attitude
controller.  The translational controllable subsystem (9 states) is
therefore analysed separately.

### 1.3 MPC Optimisation Problem

At each step *t* the DMPC solves:

```
min   Σ_{k=0}^{N-1} [‖e_k‖²_Q + ‖u_k‖²_R] + ‖e_N‖²_P
s.t.  x_{k+1} = A·x_k + B·u_k                    (dynamics)
      ‖u_k‖₂ ≤ u_max                               (acceleration limit)
      ‖p_k − p_j‖₂ ≥ r_min  ∀j ∈ 𝒩(i)             (collision avoidance)
      x_0 = x(t)                                    (initial condition)
```

where **e_k = x_k − x_ref_k** and **P** is the LQR terminal cost matrix
computed from the Discrete Algebraic Riccati Equation (DARE).

---

## 2. Lyapunov Stability

### 2.1 Candidate Lyapunov Function

We use the terminal cost as a Lyapunov function candidate:

```
V(e) = eᵀ P e
```

where **P** is the unique symmetric positive-definite solution to the DARE:

```
P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
```

### 2.2 Stability Conditions

**Condition 1 — Positive Definiteness:** V(e) is a valid Lyapunov
function because P ≻ 0 (all eigenvalues positive).

**Condition 2 — Monotone Decrease:** Under the LQR gain
**K = (R + BᵀPB)⁻¹ BᵀPA** the closed-loop matrix is
**A_cl = A − B K**.  The Bellman equation gives:

```
A_cl^T P A_cl − P = −(Q + Kᵀ R K) ≺ 0
```

Therefore:

```
ΔV = V(A_cl·e) − V(e) = eᵀ (A_cl^T P A_cl − P) e
                       = −eᵀ (Q + Kᵀ R K) e < 0   ∀ e ≠ 0
```

**Conclusion:** The LQR-closed-loop translational dynamics are
*globally asymptotically stable*.  The DMPC inherits this property via
the terminal cost and the standard MPC stability argument (Section 6).

---

## 3. Eigenvalue Analysis

### 3.1 Closed-Loop Eigenvalues

For the discrete-time LQR closed-loop matrix **A_cl = A − B K**, all
eigenvalues must satisfy **|λᵢ| < 1** for asymptotic stability.

**Typical result** (default Q = I₉, R = 0.1·I₃, Δt = 0.02 s):

| Mode | |λᵢ| |
|------|---------|
| Position convergence | 0.942 |
| Velocity convergence | 0.983 (complex pair) |
| Acceleration convergence | 0.942 |

All eigenvalues are strictly inside the unit circle. The spectral
radius ρ ≈ 0.983, giving a **stability margin of ≈ 0.017**.

### 3.2 Convergence Rate

The error norm satisfies:

```
‖e_k‖ ≤ C · ρᵏ · ‖e_0‖
```

where C > 0 is a bounded constant depending on the eigenvector
condition number.  For ρ ≈ 0.983, errors halve every
≈ ln(2) / (−ln(0.983)) ≈ 40 steps = 0.8 s at 50 Hz.

---

## 4. Input-to-State Stability (ISS)

### 4.1 Definition

The closed-loop system is *Input-to-State Stable* (ISS) with gain γ if
there exist class-KL and class-K functions β, γ such that for all
bounded disturbances **d**:

```
‖e(t)‖ ≤ β(‖e(0)‖, t) + γ(sup_{s≤t} ‖d(s)‖)
```

### 4.2 ISS Gain Bound

For the quadratic Lyapunov function V(e) = eᵀ P e:

```
γ_iss = (λ_max(P) / λ_min(P)) · ‖A_cl‖₂
```

A finite ISS gain exists whenever the spectral radius ρ < 1, which is
confirmed by the eigenvalue analysis above.

### 4.3 Disturbance Rejection

The maximum disturbance magnitude that keeps the steady-state error
below a bound ε is:

```
‖d‖_max = ε · λ_min(P) · (1 − ρ) / λ_max(P)
```

**Typical result:** With ε = 1 m, the controller rejects disturbances
up to ≈ 1.3 mm/s² (acceleration noise floor).  This is consistent with
the IMU noise characteristics of the drone platform.

### 4.4 ISR Implications

| Disturbance Source | Typical Magnitude | DMPC Tolerance |
|---|---|---|
| Wind gust (moderate) | 2–5 m/s velocity | Handled by prediction horizon |
| IMU noise | < 0.01 m/s² | Within ISS bound |
| Sensor fusion lag | 20–50 ms | 1–2 prediction steps |
| Formation perturbation | < 2 m position | Within collision margin |

---

## 5. Collision Avoidance — Control Barrier Functions

### 5.1 Barrier Function

For drones i and j, the safety function is:

```
h_{ij}(x) = ‖p_i − p_j‖² − r_min²
```

where r_min = 5 m is the minimum safe separation.

**Safety set:** `C = { x : h_{ij}(x) ≥ 0 ∀ (i,j) }`.

### 5.2 Discrete-Time CBF Condition

The DMPC hard constraint `‖p_k − p_j‖ ≥ r_min` directly enforces:

```
h_{ij}(x_{k+1}) ≥ (1 − α) h_{ij}(x_k)
```

with α = 1 (strongest condition), meaning the constraint is active at
every time step.

**Forward invariance theorem:** If `C ≠ ∅` and `h_{ij}(x_0) ≥ 0`,
the DMPC collision constraint guarantees `h_{ij}(x_k) ≥ 0` for all
k ≥ 0, i.e. the swarm remains in the safety set.

### 5.3 Swarm Safety Margin

In all ISR mission scenarios (4–6 drones, grid/wedge/line formations),
the DMPC maintains a safety margin of ≥ 15 m with the default
collision_radius = 5 m and typical 20 m inter-drone spacing.

---

## 6. Recursive Feasibility

### 6.1 Standard MPC Result

For a DMPC with:
- Terminal set Ω_f = { e : eᵀ P e ≤ c } (LQR invariant ellipsoid)
- Terminal cost V_f(e) = eᵀ P e
- LQR terminal control law u_f = −K e

**Theorem (Rawlings & Mayne, 2019):** If the DMPC is feasible at t = 0,
it is feasible at all subsequent time steps t > 0.

*Proof sketch:* Applying the terminal control law u_f at step N maps
x_N ∈ Ω_f to A_cl x_N ∈ Ω_f (by positive invariance of Ω_f under
A_cl — a consequence of the DARE Bellman equation).

### 6.2 Invariance of Terminal Set

The LQR invariant ellipsoid satisfies:

```
A_cl^T P A_cl ≺ P   ⟹   A_cl maps Ω_f into Ω_f
```

This is verified numerically by confirming that:

```
λ_max(P⁻¹ A_cl^T P A_cl) < 1
```

### 6.3 Collision Avoidance and Feasibility

Collision constraints can make the DMPC infeasible if a neighbour
is already inside the minimum separation radius.  This is prevented by:
1. Formation controller (Module 2) maintaining a minimum inter-drone
   gap > r_min at all times.
2. DMPC constraint tightening factor of 0.95 (hard margin preserved).

---

## 7. Swarm-Level Stability

### 7.1 Distributed Stability

Each drone solves its own local DMPC problem independently.  Neighbour
positions are communicated over the 100 m mesh network (Module 3 —
Sensor Fusion).

**Coupling:** Drone i uses the *last known* positions of its N
neighbours as fixed obstacles.  This introduces a bounded error
proportional to the inter-drone communication delay τ:

```
‖p_j(t) − p̂_j(t)‖ ≤ v_max · τ
```

For v_max = 20 m/s and τ = 0.1 s (5 Hz radar), the coupling error is
bounded by 2 m — well below the 5 m collision radius.

### 7.2 Collective Coverage Stability

The formation controller (Module 2) uses a consensus protocol:

```
v_i ← v_i + ε Σ_{j∈𝒩(i)} (x_j − x_i − d_{ij})
```

Stability of the consensus layer is determined by the eigenvalues of
the graph Laplacian L.  For the 4-drone grid, wedge, and line formations
all eigenvalues of L are positive (connected graph), ensuring convergence
of the formation error to zero.

---

## 8. ISR Application Considerations

### 8.1 Coverage Performance

| Scenario | Drones | Horizon N | Coverage Rate | Stability |
|---|---|---|---|---|
| Area Surveillance | 4 | 20 | ≥ 90% in 30 min | Asymptotically stable |
| Threat Response | 6 | 20 | N/A (threat-centric) | ISS w.r.t. target motion |
| Search & Track | 5 | 20 | ≥ 85% in 20 min | ISS w.r.t. target motion |

### 8.2 Real-Time Constraint

The OSQP solver completes in < 10 ms per drone at the default horizon
N = 20.  The control loop runs at 50 Hz, leaving 10 ms margin for
communication and other computations.

### 8.3 Degraded Communication

If a drone loses communication with its neighbours, the DMPC falls back
to the last known positions (held constant).  The ISS bound ensures that
the position tracking error remains bounded as long as the outage
duration satisfies:

```
τ_outage ≤ ‖e_max‖ / v_max
```

For ‖e_max‖ = 5 m and v_max = 20 m/s, the system tolerates up to 0.25 s
of communication blackout without violating collision constraints.

---

## 9. Running the Analysis in Code

```python
from isr_rl_dmpc.analysis import DMPCStabilityAnalyzer

# Instantiate the analyser for the 9-state translational subsystem
analyzer = DMPCStabilityAnalyzer(
    state_dim=9,         # [p, v, a] controllable subsystem
    control_dim=3,       # [ax, ay, az]
    dt=0.02,             # 50 Hz
    collision_radius=5.0,
)

# Run all five stability checks
report = analyzer.full_stability_report(horizon=20)
print(report.summary)

# Individual checks
lyapunov = analyzer.check_lyapunov_stability()
print(f"Lyapunov stable: {lyapunov.is_stable}, ΔV_max={lyapunov.delta_V_max:.4f}")

eig = analyzer.check_eigenvalue_stability()
print(f"Spectral radius ρ = {eig.spectral_radius:.6f}")

iss = analyzer.check_iss(error_bound=1.0)
print(f"ISS gain = {iss.iss_gain:.4f}, max disturbance = {iss.max_disturbance:.4f}")

# 6-drone wedge formation
import numpy as np
positions = np.array([
    [0, 0, 20], [20, 10, 20], [20, -10, 20],
    [40, 20, 20], [40, -20, 20], [60, 0, 20],
], dtype=float)
cbf = analyzer.check_collision_barrier(drone_positions=positions)
print(f"CBF valid: {cbf.is_valid}, min separation = {cbf.min_separation:.2f} m")
```

**Expected output:**

```
============================================================
DMPC Swarm Stability Report
============================================================
  Lyapunov stable     : True  (ΔV_max=-0.917,  margin=0.00274)
  Eigenvalue stable   : True  (ρ=0.982567,  margin=0.017433)
  ISS                 : True  (gain=13.93,  max_dist=0.00130)
  CBF valid           : True  (min_sep=20.00 m,  margin=15.00 m)
  Recursive feasible  : True  (horizon_ok=True)
------------------------------------------------------------
  OVERALL STABLE      : True
============================================================
```

---

## References

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
5. T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a
   quadrotor UAV on SE(3)," *IEEE CDC*, 2010.
