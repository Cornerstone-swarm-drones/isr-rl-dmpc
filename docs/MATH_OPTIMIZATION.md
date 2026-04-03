# DMPC Math & Control Optimisation Guide

This document explains how to improve the mathematical performance and real-world
control fidelity of the ISR-DMPC swarm system.  The changes range from
drop-in parameter tweaks to deeper algorithmic enhancements.

---

## Table of Contents

1. [Current Mathematical Formulation](#1-current-mathematical-formulation)
2. [Cost Matrix Tuning](#2-cost-matrix-tuning)
3. [Dynamics Model Improvements](#3-dynamics-model-improvements)
4. [Collision Avoidance — from Soft to CBF](#4-collision-avoidance--from-soft-to-cbf)
5. [Warm-Starting the OSQP Solver](#5-warm-starting-the-osqp-solver)
6. [Distributed Consensus ADMM](#6-distributed-consensus-admm)
7. [Attitude Loop Improvements](#7-attitude-loop-improvements)
8. [Wind & Disturbance Rejection](#8-wind--disturbance-rejection)
9. [State Estimation for Hardware](#9-state-estimation-for-hardware)
10. [Computational Performance](#10-computational-performance)
11. [Summary Checklist](#11-summary-checklist)

---

## 1  Current Mathematical Formulation

The ISR-DMPC controller solves the following finite-horizon QP at every
time step (50 Hz):

```
min   Σ_{k=0}^{N-1} [ ‖x_k − x_ref_k‖²_Q  +  ‖u_k‖²_R ]  +  ‖x_N − x_ref_N‖²_P

s.t.  x_{k+1} = A x_k + B u_k          (constant linearised dynamics)
      ‖u_k‖₂ ≤ u_max                    (acceleration saturation)
      ‖p_k − p_j‖₂ ≥ r_min ∀ j ∈ 𝒩   (collision avoidance, soft)
      x_0 = x(t)                         (initial condition)
```

State vector (dim = 11):  `x = [p(3), v(3), a(3), ψ, ψ̇]`  
Control vector (dim = 3): `u = [ax, ay, az]` (desired acceleration)  
Terminal cost matrix P:   solved from the Discrete Algebraic Riccati Equation (DARE).

**Limitations of the current formulation:**

| Issue | Impact |
|---|---|
| Constant A, B matrices | Ignores drag, inertia coupling, and yaw dynamics |
| Soft collision constraints (0.9× safety margin) | Can be violated when neighbours are near |
| No warm-start between steps | OSQP restarts cold each time step (5-10× slower) |
| Single integrator for yaw | Yaw tracking error accumulates under wind |
| No disturbance observer | Wind causes steady-state position error |

---

## 2  Cost Matrix Tuning

### 2.1  Per-State-Channel Q Weighting

The current `Q = I₁₁` treats position, velocity, acceleration, and yaw
equally.  In practice, position errors are most mission-critical:

```python
# Recommended Q for ISR reconnaissance (state dim = 11)
# [p(3), v(3), a(3), yaw, yaw_rate]
import numpy as np
Q = np.diag([
    10.0, 10.0, 15.0,   # position  (z weighted higher for altitude hold)
    2.0,  2.0,  3.0,    # velocity
    0.5,  0.5,  0.5,    # acceleration (de-emphasise, let R handle)
    5.0,                 # yaw
    0.5,                 # yaw rate
])
```

Pass to `DMPCAgent` via:
```python
agent = DMPCAgent(Q=Q, R=np.eye(3) * 0.15)
```

### 2.2  R Matrix — Control Effort Penalty

Increasing `R` smooths actuator commands and reduces motor wear on hardware.
A diagonal R with slightly higher z-axis penalty reduces abrupt altitude
changes:

```python
R = np.diag([0.12, 0.12, 0.20])
```

### 2.3  Terminal Cost P — DARE vs. Riccati Iteration

The current code computes P via `scipy.linalg.solve_discrete_are` once at
startup.  For hardware flights, re-run the DARE with the updated Q/R above
to get a tighter terminal set.  Alternatively, iterate the value function
offline and cache:

```python
from isr_rl_dmpc.modules.dmpc_controller import compute_lqr_terminal_cost
P = compute_lqr_terminal_cost(state_dim=11, control_dim=3, Q=Q, R=R, dt=0.02)
```

---

## 3  Dynamics Model Improvements

### 3.1  Include Drag in the A Matrix

The hector_quadrotor experiences translational drag ~0.22 × v.  Adding a
first-order drag term to A improves trajectory predictions:

```python
# Modified A matrix with viscous drag coefficient c_d
c_d = 0.22  # from drone_specs.yaml aerodynamics.drag_coefficient (approximate)
dt  = 0.02

A = np.eye(11)
A[0:3, 3:6] = dt * np.eye(3)         # dp/dv
A[3:6, 6:9] = dt * np.eye(3)         # dv/da
A[3:6, 3:6] -= dt * c_d * np.eye(3)  # drag: dv *= (1 - c_d·dt)
```

This keeps the A matrix affine and the QP convex.

### 3.2  Yaw-Coupled Dynamics

At large yaw angles the body-frame acceleration commands produce
world-frame forces that differ from the simplified model.  Add the
rotation matrix coupling in B:

```python
# B with yaw coupling (yaw angle ψ extracted from current state)
psi = x[9]
R_yaw = np.array([[np.cos(psi), -np.sin(psi), 0],
                  [np.sin(psi),  np.cos(psi), 0],
                  [0,            0,           1]])
B_full = np.zeros((11, 3))
B_full[6:9, 0:3] = dt * R_yaw
```

Update `DMPC._get_linearized_dynamics()` in `modules/dmpc_controller.py`
to accept the current yaw angle and return the updated B.

### 3.3  Higher-Order Integrator for Altitude

Replace the simple triple integrator for z with a second-order model
that accounts for hover thrust compensation:

```
ż̈ = az_cmd + g − g = az_cmd   (current, correct for hover)
```

For rapid altitude changes, add feed-forward gravity compensation in the
reference generation (`_build_reference` in `swarm_dmpc_sim_node.py`):

```python
ref[6:9] = np.array([0.0, 0.0, 0.0])  # zero acceleration reference
ref[2]   = target_altitude             # desired altitude
```

---

## 4  Collision Avoidance — from Soft to CBF

### 4.1  Current limitation

The soft constraint `‖p_k − p_j‖ ≥ 0.9 × r_min` can be violated when
the OSQP solver reaches its time budget without a feasible point.

### 4.2  Control Barrier Function (CBF) Constraint

Replace the hard collision constraint with a CBF inequality that is
always feasible and forward-invariant:

```
h(x) = ‖p − p_j‖² − r_min²   ≥ 0

CBF constraint:  ∇h · ẋ + α·h ≥ 0
```

In the linearised MPC context this becomes an affine constraint in `u`:

```python
# For each neighbour j at position p_j, for k = 0:
delta_p = x[0:3] - neighbor_pos
h = np.dot(delta_p, delta_p) - config.collision_radius**2
# Linearised CBF constraint: 2 * delta_p^T * (v + B_pos @ u) + alpha*h >= 0
cbf_A = -2 * delta_p @ B_pos    # shape (3,) × B_pos = shape (control_dim,)
cbf_b = 2 * delta_p @ state[3:6] + cbf_alpha * h
constraints.append(cbf_A @ u_var[:, 0] <= cbf_b)
```

The `stability.cbf_alpha = 0.3` in `config/dmpc_config.yaml` is the
class-K function parameter.  Values closer to 1.0 enforce the barrier
more aggressively (better safety, less manoeuvrability).

### 4.3  Inter-Agent Consensus (Buffered Voronoi Cells)

For large swarms (> 6 drones), replace pairwise constraints with
**Buffered Voronoi Cell** (BVC) decomposition.  Each drone's feasible
position set is its BVC, computed once per control step:

```
BVC_i = { p : ‖p − p_i‖ ≤ ‖p − p_j‖ − r_min, ∀ j ≠ i }
```

This reduces the number of constraints from O(N²) to O(N), maintaining
scalability for larger swarms without sacrificing collision safety.

---

## 5  Warm-Starting the OSQP Solver

CVXPY/OSQP supports warm-starting: if the previous solution is feasible
(or near-feasible), the solver converges in far fewer iterations.

```python
# In MPCSolver.solve():
self.problem.solve(
    solver=cp.OSQP,
    warm_start=True,          # ← add this
    max_iter=3000,
    eps_abs=1e-3,
    eps_rel=1e-3,
    time_limit=self.config.solver_timeout,
)
```

**Expected speedup**: 3–8× on the second and subsequent solves with
similar state/reference, reducing mean solve time from ~8 ms to ~2 ms
for a 4-drone swarm on a Raspberry Pi 4.

To preserve the CVXPY warm-start across calls, **do not** re-instantiate
`cp.Problem` each step (the current code rebuilds it for every new
collision constraint).  Instead, keep a fixed set of collision constraint
parameters and update their values:

```python
# Pre-declare (horizon × n_neighbours) collision constraint parameters
self._nbr_pos_params = [
    [cp.Parameter(3) for _ in range(config.n_neighbors)]
    for _ in range(config.horizon)
]
# Set to a "safe" sentinel (far away) initially; update each solve
for k in range(config.horizon):
    for j in range(config.n_neighbors):
        self._nbr_pos_params[k][j].value = np.array([1e6, 1e6, 1e6])
```

This avoids rebuilding the problem graph each step and enables warm-starting.

---

## 6  Distributed Consensus ADMM

The current architecture solves each drone's QP independently and uses
only the previous time-step positions of neighbours as fixed parameters.
This introduces a one-step lag in the collision constraints.

**ADMM-based DMPC** iterates between local solves and a consensus step:

```
1. Each drone i solves its local QP given shared positions {z_j}.
2. Global consensus: z_j ← average(x_j^(i)) over neighbours.
3. Repeat until ‖x^(i) − z‖ < ε  (typically 3–5 iterations at 50 Hz).
```

Implementation sketch:

```python
# In swarm_dmpc_sim_node._sim_step_callback():
for admm_iter in range(3):
    # 1. Local solve
    u_seqs = [agent.dmpc(states[i], refs[i], shared_pos) for i, agent in enumerate(agents)]
    # 2. Consensus update
    shared_pos = [(states[i][0:3] + states[j][0:3]) / 2 for i, j in neighbor_pairs]
```

ADMM improves constraint satisfaction from ~95% to >99.9% in dense swarms
at the cost of 3× compute per control step.

---

## 7  Attitude Loop Improvements

### 7.1  Feed-Forward Angular Acceleration

The current `AttitudeController.control_loop` uses a pure PD position loop.
Adding a feed-forward angular acceleration term from the DMPC output reduces
tracking latency:

```python
# In AttitudeController.control_loop():
alpha_ff = (u0 - prev_u0) / dt   # finite-difference angular accel feed-forward
tau_ff   = params.inertia @ alpha_ff
tau      = self.controller.control_law(R, omega, R_d) + tau_ff
```

### 7.2  Rate-Limit Filter on Motor Commands

On hardware, large step changes in motor commands cause ESC current spikes.
Apply a first-order filter before writing to MAVROS:

```python
alpha = 0.15   # filter coefficient (higher = faster response)
motor_thrusts_filtered = alpha * motor_thrusts + (1 - alpha) * prev_motor_thrusts
```

### 7.3  Integral Term for Steady-State Error

Under constant wind, the PD position controller accumulates steady-state
error.  Add a conditional integrator (active only when tracking error is
small, preventing integrator wind-up):

```python
if np.linalg.norm(e_p) < 2.0:   # only integrate inside 2 m capture radius
    self._e_p_integral += e_p * dt
    self._e_p_integral = np.clip(self._e_p_integral, -5.0, 5.0)

a_des = a_d - Kp * e_p - Kd * e_v - Ki * self._e_p_integral
```

Recommended gains for hector_quadrotor: `Ki = 0.4`.

---

## 8  Wind & Disturbance Rejection

### 8.1  Disturbance Observer (DOB)

A disturbance observer estimates external forces (wind, payload imbalance)
and feeds them back into the reference acceleration:

```
d̂_{k+1} = d̂_k + L (a_measured − a_predicted)

where a_predicted = u_k + d̂_k  (control + estimated disturbance)
      a_measured  = (v_k − v_{k−1}) / dt  (from IMU/EKF)
```

The observer gain `L ∈ (0, 1)` trades off noise rejection vs. disturbance
tracking speed.  `L = 0.05` is conservative for a 1.5 kg quad.

### 8.2  Dryden Model Integration in Reference Generation

The `WindModel` already implements a Dryden turbulence model.  Connect its
output to the MPC reference:

```python
# In _build_reference():
wind = self._sim.wind_model.update(self._dt)
ref[6:9] = -wind * drag_coefficient   # pre-compensate wind drag in accel reference
```

---

## 9  State Estimation for Hardware

On real hardware the DMPC needs accurate position and velocity.  Recommended
estimation stack:

| Sensor | ROS2 topic | Frequency |
|---|---|---|
| IMU (MPU-6050 or ICM-42688P) | `/imu/data` | 400 Hz |
| GPS RTK (u-blox F9P) | `/mavros/global_position/local` | 10 Hz |
| Optical Flow (PX4Flow) | `/mavros/optical_flow/velocity` | 30 Hz |
| Barometer | `/mavros/altitude` | 25 Hz |

MAVROS fuses these via PX4's EKF2 and publishes:

- **Position + velocity**: `/drone_i/mavros/local_position/pose` +
  `/drone_i/mavros/local_position/velocity_local`
- **Attitude**: `/drone_i/mavros/imu/data`

The `hardware_bridge_node` subscribes to these and reconstructs the
11-dimensional DMPC state vector:

```python
state[0:3]  = pose.position          # from EKF2
state[3:6]  = velocity.linear        # from EKF2
state[6:9]  = accel_body_rotated     # from IMU (body → world via R)
state[9]    = yaw                    # from EKF2 quaternion
state[10]   = angular_velocity.z     # from gyro
```

### Inter-Drone Position Sharing

For collision avoidance DMPC, each drone must know the positions of its
neighbours.  Options ranked by latency:

1. **Onboard UWB ranging** (DecaWave DWM1001, ~10 cm accuracy, ~10 ms latency)
2. **MAVLink HEARTBEAT + GLOBAL_POSITION_INT** messages over mesh radio
3. **ROS2 DDS multicast** (requires low-latency Wi-Fi or private 5G)

---

## 10  Computational Performance

### 10.1  Reduce Problem Size

At 50 Hz with a 20-step horizon, the QP has:
- Decision variables: `11 × 21 + 3 × 20 = 291`
- Constraints: `20 (dynamics) + 20 (saturation) + n_neighbours × 20 (collision)`

**Reducing the horizon to 15** cuts variables to 228 (22% reduction) with
minimal performance loss because the Dryden wind correlation length (200 m
at 10 m/s ≈ 20 s) is much longer than the 0.3 s horizon.

**Condensing** the QP by eliminating the state variables (substituting
x_k = A^k x_0 + … ) reduces variable count from 291 to 60 but makes the
Hessian dense.  Use only if OSQP solve time > 12 ms.

### 10.2  Parallelise with multiprocessing

On a companion computer with 4 cores, run one DMPC process per drone:

```python
from multiprocessing import Pool

def _solve_drone(args):
    agent, state, ref, neighbors = args
    return agent.act(state, ref, neighbor_states=neighbors)

with Pool(processes=n_drones) as pool:
    results = pool.map(_solve_drone, [(agents[i], states[i], refs[i], nbrs[i])
                                       for i in range(n_drones)])
```

### 10.3  Replace CVXPY with a Compiled QP Solver

CVXPY adds overhead from Python-to-C translation.  For sub-2 ms solves,
call OSQP directly:

```python
import osqp, scipy.sparse as sp

# Build P_qp, q_qp, A_qp, l_qp, u_qp  (standard OSQP form)
prob = osqp.OSQP()
prob.setup(P_qp, q_qp, A_qp, l_qp, u_qp,
           warm_starting=True, max_iter=3000,
           eps_abs=1e-3, eps_rel=1e-3, time_limit=0.015)
result = prob.solve()
```

This typically gives a 4–6× speedup over CVXPY at the cost of more
boilerplate to assemble the sparse matrices.

### 10.4  Solver Timing Budget Summary

| Platform | CVXPY+OSQP (current) | OSQP direct | Target for 50 Hz |
|---|---|---|---|
| x86-64 laptop | ~4 ms | ~0.8 ms | < 20 ms ✓ |
| Raspberry Pi 4 | ~18 ms | ~4 ms | < 20 ms ⚠ (marginal — reduce horizon to 12) |
| NVIDIA Jetson Nano | ~9 ms | ~2 ms | < 20 ms ✓ |

For Raspberry Pi 4 with CVXPY, reduce the horizon to 12 or call OSQP directly.

---

## 11  Summary Checklist

Copy this checklist to your experiment log and check off items as you validate
them in simulation before deploying to hardware.

### Quick wins (< 1 day, no architecture change)
- [ ] Tune Q diagonal: position ×10, yaw ×5, acceleration ×0.5
- [ ] Increase R_diag from 0.10 to 0.15 for smoother motor commands
- [ ] Enable OSQP warm-starting (`warm_start=True`)
- [ ] Reduce prediction horizon from 20 to 15 if on Raspberry Pi 4
- [ ] Add drag coefficient to A matrix (`c_d = 0.22`)

### Medium effort (1–3 days)
- [ ] Replace soft collision constraints with CBF constraints
- [ ] Add disturbance observer for wind rejection
- [ ] Add integral term to attitude position loop (gain `Ki = 0.4`)
- [ ] Feed MAVROS EKF2 state into DMPC state vector on hardware

### Advanced (1–2 weeks)
- [ ] Switch from CVXPY to direct OSQP calls for 4–6× speedup
- [ ] Implement ADMM consensus for tighter inter-drone coordination
- [ ] Add UWB ranging for peer position sharing (removes reliance on
  network time-sync for collision avoidance)
- [ ] Implement Buffered Voronoi Cells for O(N) collision constraints
- [ ] Add feed-forward angular acceleration to attitude loop

---

*For stability proofs and eigenvalue analysis of the closed-loop DMPC
system, see [`docs/STABILITY_ANALYSIS.md`](STABILITY_ANALYSIS.md).*
