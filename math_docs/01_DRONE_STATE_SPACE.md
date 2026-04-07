# Drone State Space and Linearised Dynamics

**Source files:**
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `DMPC._get_linearized_dynamics()`
- `src/isr_rl_dmpc/core/data_structures.py` — `DroneState`, `TargetState`

---

## Table of Contents

1. [Overview](#1-overview)
2. [DMPC State Vector (11-D)](#2-dmpc-state-vector-11-d)
3. [Estimation State Vector (18-D)](#3-estimation-state-vector-18-d)
4. [Target State Vector (11-D)](#4-target-state-vector-11-d)
5. [Quadrotor Translational Dynamics](#5-quadrotor-translational-dynamics)
6. [Discrete-Time Linearised Dynamics](#6-discrete-time-linearised-dynamics)
7. [A and B Matrices](#7-a-and-b-matrices)
8. [Yaw Sub-System](#8-yaw-sub-system)
9. [Controllability](#9-controllability)

---

## 1  Overview

The system uses **two different state representations** for two different purposes:

| Purpose | Dimension | State components |
|---------|-----------|-----------------|
| DMPC control optimisation | **11-D** | position, velocity, acceleration, yaw, yaw-rate |
| Sensor-fusion / EKF estimation | **18-D** | position, velocity, acceleration, quaternion, angular velocity, battery, health |

The 11-D state is the minimal representation required by the DMPC QP solver.
The 18-D state is the full physical state tracked by the Extended Kalman Filters.

---

## 2  DMPC State Vector (11-D)

```
x = [ p_x, p_y, p_z,        (indices 0–2, position in m)
      v_x, v_y, v_z,        (indices 3–5, velocity in m/s)
      a_x, a_y, a_z,        (indices 6–8, acceleration in m/s²)
      ψ,                    (index  9,   yaw angle in rad)
      ψ̇  ]                  (index 10,  yaw rate in rad/s)
```

The translational sub-state `[p, v, a] ∈ ℝ⁹` is **controllable** via the
acceleration input `u = [aₓ, a_y, a_z] ∈ ℝ³`.  The yaw sub-state `[ψ, ψ̇]`
is regulated independently by the geometric attitude controller (Module 8).

### Why include acceleration?

Incorporating acceleration as a state variable (rather than treating it as a
pure input) provides the MPC with inertia-awareness: the controller can
anticipate how current acceleration commands will evolve position and velocity
over the prediction horizon, giving smoother trajectories.

---

## 3  Estimation State Vector (18-D)

```
x_est = [ p_x, p_y, p_z,         (indices  0–2,  position in m)
          v_x, v_y, v_z,         (indices  3–5,  velocity in m/s)
          a_x, a_y, a_z,         (indices  6–8,  acceleration in m/s²)
          q_w, q_x, q_y, q_z,   (indices  9–12, unit quaternion)
          ω_x, ω_y, ω_z,        (indices 13–15, angular velocity in rad/s)
          E_batt,                (index  16,    battery energy in Wh)
          h  ]                   (index  17,    structural health ∈ [0,1])
```

This richer vector is used by the `DroneStateEstimator` (three parallel EKFs)
to track the full rigid-body state.  The first 11 components overlap with the
DMPC state (using Euler yaw extracted from the quaternion for index 9, and
ω_z for index 10).

---

## 4  Target State Vector (11-D)

```
x_tgt = [ p_x, p_y, p_z,        (indices 0–2,  position in m)
          v_x, v_y, v_z,        (indices 3–5,  velocity in m/s)
          a_x, a_y, a_z,        (indices 6–8,  acceleration in m/s²)
          ψ,                    (index  9,  yaw angle in rad)
          ψ̇  ]                  (index 10, yaw rate in rad/s)
```

This mirrors the DMPC drone state; the same EKF structure is re-used for
target tracking (see `core/target_state_estimation.py`).

---

## 5  Quadrotor Translational Dynamics

A quadrotor's translational dynamics in the world frame, neglecting drag and
disturbances, are:

```
p̈ = (T / m) R e₃ − g e₃
```

where:
- **p** ∈ ℝ³  — position  
- **T** — total thrust (N)  
- **m** — vehicle mass (1.477 kg for hector_quadrotor)  
- **R** ∈ SO(3) — rotation matrix (body → world)  
- **e₃** = [0, 0, 1]ᵀ — unit z-vector  
- **g** = 9.81 m/s²  

For small angles (near-hover), `R e₃ ≈ e₃` and the net body thrust minus
gravity cancels.  The **commanded acceleration** `u ∈ ℝ³` is:

```
u = p̈_cmd  =  a_des  (horizontal x, y)  +  a_z_des  (vertical)
```

The attitude controller (Module 8) inverts the thrust to realise the desired
translational acceleration; the DMPC only generates `u = [aₓ, a_y, a_z]`.

---

## 6  Discrete-Time Linearised Dynamics

The continuous-time triple-integrator model

```
ṗ = v
v̇ = a
ȧ = u
```

is discretised with Euler forward integration at step Δt:

```
p[k+1] = p[k] + Δt · v[k]
v[k+1] = v[k] + Δt · a[k]
a[k+1] = a[k] + Δt · u[k]
```

Written compactly as `x[k+1] = A x[k] + B u[k]` using only the
9-D translational sub-state `[p, v, a]`:

---

## 7  A and B Matrices

For the **full 11-D DMPC state** (translational + yaw):

```
     ┌ I₃   ΔtI₃   0     0   0 ┐
     │  0    I₃   ΔtI₃   0   0 │
A =  │  0    0     I₃    0   0 │   ∈ ℝ¹¹ˣ¹¹
     │  0    0     0     1  Δt │
     └  0    0     0     0   1 ┘
```

```
     ┌  0  ┐
     │  0  │
B =  │ ΔtI₃│   ∈ ℝ¹¹ˣ³
     │  0  │
     └  0  ┘
```

where **I₃** is the 3×3 identity matrix.

### Block interpretation

| Block | Rows | Cols | Meaning |
|-------|------|------|---------|
| `A[0:3, 3:6] = ΔtI₃` | position | velocity | `p += Δt·v` |
| `A[3:6, 6:9] = ΔtI₃` | velocity | acceleration | `v += Δt·a` |
| `A[9, 10] = Δt` | yaw | yaw-rate | `ψ += Δt·ψ̇` |
| `B[6:9, 0:3] = ΔtI₃` | acceleration | input | `a += Δt·u` |

The yaw dynamics in `A[9:11, 9:11]` form a 2×2 integrator; yaw is not
driven by the translational input `u`, so `B[9:11, :] = 0`.

### Python code (from `dmpc_controller.py`)

```python
A = np.eye(11)
A[0:3, 3:6] = dt * np.eye(3)   # dp/dv
A[3:6, 6:9] = dt * np.eye(3)   # dv/da

B = np.zeros((11, 3))
B[6:9, 0:3] = dt * np.eye(3)   # da/du
```

---

## 8  Yaw Sub-System

The yaw angle ψ and yaw rate ψ̇ evolve as a **decoupled 2-D linear system**:

```
[ ψ[k+1]  ]   [ 1  Δt ] [ ψ[k]  ]
[ ψ̇[k+1] ] = [ 0   1 ] [ ψ̇[k] ]
```

Because the translational control input **u** does not appear in these
equations, yaw is uncontrollable from the DMPC perspective.  Yaw is instead
regulated by the geometric attitude controller through a separate torque
command τᵤ.

---

## 9  Controllability

**Theorem:** The pair (A₉, B₉) — the 9-D translational sub-system — is
**completely controllable**.

*Proof sketch:* The controllability matrix

```
C = [B₉  A₉B₉  A₉²B₉  ···  A₉⁸B₉] ∈ ℝ⁹ˣ²⁷
```

has rank 9.  This follows because the triple-integrator chain
`p ← v ← a ← u` is in Brunovsky canonical form, which is always
controllable for any Δt > 0.

**Implication:** There exists a stabilising LQR gain K such that the
closed-loop matrix `A_cl = A − BK` is Schur stable (all eigenvalues
strictly inside the unit circle), which is the foundation for the DARE
terminal cost derivation in the DMPC (see
[03_DMPC_FORMULATION.md](03_DMPC_FORMULATION.md)).
