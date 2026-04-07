# Drone State Space and Linearised Dynamics

**Source files:**
- `src/isr_rl_dmpc/modules/dmpc_controller.py` — `DMPC._get_linearized_dynamics()`
- `src/isr_rl_dmpc/core/data_structures.py` — `DroneState`, `TargetState`

---

## Table of Contents

1. [Overview](#1-overview)
1. [DMPC State Vector (11-D)](#2-dmpc-state-vector-11-d)
1. [Estimation State Vector (18-D)](#3-estimation-state-vector-18-d)
1. [Target State Vector (11-D)](#4-target-state-vector-11-d)
1. [Quadrotor Translational Dynamics](#5-quadrotor-translational-dynamics)
1. [Discrete-Time Linearised Dynamics](#6-discrete-time-linearised-dynamics)
1. [A and B Matrices](#7-a-and-b-matrices)
1. [Yaw Sub-System](#8-yaw-sub-system)
1. [Controllability](#9-controllability)

---

## 1. Overview

The system uses **two different state representations** for two different purposes:

| Purpose | Dimension | State components |
| :--- | :--- | :--- |
| DMPC control optimisation | **11-D** | position, velocity, acceleration, yaw, yaw-rate |
| Sensor-fusion / EKF estimation | **18-D** | position, velocity, acceleration, quaternion, angular velocity, battery, health |

The 11-D state is the minimal representation required by the DMPC QP solver.
The 18-D state is the full physical state tracked by the Extended Kalman Filters.

---

## 2. DMPC State Vector (11-D)

$$
\mathbf{x} = \bigl[p_x,\; p_y,\; p_z,\; v_x,\; v_y,\; v_z,\;
  a_x,\; a_y,\; a_z,\; \psi,\; \dot\psi\bigr]^\top \in \mathbb{R}^{11}
$$

| Indices | Symbol | Units | Description |
| :--- | :--- | :--- | :--- |
| 0–2 | $\mathbf{p}$ | m | Position in world frame |
| 3–5 | $\mathbf{v}$ | m/s | Linear velocity |
| 6–8 | $\mathbf{a}$ | m/s² | Linear acceleration |
| 9 | $\psi$ | rad | Yaw angle |
| 10 | $\dot\psi$ | rad/s | Yaw rate |

The translational sub-state $[\mathbf{p}, \mathbf{v}, \mathbf{a}] \in \mathbb{R}^9$ is
**controllable** via the acceleration input $\mathbf{u} = [a_x, a_y, a_z]^\top \in \mathbb{R}^3$.
The yaw sub-state $[\psi, \dot\psi]$ is regulated independently by the geometric
attitude controller (Module 8).

### Why include acceleration?

Incorporating acceleration as a state variable (rather than treating it as a
pure input) provides the MPC with inertia-awareness: the controller can
anticipate how current acceleration commands will evolve position and velocity
over the prediction horizon, giving smoother trajectories.

---

## 3. Estimation State Vector (18-D)

$$
\mathbf{x}_{\text{est}} = \bigl[
  p_x,\; p_y,\; p_z,\;
  v_x,\; v_y,\; v_z,\;
  a_x,\; a_y,\; a_z,\;
  q_w,\; q_x,\; q_y,\; q_z,\;
  \omega_x,\; \omega_y,\; \omega_z,\;
  E_{\text{batt}},\; h
\bigr]^\top \in \mathbb{R}^{18}
$$

This richer vector is used by the `DroneStateEstimator` (three parallel EKFs)
to track the full rigid-body state.  The first 11 components overlap with the
DMPC state (using Euler yaw extracted from the quaternion for index 9, and
$\omega_z$ for index 10).

---

## 4. Target State Vector (11-D)

$$
\mathbf{x}_{\text{tgt}} = \bigl[p_x,\; p_y,\; p_z,\;
  v_x,\; v_y,\; v_z,\;
  a_x,\; a_y,\; a_z,\;
  \psi,\; \dot\psi\bigr]^\top \in \mathbb{R}^{11}
$$

This mirrors the DMPC drone state; the same EKF structure is re-used for
target tracking (see `core/target_state_estimation.py`).

---

## 5. Quadrotor Translational Dynamics

A quadrotor's translational dynamics in the world frame, neglecting drag and
disturbances, are:

$$
\ddot{\mathbf{p}} = \frac{T}{m} R\, \mathbf{e}_3 - g\, \mathbf{e}_3
$$

where:
- $\mathbf{p} \in \mathbb{R}^3$ — position
- $T$ — total thrust (N)
- $m$ — vehicle mass (1.477 kg for hector\_quadrotor)
- $R \in \mathrm{SO}(3)$ — rotation matrix (body → world)
- $\mathbf{e}_3 = [0, 0, 1]^\top$ — unit z-vector
- $g = 9.81\;\text{m/s}^2$

For small angles (near-hover), $R\mathbf{e}_3 \approx \mathbf{e}_3$ and the net body
thrust minus gravity cancels.  The **commanded acceleration** $\mathbf{u} \in \mathbb{R}^3$ is:

$$
\mathbf{u} = \ddot{\mathbf{p}}_{\text{cmd}} = \mathbf{a}_{\text{des}}
$$

The attitude controller (Module 8) inverts the thrust to realise the desired
translational acceleration; the DMPC only generates $\mathbf{u} = [a_x, a_y, a_z]^\top$.

---

## 6. Discrete-Time Linearised Dynamics

The continuous-time triple-integrator model

$$
\dot{\mathbf{p}} = \mathbf{v}, \qquad
\dot{\mathbf{v}} = \mathbf{a}, \qquad
\dot{\mathbf{a}} = \mathbf{u}
$$

is discretised with Euler forward integration at step $\Delta t$:

$$
\mathbf{p}[k{+}1] = \mathbf{p}[k] + \Delta t\;\mathbf{v}[k]
$$

$$
\mathbf{v}[k{+}1] = \mathbf{v}[k] + \Delta t\;\mathbf{a}[k]
$$

$$
\mathbf{a}[k{+}1] = \mathbf{a}[k] + \Delta t\;\mathbf{u}[k]
$$

Written compactly as $\mathbf{x}[k{+}1] = A\,\mathbf{x}[k] + B\,\mathbf{u}[k]$
using the 9-D translational sub-state $[\mathbf{p}, \mathbf{v}, \mathbf{a}]$.

---

## 7. A and B Matrices

For the **full 11-D DMPC state** (translational + yaw):

$$
A = \begin{bmatrix}
I_3 & \Delta t\,I_3 & 0 & 0 & 0 \\
0 & I_3 & \Delta t\,I_3 & 0 & 0 \\
0 & 0 & I_3 & 0 & 0 \\
0 & 0 & 0 & 1 & \Delta t \\
0 & 0 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{11 \times 11}
$$

$$
B = \begin{bmatrix} 0 \\ 0 \\ \Delta t\,I_3 \\ 0 \\ 0 \end{bmatrix}
\in \mathbb{R}^{11 \times 3}
$$

where $I_3$ is the $3 \times 3$ identity matrix.

### Block interpretation

| Block | Rows | Cols | Meaning |
| :--- | :--- | :--- | :--- |
| $A[0{:}3,\;3{:}6] = \Delta t\,I_3$ | position | velocity | $\mathbf{p} \mathrel{+}= \Delta t\,\mathbf{v}$ |
| $A[3{:}6,\;6{:}9] = \Delta t\,I_3$ | velocity | acceleration | $\mathbf{v} \mathrel{+}= \Delta t\,\mathbf{a}$ |
| $A[9,\;10] = \Delta t$ | yaw | yaw-rate | $\psi \mathrel{+}= \Delta t\,\dot\psi$ |
| $B[6{:}9,\;0{:}3] = \Delta t\,I_3$ | acceleration | input | $\mathbf{a} \mathrel{+}= \Delta t\,\mathbf{u}$ |

The yaw dynamics in $A[9{:}11,\;9{:}11]$ form a $2 \times 2$ integrator; yaw is not
driven by the translational input $\mathbf{u}$, so $B[9{:}11,\;:] = 0$.

### Python code (from `dmpc_controller.py`)

```python
A = np.eye(11)
A[0:3, 3:6] = dt * np.eye(3)   # dp/dv
A[3:6, 6:9] = dt * np.eye(3)   # dv/da

B = np.zeros((11, 3))
B[6:9, 0:3] = dt * np.eye(3)   # da/du
```

---

## 8. Yaw Sub-System

The yaw angle $\psi$ and yaw rate $\dot\psi$ evolve as a **decoupled 2-D linear system**:

$$
\begin{bmatrix} \psi[k{+}1] \\ \dot\psi[k{+}1] \end{bmatrix}
= \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} \psi[k] \\ \dot\psi[k] \end{bmatrix}
$$

Because the translational control input $\mathbf{u}$ does not appear in these
equations, yaw is uncontrollable from the DMPC perspective.  Yaw is instead
regulated by the geometric attitude controller through a separate torque command $\tau_\psi$.

---

## 9. Controllability

**Theorem:** The pair $(A_9, B_9)$ — the 9-D translational sub-system — is
**completely controllable**.

*Proof sketch:* The controllability matrix

$$
\mathcal{C} = \bigl[B_9 \;\; A_9 B_9 \;\; A_9^2 B_9 \;\; \cdots \;\; A_9^8 B_9\bigr]
\in \mathbb{R}^{9 \times 27}
$$

has rank 9.  This follows because the triple-integrator chain
$\mathbf{p} \leftarrow \mathbf{v} \leftarrow \mathbf{a} \leftarrow \mathbf{u}$
is in Brunovsky canonical form, which is always controllable for any $\Delta t > 0$.

**Implication:** There exists a stabilising LQR gain $K$ such that the
closed-loop matrix $A_{\text{cl}} = A - BK$ is Schur stable (all eigenvalues
strictly inside the unit circle), which is the foundation for the DARE
terminal cost derivation in the DMPC (see
[03_DMPC_FORMULATION.md](03_DMPC_FORMULATION.md)).
