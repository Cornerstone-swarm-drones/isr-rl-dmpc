# Geometric Attitude Control on SO(3)

**Source file:**
- `src/isr_rl_dmpc/modules/attitude_controller.py` — `GeometricController`, `AttitudeController`, `DroneParameters`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quadrotor Rigid-Body Model](#2-quadrotor-rigid-body-model)
3. [Quaternion Kinematics](#3-quaternion-kinematics)
4. [SO(3) Attitude Error](#4-so3-attitude-error)
5. [Position and Velocity Error](#5-position-and-velocity-error)
6. [Control Law](#6-control-law)
7. [Motor Mixing (X-Quad)](#7-motor-mixing-x-quad)
8. [Default Gain Parameters](#8-default-gain-parameters)
9. [References](#9-references)

---

## 1  Overview

The Attitude Controller (Module 8) translates DMPC acceleration commands
`u = [aₓ, a_y, a_z]` into motor thrust commands `[T₁, T₂, T₃, T₄]`.

The control architecture is a **cascade PD loop**:

```
1. Position loop  →  desired acceleration a_des (3-D)
2. Attitude loop  →  desired torque τ (3-D)  [Geometric SO(3) control]
3. Motor mixing   →  individual motor thrusts [T₁, T₂, T₃, T₄]
```

All control gains are fixed at construction time (no online adaptation).

---

## 2  Quadrotor Rigid-Body Model

### Translational Dynamics

$$
m\,\ddot{\mathbf{p}} = f_{\text{total}}\,R\,\mathbf{e}_3 - m\,g\,\mathbf{e}_3
$$

where:
- $m$ — mass (default 1.0 kg; hector\_quadrotor uses 1.477 kg)
- $f_{\text{total}} = T_1 + T_2 + T_3 + T_4$ — total thrust
- $R \in \mathrm{SO}(3)$ — rotation matrix (body → world)
- $\mathbf{e}_3 = [0, 0, 1]^\top$
- $g = 9.81\;\text{m/s}^2$

### Rotational Dynamics (Euler's Equation)

$$
J\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (J\,\boldsymbol{\omega})
$$

where:
- $J \in \mathbb{R}^{3 \times 3}$ — inertia tensor (diagonal for symmetric quad)
- $\boldsymbol{\omega} \in \mathbb{R}^3$ — angular velocity in body frame (rad/s)
- $\boldsymbol{\tau} \in \mathbb{R}^3$ — net torque in body frame (N·m)

Default inertia (`DroneParameters.__post_init__`):

$$
J = \mathrm{diag}(0.0083,\; 0.0083,\; 0.0166)\;\text{kg·m}^2
$$

---

## 3  Quaternion Kinematics

The rotation matrix $R$ is parameterised by the unit quaternion
$\mathbf{q} = [q_w, q_x, q_y, q_z]^\top$ (scalar-first convention).

### Quaternion to Rotation Matrix

Using Rodriguez' formula:

$$
R = (q_w^2 - \mathbf{q}_v^\top \mathbf{q}_v)\,I_3
  + 2\,\mathbf{q}_v\,\mathbf{q}_v^\top
  + 2\,q_w\,[\mathbf{q}_v]_\times
$$

where $\mathbf{q}_v = [q_x, q_y, q_z]^\top$ and $[\mathbf{q}_v]_\times$ is the skew-symmetric
cross-product matrix:

$$
[\mathbf{q}_v]_\times = \begin{bmatrix} 0 & -q_z & q_y \\ q_z & 0 & -q_x \\ -q_y & q_x & 0 \end{bmatrix}
$$

### Quaternion Kinematics Equation

$$
\dot{\mathbf{q}} = \tfrac{1}{2}\,\mathbf{q} \otimes [0, \boldsymbol{\omega}]^\top
= \frac{1}{2} \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix} \mathbf{q}
$$

The quaternion must always satisfy the unit-norm constraint $\|\mathbf{q}\| = 1$.

---

## 4  SO(3) Attitude Error

### Why SO(3) Instead of Euler Angles?

Euler-angle representations suffer from **gimbal lock** (coordinate singularities)
and can exhibit large numerical errors near ±90° pitch.  Geometric control on
SO(3) is globally defined and avoids these issues.

### Attitude Error Vector

Given current rotation matrix $R$ and desired rotation matrix $R_d$:

$$
R_e = R_d^\top R \quad \text{(relative rotation: } R_d\text{-frame to }R\text{-frame)}
$$

$$
\mathbf{e}_R = \tfrac{1}{2}\mathrm{vex}(R_e - R_e^\top) \in \mathbb{R}^3
  \quad \text{(attitude error vector)}
$$

where $\mathrm{vex}(\cdot)$ extracts the axial vector from a skew-symmetric matrix.

In code:

```python
def attitude_error(self, R, R_d):
    R_e = R_d.T @ R
    return np.array([R_e[2, 1] - R_e[1, 2],
                     R_e[0, 2] - R_e[2, 0],
                     R_e[1, 0] - R_e[0, 1]]) / 2
```

### Angular Velocity Error

$$
\mathbf{e}_\omega = \boldsymbol{\omega} - R^\top R_d\,\boldsymbol{\omega}_d
$$

where $\boldsymbol{\omega}_d$ is the desired angular velocity (zero for stationary hover).

---

## 5  Position and Velocity Error

The outer (position) loop computes a desired acceleration $\mathbf{a}_{\text{des}}$ from the
DMPC reference:

$$
\mathbf{e}_p = \mathbf{p} - \mathbf{p}_{\text{ref}}, \qquad
\mathbf{e}_v = \mathbf{v} - \mathbf{v}_{\text{ref}}
$$

$$
\mathbf{a}_{\text{des}} = \mathbf{a}_{\text{ref}} - K_{p,\text{pos}}\,\mathbf{e}_p - K_{d,\text{pos}}\,\mathbf{e}_v
$$

### Desired Thrust Direction

From Newton's second law, the total thrust vector in the world frame must be:

$$
\mathbf{f}_{\text{des}} = m\,(\mathbf{a}_{\text{des}} + g\,\mathbf{e}_3)
$$

The desired body z-axis (thrust direction) is:

$$
\mathbf{b}_{3,\text{des}} = \mathbf{f}_{\text{des}} / \|\mathbf{f}_{\text{des}}\|
$$

This is combined with a desired yaw angle $\psi_{\text{des}}$ to form the full desired
rotation matrix $R_d$ (via the Gram-Schmidt process on $\mathbf{b}_{1,\text{des}}$ and $\mathbf{b}_{3,\text{des}}$).

---

## 6  Control Law

### Torque Command (Geometric PD)

$$
\boldsymbol{\tau} = -K_{p,\text{att}}\,\mathbf{e}_R - K_{d,\text{att}}\,\mathbf{e}_\omega
  + \boldsymbol{\omega} \times (J\,\boldsymbol{\omega})
$$

The feed-forward gyroscopic term $\boldsymbol{\omega} \times (J\,\boldsymbol{\omega})$ compensates
for Coriolis and centrifugal effects, improving tracking at high angular rates.

### Total Thrust

$$
f_{\text{total}} = \mathbf{f}_{\text{des}} \cdot (R\,\mathbf{e}_3)
  \quad \text{(project desired force onto body z-axis)}
$$

$$
f_{\text{total}} \leftarrow \max(f_{\text{total}},\; 0) \quad \text{(thrust cannot be negative)}
$$

### Control Output

The attitude loop outputs $[f_{\text{total}}, \tau_x, \tau_y, \tau_z]$ which is then mapped to
individual motor thrusts via the motor mixing matrix.

---

## 7  Motor Mixing (X-Quad)

For an **X-configuration quadrotor** (motors at ±45° from the longitudinal
axis), the relationship between individual motor thrusts $[T_1, T_2, T_3, T_4]$
and body-frame wrench $[F, \tau_x, \tau_y, \tau_z]$ is:

$$
\begin{bmatrix} F \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
= \begin{bmatrix}
  k_f & k_f & k_f & k_f \\
  -k_f L & k_f L & k_f L & -k_f L \\
  k_f L & k_f L & -k_f L & -k_f L \\
  -k_d & k_d & -k_d & k_d
\end{bmatrix}
\begin{bmatrix} T_1 \\ T_2 \\ T_3 \\ T_4 \end{bmatrix}
$$

where:
- $k_f$ — thrust coefficient (N/(rad/s)²)
- $k_d$ — drag–torque coefficient (N·m/(rad/s)²)
- $L = \ell / \sqrt{2}$ — effective moment arm ($\ell$ = arm length)

The **mixing matrix** $M$ maps thrusts to wrench; the inverse maps wrench to thrusts:

$$
\begin{bmatrix} T_1 \\ T_2 \\ T_3 \\ T_4 \end{bmatrix}
= M^{-1} \begin{bmatrix} F \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
$$

M is square and invertible for the X-configuration.  Individual thrusts are
clipped to `[0, T_max]` to respect motor saturation.

### Motor Speed Conversion

Motor angular speed `Ω_i` from thrust `T_i`:

```
T_i = k_f Ω_i²   →   Ω_i = √(T_i / k_f)
```

PWM duty cycle:

```
PWM_i = Ω_i / Ω_max ∈ [0, 1]
```

---

## 8  Default Gain Parameters

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| Attitude proportional gain | Kp_att | 4.5 | — |
| Attitude derivative gain | Kd_att | 1.5 | — |
| Position proportional gain | Kp_pos | 2.0 | — |
| Position derivative gain | Kd_pos | 1.5 | — |
| Motor constant | k_f | 8.27×10⁻⁶ | N/(rad/s)² |
| Maximum motor speed | Ω_max | 800 | rad/s |
| Maximum thrust per motor | T_max | 25 | N |
| Arm length | L | 0.215 | m |
| Inertia (Ixx = Iyy) | — | 0.0083 | kg·m² |
| Inertia (Izz) | — | 0.0166 | kg·m² |

These gains are tuned for a ~1 kg quadrotor at hover with a 50 Hz control loop.
For the hector_quadrotor airframe (1.477 kg), scale `Kp_pos` and `Kd_pos`
proportionally or re-run LQR tuning via the DARE.

---

## 9  References

1. T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a
   quadrotor UAV on SE(3)," *IEEE Conference on Decision and Control (CDC)*,
   2010, pp. 5420–5425.
2. D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control
   for quadrotors," *IEEE ICRA*, 2011, pp. 2520–2525.
3. F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
   Determination and Control*, Springer, 2014. (Chapters 5–6 for SO(3) attitude
   control.)
