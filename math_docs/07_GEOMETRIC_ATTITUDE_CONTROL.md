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

```
m p̈ = f_total e₃ R − m g e₃
```

where:
- `m` — mass (default 1.0 kg; hector_quadrotor uses 1.477 kg)
- `f_total = T₁ + T₂ + T₃ + T₄` — total thrust
- `R ∈ SO(3)` — rotation matrix (body → world)
- `e₃ = [0, 0, 1]ᵀ`
- `g = 9.81 m/s²`

### Rotational Dynamics (Euler's Equation)

```
J ω̇ = τ − ω × (J ω)
```

where:
- `J ∈ ℝ³ˣ³` — inertia tensor (diagonal for symmetric quad)
- `ω ∈ ℝ³` — angular velocity in body frame (rad/s)
- `τ ∈ ℝ³` — net torque in body frame (N·m)

Default inertia (`DroneParameters.__post_init__`):

```
J = diag(0.0083, 0.0083, 0.0166)   kg·m²
```

---

## 3  Quaternion Kinematics

The rotation matrix R is parameterised by the unit quaternion
`q = [q_w, q_x, q_y, q_z]ᵀ` (scalar-first convention).

### Quaternion to Rotation Matrix

```
R = I + 2q_w [q_v]× + 2 [q_v]×²
```

where `q_v = [q_x, q_y, q_z]ᵀ` and `[q_v]×` is the 3×3 skew-symmetric
cross-product matrix:

```
[q_v]× = [  0   −q_z   q_y ]
          [  q_z   0   −q_x ]
          [ −q_y  q_x   0  ]
```

Equivalently, using Rodriguez' formula:

```
R = (q_w² − q_vᵀ q_v) I₃ + 2 q_v q_vᵀ + 2 q_w [q_v]×
```

The implementation delegates to `scipy.spatial.transform.Rotation` for
numerical accuracy.

### Quaternion Kinematics Equation

```
q̇ = ½ q ⊗ [0, ω]ᵀ
```

Expanded to matrix form:

```
[q̇_w]   ½ [  0    −ω_x  −ω_y  −ω_z ] [q_w]
[q̇_x] =   [ ω_x    0     ω_z  −ω_y ] [q_x]
[q̇_y]     [ ω_y  −ω_z    0     ω_x ] [q_y]
[q̇_z]     [ ω_z   ω_y  −ω_x    0   ] [q_z]
```

The quaternion must always satisfy the unit-norm constraint `‖q‖ = 1`.

---

## 4  SO(3) Attitude Error

### Why SO(3) Instead of Euler Angles?

Euler-angle representations suffer from **gimbal lock** (coordinate singularities)
and can exhibit large numerical errors near ±90° pitch.  Geometric control on
SO(3) is globally defined and avoids these issues.

### Attitude Error Vector

Given current rotation matrix **R** and desired rotation matrix **R_d**:

```
R_e = R_dᵀ R              (relative rotation: R_d-frame to R-frame)

e_R = ½ vex(R_e − R_eᵀ)  (attitude error vector, ∈ ℝ³)
```

where `vex(·)` extracts the axial vector from a skew-symmetric matrix:

```
vex([  0   −a₃   a₂ ]) = [a₁, a₂, a₃]ᵀ
    [  a₃   0   −a₁ ]
    [ −a₂   a₁   0  ]
```

In code:

```python
def attitude_error(self, R, R_d):
    R_e = R_d.T @ R
    return np.array([R_e[2, 1] - R_e[1, 2],
                     R_e[0, 2] - R_e[2, 0],
                     R_e[1, 0] - R_e[0, 1]]) / 2
```

### Angular Velocity Error

```
e_ω = ω − Rᵀ R_d ω_d
```

where `ω_d` is the desired angular velocity (zero for stationary hover).

---

## 5  Position and Velocity Error

The outer (position) loop computes a desired acceleration `a_des` from the
DMPC reference:

```
e_p = p − p_ref     (position error)
e_v = v − v_ref     (velocity error)

a_des = a_ref − Kp_pos · e_p − Kd_pos · e_v
```

### Desired Thrust Direction

From Newton's second law, the total thrust vector in the world frame must be:

```
f_des = m (a_des + g e₃)
```

The desired body z-axis (thrust direction) is:

```
b₃_des = f_des / ‖f_des‖
```

This is combined with a desired yaw angle ψ_des to form the full desired
rotation matrix R_d (via the Gram-Schmidt process on b₁_des and b₃_des).

---

## 6  Control Law

### Torque Command (Geometric PD)

```
τ = −Kp_att · e_R − Kd_att · e_ω + ω × (J ω)
```

The feed-forward gyroscopic term `ω × (J ω)` compensates for Coriolis and
centrifugal effects, improving tracking at high angular rates.

### Total Thrust

```
f_total = f_des · (R e₃)    (project desired force onto body z-axis)
f_total = max(f_total, 0)   (thrust cannot be negative)
```

### Control Output

The attitude loop outputs `[f_total, τ_x, τ_y, τ_z]` which is then mapped to
individual motor thrusts via the motor mixing matrix.

---

## 7  Motor Mixing (X-Quad)

For an **X-configuration quadrotor** (motors at ±45° from the longitudinal
axis), the relationship between individual motor thrusts `[T₁, T₂, T₃, T₄]`
and body-frame wrench `[F, τ_x, τ_y, τ_z]` is:

```
[ F  ]   [ k_f   k_f   k_f   k_f  ] [T₁]
[ τ_x ] = [ −k_f·L  k_f·L  k_f·L  −k_f·L] [T₂]
[ τ_y ]   [ k_f·L  k_f·L  −k_f·L  −k_f·L] [T₃]
[ τ_z ]   [ −k_d  k_d   −k_d   k_d  ] [T₄]
```

where:
- `k_f` — thrust coefficient (N/(rad/s)²)
- `k_d` — drag–torque coefficient (N·m/(rad/s)²)
- `L = arm_length / √2` — effective moment arm (m)

The **mixing matrix M** maps thrusts to wrench; the inverse maps wrench to
thrusts:

```
[T₁]       [ F  ]
[T₂] = M⁻¹ [ τ_x]
[T₃]       [ τ_y]
[T₄]       [ τ_z]
```

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
