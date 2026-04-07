# Extended Kalman Filter — Drone and Target State Estimation

**Source files:**
- `src/isr_rl_dmpc/core/drone_state_estimation.py` — `PositionVelocityEKF`, `AttitudeEKF`, `DroneStateEstimator`
- `src/isr_rl_dmpc/core/target_state_estimation.py` — `TargetTrackingEKF`, `TargetStateEstimator`
- `src/isr_rl_dmpc/modules/sensor_fusion.py` — `SensorFusionManager`

---

## Table of Contents

1. [Overview](#1-overview)
2. [EKF Fundamentals](#2-ekf-fundamentals)
3. [Drone State Estimation (18-D)](#3-drone-state-estimation-18-d)
   - 3.1 [Position/Velocity EKF (6-D)](#31-positionvelocity-ekf-6-d)
   - 3.2 [Attitude EKF — Quaternion (4-D)](#32-attitude-ekf--quaternion-4-d)
   - 3.3 [Angular Velocity Filter (3-D)](#33-angular-velocity-filter-3-d)
   - 3.4 [Battery and Health (2-D direct)](#34-battery-and-health-2-d-direct)
4. [Target Tracking EKF (11-D)](#4-target-tracking-ekf-11-d)
5. [Multi-Sensor Measurement Models](#5-multi-sensor-measurement-models)
   - 5.1 [GPS / RTK](#51-gps--rtk)
   - 5.2 [Radar (4-D)](#52-radar-4-d)
   - 5.3 [Optical Bearing (2-D / 3-D)](#53-optical-bearing-2-d--3-d)
   - 5.4 [RF Fingerprinting (3-D)](#54-rf-fingerprinting-3-d)
   - 5.5 [Acoustic TDOA (3-D)](#55-acoustic-tdoa-3-d)
6. [Adaptive Sensor Fusion](#6-adaptive-sensor-fusion)
7. [Covariance Propagation](#7-covariance-propagation)

---

## 1  Overview

State estimation fuses noisy, asynchronous sensor measurements into smooth,
low-latency state estimates.  The system uses three parallel EKFs for the
drone's own state and one EKF per tracked target.

```
IMU (400 Hz) ──► PositionVelocityEKF (6-D) ─┐
GPS (10 Hz) ─────────────────────────────────┤
                                              ├─► DroneStateEstimator ──► 18-D state
Gyro (400 Hz) ──► AttitudeEKF (4-D) ─────────┤
Magnetometer ────────────────────────────────┤
                                              │
Angular Velocity Filter (3-D) ───────────────┘

Radar ──────────┐
Optical ────────┤
RF ─────────────┼─► TargetTrackingEKF (11-D per target)
Acoustic TDOA ──┘
```

---

## 2  EKF Fundamentals

The (Extended) Kalman Filter alternates between two steps:

### Predict Step

Given process model `f(x)` and process noise covariance **Q**:

```
x̂⁻[k] = f(x̂[k-1], u)            (propagate mean)
P⁻[k]  = F P[k-1] Fᵀ + Q         (propagate covariance)
```

where **F** = ∂f/∂x is the Jacobian of the process model.
For linear models f(x) = Ax, F = A exactly.

### Update Step

Given measurement model `h(x)`, measurement **z**, and noise covariance **R_meas**:

```
ŷ  = z − h(x̂⁻)                   (innovation)
S  = H P⁻ Hᵀ + R_meas            (innovation covariance)
K  = P⁻ Hᵀ S⁻¹                   (Kalman gain)

x̂  = x̂⁻ + K ŷ                    (corrected mean)
P   = (I − KH) P⁻                 (corrected covariance)
```

where **H** = ∂h/∂x is the measurement Jacobian.

**Key property:** The Kalman gain **K** optimally weights the prior estimate
against the new measurement based on their respective uncertainties (P⁻ and R_meas).

---

## 3  Drone State Estimation (18-D)

The full 18-D drone state `[p(3), v(3), a(3), q(4), ω(3), E(1), h(1)]` is
estimated by three specialised sub-filters fused inside `DroneStateEstimator`.

### 3.1  Position/Velocity EKF (6-D)

**State:** `x_pv = [p_x, p_y, p_z, v_x, v_y, v_z]ᵀ ∈ ℝ⁶`

#### Process Model (Euler integration)

The IMU accelerometer provides body-frame acceleration `a_body`.  After
rotating to the world frame (`a_world = R(q) a_body − g e₃`):

```
p[k+1] = p[k] + Δt v[k] + ½ Δt² a_world
v[k+1] = v[k] + Δt a_world
```

State transition matrix:

```
F_pv = [ I₃   ΔtI₃ ] ∈ ℝ⁶ˣ⁶
       [  0    I₃  ]
```

Covariance prediction:

```
P⁻ = F_pv P F_pvᵀ + Q_pv
```

where `Q_pv` reflects position drift and velocity uncertainty.

#### GPS Update

Full-state GPS measurement `z = [p_GPS; v_GPS] ∈ ℝ⁶`:

```
H_gps = I₆

R_gps = diag(σ_pos², σ_pos², σ_pos², σ_vel², σ_vel², σ_vel²)
```

Default noise: `σ_pos = 5.0 m`,  `σ_vel = 1.0 m/s`.

Innovation and update follow the standard Kalman equations above.

### 3.2  Attitude EKF — Quaternion (4-D)

**State:** `q = [q_w, q_x, q_y, q_z]ᵀ` (unit quaternion, scalar-first)

#### Quaternion Kinematics

The quaternion derivative is:

```
q̇ = ½ q ⊗ [0, ω]ᵀ
```

where `⊗` is the quaternion product and `ω ∈ ℝ³` is the gyro reading.

Discrete-time prediction (first-order):

```
q[k+1] = q[k] + ½ Ω(ω) q[k] Δt
q[k+1] ← q[k+1] / ‖q[k+1]‖    (renormalise)
```

where `Ω(ω)` is the 4×4 skew-symmetric matrix:

```
Ω(ω) = [  0  −ω_x  −ω_y  −ω_z ]
        [ ω_x   0    ω_z  −ω_y ]
        [ ω_y  −ω_z   0    ω_x ]
        [ ω_z   ω_y  −ω_x   0  ]
```

#### Accelerometer Correction (roll/pitch)

When approximately static, the accelerometer points in the direction of
gravity.  The expected direction is `ĝ = [0, 0, 1]ᵀ` (up in world frame).

Roll/pitch correction via cross-product error:

```
e = a_norm × ĝ
Δq = k_a [0, e]
q ← normalise(q + Δq)
```

where `k_a = 0.01` is the accelerometer correction gain.

#### Magnetometer Correction (yaw)

Expected magnetic north direction: `m_expected = [1, 0, 0]ᵀ`.

```
e = m_norm × m_expected
Δq = k_m [0, e]
q ← normalise(q + Δq)
```

where `k_m = 0.01` is the magnetometer correction gain.

### 3.3  Angular Velocity Filter (3-D)

A simple bias-subtraction model with low-pass bias estimation:

```
ω_est[k]  = ω_gyro[k] − b[k]

b[k+1] = (1−α) b[k] + α ω_gyro[k]   (when stationary, α = 0.1)
```

The bias `b` accounts for gyroscope temperature drift and is estimated
during stationary calibration before flight.

### 3.4  Battery and Health (2-D direct)

**Battery** is modelled as a first-order discharge:

```
E[k+1] = max(0,  E[k] − P_draw Δt / 3600)
```

where `P_draw` is the power draw in Watts.  A fuel-gauge update applies
an α-filter to fuse the on-board measurement:

```
E ← (1−α) E + α E_measured,   α = 0.5
```

**Health** is a direct measurement (motor diagnostics) clipped to [0, 1].

---

## 4  Target Tracking EKF (11-D)

**State:** `x_tgt = [p(3), v(3), a(3), ψ, ψ̇]ᵀ ∈ ℝ¹¹`

This mirrors the DMPC drone state and uses the same triple-integrator
dynamics (Section 6 of `01_DRONE_STATE_SPACE.md`).

#### Process Model

```
F_tgt = A₁₁   (same matrix as DMPC A matrix)
Q_tgt = diag(process noise for each state component)
```

#### Measurement Jacobians

Non-linear measurement models require the Jacobian H to be computed at the
current state estimate `x̂`.  See [Section 5](#5-multi-sensor-measurement-models) for each sensor type.

#### Multiple Targets

One EKF instance is maintained per tracked target.  Targets are created when
a new detection cannot be associated with any existing track, and deleted when
the covariance exceeds a health threshold (indicating track loss).

---

## 5  Multi-Sensor Measurement Models

### 5.1  GPS / RTK

Applied to the drone's own position/velocity EKF (Section 3.1 above).

```
h_GPS(x) = [p_x, p_y, p_z, v_x, v_y, v_z]ᵀ
H_GPS = I₆
```

### 5.2  Radar (4-D)

Radar measures range, range-rate, azimuth, and elevation from sensor position `s`:

```
δ = p_tgt − s         (relative position vector)
r = ‖δ‖              (range)
ṙ = δᵀ v / r         (range-rate via radial projection)
az = atan2(δ_y, δ_x) (azimuth angle)
el = asin(δ_z / r)   (elevation angle)

h_radar(x) = [r, ṙ, az, el]ᵀ
```

The Jacobian **H_radar** = ∂h_radar / ∂x is computed numerically via
finite differences at the current estimate.

Measurement noise covariance (typical values):

```
R_radar = diag(σ_r², σ_ṙ², σ_az², σ_el²)
        = diag(5², 1², 0.01², 0.01²)
```

### 5.3  Optical Bearing (2-D / 3-D)

Camera provides azimuth and elevation (and optionally range):

```
h_opt(x) = [az, el]ᵀ          (2-D, no range)
h_opt(x) = [az, el, r]ᵀ       (3-D, with LiDAR)
```

Noise covariance:
```
R_opt = diag(σ_az², σ_el²) = diag(0.02², 0.02²)
```

### 5.4  RF Fingerprinting (3-D)

RF fingerprinting provides a direct position estimate:

```
h_RF(x) = [p_x, p_y, p_z]ᵀ
H_RF = [I₃ | 0₃ₓ₈]
```

Measurement noise proportional to RF confidence `c ∈ [0, 1]`:

```
R_RF = (1/c) diag(σ_RF², σ_RF², σ_RF²),   σ_RF = 10 m
```

### 5.5  Acoustic TDOA (3-D)

Hyperbolic trilateration from time-difference-of-arrival converts to a
direct position estimate with confidence-scaled noise (same structure as RF).

---

## 6  Adaptive Sensor Fusion

When multiple sensor types are available simultaneously, an **adaptive
weighting** scheme scales each sensor's noise covariance by its current
confidence score `c_s ∈ [0, 1]`:

```
R_s_eff = R_s / c_s
```

This automatically de-weights degraded sensors.  The standard EKF update
is then applied sequentially for each sensor type, or batch-applied via a
stacked measurement vector.

**Sequential vs. batch update:**  The implementation uses sequential updates
(one sensor at a time) to preserve numerical stability and allow independent
sensor dropout without re-building the full measurement matrix.

---

## 7  Covariance Propagation

### Symmetrisation

After every predict and update step the covariance matrix is symmetrised to
eliminate floating-point asymmetry accumulation:

```
P ← (P + Pᵀ) / 2
```

### Positive Definiteness

A small regularisation term can be added if eigenvalues approach zero:

```
P ← P + ε I,   ε ≈ 1e-9
```

This is applied in the position/velocity EKF GPS update to keep `S = HPHᵀ + R`
invertible even when P becomes near-singular.

### Assembled 18×18 Covariance

The full 18-D covariance is a **block-diagonal** assembly of the three
sub-filter covariances:

```
P_18 = block_diag(P_pv(6×6), P_a(3×3), P_att(4×4), P_av(3×3), P_batt(1×1), P_h(1×1))
```

Acceleration uncertainty is approximated as `P_a ≈ P_v / 10` (velocity
sub-block of the position/velocity EKF, scaled down).
