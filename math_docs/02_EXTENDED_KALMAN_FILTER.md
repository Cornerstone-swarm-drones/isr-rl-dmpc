# Extended Kalman Filter — Drone and Target State Estimation

**Source files:**
- `src/isr_rl_dmpc/core/drone_state_estimation.py` — `PositionVelocityEKF`, `AttitudeEKF`, `DroneStateEstimator`
- `src/isr_rl_dmpc/core/target_state_estimation.py` — `TargetTrackingEKF`, `TargetStateEstimator`
- `src/isr_rl_dmpc/modules/sensor_fusion.py` — `SensorFusionManager`

---

## Table of Contents

1. [Overview](#1-overview)
1. [EKF Fundamentals](#2-ekf-fundamentals)
1. [Drone State Estimation (18-D)](#3-drone-state-estimation-18-d)
   - 3.1 [Position/Velocity EKF (6-D)](#31-positionvelocity-ekf-6-d)
   - 3.2 [Attitude EKF — Quaternion (4-D)](#32-attitude-ekf--quaternion-4-d)
   - 3.3 [Angular Velocity Filter (3-D)](#33-angular-velocity-filter-3-d)
   - 3.4 [Battery and Health (2-D direct)](#34-battery-and-health-2-d-direct)
1. [Target Tracking EKF (11-D)](#4-target-tracking-ekf-11-d)
1. [Multi-Sensor Measurement Models](#5-multi-sensor-measurement-models)
   - 5.1 [GPS / RTK](#51-gps--rtk)
   - 5.2 [Radar (4-D)](#52-radar-4-d)
   - 5.3 [Optical Bearing (2-D / 3-D)](#53-optical-bearing-2-d--3-d)
   - 5.4 [RF Fingerprinting (3-D)](#54-rf-fingerprinting-3-d)
   - 5.5 [Acoustic TDOA (3-D)](#55-acoustic-tdoa-3-d)
1. [Adaptive Sensor Fusion](#6-adaptive-sensor-fusion)
1. [Covariance Propagation](#7-covariance-propagation)

---

## 1. Overview

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

## 2. EKF Fundamentals

The (Extended) Kalman Filter alternates between two steps:

### Predict Step

Given process model $f(\boldsymbol{x})$ and process noise covariance $Q$:

$$
\hat{\boldsymbol{x}}^-[k] = f(\hat{\boldsymbol{x}}[k{-}1],\, \boldsymbol{u})
  \quad \text{(propagate mean)}
$$

$$
P^-[k] = F\,P[k{-}1]\,F^\top + Q
  \quad \text{(propagate covariance)}
$$

where $F = \partial f / \partial \boldsymbol{x}$ is the Jacobian of the process model.
For linear models $f(\boldsymbol{x}) = A\boldsymbol{x}$, $F = A$ exactly.

### Update Step

Given measurement model $h(\boldsymbol{x})$, measurement $\boldsymbol{z}$, and noise covariance $R_{\text{meas}}$:

$$
\hat{\boldsymbol{y}} = \boldsymbol{z} - h(\hat{\boldsymbol{x}}^-)
  \quad \text{(innovation)}
$$

$$
S = H\,P^-\,H^\top + R_{\text{meas}}
  \quad \text{(innovation covariance)}
$$

$$
K = P^-\,H^\top S^{-1}
  \quad \text{(Kalman gain)}
$$

$$
\hat{\boldsymbol{x}} = \hat{\boldsymbol{x}}^- + K\,\hat{\boldsymbol{y}}
  \quad \text{(corrected mean)}
$$

$$
P = (I - KH)\,P^-
  \quad \text{(corrected covariance)}
$$

where $H = \partial h / \partial \boldsymbol{x}$ is the measurement Jacobian.

**Key property:** The Kalman gain $K$ optimally weights the prior estimate
against the new measurement based on their respective uncertainties ($P^-$ and $R_{\text{meas}}$).

---

## 3. Drone State Estimation (18-D)

The full 18-D drone state $[\boldsymbol{p}(3), \boldsymbol{v}(3), \boldsymbol{a}(3), \boldsymbol{q}(4), \boldsymbol{\omega}(3), E(1), h(1)]$
is estimated by three specialised sub-filters fused inside `DroneStateEstimator`.

### 3.1. Position/Velocity EKF (6-D)

**State:** $\boldsymbol{x}_{pv} = [p_x, p_y, p_z, v_x, v_y, v_z]^\top \in \mathbb{R}^6$

#### Process Model (Euler integration)

The IMU accelerometer provides body-frame acceleration $\boldsymbol{a}_{\text{body}}$.
After rotating to the world frame ($\boldsymbol{a}_{\text{world}} = R(\boldsymbol{q})\,\boldsymbol{a}_{\text{body}} - g\,\boldsymbol{e}_3$):

$$
\boldsymbol{p}[k{+}1] = \boldsymbol{p}[k] + \Delta t\,\boldsymbol{v}[k] + \tfrac{1}{2}\Delta t^2\,\boldsymbol{a}_{\text{world}}
$$

$$
\boldsymbol{v}[k{+}1] = \boldsymbol{v}[k] + \Delta t\,\boldsymbol{a}_{\text{world}}
$$

State transition matrix:

$$
F_{pv} = \begin{bmatrix} I_3 & \Delta t\,I_3 \\ 0 & I_3 \end{bmatrix} \in \mathbb{R}^{6 \times 6}
$$

Covariance prediction:

$$
P^- = F_{pv}\,P\,F_{pv}^\top + Q_{pv}
$$

#### GPS Update

Full-state GPS measurement $\boldsymbol{z} = [\boldsymbol{p}_{\text{GPS}};\; \boldsymbol{v}_{\text{GPS}}] \in \mathbb{R}^6$:

$$
H_{\text{gps}} = I_6, \qquad
R_{\text{gps}} = \mathrm{diag}(\sigma_p^2, \sigma_p^2, \sigma_p^2, \sigma_v^2, \sigma_v^2, \sigma_v^2)
$$

Default noise: $\sigma_p = 5.0\;\text{m}$, $\sigma_v = 1.0\;\text{m/s}$.

### 3.2. Attitude EKF — Quaternion (4-D)

**State:** $\boldsymbol{q} = [q_w, q_x, q_y, q_z]^\top$ (unit quaternion, scalar-first)

#### Quaternion Kinematics

The quaternion derivative is:

$$
\dot{\boldsymbol{q}} = \tfrac{1}{2}\,\boldsymbol{q} \otimes [0, \boldsymbol{\omega}]^\top
$$

where $\otimes$ is the quaternion product and $\boldsymbol{\omega} \in \mathbb{R}^3$ is the gyro reading.

Discrete-time prediction (first-order):

$$
\boldsymbol{q}[k{+}1] = \boldsymbol{q}[k] + \tfrac{1}{2}\,\Omega(\boldsymbol{\omega})\,\boldsymbol{q}[k]\,\Delta t
$$

$$
\boldsymbol{q}[k{+}1] \leftarrow \boldsymbol{q}[k{+}1] / \|\boldsymbol{q}[k{+}1]\|
  \quad \text{(renormalise)}
$$

where $\Omega(\boldsymbol{\omega})$ is the $4 \times 4$ skew-symmetric matrix:

$$
\Omega(\boldsymbol{\omega}) = \begin{bmatrix}
  0 & -\omega_x & -\omega_y & -\omega_z \\
  \omega_x & 0 & \omega_z & -\omega_y \\
  \omega_y & -\omega_z & 0 & \omega_x \\
  \omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
$$

#### Accelerometer Correction (roll/pitch)

Roll/pitch correction via cross-product error:

$$
\boldsymbol{e} = \boldsymbol{a}_{\text{norm}} \times \hat{\boldsymbol{g}}, \qquad
\Delta\boldsymbol{q} = k_a\,[0, \boldsymbol{e}], \qquad
\boldsymbol{q} \leftarrow \mathrm{normalise}(\boldsymbol{q} + \Delta\boldsymbol{q})
$$

where $k_a = 0.01$ is the accelerometer correction gain.

#### Magnetometer Correction (yaw)

$$
\boldsymbol{e} = \boldsymbol{m}_{\text{norm}} \times \boldsymbol{m}_{\text{expected}}, \qquad
\Delta\boldsymbol{q} = k_m\,[0, \boldsymbol{e}], \qquad
\boldsymbol{q} \leftarrow \mathrm{normalise}(\boldsymbol{q} + \Delta\boldsymbol{q})
$$

where $k_m = 0.01$ is the magnetometer correction gain.

### 3.3. Angular Velocity Filter (3-D)

A simple bias-subtraction model with low-pass bias estimation:

$$
\hat{\boldsymbol{\omega}}[k] = \boldsymbol{\omega}_{\text{gyro}}[k] - \boldsymbol{b}[k]
$$

$$
\boldsymbol{b}[k{+}1] = (1-\alpha)\,\boldsymbol{b}[k] + \alpha\,\boldsymbol{\omega}_{\text{gyro}}[k]
  \quad (\text{stationary calibration, } \alpha = 0.1)
$$

### 3.4. Battery and Health (2-D direct)

**Battery** is modelled as a first-order discharge:

$$
E[k{+}1] = \max(0,\; E[k] - P_{\text{draw}}\,\Delta t / 3600)
$$

A fuel-gauge update applies an $\alpha$-filter to fuse the on-board measurement:

$$
E \leftarrow (1-\alpha)\,E + \alpha\,E_{\text{measured}}, \quad \alpha = 0.5
$$

**Health** is a direct measurement (motor diagnostics) clipped to $[0, 1]$.

---

## 4. Target Tracking EKF (11-D)

**State:** $\boldsymbol{x}_{\text{tgt}} = [\boldsymbol{p}(3), \boldsymbol{v}(3), \boldsymbol{a}(3), \psi, \dot\psi]^\top \in \mathbb{R}^{11}$

This mirrors the DMPC drone state and uses the same triple-integrator
dynamics (Section 6 of `01_DRONE_STATE_SPACE.md`).

#### Process Model

$$
F_{\text{tgt}} = A_{11} \quad \text{(same matrix as DMPC } A \text{ matrix)}
$$

Non-linear measurement models require the Jacobian $H$ to be computed at the
current state estimate $\hat{\boldsymbol{x}}$.  See [Section 5](#5-multi-sensor-measurement-models).

---

## 5. Multi-Sensor Measurement Models

### 5.1. GPS / RTK

$$
h_{\text{GPS}}(\boldsymbol{x}) = [p_x, p_y, p_z, v_x, v_y, v_z]^\top, \qquad H_{\text{GPS}} = I_6
$$

### 5.2. Radar (4-D)

Radar measures range, range-rate, azimuth, and elevation from sensor position $\boldsymbol{s}$:

$$
\boldsymbol{\delta} = \boldsymbol{p}_{\text{tgt}} - \boldsymbol{s}, \qquad
r = \|\boldsymbol{\delta}\|, \qquad
\dot{r} = \frac{\boldsymbol{\delta}^\top \boldsymbol{v}}{r}
$$

$$
\alpha_z = \mathrm{atan2}(\delta_y, \delta_x), \qquad
\text{el} = \arcsin(\delta_z / r)
$$

$$
h_{\text{radar}}(\boldsymbol{x}) = [r,\; \dot{r},\; \alpha_z,\; \text{el}]^\top
$$

$$
R_{\text{radar}} = \mathrm{diag}(\sigma_r^2, \sigma_{\dot{r}}^2, \sigma_{\alpha}^2, \sigma_{\text{el}}^2)
= \mathrm{diag}(25, 1, 10^{-4}, 10^{-4})
$$

### 5.3. Optical Bearing (2-D / 3-D)

$$
h_{\text{opt}}(\boldsymbol{x}) = [\alpha_z,\; \text{el}]^\top \quad \text{(2-D, no range)}
$$

$$
R_{\text{opt}} = \mathrm{diag}(\sigma_\alpha^2, \sigma_{\text{el}}^2)
= \mathrm{diag}(4 \times 10^{-4}, 4 \times 10^{-4})
$$

### 5.4. RF Fingerprinting (3-D)

$$
h_{\text{RF}}(\boldsymbol{x}) = [p_x, p_y, p_z]^\top, \qquad
H_{\text{RF}} = [I_3 \mid \boldsymbol{0}_{3 \times 8}]
$$

$$
R_{\text{RF}} = \frac{1}{c}\,\mathrm{diag}(\sigma_{\text{RF}}^2, \sigma_{\text{RF}}^2, \sigma_{\text{RF}}^2),
\quad \sigma_{\text{RF}} = 10\;\text{m}
$$

### 5.5. Acoustic TDOA (3-D)

Hyperbolic trilateration from time-difference-of-arrival converts to a
direct position estimate with confidence-scaled noise (same structure as RF).

---

## 6. Adaptive Sensor Fusion

When multiple sensor types are available simultaneously, an **adaptive
weighting** scheme scales each sensor's noise covariance by its current
confidence score $c_s \in [0, 1]$:

$$
R_{s,\text{eff}} = R_s / c_s
$$

This automatically de-weights degraded sensors.

---

## 7. Covariance Propagation

### Symmetrisation

After every predict and update step the covariance matrix is symmetrised to
eliminate floating-point asymmetry accumulation:

$$
P \leftarrow (P + P^\top) / 2
$$

### Positive Definiteness

A small regularisation term can be added if eigenvalues approach zero:

$$
P \leftarrow P + \varepsilon\,I, \quad \varepsilon \approx 10^{-9}
$$

### Assembled 18×18 Covariance

The full 18-D covariance is a **block-diagonal** assembly of the three
sub-filter covariances:

$$
P_{18} = \mathrm{block\_diag}\!\bigl(P_{pv}(6 \times 6),\;
  P_a(3 \times 3),\; P_{\text{att}}(4 \times 4),\;
  P_{\omega}(3 \times 3),\; P_E(1 \times 1),\; P_h(1 \times 1)\bigr)
$$


