# Math Documentation — ISR-DMPC

This directory contains self-contained mathematical reference documents for every
algorithm, model, and control method used in the **ISR-DMPC** system.  Each
document introduces the problem, derives the equations from first principles, and
links back to the corresponding source file(s).

---

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md) | Drone state vectors, discrete-time linearised dynamics, integrator chain |
| 2 | [02_EXTENDED_KALMAN_FILTER.md](02_EXTENDED_KALMAN_FILTER.md) | EKF predict/update equations, 18-D drone estimation, 11-D target tracking, multi-sensor fusion |
| 3 | [03_DMPC_FORMULATION.md](03_DMPC_FORMULATION.md) | DMPC QP optimisation problem, cost matrices, DARE terminal cost, OSQP solver |
| 4 | [04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) | Lyapunov stability, eigenvalue analysis, ISS, Control Barrier Functions, recursive feasibility |
| 5 | [05_FORMATION_CONSENSUS.md](05_FORMATION_CONSENSUS.md) | Distributed consensus protocol, graph Laplacian, all six formation geometries |
| 6 | [06_TASK_ALLOCATION.md](06_TASK_ALLOCATION.md) | Bipartite assignment problem, Hungarian algorithm, cost-matrix construction |
| 7 | [07_GEOMETRIC_ATTITUDE_CONTROL.md](07_GEOMETRIC_ATTITUDE_CONTROL.md) | Quaternion kinematics, SO(3) attitude error, motor mixing |
| 8 | [08_COVERAGE_PLANNING.md](08_COVERAGE_PLANNING.md) | Grid decomposition, waypoint generation, coverage path strategies |

---

## How These Relate to the Source Code

```
math_docs/                           src/isr_rl_dmpc/
├── 01_DRONE_STATE_SPACE       ←→   modules/dmpc_controller.py  (DMPC._get_linearized_dynamics)
│                              ←→   core/data_structures.py      (DroneState)
├── 02_EXTENDED_KALMAN_FILTER  ←→   core/drone_state_estimation.py
│                              ←→   core/target_state_estimation.py
│                              ←→   modules/sensor_fusion.py
├── 03_DMPC_FORMULATION        ←→   modules/dmpc_controller.py   (DMPC, MPCSolver)
├── 04_LYAPUNOV_AND_STABILITY  ←→   analysis/stability_analysis.py
│                              ←→   modules/dmpc_controller.py   (compute_lqr_terminal_cost)
├── 05_FORMATION_CONSENSUS     ←→   modules/formation_controller.py
├── 06_TASK_ALLOCATION         ←→   modules/task_allocator.py
├── 07_GEOMETRIC_ATTITUDE_CTRL ←→   modules/attitude_controller.py
└── 08_COVERAGE_PLANNING       ←→   modules/mission_planner.py
```

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| **x** | State vector |
| **u** | Control (input) vector |
| A, B  | Discrete-time system matrices |
| Q, R, P | LQR / MPC cost matrices |
| P_cov | Covariance matrix (estimation context) |
| K | Kalman gain or LQR gain (context-dependent) |
| N | MPC prediction horizon |
| Δt (dt) | Discrete time step (0.02 s = 50 Hz) |
| ρ | Spectral radius |
| λ | Eigenvalue |
| ‖·‖ | Euclidean norm (unless subscripted) |
| ‖·‖_Q | Quadratic norm: ‖x‖²_Q = xᵀQx |
| ℝⁿ | n-dimensional real vector space |
| SO(3) | Special orthogonal group in 3D |
| 𝒩(i) | Neighbours of drone i |
