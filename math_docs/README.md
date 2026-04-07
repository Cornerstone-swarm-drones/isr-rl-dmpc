# Math Documentation — ISR-RL-DMPC

This directory contains self-contained mathematical reference documents for every
algorithm, model, and control method used in the **ISR-RL-DMPC** system.  The
system implements a **Multi-Agent Reinforcement Learning (MARL) framework using
MAPPO to dynamically tune the cost parameters of a DMPC layer**, where
coordination between drones is achieved through **ADMM** to ensure consensus on
shared constraints and objectives.

Each document introduces the problem, derives the equations from first principles,
and links back to the corresponding source file(s).

---

## Contents

| # | File | Topic |
| :--- | :--- | :--- |
| 1 | [01_DRONE_STATE_SPACE.md](01_DRONE_STATE_SPACE.md) | Drone state vectors, discrete-time linearised dynamics, integrator chain |
| 2 | [02_EXTENDED_KALMAN_FILTER.md](02_EXTENDED_KALMAN_FILTER.md) | EKF predict/update equations, 18-D drone estimation, 11-D target tracking, multi-sensor fusion |
| 3 | [03_DMPC_FORMULATION.md](03_DMPC_FORMULATION.md) | DMPC QP optimisation, MARL-adaptive cost matrices, DARE terminal cost, OSQP solver |
| 4 | [04_LYAPUNOV_AND_STABILITY.md](04_LYAPUNOV_AND_STABILITY.md) | Lyapunov stability, eigenvalue analysis, ISS, Control Barrier Functions, recursive feasibility |
| 5 | [05_FORMATION_CONSENSUS.md](05_FORMATION_CONSENSUS.md) | Distributed consensus protocol, graph Laplacian, ADMM formation consensus |
| 6 | [06_TASK_ALLOCATION.md](06_TASK_ALLOCATION.md) | Bipartite assignment problem, Hungarian algorithm, cost-matrix construction |
| 7 | [07_GEOMETRIC_ATTITUDE_CONTROL.md](07_GEOMETRIC_ATTITUDE_CONTROL.md) | Quaternion kinematics, SO(3) attitude error, motor mixing |
| 8 | [08_COVERAGE_PLANNING.md](08_COVERAGE_PLANNING.md) | Grid decomposition, waypoint generation, coverage path strategies |
| 9 | [09_MAPPO_AGENT.md](09_MAPPO_AGENT.md) | MARL architecture, MAPPO policy gradient, centralised critic, CTDE |
| 10 | [10_ADMM_CONSENSUS.md](10_ADMM_CONSENSUS.md) | ADMM derivation, dual decomposition, swarm consensus convergence |

---

## How These Relate to the Source Code

```
math_docs/                           src/isr_rl_dmpc/
├── 01_DRONE_STATE_SPACE       ←→   modules/dmpc_controller.py  (_get_linearized_dynamics)
│                              ←→   core/data_structures.py      (DroneState)
├── 02_EXTENDED_KALMAN_FILTER  ←→   core/drone_state_estimation.py
│                              ←→   core/target_state_estimation.py
│                              ←→   modules/sensor_fusion.py
├── 03_DMPC_FORMULATION        ←→   modules/dmpc_controller.py   (DMPC, MPCSolver)
├── 04_LYAPUNOV_AND_STABILITY  ←→   analysis/stability_analysis.py
│                              ←→   modules/dmpc_controller.py   (compute_lqr_terminal_cost)
├── 05_FORMATION_CONSENSUS     ←→   modules/formation_controller.py
│                              ←→   modules/admm_consensus.py
├── 06_TASK_ALLOCATION         ←→   modules/task_allocator.py
├── 07_GEOMETRIC_ATTITUDE_CTRL ←→   modules/attitude_controller.py
├── 08_COVERAGE_PLANNING       ←→   modules/mission_planner.py
├── 09_MAPPO_AGENT             ←→   agents/mappo_agent.py
│                              ←→   gym_env/marl_env.py
└── 10_ADMM_CONSENSUS          ←→   modules/admm_consensus.py
```

---

## Notation Conventions

| Symbol | Meaning |
| :--- | :--- |
| $\boldsymbol{x}$ | State vector |
| $\boldsymbol{u}$ | Control (input) vector |
| $A, B$ | Discrete-time system matrices |
| $Q, R, P$ | LQR / MPC cost matrices |
| $\boldsymbol{q}_s, \boldsymbol{r}_s$ | MAPPO-output Q and R scale vectors |
| $P_{\text{cov}}$ | Covariance matrix (estimation context) |
| $K$ | Kalman gain or LQR gain (context-dependent) |
| $N$ | MPC prediction horizon |
| $\Delta t$ | Discrete time step (0.02 s = 50 Hz) |
| $\rho$ | Spectral radius or ADMM penalty parameter |
| $\lambda$ | Eigenvalue |
| $\lVert\cdot\rVert$ | Euclidean norm (unless subscripted) |
| $\lVert\cdot\rVert_Q$ | Quadratic norm: $\lVert x\rVert_Q^2 = x^\top Q x$ |
| $\mathbb{R}^n$ | $n$-dimensional real vector space |
| $\mathrm{SO}(3)$ | Special orthogonal group in 3D |
| $\mathcal{N}(i)$ | Neighbours of drone $i$ |
| $\pi_\theta$ | MAPPO stochastic policy with parameters $\theta$ |
| $V_\phi$ | Centralised value function with parameters $\phi$ |
