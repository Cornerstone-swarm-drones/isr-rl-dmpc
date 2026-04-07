"""
Module 7 - DMPC Controller: Pure CVXPY/OSQP Convex Optimisation

Implements a purely optimisation-based Distributed Model Predictive
Controller (DMPC) for swarm drone ISR missions.  All learning layers
(neural network cost-weight adaptation, residual dynamics networks,
and terminal value networks) have been removed.  The terminal cost
matrix P is instead computed analytically by solving the discrete-time
algebraic Riccati equation (DARE) via :func:`scipy.linalg.solve_discrete_are`.

Mathematical problem solved at each time step:

  min   Σ_{k=0}^{N-1} [||e_k||²_Q + ||u_k||²_R] + ||e_N||²_P
  s.t.  x_{k+1} = A x_k + B u_k          (linearised dynamics)
        |u_{k,ℓ}| ≤ u_max  ∀ k,ℓ          (per-axis box saturation)
        CBF(p_k, p_j) ≥ 0   ∀j∈𝒩          (linearised collision barrier)
        x_0 = x(t)                         (initial condition)

where e_k = x_k − x_ref_k is the tracking error.

Collision avoidance uses a linearised Control Barrier Function (CBF)
constraint that is affine in the decision variables and hence compatible
with the OSQP QP solver.  See 03_DMPC_FORMULATION.md §11 and
04_LYAPUNOV_AND_STABILITY.md §5 for the derivation.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are


@dataclass
class DMPCConfig:
    """Configuration for pure DMPC."""

    horizon: int = 20
    dt: float = 0.02
    state_dim: int = 11  # [p(3), v(3), a(3), yaw_angle, yaw_rate]
    control_dim: int = 3  # [ax, ay, az]
    n_neighbors: int = 4

    # Cost matrices
    Q_base: np.ndarray = field(default_factory=lambda: np.eye(11) * 1.0)
    R_base: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    # P_base: will be computed from DARE if not overridden (set to None to auto-compute)
    P_base: Optional[np.ndarray] = None

    # Safety & constraints
    accel_max: float = 10.0
    collision_radius: float = 5.0

    # CVXPY solver settings
    solver_timeout: float = 0.01  # 10 ms solver time budget

    def __post_init__(self) -> None:
        if self.P_base is None:
            # Compute LQR terminal cost via DARE
            self.P_base = compute_lqr_terminal_cost(
                self.state_dim, self.control_dim, self.Q_base, self.R_base, self.dt
            )


def compute_lqr_terminal_cost(
    state_dim: int,
    control_dim: int,
    Q: np.ndarray,
    R: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Compute the LQR terminal cost matrix P by solving the DARE.

    The discrete-time system uses the same integrator structure as
    :class:`MPCSolver` (positions ← velocities ← accelerations).

    Args:
        state_dim:   Number of state variables.
        control_dim: Number of control inputs.
        Q:           State-tracking cost matrix (state_dim × state_dim).
        R:           Input cost matrix (control_dim × control_dim).
        dt:          Discretisation time step in seconds.

    Returns:
        P: Positive-definite terminal cost matrix (state_dim × state_dim).
    """
    A = np.eye(state_dim)
    if state_dim >= 9:
        A[0:3, 3:6] = dt * np.eye(3)  # dp/dv
        A[3:6, 6:9] = dt * np.eye(3)  # dv/da
    B = np.zeros((state_dim, control_dim))
    if state_dim >= 9:
        B[6:9, 0:3] = dt * np.eye(3)  # da/du
    try:
        P = solve_discrete_are(A, B, Q, R)
        # Symmetrise to eliminate floating-point asymmetry
        P = (P + P.T) / 2.0
    except Exception:
        # Fall back to a scaled identity if DARE fails
        P = np.eye(state_dim) * 10.0
    return P


class MPCSolver:
    """CVXPY / OSQP real-time convex MPC solver.

    The QP is built once at construction (base constraints only).  Collision
    avoidance is handled via a linearised Control Barrier Function (CBF)
    constraint that is **affine** in the decision variables, ensuring the
    problem remains a valid convex QP at every call.

    From 04_LYAPUNOV_AND_STABILITY.md §5, the discrete-time CBF condition
    applied to each prediction step k and neighbour j is:

        h_k  = ‖p_nom_k − p_j‖² − r_min²     (barrier value at nominal trajectory)
        d_k  = p_nom_k − p_j                   (separation vector)
        dist_k = ‖d_k‖                         (scalar distance, clipped to ε)

    Linearised constraint (affine in x_var[:3, k]):

        d_k @ x_var[:3, k]  ≥  dist_k * r_min + d_k @ p_j

    The nominal trajectory p_nom_k is obtained by propagating x0 through
    the autonomous dynamics (u = 0): p_nom_{k+1} = (A @ x_nom_k)[:3].
    """

    def __init__(self, config: DMPCConfig):
        """
        Initialise the CVXPY QP problem.

        min  Σ_{k=0}^{N-1} [‖e_k‖²_Q + ‖u_k‖²_R] + ‖e_N‖²_P
        s.t. x_{k+1} = A x_k + B u_k        (linearised dynamics)
             |u_{k,ℓ}| ≤ u_max  ∀ k,ℓ       (per-axis box saturation)
             x_0      = x(t)                 (initial condition)
        Collision avoidance CBF constraints are added fresh each solve().
        """
        self.config = config
        self.horizon = config.horizon

        # Decision variables
        self.x_var = cp.Variable((config.state_dim, config.horizon + 1))
        self.u_var = cp.Variable((config.control_dim, config.horizon))

        # Parameters (updated each solve)
        self.x0_param = cp.Parameter(config.state_dim)
        self.A_param = cp.Parameter((config.state_dim, config.state_dim))
        self.B_param = cp.Parameter((config.state_dim, config.control_dim))
        self.Q_param = cp.Parameter((config.state_dim, config.state_dim), PSD=True)
        self.R_param = cp.Parameter((config.control_dim, config.control_dim), PSD=True)
        self.P_param = cp.Parameter((config.state_dim, config.state_dim), PSD=True)
        self.x_ref_param = cp.Parameter((config.state_dim, config.horizon + 1))

        # Build cost function
        cost = 0
        for k in range(self.horizon):
            x_err = self.x_var[:, k] - self.x_ref_param[:, k]
            u_err = self.u_var[:, k]
            cost += cp.quad_form(x_err, self.Q_param)
            cost += cp.quad_form(u_err, self.R_param)

        # Terminal cost
        x_err_T = self.x_var[:, -1] - self.x_ref_param[:, -1]
        cost += cp.quad_form(x_err_T, self.P_param)

        self._objective = cp.Minimize(cost)

        # Build BASE constraints (initial condition + dynamics + saturation).
        # These never change across calls; collision CBF constraints are added
        # fresh in solve() using a newly constructed Problem.
        self._base_constraints: List = []

        # Initial condition
        self._base_constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics: x_{k+1} = A x_k + B u_k
        for k in range(self.horizon):
            self._base_constraints.append(
                self.x_var[:, k + 1]
                == self.A_param @ self.x_var[:, k] + self.B_param @ self.u_var[:, k]
            )

        # Control saturation: per-axis box constraints |u_{k,ℓ}| ≤ u_max
        # (linear constraints — required for OSQP; see 03_DMPC_FORMULATION.md §11)
        for k in range(self.horizon):
            self._base_constraints.append(
                self.u_var[:, k] <= config.accel_max
            )
            self._base_constraints.append(
                self.u_var[:, k] >= -config.accel_max
            )

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        neighbor_positions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve the MPC QP for the current state.

        Args:
            x0:               Current state [state_dim].
            x_ref:            Reference trajectory of shape
                              ``(horizon+1, state_dim)``.
            A, B:             Linearised dynamics matrices.
            Q, R, P:          Cost matrices.
            neighbor_positions: List of neighbour 3-D position vectors for
                              linearised CBF collision avoidance.

        Returns:
            u_opt: Optimal control sequence ``(horizon, control_dim)``.
            info:  Dict with keys ``status``, ``solve_time``, ``objective``,
                   and optionally ``x_trajectory``.
        """
        # Update parameters
        self.x0_param.value = x0
        self.A_param.value = A
        self.B_param.value = B
        self.Q_param.value = Q
        self.R_param.value = R
        self.P_param.value = P
        self.x_ref_param.value = x_ref.T  # (state_dim, horizon+1)

        # Build linearised CBF collision constraints fresh each call.
        # Constraint (affine in x_var[:3, k]):
        #   d_k @ x_var[:3, k]  ≥  dist_k * r_min + d_k @ p_j
        # where d_k = p_nom_k − p_j, dist_k = ‖d_k‖.
        _MIN_SEPARATION = 1e-3  # degenerate-direction guard [m]
        collision_constraints: List = []
        r_min = self.config.collision_radius * 0.9  # 10 % safety margin
        if neighbor_positions:
            x_nom = x0.copy()
            for k in range(self.horizon):
                p_nom_k = x_nom[:3]
                for p_j in neighbor_positions:
                    d = p_nom_k - p_j
                    dist = float(np.linalg.norm(d))
                    if dist < _MIN_SEPARATION:
                        # Numerically degenerate — skip this step/neighbour pair
                        continue
                    rhs = float(dist * r_min + d @ p_j)
                    collision_constraints.append(
                        d @ self.x_var[:3, k] >= rhs
                    )
                # Propagate nominal trajectory (u = 0)
                x_nom = A @ x_nom

        # Assemble and solve a fresh Problem (avoids constraint accumulation)
        all_constraints = self._base_constraints + collision_constraints
        problem = cp.Problem(self._objective, all_constraints)

        try:
            problem.solve(
                solver=cp.OSQP,
                max_iter=3000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                time_limit=self.config.solver_timeout,
            )
        except Exception as e:
            print(f"Solver warning: {e}")
            return np.zeros((self.horizon, self.config.control_dim)), {
                "status": "error",
                "solve_time": 0,
                "objective": np.inf,
            }

        if problem.status in ("optimal", "optimal_inaccurate"):
            u_val = self.u_var.value
            x_val = self.x_var.value
            u_opt = np.array(u_val).T if u_val is not None else \
                np.zeros((self.horizon, self.config.control_dim))
            solve_time = (
                problem.solver_stats.solve_time
                if problem.solver_stats is not None
                else 0.0
            )
            return u_opt, {
                "status": problem.status,
                "solve_time": solve_time,
                "objective": problem.value if problem.value is not None else np.inf,
                "x_trajectory": np.array(x_val).T if x_val is not None else None,
            }
        else:
            print(f"Solver status: {problem.status}")
            return np.zeros((self.horizon, self.config.control_dim)), {
                "status": problem.status,
                "solve_time": 0,
                "objective": np.inf,
            }


class DMPC:
    """
    Pure DMPC controller using CVXPY/OSQP.

    All decisions are made through convex optimisation — no neural
    networks or learning layers.  The terminal cost matrix P is
    computed once from the DARE and kept fixed throughout operation.

    Usage::

        config = DMPCConfig()
        controller = DMPC(config)
        u_opt, info = controller(x0, x_ref)
        # info keys: 'status', 'solve_time', 'objective', 'x_trajectory'
    """

    def __init__(self, config: DMPCConfig) -> None:
        self.config = config

        # CVXPY/OSQP solver
        self.cvxpy_solver = MPCSolver(config)

        # Fixed cost matrices (no NN scaling)
        self.Q = config.Q_base.copy()
        self.R = config.R_base.copy()
        self.P = config.P_base.copy()

    def _get_linearized_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the constant linearised dynamics matrices A and B.

        State layout: x = [p(3), v(3), a(3), yaw, yaw_rate]
        """
        A = np.eye(self.config.state_dim)
        A[0:3, 3:6] = self.config.dt * np.eye(3)  # dp/dv
        A[3:6, 6:9] = self.config.dt * np.eye(3)  # dv/da

        B = np.zeros((self.config.state_dim, self.config.control_dim))
        B[6:9, 0:3] = self.config.dt * np.eye(3)  # da/du

        return A, B

    def __call__(
        self,
        x: np.ndarray,
        x_ref: np.ndarray,
        neighbor_states: Optional[List[np.ndarray]] = None,
        q_scale: Optional[np.ndarray] = None,
        r_scale: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute the optimal control sequence for the current state.

        Args:
            x:               Current state vector of shape ``(state_dim,)``.
            x_ref:           Reference trajectory of shape
                             ``(horizon, state_dim)`` or
                             ``(horizon+1, state_dim)``.
                             If only ``horizon`` steps are given, the last
                             step is replicated to fill the terminal slot.
            neighbor_states: List of neighbour state vectors used for
                             collision avoidance.  Only the first 3
                             components (position) are used.
            q_scale:         Per-state cost scale vector of shape
                             ``(state_dim,)`` from MARL policy.
                             ``Q_eff = Q ⊙ diag(q_scale)`` (Hadamard product
                             with diagonal matrix — see 03_DMPC_FORMULATION.md
                             §2).  Uses base Q when ``None``.
            r_scale:         Per-input cost scale vector of shape
                             ``(control_dim,)`` from MARL policy.
                             ``R_eff = R ⊙ diag(r_scale)``.  Uses base R
                             when ``None``.

        Returns:
            u_opt: Optimal control sequence ``(horizon, control_dim)``.
            info:  Dictionary with keys ``status``, ``solve_time``,
                   ``objective``, and optionally ``x_trajectory``.
        """
        A, B = self._get_linearized_dynamics()

        # Normalise x_ref to shape (horizon+1, state_dim)
        x_ref = np.asarray(x_ref, dtype=np.float64)
        if x_ref.shape[0] == self.config.horizon:
            x_ref = np.vstack([x_ref, x_ref[-1:]])  # replicate terminal step

        neighbor_positions: Optional[List[np.ndarray]] = None
        if neighbor_states is not None:
            neighbor_positions = [s[:3] for s in neighbor_states]

        # Apply MARL-supplied cost scales when provided.
        # Q_eff = Q ⊙ diag(q_scale): column-scale Q so each state channel's
        # contribution is weighted by the corresponding q_scale element.
        # For the default Q = I this gives diag(q_scale), which is PSD iff
        # all q_scale entries are positive (guaranteed by the [0.1, 10] clip).
        if q_scale is not None:
            qs = np.clip(np.asarray(q_scale, dtype=np.float64), 0.1, 10.0)
            Q_eff = self.Q * qs  # broadcasts: Q[i,j] * qs[j] — column scaling
            Q_eff = (Q_eff + Q_eff.T) / 2.0  # symmetrise for numerical safety
        else:
            Q_eff = self.Q

        if r_scale is not None:
            rs = np.clip(np.asarray(r_scale, dtype=np.float64), 0.1, 10.0)
            R_eff = self.R * rs  # column scaling
            R_eff = (R_eff + R_eff.T) / 2.0
        else:
            R_eff = self.R

        return self.cvxpy_solver.solve(
            x0=x,
            x_ref=x_ref,
            A=A,
            B=B,
            Q=Q_eff,
            R=R_eff,
            P=self.P,
            neighbor_positions=neighbor_positions,
        )

    def save_config(self, path: str) -> None:
        """Persist cost matrices and configuration to a NumPy archive."""
        np.savez(
            path,
            Q=self.Q,
            R=self.R,
            P=self.P,
            horizon=np.array(self.config.horizon),
            dt=np.array(self.config.dt),
            accel_max=np.array(self.config.accel_max),
            collision_radius=np.array(self.config.collision_radius),
        )
        print(f"DMPC config saved to {path}")

    def load_config(self, path: str) -> None:
        """Restore cost matrices from a NumPy archive."""
        data = np.load(path)
        self.Q = data["Q"]
        self.R = data["R"]
        self.P = data["P"]
        print(f"DMPC config loaded from {path}")


if __name__ == "__main__":
    config = DMPCConfig()
    controller = DMPC(config)

    x0 = np.random.randn(11).astype(np.float32)
    x_ref = np.random.randn(21, 11).astype(np.float32)

    u_opt, info = controller(x0, x_ref)

    print("✓ Pure DMPC solve successful")
    print(f"  Status:       {info['status']}")
    print(f"  Solve time:   {info['solve_time']:.6f}s")
    print(f"  Control shape:{u_opt.shape}")
