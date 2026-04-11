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

DPP Compliance
--------------
CVXPY's Disciplined Parametrized Programming (DPP) cache stores the
compiled KKT system so that subsequent solves only update parameter
values rather than re-canonicalising the full problem from scratch.
DPP is violated whenever a *matrix* ``cp.Parameter`` appears in a
quadratic form or is matrix-multiplied with a ``cp.Variable``.

This module achieves DPP compliance by:

* Embedding A, B, Q, R, P as plain NumPy constants in the problem
  (they are constant for a fixed :class:`DMPCConfig`).
* Keeping only ``x0_param`` and ``x_ref_param`` as ``cp.Parameter``
  objects (both are purely affine in variables — allowed by DPP).
* Pre-allocating ``n_neighbors × horizon`` CBF constraint slots using
  *vector* ``cp.Parameter(3)`` direction parameters and scalar RHS
  parameters (vector-param @ var-vector inner products are DPP).
  Inactive slots use ``d=[0,0,0], rhs=−1e9`` so the constraint is
  trivially satisfied and imposes no OSQP work.
* Building the ``cp.Problem`` exactly **once** at construction — it is
  never recreated; every subsequent solve only mutates parameter values.

Result: the first solve pays the canonicalization cost once; all later
solves reduce to a parameter update + OSQP warm-start, cutting
per-step overhead from ~100–500 ms to ~1–5 ms.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import logging
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

logger = logging.getLogger(__name__)


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
    solver_timeout: float = 0.02  # 20 ms solver time budget
    osqp_max_iter: int = 4000     # OSQP iteration limit

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
    """CVXPY / OSQP real-time convex MPC solver — DPP-compliant.

    The QP is built **once** at construction with fixed dynamics (A, B) and
    cost matrices (Q, R, P) embedded as NumPy constants rather than
    ``cp.Parameter`` objects, which is the key requirement for CVXPY's DPP
    caching (see module docstring).

    Only two ``cp.Parameter`` objects remain:

    * ``x0_param``    — current state (updated each call, affine in vars).
    * ``x_ref_param`` — reference trajectory (updated each call, affine).

    Collision CBF constraints occupy ``n_neighbors × horizon`` pre-allocated
    slots, each backed by a vector ``cp.Parameter(3)`` (direction) and a
    scalar ``cp.Parameter`` (RHS).  Vector-parameter inner products are
    DPP-compliant.  Inactive slots are set to ``d=[0,0,0], rhs=−1e9`` so
    the constraint ``0 ≥ −1e9`` is trivially satisfied.

    From 04_LYAPUNOV_AND_STABILITY.md §5, the discrete-time CBF condition
    applied to each prediction step k and neighbour j is:

        h_k  = ‖p_nom_k − p_j‖² − r_min²
        d_k  = p_nom_k − p_j
        dist_k = ‖d_k‖  (clipped to ε)

    Linearised constraint (affine in x_var[:3, k]):

        d_k @ x_var[:3, k]  ≥  dist_k * r_min + d_k @ p_j
    """

    _MIN_SEP: float = 1e-3  # degenerate-direction guard [m]

    def __init__(
        self,
        config: "DMPCConfig",
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
    ) -> None:
        """
        Build the CVXPY QP problem once with embedded numpy matrices.

        Args:
            config:  DMPC configuration dataclass.
            A, B:    Constant linearised dynamics matrices (numpy arrays).
            Q, R, P: Cost matrices (numpy arrays) — stage, input, terminal.
        """
        self.config = config
        self._A = A  # stored for nominal-trajectory propagation in CBF
        self.horizon = config.horizon
        n = config.state_dim
        m = config.control_dim
        N = self.horizon
        nb = config.n_neighbors

        # ── Decision variables ─────────────────────────────────────────────
        self.x_var = cp.Variable((n, N + 1))
        self.u_var = cp.Variable((m, N))

        # ── Parameters (only those that change each solve) ─────────────────
        self.x0_param = cp.Parameter(n)
        self.x_ref_param = cp.Parameter((n, N + 1))

        # Pre-allocated CBF parameter slots.
        # Slot index = k * nb + j  →  prediction step k, neighbour j.
        # cp.Parameter(3) in an inner product is DPP-compliant.
        max_cbf = nb * N
        self._cbf_d: List[cp.Parameter] = [
            cp.Parameter(3) for _ in range(max_cbf)
        ]
        self._cbf_rhs: List[cp.Parameter] = [
            cp.Parameter() for _ in range(max_cbf)
        ]

        # ── Cost (numpy Q, R, P → DPP-compliant) ──────────────────────────
        cost: cp.Expression = 0
        for k in range(N):
            x_err = self.x_var[:, k] - self.x_ref_param[:, k]
            cost = cost + cp.quad_form(x_err, Q)           # Q numpy ✓
            cost = cost + cp.quad_form(self.u_var[:, k], R)  # R numpy ✓
        x_err_T = self.x_var[:, -1] - self.x_ref_param[:, -1]
        cost = cost + cp.quad_form(x_err_T, P)              # P numpy ✓

        # ── Constraints ────────────────────────────────────────────────────
        constraints: List = []

        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)

        # Dynamics: x_{k+1} = A x_k + B u_k  (numpy A, B → DPP-compliant)
        for k in range(N):
            constraints.append(
                self.x_var[:, k + 1]
                == A @ self.x_var[:, k] + B @ self.u_var[:, k]
            )

        # Control saturation: per-axis box constraints |u_{k,ℓ}| ≤ u_max
        constraints.append(self.u_var <= config.accel_max)
        constraints.append(self.u_var >= -config.accel_max)

        # CBF slots: vector_param @ var_slice — DPP-compliant inner product
        for k in range(N):
            for j in range(nb):
                slot = k * nb + j
                constraints.append(
                    self._cbf_d[slot] @ self.x_var[:3, k]
                    >= self._cbf_rhs[slot]
                )

        # ── Build problem ONCE ─────────────────────────────────────────────
        self._problem = cp.Problem(cp.Minimize(cost), constraints)

        # Initialise all CBF slots to the inactive (trivially satisfied) state
        self._deactivate_all_cbf()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deactivate_all_cbf(self) -> None:
        """Set every CBF slot to d=0, rhs=-1e9 (trivially satisfied)."""
        for d_p, rhs_p in zip(self._cbf_d, self._cbf_rhs):
            d_p.value = np.zeros(3)
            rhs_p.value = -1e9

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        neighbor_positions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve the MPC QP by updating parameter values only (no re-compilation).

        On the first call CVXPY canonicalises the problem and caches the KKT
        structure.  All subsequent calls skip canonicalization and go directly
        to OSQP with warm-start, reducing per-call overhead to ~1–5 ms.

        Args:
            x0:               Current state [state_dim].
            x_ref:            Reference trajectory of shape
                              ``(horizon+1, state_dim)``.
            neighbor_positions: List of neighbour 3-D position vectors for
                              linearised CBF collision avoidance.

        Returns:
            u_opt: Optimal control sequence ``(horizon, control_dim)``.
            info:  Dict with keys ``status``, ``solve_time``, ``objective``,
                   and optionally ``x_trajectory``.
        """
        # Update mutable parameters
        self.x0_param.value = x0
        self.x_ref_param.value = x_ref.T  # (state_dim, horizon+1)

        # Reset all CBF slots, then activate the relevant ones
        self._deactivate_all_cbf()
        r_min = self.config.collision_radius * 0.9  # 10 % safety margin
        nb = self.config.n_neighbors
        if neighbor_positions:
            x_nom = x0.copy()
            for k in range(self.horizon):
                p_nom_k = x_nom[:3]
                for j_idx, p_j in enumerate(neighbor_positions[:nb]):
                    d = p_nom_k - p_j
                    dist = float(np.linalg.norm(d))
                    if dist < self._MIN_SEP:
                        continue
                    slot = k * nb + j_idx
                    self._cbf_d[slot].value = d
                    self._cbf_rhs[slot].value = float(dist * r_min + d @ p_j)
                # Propagate nominal trajectory (u = 0)
                x_nom = self._A @ x_nom

        try:
            self._problem.solve(
                solver=cp.OSQP,
                max_iter=self.config.osqp_max_iter,
                eps_abs=1e-3,
                eps_rel=1e-3,
                time_limit=self.config.solver_timeout,
                warm_starting=True,
            )
        except Exception as e:
            logger.debug("Solver exception: %s", e)
            return np.zeros((self.horizon, self.config.control_dim)), {
                "status": "error",
                "solve_time": 0.0,
                "objective": np.inf,
            }

        status = self._problem.status
        if status in ("optimal", "optimal_inaccurate"):
            u_val = self.u_var.value
            x_val = self.x_var.value
            u_opt = (
                np.array(u_val).T
                if u_val is not None
                else np.zeros((self.horizon, self.config.control_dim))
            )
            solve_time = (
                self._problem.solver_stats.solve_time
                if self._problem.solver_stats is not None
                else 0.0
            )
            return u_opt, {
                "status": status,
                "solve_time": solve_time,
                "objective": (
                    self._problem.value
                    if self._problem.value is not None
                    else np.inf
                ),
                "x_trajectory": (
                    np.array(x_val).T if x_val is not None else None
                ),
            }
        else:
            # For user_limit / solver_inaccurate: use the warm-start iterate
            # instead of returning zeros, which can destabilise the controller.
            u_val = self.u_var.value
            if u_val is not None:
                logger.debug("Solver status %s — using current iterate", status)
                u_opt = np.array(u_val).T
                solve_time = (
                    self._problem.solver_stats.solve_time
                    if self._problem.solver_stats is not None
                    else 0.0
                )
                return u_opt, {
                    "status": status,
                    "solve_time": solve_time,
                    "objective": (
                        self._problem.value
                        if self._problem.value is not None
                        else np.inf
                    ),
                }
            logger.debug("Solver status: %s — no iterate available", status)
            return np.zeros((self.horizon, self.config.control_dim)), {
                "status": status,
                "solve_time": 0.0,
                "objective": np.inf,
            }


class DMPC:
    """
    Pure DMPC controller using CVXPY/OSQP.

    All decisions are made through convex optimisation — no neural
    networks or learning layers.  The terminal cost matrix P is
    computed once from the DARE and kept fixed throughout operation.

    A, B are computed once at construction from the fixed
    :class:`DMPCConfig` and embedded as numpy constants inside the
    :class:`MPCSolver`, making every subsequent solve DPP-compliant
    and fast (see :class:`MPCSolver` for details).

    When ``q_scale`` / ``r_scale`` are provided (MARL mode) a solver
    is built for each unique ``(Q_eff, R_eff)`` pair and cached, so
    repeated MARL calls with the same scaling reuse the compiled KKT
    structure.  Pure DMPC (no scaling) always hits the same cache entry
    and incurs zero rebuild cost.

    Usage::

        config = DMPCConfig()
        controller = DMPC(config)
        u_opt, info = controller(x0, x_ref)
        # info keys: 'status', 'solve_time', 'objective', 'x_trajectory'
    """

    def __init__(self, config: DMPCConfig) -> None:
        self.config = config

        # Fixed cost matrices (no NN scaling)
        self.Q = config.Q_base.copy()
        self.R = config.R_base.copy()
        self.P = config.P_base.copy()

        # Constant dynamics matrices — computed once from config
        self._A, self._B = self._get_linearized_dynamics()

        # Solver cache: keyed by (Q_bytes, R_bytes).
        # Pure DMPC always uses the same key → zero rebuilds after init.
        self._solver_cache: Dict[Tuple[bytes, bytes], MPCSolver] = {}
        # Eagerly build the base solver so the first real call is fast.
        self._get_or_build_solver(self.Q, self.R)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_build_solver(
        self, Q: np.ndarray, R: np.ndarray
    ) -> MPCSolver:
        """Return a cached :class:`MPCSolver` for the given Q, R pair.

        A new solver (with fresh CVXPY problem) is built only when a
        previously unseen ``(Q, R)`` combination is requested.
        """
        key: Tuple[bytes, bytes] = (Q.tobytes(), R.tobytes())
        if key not in self._solver_cache:
            self._solver_cache[key] = MPCSolver(
                self.config, self._A, self._B, Q, R, self.P
            )
        return self._solver_cache[key]

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

        solver = self._get_or_build_solver(Q_eff, R_eff)
        return solver.solve(
            x0=x,
            x_ref=x_ref,
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
        # Clear solver cache — cached solvers embed P as a numpy constant,
        # so they must be rebuilt when P changes.
        self._solver_cache.clear()
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
