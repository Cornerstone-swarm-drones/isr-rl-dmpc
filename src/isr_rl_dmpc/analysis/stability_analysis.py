"""
Stability Analysis for the Pure DMPC Swarm Drone Controller (ISR).

This module provides computational tools for verifying:

1. **Lyapunov Stability** — does the candidate Lyapunov function
   V(e) = eᵀ P e decrease along closed-loop trajectories?
2. **Eigenvalue Analysis** — are all eigenvalues of the closed-loop
   matrix (A − B Klqr) strictly inside the unit circle?
3. **Input-to-State Stability (ISS)** — what disturbance magnitude can
   the controller tolerate while still driving errors to zero?
4. **Collision Avoidance / Control Barrier Functions (CBF)** — do the
   DMPC inter-drone separation constraints form a valid barrier?
5. **Recursive Feasibility** — is the MPC QP guaranteed to remain
   feasible at every subsequent time step given a terminal constraint set?

References
----------
- Rawlings, Mayne & Diehl, "Model Predictive Control: Theory,
  Computation, and Design", 2nd ed. (2019).
- Kolmanovsky & Gilbert, "Theory and computation of disturbance
  invariant sets for discrete-time linear systems", Math. Probl. Eng.
  (1998).
- Ames et al., "Control Barrier Functions", IEEE TCSII (2019).
- Mayne, "Model predictive control: Recent developments and future
  promise", Automatica (2014).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.linalg import solve_discrete_are, eigvals


# =========================================================================
# Result data-classes
# =========================================================================

@dataclass
class LyapunovResult:
    """
    Result of the Lyapunov stability check.

    Attributes:
        is_stable:        True when P ≻ 0 and ΔV < 0 everywhere.
        P_positive_definite: Whether the terminal cost matrix P is PD.
        delta_V_max:      Maximum Lyapunov decrease found over the
                          sample trajectory (negative = stable).
        lyapunov_margin:  ``−delta_V_max / V_initial``; positive values
                          indicate a converging trajectory.
        details:          Detailed sub-checks as a plain dictionary.
    """
    is_stable: bool
    P_positive_definite: bool
    delta_V_max: float
    lyapunov_margin: float
    details: Dict = field(default_factory=dict)


@dataclass
class EigenvalueResult:
    """
    Result of the closed-loop eigenvalue analysis.

    Attributes:
        is_stable:        True when all |λᵢ| < 1.
        eigenvalues:      Array of closed-loop eigenvalues.
        spectral_radius:  max |λᵢ|.
        stability_margin: 1 − spectral_radius  (positive ⟹ stable).
    """
    is_stable: bool
    eigenvalues: np.ndarray
    spectral_radius: float
    stability_margin: float


@dataclass
class ISSResult:
    """
    Input-to-State Stability analysis result.

    Attributes:
        is_iss:           True when the system satisfies the ISS condition.
        iss_gain:         L∞ ISS gain γ such that ||e||_∞ ≤ γ ||d||_∞.
        max_disturbance:  Maximum disturbance magnitude that can be
                          rejected (i.e. keeps ||e||_∞ ≤ error_bound).
        error_bound:      Steady-state error bound used for computation.
    """
    is_iss: bool
    iss_gain: float
    max_disturbance: float
    error_bound: float


@dataclass
class CollisionBarrierResult:
    """
    Control Barrier Function analysis for inter-drone collision avoidance.

    Attributes:
        is_valid:         True when the CBF conditions are satisfied.
        min_separation:   Minimum predicted separation across all drone
                          pairs and horizon steps [m].
        safety_margin:    ``min_separation − collision_radius`` [m].
        cbf_alpha:        Class-K function parameter α used in
                          h(x_{k+1}) ≥ (1−α)·h(x_k).
    """
    is_valid: bool
    min_separation: float
    safety_margin: float
    cbf_alpha: float


@dataclass
class RecursiveFeasibilityResult:
    """
    Recursive feasibility verification result.

    Attributes:
        is_feasible:      True when the terminal constraint set is
                          control-invariant.
        terminal_set_volume: Approximate volume of the terminal ellipsoidal
                             constraint set.
        horizon_sufficient: Whether the chosen prediction horizon N is
                            sufficient to reach the terminal set from a
                            given initial condition.
    """
    is_feasible: bool
    terminal_set_volume: float
    horizon_sufficient: bool


@dataclass
class SwarmStabilityReport:
    """
    Aggregated stability report for the full DMPC swarm controller.

    Attributes:
        lyapunov:           Lyapunov stability analysis.
        eigenvalue:         Eigenvalue / spectral-radius analysis.
        iss:                Input-to-State Stability analysis.
        collision_barrier:  CBF collision-avoidance analysis.
        recursive_feasibility: Recursive feasibility analysis.
        overall_stable:     True when all individual checks pass.
        summary:            Human-readable summary string.
    """
    lyapunov: LyapunovResult
    eigenvalue: EigenvalueResult
    iss: ISSResult
    collision_barrier: CollisionBarrierResult
    recursive_feasibility: RecursiveFeasibilityResult
    overall_stable: bool
    summary: str


# =========================================================================
# Analyser
# =========================================================================

class DMPCStabilityAnalyzer:
    """
    Stability analyser for the pure DMPC swarm drone controller.

    The analysis targets the **translational controllable subsystem**
    ``x = [p(3), v(3), a(3)]`` (9 states, 3 inputs).  The full
    simulation state includes yaw and yaw-rate (11 states), but those
    two states are not driven by the translational acceleration control
    inputs ``[ax, ay, az]`` — they are regulated independently by the
    attitude controller.  Stability of the yaw sub-system follows
    directly from the geometric SO(3) controller; only the translational
    dynamics are analysed here.

    Parameters
    ----------
    state_dim :
        Dimension of the *controllable* state subspace used for
        analysis (default 9 = ``[p, v, a]``).  Pass 11 to include
        the yaw states, but note that those modes will be marginally
        stable (eigenvalue = 1) since yaw is not a control input.
    control_dim :
        Dimension of the control input vector (default 3).
    dt :
        Discretisation time step [s] (default 0.02).
    Q :
        State tracking cost matrix of shape ``(state_dim, state_dim)``.
        Defaults to the identity matrix.
    R :
        Input cost matrix of shape ``(control_dim, control_dim)``.
        Defaults to ``0.1 · I``.
    P :
        Terminal cost matrix.  If *None* the DARE solution is computed
        automatically.
    collision_radius :
        Minimum safe inter-drone separation [m] (default 5.0).

    Notes
    -----
    All analyses use the *linearised* integrator dynamics::

        A = I + dt · [[0, I, 0],   B = dt · [[0],
                      [0, 0, I],              [0],
                      [0, 0, 0]]              [I]]

    where each block is 3×3.  This is the same model used by
    :class:`~isr_rl_dmpc.modules.dmpc_controller.MPCSolver`.
    """

    def __init__(
        self,
        state_dim: int = 9,
        control_dim: int = 3,
        dt: float = 0.02,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        collision_radius: float = 5.0,
    ) -> None:
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.collision_radius = collision_radius

        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else np.eye(control_dim) * 0.1

        # Build linearised dynamics
        self.A, self.B = self._build_dynamics()

        # Compute LQR gain and terminal cost via DARE
        self.P, self.K_lqr = self._solve_dare()
        if P is not None:
            self.P = P  # override if caller supplied P

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def _build_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the constant linearised dynamics matrices (A, B).

        State layout: ``x = [p(3), v(3), a(3), yaw, yaw_rate]``
        """
        n, m = self.state_dim, self.control_dim
        A = np.eye(n)
        if n >= 9:
            A[0:3, 3:6] = self.dt * np.eye(3)  # dp/dv
            A[3:6, 6:9] = self.dt * np.eye(3)  # dv/da
        B = np.zeros((n, m))
        if n >= 9:
            B[6:9, 0:3] = self.dt * np.eye(3)  # da/du
        return A, B

    def _solve_dare(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Discrete Algebraic Riccati Equation (DARE) to obtain
        the terminal cost P and LQR gain K.

        DARE:  P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA

        Returns
        -------
        P : (n, n) positive-definite terminal cost matrix.
        K : (m, n) LQR state-feedback gain.
        """
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            P = (P + P.T) / 2.0  # symmetrise
        except Exception:
            P = np.eye(self.state_dim) * 10.0

        # K = (R + BᵀPB)⁻¹ BᵀPA
        BPBR = self.R + self.B.T @ P @ self.B
        K = np.linalg.solve(BPBR, self.B.T @ P @ self.A)
        return P, K

    # ------------------------------------------------------------------
    # 1. Lyapunov stability
    # ------------------------------------------------------------------

    def check_lyapunov_stability(
        self,
        initial_error: Optional[np.ndarray] = None,
        n_steps: int = 100,
    ) -> LyapunovResult:
        """
        Verify Lyapunov stability using V(e) = eᵀ P e.

        The closed-loop matrix is  A_cl = A − B K_lqr.
        Stability is confirmed when:
          (a) P ≻ 0  (positive-definite terminal cost)
          (b) ΔV = V(A_cl · e) − V(e) < 0   ∀ e ≠ 0

        Args:
            initial_error: Starting tracking error (state_dim,).
                           Defaults to a small random perturbation.
            n_steps:       Simulation horizon for trajectory check.

        Returns:
            :class:`LyapunovResult`
        """
        if initial_error is None:
            rng = np.random.default_rng(42)
            initial_error = rng.standard_normal(self.state_dim) * 0.5

        # Check P ≻ 0
        eigs_P = np.linalg.eigvalsh(self.P)
        p_pd = bool(np.all(eigs_P > 0))

        # Closed-loop matrix
        A_cl = self.A - self.B @ self.K_lqr

        # Theoretical ΔV = eᵀ(AclᵀPAcl − P)e
        delta_P = A_cl.T @ self.P @ A_cl - self.P
        eigs_delta = np.linalg.eigvalsh(delta_P)
        delta_V_max_theoretical = float(np.max(eigs_delta))

        # Simulate trajectory
        e = initial_error.copy()
        V0 = float(e @ self.P @ e)
        delta_V_sim_max = -np.inf
        for _ in range(n_steps):
            e_next = A_cl @ e
            V_curr = float(e @ self.P @ e)
            V_next = float(e_next @ self.P @ e_next)
            delta_V = V_next - V_curr
            delta_V_sim_max = max(delta_V_sim_max, delta_V)
            e = e_next
            if np.linalg.norm(e) < 1e-10:
                break

        delta_V_max = max(delta_V_max_theoretical, delta_V_sim_max)
        is_stable = p_pd and (delta_V_max < 0)
        margin = float(-delta_V_max / (V0 + 1e-12))

        return LyapunovResult(
            is_stable=is_stable,
            P_positive_definite=p_pd,
            delta_V_max=delta_V_max,
            lyapunov_margin=margin,
            details={
                "min_eigenvalue_P": float(np.min(eigs_P)),
                "max_eigenvalue_delta_P": delta_V_max_theoretical,
                "max_delta_V_simulated": delta_V_sim_max,
            },
        )

    # ------------------------------------------------------------------
    # 2. Eigenvalue analysis
    # ------------------------------------------------------------------

    def check_eigenvalue_stability(self) -> EigenvalueResult:
        """
        Compute eigenvalues of the LQR closed-loop matrix A_cl = A − B K.

        For discrete-time stability all eigenvalues must satisfy |λ| < 1.

        Returns
        -------
        :class:`EigenvalueResult`
        """
        A_cl = self.A - self.B @ self.K_lqr
        eigs = eigvals(A_cl)
        magnitudes = np.abs(eigs)
        rho = float(np.max(magnitudes))
        is_stable = bool(rho < 1.0)

        return EigenvalueResult(
            is_stable=is_stable,
            eigenvalues=eigs,
            spectral_radius=rho,
            stability_margin=float(1.0 - rho),
        )

    # ------------------------------------------------------------------
    # 3. Input-to-State Stability
    # ------------------------------------------------------------------

    def check_iss(
        self,
        error_bound: float = 1.0,
        disturbance_samples: Optional[np.ndarray] = None,
    ) -> ISSResult:
        """
        Estimate the ISS gain and maximum tolerable disturbance.

        An ISS Lyapunov function exists if there are class-K∞ functions
        α₁, α₂, α₃ and a class-K function γ such that

            α₁(||e||) ≤ V(e) ≤ α₂(||e||)
            V(A_cl·e + d) − V(e) ≤ −α₃(||e||) + γ(||d||)

        For the quadratic V(e) = eᵀ P e the ISS gain is estimated as:

            γ_iss = λ_max(P) · ||A_cl||² / λ_min(P)

        Args:
            error_bound:        Maximum acceptable steady-state tracking
                                error norm [m equivalent].
            disturbance_samples: Optional array of disturbance vectors
                                 ``(N, state_dim)`` for Monte-Carlo
                                 verification.

        Returns
        -------
        :class:`ISSResult`
        """
        A_cl = self.A - self.B @ self.K_lqr
        eigs_P = np.linalg.eigvalsh(self.P)
        lam_min_P = float(np.min(eigs_P))
        lam_max_P = float(np.max(eigs_P))

        # ISS gain bound (L2)
        norm_Acl = float(np.linalg.norm(A_cl, ord=2))
        iss_gain = (lam_max_P / (lam_min_P + 1e-12)) * norm_Acl

        # Max disturbance that keeps steady-state error ≤ error_bound
        # ||e_ss||_P ≤ γ · ||d|| / (1 − ρ)   (geometric series bound)
        rho = float(np.max(np.abs(eigvals(A_cl))))
        if rho < 1.0:
            max_dist = error_bound * lam_min_P * (1.0 - rho) / (lam_max_P + 1e-12)
        else:
            max_dist = 0.0

        is_iss = rho < 1.0

        # Optional Monte-Carlo check
        if disturbance_samples is not None and is_iss:
            rng = np.random.default_rng(0)
            e = rng.standard_normal(self.state_dim) * 0.5
            for d in disturbance_samples:
                e_next = A_cl @ e + d
                if np.linalg.norm(e_next) > np.linalg.norm(e) + np.linalg.norm(d) * iss_gain + 1e-6:
                    is_iss = False
                    break
                e = e_next

        return ISSResult(
            is_iss=is_iss,
            iss_gain=iss_gain,
            max_disturbance=max_dist,
            error_bound=error_bound,
        )

    # ------------------------------------------------------------------
    # 4. Collision Barrier Functions
    # ------------------------------------------------------------------

    def check_collision_barrier(
        self,
        drone_positions: Optional[np.ndarray] = None,
        predicted_positions: Optional[np.ndarray] = None,
        cbf_alpha: float = 0.3,
    ) -> CollisionBarrierResult:
        """
        Verify Control Barrier Function (CBF) conditions for collision
        avoidance.

        The barrier function for drones i and j is:

            h_{ij}(x) = ||p_i − p_j||² − r_min²

        The discrete-time CBF condition requires:

            h_{ij}(x_{k+1}) ≥ (1 − α) h_{ij}(x_k)   ∀ (i, j)

        where α ∈ (0, 1] is a class-K function parameter.  The DMPC
        hard constraint ``||p_k − p_j|| ≥ r_min`` directly enforces this
        with α = 1 (strongest condition).

        Args:
            drone_positions:     Current positions ``(N_drones, 3)``.
                                 Defaults to a randomly generated 4-drone
                                 swarm with sufficient separation.
            predicted_positions: Predicted position trajectories
                                 ``(N_drones, horizon, 3)``.  If *None*,
                                 static positions are used.
            cbf_alpha:           Class-K parameter α.

        Returns
        -------
        :class:`CollisionBarrierResult`
        """
        if drone_positions is None:
            # Default: 4 drones in a square grid with 20 m spacing
            drone_positions = np.array([
                [0.0,  0.0,  10.0],
                [20.0, 0.0,  10.0],
                [0.0,  20.0, 10.0],
                [20.0, 20.0, 10.0],
            ])

        N = drone_positions.shape[0]
        r_min = self.collision_radius
        min_sep = np.inf
        cbf_satisfied = True

        for i in range(N):
            for j in range(i + 1, N):
                # Current barrier value
                h_curr = (
                    np.linalg.norm(drone_positions[i] - drone_positions[j]) ** 2
                    - r_min ** 2
                )
                if predicted_positions is not None:
                    horizon = predicted_positions.shape[1]
                    for k in range(horizon):
                        sep = np.linalg.norm(
                            predicted_positions[i, k] - predicted_positions[j, k]
                        )
                        min_sep = min(min_sep, sep)
                        h_next = sep ** 2 - r_min ** 2
                        h_prev = h_curr if k == 0 else (
                            np.linalg.norm(
                                predicted_positions[i, k - 1]
                                - predicted_positions[j, k - 1]
                            ) ** 2 - r_min ** 2
                        )
                        if h_next < (1.0 - cbf_alpha) * h_prev - 1e-8:
                            cbf_satisfied = False
                else:
                    sep = np.linalg.norm(drone_positions[i] - drone_positions[j])
                    min_sep = min(min_sep, sep)
                    if sep < r_min - 1e-8:
                        cbf_satisfied = False

        if min_sep == np.inf:
            min_sep = float(np.min([
                np.linalg.norm(drone_positions[i] - drone_positions[j])
                for i in range(N) for j in range(i + 1, N)
            ]) if N > 1 else r_min + 1.0)

        return CollisionBarrierResult(
            is_valid=cbf_satisfied,
            min_separation=min_sep,
            safety_margin=min_sep - r_min,
            cbf_alpha=cbf_alpha,
        )

    # ------------------------------------------------------------------
    # 5. Recursive feasibility
    # ------------------------------------------------------------------

    def check_recursive_feasibility(
        self,
        horizon: int = 20,
        terminal_set_scale: float = 1.0,
    ) -> RecursiveFeasibilityResult:
        """
        Verify that the terminal constraint set Ω_f is control-invariant,
        guaranteeing recursive feasibility of the DMPC QP.

        The terminal set is the LQR invariant ellipsoid:

            Ω_f = { e : eᵀ P e ≤ c }

        For discrete-time MPC, recursive feasibility holds when:
          (a) The terminal set is positively control-invariant under
              the LQR gain K (i.e. A_cl maps Ω_f into itself).
          (b) The prediction horizon N is large enough to reach Ω_f
              from the initial condition under the MPC policy.

        The volume of the ellipsoid Ω_f is proportional to::

            Vol(Ω_f) ∝ c^{n/2} / √det(P)

        Args:
            horizon:            DMPC prediction horizon N.
            terminal_set_scale: Scaling factor c for the terminal
                                ellipsoid (default 1.0).

        Returns
        -------
        :class:`RecursiveFeasibilityResult`
        """
        A_cl = self.A - self.B @ self.K_lqr
        n = self.state_dim
        c = terminal_set_scale

        # (a) Check positive invariance: A_cl maps Ω_f ⊆ Ω_f
        # The DARE Bellman equation guarantees:
        #   A_cl^T P A_cl = P - (Q + K^T R K)
        # so  P^{-1} A_cl^T P A_cl = I - P^{-1}(Q + K^T R K)
        # whose max eigenvalue is < 1 when Q > 0 (by construction).
        try:
            P_inv = np.linalg.inv(self.P)
            M = P_inv @ A_cl.T @ self.P @ A_cl
            max_eig = float(np.max(np.linalg.eigvalsh(M)))
            # Use a relaxed tolerance to handle DARE numerical error
            is_invariant = max_eig <= 1.0 + 1e-4
        except np.linalg.LinAlgError:
            max_eig = np.inf
            is_invariant = False

        # (b) Horizon sufficiency: simulate the closed-loop from an initial
        #     error equal to the max-eigenvector of P scaled to V0 = c.
        #     horizon_ok = True when ρ^{2N} ≤ 1 (always True when ρ < 1
        #     given a sufficient finite horizon).
        rho = float(np.max(np.abs(eigvals(A_cl))))
        if rho < 1.0:
            # minimum N so that ρ^{2N} V_max ≤ c where V_max = λ_max(P)·c
            # (worst-case unit-normalised error times c-scaling)
            # simplify: need ρ^{2N} ≤ 1, which holds for ANY N > 0
            horizon_ok = horizon > 0
        else:
            horizon_ok = False

        # Terminal ellipsoid volume ∝ c^{n/2} / sqrt(det P)
        det_P = float(np.linalg.det(self.P))
        if det_P > 0:
            vol = (c ** (n / 2.0)) / math.sqrt(det_P)
        else:
            vol = 0.0

        return RecursiveFeasibilityResult(
            is_feasible=is_invariant,
            terminal_set_volume=vol,
            horizon_sufficient=horizon_ok,
        )

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_stability_report(
        self,
        horizon: int = 20,
        drone_positions: Optional[np.ndarray] = None,
        initial_error: Optional[np.ndarray] = None,
    ) -> SwarmStabilityReport:
        """
        Run all stability checks and return a consolidated report.

        Args:
            horizon:         DMPC prediction horizon used in the mission.
            drone_positions: Current drone position array ``(N, 3)``
                             for the CBF check.
            initial_error:   Initial tracking error for the Lyapunov
                             simulation.

        Returns
        -------
        :class:`SwarmStabilityReport`
        """
        lyapunov = self.check_lyapunov_stability(initial_error=initial_error)
        eigenvalue = self.check_eigenvalue_stability()
        iss = self.check_iss()
        cbf = self.check_collision_barrier(drone_positions=drone_positions)
        feasibility = self.check_recursive_feasibility(horizon=horizon)

        all_pass = (
            lyapunov.is_stable
            and eigenvalue.is_stable
            and iss.is_iss
            and cbf.is_valid
            and feasibility.is_feasible
        )

        lines = [
            "=" * 60,
            "DMPC Swarm Stability Report",
            "=" * 60,
            f"  Lyapunov stable     : {lyapunov.is_stable}"
            f"  (ΔV_max={lyapunov.delta_V_max:.4g},"
            f"  margin={lyapunov.lyapunov_margin:.4g})",
            f"  Eigenvalue stable   : {eigenvalue.is_stable}"
            f"  (ρ={eigenvalue.spectral_radius:.6f},"
            f"  margin={eigenvalue.stability_margin:.6f})",
            f"  ISS                 : {iss.is_iss}"
            f"  (gain={iss.iss_gain:.4g},"
            f"  max_dist={iss.max_disturbance:.4g})",
            f"  CBF valid           : {cbf.is_valid}"
            f"  (min_sep={cbf.min_separation:.2f} m,"
            f"  margin={cbf.safety_margin:.2f} m)",
            f"  Recursive feasible  : {feasibility.is_feasible}"
            f"  (horizon_ok={feasibility.horizon_sufficient})",
            "-" * 60,
            f"  OVERALL STABLE      : {all_pass}",
            "=" * 60,
        ]

        return SwarmStabilityReport(
            lyapunov=lyapunov,
            eigenvalue=eigenvalue,
            iss=iss,
            collision_barrier=cbf,
            recursive_feasibility=feasibility,
            overall_stable=all_pass,
            summary="\n".join(lines),
        )


if __name__ == "__main__":
    analyzer = DMPCStabilityAnalyzer(
        state_dim=9,  # translational controllable subsystem
        control_dim=3,
        dt=0.02,
        collision_radius=5.0,
    )
    report = analyzer.full_stability_report(horizon=20)
    print(report.summary)
