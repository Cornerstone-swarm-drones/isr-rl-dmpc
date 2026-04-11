"""
MARLDMPCEnv — Multi-Agent Reinforcement Learning environment for MAPPO-adaptive DMPC.

Each agent (drone) observes a 40-D local observation and outputs a 14-D action
vector of multiplicative cost-scale factors for its DMPC Q and R matrices.

Observation space (40-D per agent):
  [0:11]  Own DMPC state   [p(3), v(3), a(3), yaw, yaw_rate]
  [11:14] Reference position
  [14:17] Reference velocity
  [17:20] Tracking error   e_p = p - p_ref
  [20:26] Nearest-neighbour relative state [Δp(3), Δv(3)]
  [26:29] Mean swarm position offset
  [29]    Battery level (normalised)
  [30]    Structural health (normalised)
  [31:34] Last applied control u
  [34:37] ADMM primal residual (per axis)
  [37]    DMPC solve time (normalised)
  [38]    Collision margin (normalised)
  [39]    Mission progress t / T_max

Action space (14-D per agent):
  [0:11] q_scale — multiplicative scales for Q diagonal (state_dim=11)
  [11:14] r_scale — multiplicative scales for R diagonal (control_dim=3)
  All values clipped to [0.1, 10.0].

See docs/GYM_DESIGN.md and math_docs/09_MAPPO_AGENT.md for full derivations.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from isr_rl_dmpc.modules.dmpc_controller import DMPC, DMPCConfig
from isr_rl_dmpc.modules.admm_consensus import ADMMConsensus, ADMMConfig
from isr_rl_dmpc.gym_env.simulator import (
    EnvironmentSimulator,
    DroneConfig,
    TargetConfig,
    EnvironmentConfig,
    TargetType,
)

# Observation and action dimensions (fixed by architecture)
OBS_DIM = 40
ACT_DIM = 14  # 11 q_scales + 3 r_scales
STATE_DIM = 11
CONTROL_DIM = 3

# ── Scenario-specific reward weights ──────────────────────────────────────
# Keys: W_TRACK, W_FORM, W_SAFE, W_EFF, W_COV
# Rationale:
#   W_TRACK  — tracking reward (positive, exp-decay) — higher in target-centric missions
#   W_FORM   — formation penalty (only separation violations) — kept low to allow spreading
#   W_SAFE   — collision safety penalty — always high
#   W_EFF    — control effort penalty — very small, just a tie-breaker
#   W_COV    — coverage reward (delta + absolute) — highest in area_surveillance
_SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "area_surveillance": {
        "W_TRACK": 3.0,
        "W_FORM":  0.5,
        "W_SAFE":  10.0,
        "W_EFF":   0.05,
        "W_COV":   20.0,
    },
    "threat_response": {
        "W_TRACK": 8.0,
        "W_FORM":  0.5,
        "W_SAFE":  10.0,
        "W_EFF":   0.05,
        "W_COV":   3.0,
    },
    "search_and_track": {
        "W_TRACK": 5.0,
        "W_FORM":  0.5,
        "W_SAFE":  10.0,
        "W_EFF":   0.05,
        "W_COV":   12.0,
    },
}
_DEFAULT_WEIGHTS: Dict[str, float] = _SCENARIO_WEIGHTS["area_surveillance"]


class MARLDMPCEnv(gym.Env):
    """
    Gymnasium environment for MAPPO-adaptive DMPC swarm control.

    Follows the Centralised Training / Decentralised Execution (CTDE)
    paradigm.  During training a single `PPO` instance from
    Stable-Baselines3 trains a *shared* actor policy that is applied to
    each drone independently.

    The environment flattens the per-drone observations into a single
    observation vector ``(num_drones * OBS_DIM,)`` and a single action
    vector ``(num_drones * ACT_DIM,)``.  This makes it compatible with
    the standard SB3 `PPO` interface without requiring a custom
    multi-agent wrapper.

    Reference trajectory planning
    ------------------------------
    Each scenario gets a dedicated trajectory planner that updates
    ``_references`` at the start of every ``step()`` call:

    * **area_surveillance** — Boustrophedon (lawnmower) coverage pattern.
      The mission area is divided into equal strips, one per drone.  Each
      drone sweeps its strip back-and-forth at cruise speed.
    * **threat_response** — Circular perimeter patrol.  Drones are
      phase-distributed around the defended area.  When a target is
      detected the nearest drone switches to an intercept reference.
    * **search_and_track** — Expanding-square search, one quadrant per
      drone.  Once a target is detected the responsible drone follows a
      predictive tracking reference.

    Args:
        num_drones:       Number of drones (agents).
        max_targets:      Maximum number of tracked targets.
        mission_duration: Episode length in steps (50 Hz → 1 step = 0.02 s).
        horizon:          DMPC prediction horizon (steps).
        dt:               DMPC discretisation step [s].
        admm_iters:       ADMM inner iterations per environment step.
        render_mode:      ``'human'``, ``'rgb_array'``, or ``None``.
        accel_max:        Maximum drone acceleration [m/s²].
        collision_radius: Minimum inter-drone separation [m].
        scenario:         Mission scenario name; selects reward weights and
                          trajectory planner.  One of
                          ``"area_surveillance"``, ``"threat_response"``,
                          ``"search_and_track"``.
        area_size:        Mission area dimensions ``(width_m, height_m)``.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Formation defaults
    FORMATION_SPACING_FACTOR = 2.5  # multiplier on collision_radius
    DEFAULT_ALTITUDE = 30.0  # metres AGL

    # Coverage grid
    GRID_CELL_SIZE = 20.0   # m — drone sensor footprint cell size
    COVERAGE_RADIUS = 25.0  # m — radius a drone covers around its position

    # Waypoint planner
    CRUISE_SPEED = 8.0          # m/s target reference speed
    WP_ACCEPT_RADIUS = 15.0     # m — distance to accept waypoint as reached

    def __init__(
        self,
        num_drones: int = 4,
        max_targets: int = 3,
        mission_duration: int = 1000,
        horizon: int = 20,
        dt: float = 0.02,
        admm_iters: int = 5,
        render_mode: Optional[str] = None,
        accel_max: float = 8.0,
        collision_radius: float = 3.0,
        solver_timeout: float = 0.02,
        osqp_max_iter: int = 4000,
        scenario: str = "area_surveillance",
        area_size: Tuple[float, float] = (400.0, 400.0),
    ) -> None:
        super().__init__()

        self.num_drones = num_drones
        self.max_targets = max_targets
        self.mission_duration = mission_duration
        self.horizon = horizon
        self.dt = dt
        self.admm_iters = admm_iters
        self.render_mode = render_mode
        self.accel_max = accel_max
        self.collision_radius = collision_radius
        self.scenario = scenario
        self._area_size: Tuple[float, float] = (float(area_size[0]), float(area_size[1]))

        # ── Scenario-specific reward weights ────────────────────────────
        self._weights: Dict[str, float] = _SCENARIO_WEIGHTS.get(
            scenario, _DEFAULT_WEIGHTS
        )

        # ── Coverage grid ────────────────────────────────────────────────
        self._grid_cols = max(1, int(self._area_size[0] / self.GRID_CELL_SIZE))
        self._grid_rows = max(1, int(self._area_size[1] / self.GRID_CELL_SIZE))
        self._coverage_grid = np.zeros(
            self._grid_rows * self._grid_cols, dtype=np.float32
        )
        self._prev_coverage: float = 0.0

        # ── Waypoint planner state ───────────────────────────────────────
        # _waypoints[i]: (K_i, 3) array of 3-D waypoints for drone i
        self._waypoints: List[np.ndarray] = []
        self._wp_idx: np.ndarray = np.zeros(num_drones, dtype=int)

        # ── Per-step reward component accumulators (exposed in info) ────
        self._step_reward_components: Dict[str, float] = dict.fromkeys(
            ("r_track", "r_form", "r_safe", "r_eff", "r_cov"), 0.0
        )

        # ── Gymnasium spaces ────────────────────────────────────────────
        total_obs = num_drones * OBS_DIM
        total_act = num_drones * ACT_DIM

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.1,
            high=10.0,
            shape=(total_act,),
            dtype=np.float32,
        )

        # ── Physics simulator ────────────────────────────────────────────
        self._simulator = EnvironmentSimulator(
            num_drones=num_drones,
            max_targets=max_targets,
            drone_config=DroneConfig(),
            target_config=TargetConfig(),
            env_config=EnvironmentConfig(),
        )

        # ── Per-drone DMPC controllers ───────────────────────────────────
        dmpc_cfg = DMPCConfig(
            horizon=horizon,
            dt=dt,
            state_dim=STATE_DIM,
            control_dim=CONTROL_DIM,
            accel_max=accel_max,
            collision_radius=collision_radius,
            solver_timeout=solver_timeout,
            osqp_max_iter=osqp_max_iter,
        )
        self._dmpc: List[DMPC] = [DMPC(dmpc_cfg) for _ in range(num_drones)]

        # ── ADMM consensus ───────────────────────────────────────────────
        admm_cfg = ADMMConfig(rho=1.0, max_iters=admm_iters)
        self._admm = ADMMConsensus(
            num_drones=num_drones,
            dim=CONTROL_DIM,
            config=admm_cfg,
        )

        # ── Internal state ───────────────────────────────────────────────
        self._step_count: int = 0
        # Drone states: (num_drones, STATE_DIM)
        self._drone_states: np.ndarray = np.zeros((num_drones, STATE_DIM), dtype=np.float64)
        # Reference trajectories: (num_drones, horizon+1, STATE_DIM)
        self._references: np.ndarray = np.zeros(
            (num_drones, horizon + 1, STATE_DIM), dtype=np.float64
        )
        # Last controls per drone: (num_drones, CONTROL_DIM)
        self._last_controls: np.ndarray = np.zeros(
            (num_drones, CONTROL_DIM), dtype=np.float64
        )
        # Battery and health levels per drone
        self._battery: np.ndarray = np.ones(num_drones, dtype=np.float64)
        self._health: np.ndarray = np.ones(num_drones, dtype=np.float64)
        # Solve times (normalised)
        self._solve_times: np.ndarray = np.zeros(num_drones, dtype=np.float64)
        # ADMM primal residuals: (num_drones, CONTROL_DIM)
        self._admm_residuals: np.ndarray = np.zeros(
            (num_drones, CONTROL_DIM), dtype=np.float64
        )

    # ──────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        self._step_count = 0
        self._admm.reset()
        self._last_controls[:] = 0.0
        self._solve_times[:] = 0.0
        self._admm_residuals[:] = 0.0
        self._battery[:] = 1.0
        self._health[:] = 1.0

        # Reset coverage grid
        self._coverage_grid[:] = 0.0
        self._prev_coverage = 0.0

        # Reset reward component accumulators
        for key in self._step_reward_components:
            self._step_reward_components[key] = 0.0

        # Reset physics simulator and place drones in a spaced formation
        self._simulator.reset()
        self._place_drones_in_formation()
        self._sync_states_from_sim()

        # Spawn targets for scenarios that require them
        if self.max_targets > 0:
            self._spawn_targets()

        # Build scenario-specific waypoints
        self._init_waypoints()

        # Generate initial references toward first waypoint
        for i in range(self.num_drones):
            if self._waypoints and len(self._waypoints[i]) > 0:
                self._references[i] = self._build_trajectory_to_waypoint(
                    self._drone_states[i], self._waypoints[i][0]
                )
            else:
                for k in range(self.horizon + 1):
                    self._references[i, k] = self._drone_states[i].copy()

        obs = self._build_observation()
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance environment by one step.

        Args:
            action: Flat action vector ``(num_drones * ACT_DIM,)`` where
                    each chunk of ACT_DIM values contains
                    ``[q_scale(11), r_scale(3)]`` for one drone.

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, 0.1, 10.0)
        actions_per_drone = action.reshape(self.num_drones, ACT_DIM)

        # ── Update mission references BEFORE DMPC solve ──────────────────
        self._update_references()

        controls = np.zeros((self.num_drones, CONTROL_DIM), dtype=np.float64)
        solve_times = np.zeros(self.num_drones, dtype=np.float64)
        raw_proposals = np.zeros((self.num_drones, CONTROL_DIM), dtype=np.float64)

        # ── DMPC solve per drone ─────────────────────────────────────────
        for i in range(self.num_drones):
            q_scale = actions_per_drone[i, :STATE_DIM]
            r_scale = actions_per_drone[i, STATE_DIM:]

            neighbour_states = [
                self._drone_states[j]
                for j in range(self.num_drones)
                if j != i
            ]

            u_seq, dmpc_info = self._dmpc[i](
                self._drone_states[i],
                self._references[i],
                neighbour_states,
                q_scale=q_scale,
                r_scale=r_scale,
            )

            u0 = u_seq[0] if u_seq.ndim == 2 and u_seq.shape[0] > 0 else np.zeros(CONTROL_DIM)
            raw_proposals[i] = u0
            solve_times[i] = float(dmpc_info.get("solve_time", 0.0))

        # ── ADMM consensus on acceleration proposals ─────────────────────
        consensus_ref = self._admm.step(raw_proposals)
        admm_residuals = self._admm.get_primal_residuals()  # (num_drones, CONTROL_DIM)

        # Blend per-drone proposals with consensus (soft coupling)
        for i in range(self.num_drones):
            controls[i] = 0.8 * raw_proposals[i] + 0.2 * consensus_ref

        # ── Apply controls to simulator ──────────────────────────────────
        # _accel_to_motor_cmds returns (num_drones, 4); simulator.step expects
        # the same shape so that motor_commands[drone_id] gives a (4,) vector.
        motor_cmds = self._accel_to_motor_cmds(controls)  # (num_drones, 4)
        self._simulator.step(motor_cmds)
        terminated = not all(d.is_active for d in self._simulator.drones)
        truncated = False
        self._sync_states_from_sim()

        # Sync battery level from the physics simulator so that there is
        # only one ground-truth battery model (the one in DronePhysics).
        for i in range(self.num_drones):
            drone = self._simulator.drones[i]
            self._battery[i] = max(
                drone.battery_energy / drone.config.battery_capacity, 0.0
            )

        # ── Store diagnostics ────────────────────────────────────────────
        self._last_controls = controls.copy()
        max_st = float(np.max(solve_times)) if np.max(solve_times) > 0.0 else 1e-3
        self._solve_times = solve_times / max_st
        self._admm_residuals = admm_residuals

        self._step_count += 1

        # ── Update coverage grid (after sync so positions are current) ───
        self._update_coverage()

        # ── Reward ───────────────────────────────────────────────────────
        reward = self._compute_reward(controls)

        # ── Termination ──────────────────────────────────────────────────
        terminated_rl = self._check_collision() or terminated
        truncated_rl = self._step_count >= self.mission_duration or truncated

        obs = self._build_observation()
        info = {
            "step": self._step_count,
            "solve_times": solve_times.tolist(),
            "battery": self._battery.tolist(),
            "coverage": float(np.mean(self._coverage_grid)),
            "reward_components": dict(self._step_reward_components),
        }

        return obs, reward, terminated_rl, truncated_rl, info

    def render(self) -> Optional[np.ndarray]:  # type: ignore[override]
        """Render is delegated to the physics simulator."""
        if hasattr(self._simulator, "render"):
            return self._simulator.render()
        return None

    def close(self) -> None:
        if hasattr(self._simulator, "close"):
            self._simulator.close()

    # ──────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """Construct the flat observation for all drones."""
        obs_list = []
        progress = self._step_count / max(self.mission_duration, 1)

        for i in range(self.num_drones):
            state = self._drone_states[i]  # (STATE_DIM,)
            ref = self._references[i, 0]   # (STATE_DIM,)
            ref_pos = ref[:3]
            ref_vel = ref[3:6]
            track_err = state[:3] - ref_pos

            # Nearest neighbour
            nn_rel = np.zeros(6, dtype=np.float64)
            if self.num_drones > 1:
                dists = [
                    np.linalg.norm(self._drone_states[j][:3] - state[:3])
                    for j in range(self.num_drones)
                    if j != i
                ]
                nn_idx_rel = int(np.argmin(dists))
                nn_idx = [j for j in range(self.num_drones) if j != i][nn_idx_rel]
                nn_rel[:3] = self._drone_states[nn_idx][:3] - state[:3]
                nn_rel[3:6] = self._drone_states[nn_idx][3:6] - state[3:6]

            # Mean swarm offset
            other_pos = np.stack(
                [self._drone_states[j][:3] for j in range(self.num_drones) if j != i],
                axis=0,
            ) if self.num_drones > 1 else state[:3][None, :]
            mean_offset = np.mean(other_pos, axis=0) - state[:3]

            # Collision margin (normalised)
            if self.num_drones > 1:
                min_dist = float(min(
                    np.linalg.norm(self._drone_states[j][:3] - state[:3])
                    for j in range(self.num_drones) if j != i
                ))
            else:
                min_dist = self.collision_radius * 10.0
            coll_margin = (min_dist - self.collision_radius) / max(self.collision_radius, 1e-6)

            single_obs = np.concatenate([
                state,                                   # 0–10  (11)
                ref_pos,                                 # 11–13 (3)
                ref_vel,                                 # 14–16 (3)
                track_err,                               # 17–19 (3)
                nn_rel,                                  # 20–25 (6)
                mean_offset,                             # 26–28 (3)
                [self._battery[i]],                      # 29    (1)
                [self._health[i]],                       # 30    (1)
                self._last_controls[i],                  # 31–33 (3)
                self._admm_residuals[i],                 # 34–36 (3)
                [self._solve_times[i]],                  # 37    (1)
                [float(np.clip(coll_margin, -1.0, 10.0))],  # 38 (1)
                [progress],                              # 39    (1)
            ]).astype(np.float32)

            assert single_obs.shape == (OBS_DIM,), (
                f"Observation shape mismatch: {single_obs.shape} != ({OBS_DIM},)"
            )
            obs_list.append(single_obs)

        return np.concatenate(obs_list).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(self, controls: np.ndarray) -> float:
        """Scalar reward combining tracking, formation, safety, efficiency, and coverage.

        Design principles
        -----------------
        * **Tracking** — ``exp(-α · ‖e_p‖²)`` is in **(0, 1]**, always positive,
          so good tracking behaviour is directly reinforced.  The old
          ``exp(…) - 1`` formulation was always ≤ 0, preventing the agent
          from ever earning a positive reward.
        * **Formation** — penalises only separation *below* the desired
          spacing (collision risk) with a linear term, and penalises
          extreme spread only beyond ``3 × desired``.  This lets drones
          spread out during coverage sweeps without being punished for it.
        * **Safety** — CBF-style continuous penalty when any pair gets
          closer than ``collision_radius``.
        * **Efficiency** — small control-effort penalty, normalised to
          ``[-1, 0]`` regardless of ``accel_max``.
        * **Coverage** — combination of incremental (delta) and absolute
          coverage fraction, shared across all drones each step.
        * All components are weighted by the scenario-specific
          ``_SCENARIO_WEIGHTS`` table.
        """
        W = self._weights

        # ── Coverage (swarm-level, computed once per step) ───────────────
        current_coverage = float(np.mean(self._coverage_grid))
        delta_cov = max(0.0, current_coverage - self._prev_coverage)
        # Incremental reward for new cells + small absolute level bonus
        r_cov = float(np.clip(delta_cov * 10.0 + current_coverage * 0.1, 0.0, 2.0))
        self._prev_coverage = current_coverage

        desired_spacing = self.collision_radius * self.FORMATION_SPACING_FACTOR

        total = 0.0
        sum_r_track = 0.0
        sum_r_form = 0.0
        sum_r_safe = 0.0
        sum_r_eff = 0.0

        for i in range(self.num_drones):
            state = self._drone_states[i]
            ref_pos = self._references[i, 0, :3]
            track_err = np.linalg.norm(state[:3] - ref_pos)

            # Tracking: positive exponential, range (0, 1]
            r_track = float(np.exp(-0.01 * track_err ** 2))

            # Formation: penalty only for violating minimum separation.
            # Drones are allowed (and expected) to spread out for coverage.
            r_form = 0.0
            n_neighbours_checked = 0
            for j in range(self.num_drones):
                if j != i:
                    dist = float(np.linalg.norm(
                        self._drone_states[j][:3] - state[:3]
                    ))
                    n_neighbours_checked += 1
                    if dist < desired_spacing:
                        # Linear penalty proportional to how much closer than desired
                        r_form -= (desired_spacing - dist) / desired_spacing

            # Normalise by actual number of neighbours checked
            r_form /= max(n_neighbours_checked, 1)

            # Safety: CBF-style continuous penalty for proximity < collision_radius
            r_safe = 0.0
            for j in range(self.num_drones):
                if j != i:
                    dist = float(np.linalg.norm(
                        self._drone_states[j][:3] - state[:3]
                    ))
                    if dist < self.collision_radius:
                        r_safe += min(0.0, dist - self.collision_radius)

            # Efficiency: normalised to [-1, 0]
            u_sq = float(np.dot(controls[i], controls[i]))
            r_eff = -u_sq / max(self.accel_max ** 2 * CONTROL_DIM, 1e-6)

            total += (
                W["W_TRACK"] * r_track
                + W["W_FORM"]  * r_form
                + W["W_SAFE"]  * r_safe
                + W["W_EFF"]   * r_eff
            )
            sum_r_track += r_track
            sum_r_form  += r_form
            sum_r_safe  += r_safe
            sum_r_eff   += r_eff

        # Coverage is a swarm-level reward, added once
        total += W["W_COV"] * r_cov

        per_drone = float(total / self.num_drones)
        n = float(self.num_drones)

        # Store per-step components for diagnostics (averaged over drones)
        self._step_reward_components["r_track"] = W["W_TRACK"] * sum_r_track / n
        self._step_reward_components["r_form"]  = W["W_FORM"]  * sum_r_form  / n
        self._step_reward_components["r_safe"]  = W["W_SAFE"]  * sum_r_safe  / n
        self._step_reward_components["r_eff"]   = W["W_EFF"]   * sum_r_eff   / n
        self._step_reward_components["r_cov"]   = W["W_COV"]   * r_cov / n

        return per_drone

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _place_drones_in_formation(self) -> None:
        """Place drones in a grid formation with sufficient spacing.

        Spacing is at least ``2 * collision_radius`` so that CBF constraints
        are comfortably satisfied from the first time-step.
        """
        spacing = self.collision_radius * self.FORMATION_SPACING_FACTOR
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            x = col * spacing
            y = row * spacing
            z = self.DEFAULT_ALTITUDE
            self._simulator.set_drone_initial_state(
                i, np.array([x, y, z], dtype=np.float64)
            )

    def _sync_states_from_sim(self) -> None:
        """Pull drone states from the physics simulator.

        The simulator's ``DronePhysics.get_state_vector()`` returns an 18-D
        vector::

            [pos(3), vel(3), quat(4), ang_vel(3), battery, health,
             active, motor_avg, temp]

        The DMPC state layout is 11-D::

            [pos(3), vel(3), accel(3), yaw, yaw_rate]

        This helper performs the correct mapping, extracting acceleration
        from ``DronePhysics.acceleration`` and computing yaw from the
        quaternion.
        """
        for i in range(self.num_drones):
            drone = self._simulator.drones[i]
            # Position & velocity come directly from the drone
            pos = drone.position  # (3,)
            vel = drone.velocity  # (3,)
            acc = drone.acceleration  # (3,)
            # Yaw from quaternion: atan2(2(wz + xy), 1 - 2(y² + z²))
            w, qx, qy, qz = drone.q
            yaw = float(np.arctan2(2.0 * (w * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))
            yaw_rate = float(drone.angular_velocity[2])  # body z-axis rate
            self._drone_states[i] = np.array([
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                acc[0], acc[1], acc[2],
                yaw, yaw_rate,
            ], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────
    # Target spawning
    # ──────────────────────────────────────────────────────────────────

    def _spawn_targets(self) -> None:
        """Spawn mobile targets within the mission area.

        Positions are drawn uniformly, keeping a 50 m margin from the
        boundary.  All targets are classified as HOSTILE with a random
        walking-speed velocity.
        """
        margin = 50.0
        x_lo = margin
        x_hi = max(margin + 1.0, self._area_size[0] - margin)
        y_lo = margin
        y_hi = max(margin + 1.0, self._area_size[1] - margin)

        for t in range(self.max_targets):
            pos = np.array([
                float(self.np_random.uniform(x_lo, x_hi)),
                float(self.np_random.uniform(y_lo, y_hi)),
                0.0,  # ground target
            ])
            self._simulator.add_target(pos, TargetType.HOSTILE)

            # Random walking velocity (~3 m/s)
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            speed = 3.0
            self._simulator.targets[t].velocity = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0,
            ])

    # ──────────────────────────────────────────────────────────────────
    # Coverage grid
    # ──────────────────────────────────────────────────────────────────

    def _update_coverage(self) -> None:
        """Mark grid cells within each drone's sensor footprint as covered.

        The drone is assumed to carry a nadir-pointing camera with a
        circular footprint of radius ``COVERAGE_RADIUS`` on the ground.
        """
        r_cells = max(1, int(np.ceil(self.COVERAGE_RADIUS / self.GRID_CELL_SIZE)))

        for i in range(self.num_drones):
            px = self._drone_states[i, 0]
            py = self._drone_states[i, 1]
            cx = int(px / self.GRID_CELL_SIZE)
            cy = int(py / self.GRID_CELL_SIZE)

            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    col = cx + dx
                    row = cy + dy
                    if 0 <= col < self._grid_cols and 0 <= row < self._grid_rows:
                        self._coverage_grid[row * self._grid_cols + col] = 1.0

    # ──────────────────────────────────────────────────────────────────
    # Waypoint builders (one per scenario)
    # ──────────────────────────────────────────────────────────────────

    def _build_lawnmower_waypoints(self, drone_idx: int) -> np.ndarray:
        """Boustrophedon sweep for *area_surveillance*.

        The mission area is divided into equal-width strips along the
        x-axis (one strip per drone).  Each drone sweeps its strip from
        south to north, shifts east by one cell width, then sweeps north
        to south, and so on.
        """
        strip_w = self._area_size[0] / self.num_drones
        x_start = drone_idx * strip_w
        leg_spacing = self.GRID_CELL_SIZE  # 20 m between legs
        n_legs = max(2, int(np.ceil(strip_w / leg_spacing)))

        waypoints = []
        for leg in range(n_legs):
            x = x_start + leg * leg_spacing + leg_spacing * 0.5
            x = min(x, x_start + strip_w - leg_spacing * 0.5)
            if leg % 2 == 0:
                waypoints.append(np.array([x, 0.0, self.DEFAULT_ALTITUDE]))
                waypoints.append(np.array([x, self._area_size[1], self.DEFAULT_ALTITUDE]))
            else:
                waypoints.append(np.array([x, self._area_size[1], self.DEFAULT_ALTITUDE]))
                waypoints.append(np.array([x, 0.0, self.DEFAULT_ALTITUDE]))
        return np.array(waypoints, dtype=np.float64)

    def _build_patrol_waypoints(self, drone_idx: int) -> np.ndarray:
        """Circular perimeter patrol for *threat_response*.

        Drones are phase-distributed around the defended perimeter at
        equal angular intervals so the entire boundary is always covered.
        """
        cx = self._area_size[0] * 0.5
        cy = self._area_size[1] * 0.5
        r = min(self._area_size) * 0.4  # patrol radius

        n_pts = 8  # patrol vertices per lap
        phase = drone_idx * (2.0 * np.pi / self.num_drones)
        waypoints = []
        for i in range(n_pts + 1):  # +1 closes the loop
            angle = phase + i * (2.0 * np.pi / n_pts)
            waypoints.append(np.array([
                cx + r * np.cos(angle),
                cy + r * np.sin(angle),
                self.DEFAULT_ALTITUDE,
            ]))
        return np.array(waypoints, dtype=np.float64)

    def _build_search_waypoints(self, drone_idx: int) -> np.ndarray:
        """Expanding-square search for *search_and_track*.

        The mission area is divided into quadrants; each drone searches
        its assigned quadrant using an expanding square centred on the
        quadrant centre.
        """
        qw = self._area_size[0] * 0.5
        qh = self._area_size[1] * 0.5
        qx = (drone_idx % 2) * qw
        qy = (drone_idx // 2) * qh
        cx = qx + qw * 0.5
        cy = qy + qh * 0.5

        step = self.GRID_CELL_SIZE
        max_r = min(qw, qh) * 0.5
        waypoints = []
        r = step
        while r <= max_r:
            # One loop of the expanding square (5 corners)
            waypoints += [
                [cx + r, cy,     self.DEFAULT_ALTITUDE],
                [cx + r, cy + r, self.DEFAULT_ALTITUDE],
                [cx - r, cy + r, self.DEFAULT_ALTITUDE],
                [cx - r, cy - r, self.DEFAULT_ALTITUDE],
                [cx + r, cy - r, self.DEFAULT_ALTITUDE],
            ]
            r += step
        if not waypoints:
            waypoints = [[cx, cy, self.DEFAULT_ALTITUDE]]
        return np.array(waypoints, dtype=np.float64)

    def _init_waypoints(self) -> None:
        """Build per-drone waypoint lists for the current scenario."""
        self._wp_idx[:] = 0
        builders = {
            "area_surveillance": self._build_lawnmower_waypoints,
            "threat_response":   self._build_patrol_waypoints,
            "search_and_track":  self._build_search_waypoints,
        }
        builder = builders.get(self.scenario, self._build_lawnmower_waypoints)
        self._waypoints = [builder(i) for i in range(self.num_drones)]

    # ──────────────────────────────────────────────────────────────────
    # Reference trajectory builder
    # ──────────────────────────────────────────────────────────────────

    def _build_trajectory_to_waypoint(
        self,
        state: np.ndarray,
        waypoint: np.ndarray,
    ) -> np.ndarray:
        """Build a ``(horizon+1, STATE_DIM)`` reference trajectory.

        The trajectory is a straight-line path from the drone's current
        position to *waypoint* at ``CRUISE_SPEED``.  If the waypoint is
        reachable within the horizon the speed is reduced so the drone
        arrives precisely at the terminal step.

        Args:
            state:    Current drone state ``(STATE_DIM,)``.
            waypoint: 3-D target position ``(3,)``.

        Returns:
            Reference trajectory ``(horizon+1, STATE_DIM)``.
        """
        ref = np.zeros((self.horizon + 1, STATE_DIM), dtype=np.float64)
        pos = state[:3].copy()
        direction = waypoint[:3] - pos
        dist = float(np.linalg.norm(direction))

        if dist < 0.5:
            # Already at waypoint — hover in place
            for k in range(self.horizon + 1):
                ref[k, :3] = pos
            return ref

        unit_dir = direction / dist
        dt_horizon = self.horizon * self.dt
        speed = self.CRUISE_SPEED
        if dist < speed * dt_horizon:
            # Can reach waypoint within horizon — arrive at last step
            speed = dist / max(dt_horizon, self.dt)

        target_vel = unit_dir * speed
        cur_pos = pos.copy()
        for k in range(self.horizon + 1):
            ref[k, :3] = cur_pos
            ref[k, 3:6] = target_vel
            # ref[k, 6:9] = 0  (zero accel reference — DMPC computes this)
            next_pos = cur_pos + target_vel * self.dt
            # Don't overshoot the waypoint
            if np.linalg.norm(next_pos - pos) < dist:
                cur_pos = next_pos
            else:
                cur_pos = waypoint[:3].copy()
        return ref

    def _build_tracking_reference(
        self,
        drone_idx: int,
        fallback_waypoint: np.ndarray,
    ) -> np.ndarray:
        """Build a predictive tracking reference for a detected target.

        Assigns detected targets round-robin to drones by index.  Uses
        constant-velocity prediction over the horizon.  Falls back to the
        normal waypoint reference when no targets are detected.

        Args:
            drone_idx:        Index of the requesting drone.
            fallback_waypoint: Waypoint to use when no target is detected.

        Returns:
            Reference trajectory ``(horizon+1, STATE_DIM)``.
        """
        detected = [
            t for t in self._simulator.targets[:self._simulator.num_targets]
            if t.is_detected
        ]
        if not detected:
            return self._build_trajectory_to_waypoint(
                self._drone_states[drone_idx], fallback_waypoint
            )

        target = detected[drone_idx % len(detected)]
        t_pos = target.position.copy()
        t_vel = target.velocity.copy()

        ref = np.zeros((self.horizon + 1, STATE_DIM), dtype=np.float64)
        cur_t_pos = t_pos.copy()
        for k in range(self.horizon + 1):
            # Track target horizontally; maintain drone altitude
            ref[k, 0] = cur_t_pos[0]
            ref[k, 1] = cur_t_pos[1]
            ref[k, 2] = self.DEFAULT_ALTITUDE
            ref[k, 3] = t_vel[0]
            ref[k, 4] = t_vel[1]
            ref[k, 5] = 0.0
            cur_t_pos = cur_t_pos + t_vel * self.dt
        return ref

    # ──────────────────────────────────────────────────────────────────
    # Reference update (called at the start of each step)
    # ──────────────────────────────────────────────────────────────────

    def _update_references(self) -> None:
        """Advance per-drone reference trajectories based on waypoint progress.

        For each drone:
        1. Check whether the current waypoint has been reached
           (within ``WP_ACCEPT_RADIUS`` metres).
        2. If yes, advance to the next waypoint (cyclic).
        3. Build a new horizon-length trajectory toward the active waypoint.

        For *search_and_track* with detected targets the reference is
        replaced by a predictive tracking trajectory.
        """
        if not self._waypoints:
            return

        for i in range(self.num_drones):
            waypoints = self._waypoints[i]
            if len(waypoints) == 0:
                continue

            wp_idx = int(self._wp_idx[i])
            wp = waypoints[wp_idx]

            # Advance waypoint when within acceptance radius
            dist_to_wp = float(np.linalg.norm(self._drone_states[i, :3] - wp[:3]))
            if dist_to_wp < self.WP_ACCEPT_RADIUS:
                wp_idx = (wp_idx + 1) % len(waypoints)
                self._wp_idx[i] = wp_idx
                wp = waypoints[wp_idx]

            # Build trajectory: tracking reference for search_and_track,
            # waypoint-following otherwise
            if self.scenario == "search_and_track":
                self._references[i] = self._build_tracking_reference(i, wp)
            elif self.scenario == "threat_response":
                # If any target detected, nearest drone intercepts
                detected = [
                    t for t in self._simulator.targets[:self._simulator.num_targets]
                    if t.is_detected
                ]
                if detected:
                    # Assign closest target
                    closest = min(
                        detected,
                        key=lambda t: np.linalg.norm(
                            self._drone_states[i, :3] - t.position
                        ),
                    )
                    intercept_wp = np.array([
                        closest.position[0],
                        closest.position[1],
                        self.DEFAULT_ALTITUDE,
                    ])
                    self._references[i] = self._build_trajectory_to_waypoint(
                        self._drone_states[i], intercept_wp
                    )
                else:
                    self._references[i] = self._build_trajectory_to_waypoint(
                        self._drone_states[i], wp
                    )
            else:
                self._references[i] = self._build_trajectory_to_waypoint(
                    self._drone_states[i], wp
                )

    def _accel_to_motor_cmds(self, controls: np.ndarray) -> np.ndarray:
        """Convert per-drone acceleration commands to normalised motor commands.

        Returns an array of shape ``(num_drones, 4)`` with values in [0, 1].
        The mapping assumes symmetric rotor allocation and clips to hover ± delta.
        """
        motor_cmds = np.zeros((self.num_drones, 4), dtype=np.float32)
        hover_norm = 0.5  # normalised hover throttle
        for i in range(self.num_drones):
            ax, ay, az = controls[i]
            az_norm = az / max(self.accel_max, 1e-6)
            delta = np.clip(az_norm * 0.5, -0.4, 0.4)
            motor_cmds[i] = np.clip(
                hover_norm + delta + np.array([ax, -ax, ay, -ay]) * 0.1,
                0.0,
                1.0,
            )
        return motor_cmds

    def _check_collision(self) -> bool:
        """Return True if any pair of drones is closer than 0.5 * collision_radius."""
        threshold = 0.5 * self.collision_radius
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = np.linalg.norm(
                    self._drone_states[i][:3] - self._drone_states[j][:3]
                )
                if dist < threshold:
                    return True
        return False


def make_env(
    env_id: str = "MARLDMPCEnv-v0",
    **kwargs,
) -> MARLDMPCEnv:
    """
    Convenience factory matching the ``gym.make`` interface.

    Args:
        env_id:  Environment ID (currently only ``'MARLDMPCEnv-v0'``).
        **kwargs: Passed directly to :class:`MARLDMPCEnv`.  Supports all
                  constructor arguments including ``scenario`` and
                  ``area_size``.

    Returns:
        Configured :class:`MARLDMPCEnv` instance.
    """
    return MARLDMPCEnv(**kwargs)
