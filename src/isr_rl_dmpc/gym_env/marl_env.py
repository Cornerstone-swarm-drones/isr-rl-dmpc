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
)

# Observation and action dimensions (fixed by architecture)
OBS_DIM = 40
ACT_DIM = 14  # 11 q_scales + 3 r_scales
STATE_DIM = 11
CONTROL_DIM = 3


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
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # Reward weights
    W_TRACK = 5.0
    W_FORM = 2.0
    W_SAFE = 10.0
    W_EFF = 0.1

    # Formation defaults
    FORMATION_SPACING_FACTOR = 2.5  # multiplier on collision_radius
    DEFAULT_ALTITUDE = 30.0  # metres AGL

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

        # Reset physics simulator and place drones in a spaced formation
        self._simulator.reset()
        self._place_drones_in_formation()
        self._sync_states_from_sim()

        # Generate initial references (hover in place at the spaced positions)
        for i in range(self.num_drones):
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

            u_seq, info = self._dmpc[i](
                self._drone_states[i],
                self._references[i],
                neighbour_states,
                q_scale=q_scale,
                r_scale=r_scale,
            )

            u0 = u_seq[0] if u_seq.ndim == 2 and u_seq.shape[0] > 0 else np.zeros(CONTROL_DIM)
            raw_proposals[i] = u0
            solve_times[i] = float(info.get("solve_time", 0.0))

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
        """Scalar reward combining tracking, formation, safety, and efficiency."""
        total = 0.0
        for i in range(self.num_drones):
            state = self._drone_states[i]
            ref_pos = self._references[i, 0, :3]
            track_err = np.linalg.norm(state[:3] - ref_pos)

            # Tracking reward: exp(-α * ‖e_p‖²) - 1
            # α = 0.01 keeps the reward meaningful up to ~10 m tracking error
            r_track = float(np.exp(-0.01 * track_err ** 2) - 1.0)

            # Formation reward: penalise deviation from desired spacing
            # Use the same spacing factor as initial placement so the reward
            # starts near zero and only penalises actual drift.
            form_errs = []
            desired_spacing = self.collision_radius * self.FORMATION_SPACING_FACTOR
            for j in range(self.num_drones):
                if j != i:
                    dist = np.linalg.norm(self._drone_states[j][:3] - state[:3])
                    form_errs.append(abs(dist - desired_spacing))
            r_form = -float(np.mean(form_errs)) if form_errs else 0.0

            # Safety reward: penalty for proximity violations
            r_safe = 0.0
            for j in range(self.num_drones):
                if j != i:
                    dist = np.linalg.norm(self._drone_states[j][:3] - state[:3])
                    r_safe += min(0.0, dist - self.collision_radius)

            # Efficiency reward: normalised by accel_max² so that it is in
            # [-1, 0] regardless of the acceleration limit.
            u_sq = float(np.dot(controls[i], controls[i]))
            r_eff = -u_sq / max(self.accel_max ** 2 * CONTROL_DIM, 1e-6)

            total += (
                self.W_TRACK * r_track
                + self.W_FORM * r_form
                + self.W_SAFE * r_safe
                + self.W_EFF * r_eff
            )

        return total / self.num_drones

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
        **kwargs: Passed directly to :class:`MARLDMPCEnv`.

    Returns:
        Configured :class:`MARLDMPCEnv` instance.
    """
    return MARLDMPCEnv(**kwargs)
