"""
BeliefCoverageEnv — Phase 1 belief-based coverage for area surveillance.

This environment runs alongside the existing ``MARLDMPCEnv`` rather than
replacing it. High-level actions choose target cells; the existing simulator
and DMPC stack then turn those cell choices into safe fixed-altitude motion.

Phase 1 interpretation
----------------------
The belief signal in this environment should currently be read as a
conservative *patrol risk* proxy:

* high uncertainty => insufficiently monitored / neglected area
* high anomaly score => heuristic risk cue, not a confirmed hostile hotspot

POMDP interpretation
--------------------
This environment is still heuristic and simulator-backed, but its internals now
follow the standard POMDP split:

* hidden world state = patrol-risk state + persistent threat-like world state
* belief state = fused/local patrol-risk belief + threat belief
* transition model = risk growth plus latent world-state persistence
* observation model = forward FOV sensing
* belief update = local observation + sharing + fusion + persistence confirmation
* policy = deterministic home-strip patrol over the current belief state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from isr_rl_dmpc.core import BeliefGrid, BeliefGridConfig, LocalBeliefGrid
from isr_rl_dmpc.gym_env.reward_shaper import (
    BeliefCoverageRewardShaper,
    BeliefCoverageRewardWeights,
)
from isr_rl_dmpc.gym_env.patrol_pomdp import (
    PatrolBeliefState,
    PatrolBeliefUpdater,
    PatrolHiddenWorldState,
    PatrolObservationModel,
    PatrolTransitionModel,
)
from isr_rl_dmpc.gym_env.sensor_model import ForwardFOVSensorModel, VisionSensorConfig
from isr_rl_dmpc.gym_env.simulator import (
    DroneConfig,
    EnvironmentConfig,
    EnvironmentSimulator,
    TargetConfig,
)
from isr_rl_dmpc.modules.admm_consensus import ADMMConfig, ADMMConsensus
from isr_rl_dmpc.modules.dmpc_controller import DMPC, DMPCConfig
from isr_rl_dmpc.modules.mission_planner import GridDecomposer


STATE_DIM = 11
CONTROL_DIM = 3


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to ``[-pi, pi]`` for stable heading comparisons."""
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class SuspiciousZone:
    """Heuristic risk-cue source used in the first belief-coverage pass."""

    center: Tuple[float, float]
    radius: float
    score: float


@dataclass(frozen=True)
class ThreatPatchCandidate:
    """Contiguous 2x2 threat-patch candidate used by the moving threat loop."""

    cell_indices: np.ndarray
    centroid_xy: np.ndarray
    weight: float


class BeliefCoverageEnv(gym.Env):
    """Parallel Phase 1 environment for belief-based coverage."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    FORMATION_SPACING_FACTOR = 2.5
    WP_ACCEPT_RADIUS = 15.0
    CRUISE_SPEED = 10.0
    PLANAR_TRACKING_GAIN = 1.75
    PATROL_HOME_CLEAR_RISK = 0.4
    PATROL_ASSIST_TRIGGER_RISK = 0.7
    PATROL_ASSIST_CLEAR_RISK = 0.45
    PATROL_ASSIST_MARGIN = 0.1
    PATROL_HOME_RETURN_RISK = 0.55
    PATROL_MAX_DETOUR_STEPS = 4
    THREAT_PATCH_WIDTH = 2
    THREAT_CONFIRMATION_GAIN = 0.45
    THREAT_CONFIRMATION_DECAY = 0.94
    THREAT_CONFIRMATION_THRESHOLD = 0.75
    THREAT_SUSPECT_THRESHOLD = 0.25
    THREAT_MAX_TRACKERS = 2
    THREAT_SPEED_CASES = {
        "slow": 8.0,
        "medium": 14.0,
        "fast": 22.0,
    }
    THREAT_BASE_REACH_SPEED = 18.0
    THREAT_LATERAL_BIAS = 0.35
    THREAT_BASE_REACH_RADIUS_FACTOR = 0.8
    INTERCEPTOR_SPEED = 90.0
    INTERCEPTOR_HIT_RADIUS_FACTOR = 0.45
    MISSION_FAILURE_PENALTY = 5.0

    def __init__(
        self,
        num_drones: int = 4,
        mission_duration: int = 1000,
        horizon: int = 20,
        dt: float = 0.02,
        accel_max: float = 8.0,
        collision_radius: float = 3.0,
        solver_timeout: float = 0.02,
        osqp_max_iter: int = 4000,
        scenario: str = "area_surveillance",
        area_size: Tuple[float, float] = (400.0, 400.0),
        grid_resolution: float = 20.0,
        observation_mode: str = "flat",
        fixed_altitude: float = 30.0,
        communication_range: float = 250.0,
        sensor_range: float = 120.0,
        growth_rate: float = 0.03,
        global_sync_steps: int = 25,
        suspicious_zones: Optional[Sequence[Dict[str, object] | SuspiciousZone]] = None,
        base_station: Optional[Tuple[float, float]] = None,
        enable_neighbor_sharing: bool = True,
        enable_goal_projection: bool = True,
        enable_persistent_threats: bool = True,
        max_threat_cycles: int = 3,
        persistent_threat_speed_case: str = "medium",
        persistent_threat_speed: Optional[float] = None,
        reward_weights: Optional[BeliefCoverageRewardWeights] = None,
    ) -> None:
        super().__init__()

        if scenario != "area_surveillance":
            raise ValueError(
                "BeliefCoverageEnv Phase A supports only the 'area_surveillance' scenario."
            )
        if observation_mode != "flat":
            raise ValueError("BeliefCoverageEnv currently supports observation_mode='flat' only.")

        self.num_drones = int(num_drones)
        self.mission_duration = int(mission_duration)
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.accel_max = float(accel_max)
        self.collision_radius = float(collision_radius)
        self.scenario = scenario
        self.area_size = (float(area_size[0]), float(area_size[1]))
        self.grid_resolution = float(grid_resolution)
        self.observation_mode = observation_mode
        self.fixed_altitude = float(fixed_altitude)
        self.communication_range = float(communication_range)
        self.enable_neighbor_sharing = bool(enable_neighbor_sharing)
        self.enable_goal_projection = bool(enable_goal_projection)
        self.enable_persistent_threats = bool(enable_persistent_threats)
        self.max_threat_cycles = max(0, int(max_threat_cycles))
        self.persistent_threat_speed_case = str(persistent_threat_speed_case).lower()
        self.persistent_threat_speed = self._resolve_threat_speed(
            speed_case=self.persistent_threat_speed_case,
            speed_override=persistent_threat_speed,
        )
        self.policy_name = "home_strip_boustrophedon"
        if base_station is None:
            base_station = (self.area_size[0] * 0.5, self.area_size[1] * 0.5)
        self.base_station = np.array(base_station, dtype=np.float64)

        self._area_boundary = np.array(
            [
                [0.0, 0.0],
                [self.area_size[0], 0.0],
                [self.area_size[0], self.area_size[1]],
                [0.0, self.area_size[1]],
            ],
            dtype=np.float64,
        )

        decomposer = GridDecomposer(self._area_boundary, grid_resolution=self.grid_resolution)
        self._cells = decomposer.decompose_grid(n_drones=self.num_drones)
        if not self._cells:
            raise ValueError("BeliefCoverageEnv requires at least one decomposed coverage cell.")

        belief_config = BeliefGridConfig(
            growth_rate=float(growth_rate),
            global_sync_steps=max(1, int(global_sync_steps)),
        )
        self.global_belief = BeliefGrid.from_cells(
            self._cells,
            grid_resolution=self.grid_resolution,
            config=belief_config,
            name="global",
        )
        self.transition_model = PatrolTransitionModel(config=belief_config)
        self.local_beliefs = [
            LocalBeliefGrid.from_belief_grid(self.global_belief, drone_id=i)
            for i in range(self.num_drones)
        ]
        self.n_cells = self.global_belief.n_cells
        self.cell_centers_xy = self.global_belief.centers.copy()
        self._row_values = np.unique(self.global_belief.rows)
        self._col_values = np.unique(self.global_belief.cols)
        self._row_lookup = {int(row): idx for idx, row in enumerate(self._row_values)}
        self._col_lookup = {int(col): idx for idx, col in enumerate(self._col_values)}
        self.cell_centers = np.column_stack(
            [
                self.cell_centers_xy,
                np.full(self.n_cells, self.fixed_altitude, dtype=np.float64),
            ]
        )
        self.cell_priorities = self.global_belief.priorities.copy()
        self._home_cell_indices, self._assist_cell_indices = self._build_home_subregions()
        self._home_patrol_routes = [
            self._build_boustrophedon_route(home_cells)
            for home_cells in self._home_cell_indices
        ]

        self.suspicious_zones = self._normalize_suspicious_zones(suspicious_zones)
        self._threat_spawn_preference = self._build_threat_spawn_preference_field()
        self._threat_patch_candidates = self._build_threat_patch_candidates()
        self._threat_candidate_centroids = (
            np.stack([candidate.centroid_xy for candidate in self._threat_patch_candidates], axis=0)
            if self._threat_patch_candidates
            else np.zeros((0, 2), dtype=np.float64)
        )
        self.observation_model = PatrolObservationModel(
            ForwardFOVSensorModel(VisionSensorConfig(max_range=float(sensor_range)))
        )
        self.sensor_model = self.observation_model.sensor_model
        self.truth_observation_model = PatrolObservationModel(
            ForwardFOVSensorModel(
                VisionSensorConfig(
                    fov_deg=self.sensor_model.config.fov_deg,
                    max_range=self.sensor_model.config.max_range,
                    distance_decay=self.sensor_model.config.distance_decay,
                    angular_decay=self.sensor_model.config.angular_decay,
                    noise_std=0.0,
                    min_quality=self.sensor_model.config.min_quality,
                )
            )
        )
        self._truth_sensor_model = self.truth_observation_model.sensor_model
        self.belief_updater = PatrolBeliefUpdater()
        self._truth_persistent_threat = np.zeros(self.n_cells, dtype=np.float64)
        self._truth_risk_cue = self._truth_persistent_threat.copy()
        self.reward_shaper = BeliefCoverageRewardShaper(
            self.num_drones,
            weights=reward_weights,
        )
        self._truth_monitoring_risk = np.full(
            self.n_cells,
            self.global_belief.config.uncertainty_max,
            dtype=np.float64,
        )
        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )
        self._truth_last_observed_step = np.full(self.n_cells, -1, dtype=np.int32)
        self._threat_persistence_score = np.zeros(self.n_cells, dtype=np.float64)
        self._confirmed_threat_mask = np.zeros(self.n_cells, dtype=bool)
        self._active_threat_cells = np.zeros(0, dtype=np.int32)
        self._active_threat_patch_index = -1
        self._active_threat_position = np.full(2, np.nan, dtype=np.float64)
        self._active_threat_velocity = np.zeros(2, dtype=np.float64)
        self._active_threat_confirmation_level = 0.0
        self._active_threat_lateral_sign = 1.0
        self._threat_confirmed = False
        self._threat_cycles_spawned = 0
        self._threat_cycles_completed = 0
        self._threat_removed_this_step = False
        self._threat_respawned_this_step = False
        self._interceptor_dispatched_this_step = False
        self._physical_base_reached_this_step = False
        self._central_command_notified = False
        self._tracking_bias_drones = np.zeros(self.num_drones, dtype=bool)
        self._tracking_target_cells = np.full(self.num_drones, -1, dtype=np.int32)
        self._mission_failed = False
        self._mission_fail_reason = ""
        self._threat_base_eta_steps = -1
        self._threat_base_timeout_steps = -1
        self._threat_patch_history: List[int] = []
        self._threat_trace: List[np.ndarray] = []
        self._interceptor_active = False
        self._interceptor_position = self.base_station.copy()
        self._interceptor_velocity = np.zeros(2, dtype=np.float64)
        self._interceptor_target = np.full(2, np.nan, dtype=np.float64)
        self._interceptor_trace: List[np.ndarray] = [self.base_station.copy()]
        self._interceptor_dispatch_count = 0

        total_obs_dim = self.num_drones * self.n_cells * 3 + self.num_drones * 7 + 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_drones, self.n_cells, dtype=np.int64)
        )

        self._simulator = EnvironmentSimulator(
            num_drones=self.num_drones,
            max_targets=0,
            drone_config=DroneConfig(),
            target_config=TargetConfig(),
            env_config=EnvironmentConfig(timestep=self.dt),
        )
        dmpc_cfg = DMPCConfig(
            horizon=self.horizon,
            dt=self.dt,
            state_dim=STATE_DIM,
            control_dim=CONTROL_DIM,
            accel_max=self.accel_max,
            collision_radius=self.collision_radius,
            solver_timeout=float(solver_timeout),
            osqp_max_iter=int(osqp_max_iter),
        )
        self._dmpc: List[DMPC] = [DMPC(dmpc_cfg) for _ in range(self.num_drones)]
        self._admm = ADMMConsensus(
            num_drones=self.num_drones,
            dim=CONTROL_DIM,
            config=ADMMConfig(rho=1.0, max_iters=5),
        )

        self._step_count = 0
        self._drone_states = np.zeros((self.num_drones, STATE_DIM), dtype=np.float64)
        self._references = np.zeros((self.num_drones, self.horizon + 1, STATE_DIM), dtype=np.float64)
        self._current_target_cells = np.zeros(self.num_drones, dtype=np.int32)
        self._last_structured_obs: Dict[str, np.ndarray] = {}
        self._last_reward_components: Dict[str, float] = {}
        self._last_connectivity_state: Dict[str, float] = {}
        self._patrol_route_indices = np.zeros(self.num_drones, dtype=np.int32)
        self._patrol_detour_targets = np.full(self.num_drones, -1, dtype=np.int32)
        self._patrol_detour_remaining = np.zeros(self.num_drones, dtype=np.int32)
        self._patrol_detour_counts = np.zeros(self.num_drones, dtype=np.int32)
        self._time_away_from_home_steps = np.zeros(self.num_drones, dtype=np.int32)
        self._truth_monitoring_risk = np.full(
            self.n_cells,
            self.global_belief.config.uncertainty_max,
            dtype=np.float64,
        )
        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )
        self._truth_last_observed_step = np.full(self.n_cells, -1, dtype=np.int32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._step_count = 0
        self._admm.reset()
        self._simulator.reset()
        self._place_drones_in_formation()
        self._sync_states_from_sim()

        self.global_belief.reset()
        for local in self.local_beliefs:
            local.reset()
            local.clear_recent_changes()
        self._truth_monitoring_risk.fill(self.global_belief.config.uncertainty_max)
        self._truth_last_observed_step.fill(-1)
        self._reset_threat_cycle_state()
        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )

        self._reset_patrol_state()
        self._current_target_cells = self._select_initial_targets()
        self._update_target_references(self._current_target_cells)
        self._align_heading_to_targets(self._current_target_cells)
        self._time_away_from_home_steps.fill(0)
        self._patrol_detour_counts.fill(0)
        connectivity = self._compute_connectivity()
        belief_metrics = self._update_beliefs(step=0, connectivity=connectivity, grow=False, apply_global_sync=False)
        truth_metrics = self._update_truth_risk(step=0, grow=False)
        threat_metrics = self._update_threat_response(
            belief_metrics=belief_metrics,
            connectivity=connectivity,
            step=0,
        )

        total_uncertainty = self.global_belief.total_uncertainty()
        self.reward_shaper.reset(
            total_uncertainty=total_uncertainty,
            battery_levels=self._battery_levels(),
        )

        observation = self._build_observation(connectivity)
        info = self._build_info(
            connectivity=connectivity,
            reward=0.0,
            reward_components={},
            solve_times=np.zeros(self.num_drones, dtype=np.float64),
            truth_metrics=truth_metrics,
            threat_metrics=threat_metrics,
        )
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, Dict]:
        requested_cells = self._normalize_action(action)
        selected_cells = requested_cells.copy()
        if self.enable_goal_projection:
            selected_cells = self._project_action_cells(selected_cells)
        projection_override_count = int(np.sum(selected_cells != requested_cells))
        self._current_target_cells = selected_cells
        self._update_target_references(selected_cells)

        controls = np.zeros((self.num_drones, CONTROL_DIM), dtype=np.float64)
        raw_proposals = np.zeros((self.num_drones, CONTROL_DIM), dtype=np.float64)
        solve_times = np.zeros(self.num_drones, dtype=np.float64)

        for i in range(self.num_drones):
            neighbour_states = [
                self._drone_states[j]
                for j in range(self.num_drones)
                if j != i
            ]
            u_seq, info = self._dmpc[i](
                self._drone_states[i],
                self._references[i],
                neighbour_states,
            )
            u0 = u_seq[0] if u_seq.ndim == 2 and u_seq.shape[0] > 0 else np.zeros(CONTROL_DIM)
            raw_proposals[i] = u0
            solve_times[i] = float(info.get("solve_time", 0.0))

        consensus_ref = self._admm.step(raw_proposals)
        for i in range(self.num_drones):
            controls[i] = 0.8 * raw_proposals[i] + 0.2 * consensus_ref

        motor_cmds = self._accel_to_motor_cmds(controls)
        self._simulator.step(motor_cmds)
        self._apply_planar_tracking(selected_cells)
        self._sync_states_from_sim()
        self._align_heading_to_targets(selected_cells)
        self._time_away_from_home_steps += 1 - self._positions_in_home_region()

        self._step_count += 1
        connectivity = self._compute_connectivity()
        self._threat_removed_this_step = False
        self._threat_respawned_this_step = False
        self._interceptor_dispatched_this_step = False
        self._physical_base_reached_this_step = False
        truth_metrics = self._update_truth_risk(step=self._step_count, grow=True)
        belief_metrics = self._update_beliefs(
            step=self._step_count,
            connectivity=connectivity,
            grow=True,
            apply_global_sync=(self._step_count % self.global_belief.config.global_sync_steps == 0),
        )
        threat_metrics = self._update_threat_response(
            belief_metrics=belief_metrics,
            connectivity=connectivity,
            step=self._step_count,
        )

        battery_levels = self._battery_levels()
        dist_to_targets = self._distance_to_selected_targets(selected_cells)
        dist_to_base = self._distance_to_base_norm()
        reward, reward_components = self.reward_shaper.compute(
            total_uncertainty=self.global_belief.total_uncertainty(),
            uncertainty_values=self.global_belief.uncertainty,
            distance_norm=dist_to_targets,
            battery_levels=battery_levels,
            dist_to_base_norm=dist_to_base,
            connected_components=int(connectivity["component_count"]),
        )
        if self._mission_failed:
            reward -= self.MISSION_FAILURE_PENALTY
            reward_components["mission_failure_penalty"] = float(self.MISSION_FAILURE_PENALTY)

        terminated = self._check_collision() or self._mission_failed
        truncated = self._step_count >= self.mission_duration
        observation = self._build_observation(connectivity)
        info = self._build_info(
            connectivity=connectivity,
            reward=reward,
            reward_components=reward_components,
            solve_times=solve_times,
            observed_counts=belief_metrics["observed_counts"],
            requested_target_cells=requested_cells,
            projection_override_count=projection_override_count,
            shared_cell_count=int(belief_metrics["shared_cell_count"]),
            global_sync_applied=bool(belief_metrics["global_sync_applied"]),
            truth_metrics=truth_metrics,
            threat_metrics=threat_metrics,
        )
        return observation, float(reward), bool(terminated), bool(truncated), info

    def get_structured_observation(self) -> Dict[str, np.ndarray]:
        """Return a copy of the latest structured observation."""
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._last_structured_obs.items()
        }

    def score_cells_for_drone(self, drone_idx: int) -> np.ndarray:
        """Heuristic Phase B score used by the smoke runner and baseline policies."""
        local = self.local_beliefs[drone_idx]
        position = self._drone_states[drone_idx, :2]
        distances = np.linalg.norm(self.cell_centers_xy - position[None, :], axis=1)
        dist_norm = distances / max(np.linalg.norm(self.area_size), 1e-6)

        risk_score = np.maximum(local.uncertainty, local.anomaly_score)
        return (
            risk_score
            + 0.1 * self.cell_priorities
            - 0.2 * np.clip(dist_norm, 0.0, 1.0)
        )

    def select_greedy_action(self, unique: bool = True) -> np.ndarray:
        """Return a simple heuristic action for smoke tests and baseline runs."""
        chosen = np.zeros(self.num_drones, dtype=np.int32)
        used: set[int] = set()

        for drone_idx in range(self.num_drones):
            scores = self.score_cells_for_drone(drone_idx)
            ordering = np.argsort(scores)[::-1]
            selected = int(ordering[0])
            if unique:
                for candidate in ordering:
                    if int(candidate) not in used:
                        selected = int(candidate)
                        break
            chosen[drone_idx] = selected
            used.add(selected)
        return chosen

    def select_patrol_action(self) -> np.ndarray:
        """
        Deterministic Phase 1 policy over the current belief state.

        Drones sweep their home subregion in a boustrophedon pattern, allow
        only short local detours into neighboring boundary cells when nearby
        patrol risk rises, and then return to the home route. After a
        persistent threat is confirmed, only a small nearby subset receives a
        light tracking bias while the rest continue patrol. This remains a
        structured heuristic policy, not a learned POMDP controller.
        """
        risk_scores = self._current_risk_scores()
        chosen = np.zeros(self.num_drones, dtype=np.int32)

        for drone_idx in range(self.num_drones):
            patrol_target = self._select_home_patrol_target(drone_idx, risk_scores)
            home_target_risk = float(risk_scores[int(patrol_target)])
            home_cells = self._home_cell_indices[drone_idx]
            home_region_risk = (
                float(np.max(risk_scores[home_cells]))
                if home_cells.size
                else home_target_risk
            )
            local_tracking_overlap = bool(
                self._active_threat_exists()
                and np.intersect1d(
                    np.union1d(self._home_cell_indices[drone_idx], self._assist_cell_indices[drone_idx]),
                    self._active_threat_cells,
                ).size > 0
            )

            if bool(self._tracking_bias_drones[drone_idx]) and self._tracking_target_cells[drone_idx] >= 0:
                if local_tracking_overlap or home_region_risk <= self.PATROL_HOME_RETURN_RISK:
                    self._clear_patrol_detour(drone_idx)
                    chosen[drone_idx] = int(self._tracking_target_cells[drone_idx])
                    continue

            detour_target = int(self._patrol_detour_targets[drone_idx])
            if detour_target >= 0:
                detour_risk = float(risk_scores[detour_target])
                if (
                    self._patrol_detour_remaining[drone_idx] <= 0
                    or detour_risk <= self.PATROL_ASSIST_CLEAR_RISK
                    or home_region_risk >= self.PATROL_HOME_RETURN_RISK
                ):
                    self._clear_patrol_detour(drone_idx)
                else:
                    chosen[drone_idx] = detour_target
                    self._patrol_detour_remaining[drone_idx] -= 1
                    continue

            assist_target = self._select_neighbor_assist_target(drone_idx, risk_scores)
            if assist_target >= 0 and home_region_risk <= self.PATROL_HOME_CLEAR_RISK:
                assist_risk = float(risk_scores[assist_target])
                if assist_risk >= max(
                    self.PATROL_ASSIST_TRIGGER_RISK,
                    home_region_risk + self.PATROL_ASSIST_MARGIN,
                ):
                    self._patrol_detour_targets[drone_idx] = int(assist_target)
                    self._patrol_detour_remaining[drone_idx] = self.PATROL_MAX_DETOUR_STEPS - 1
                    self._record_detour_start(drone_idx)
                    chosen[drone_idx] = int(assist_target)
                    continue

            chosen[drone_idx] = int(patrol_target)

        return chosen

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        return None

    def _resolve_threat_speed(
        self,
        *,
        speed_case: str,
        speed_override: Optional[float],
    ) -> float:
        """Resolve the moving-threat speed from a named debug case or override."""
        if speed_override is not None:
            return float(max(float(speed_override), 1e-6))
        if speed_case not in self.THREAT_SPEED_CASES:
            available = ", ".join(sorted(self.THREAT_SPEED_CASES))
            raise ValueError(
                f"Unknown persistent_threat_speed_case '{speed_case}'. Expected one of: {available}"
            )
        return float(self.THREAT_SPEED_CASES[speed_case])

    def _normalize_suspicious_zones(
        self,
        suspicious_zones: Optional[Sequence[Dict[str, object] | SuspiciousZone]],
    ) -> List[SuspiciousZone]:
        zones: List[SuspiciousZone] = []
        if not suspicious_zones:
            return zones

        for zone in suspicious_zones:
            if isinstance(zone, SuspiciousZone):
                zones.append(zone)
                continue
            center = zone.get("center", [0.0, 0.0])
            zones.append(
                SuspiciousZone(
                    center=(float(center[0]), float(center[1])),
                    radius=float(zone.get("radius", 1.0)),
                    score=float(zone.get("score", 0.5)),
                )
            )
        return zones

    def _build_home_subregions(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Assign each drone a deterministic home strip plus a narrow assist band.

        The longer map axis is partitioned into contiguous strips. Each drone
        owns one strip as its home patrol region and may assist only in the
        immediate neighboring boundary strip cells.
        """
        split_by_cols = self.area_size[0] >= self.area_size[1]
        axis_values = self.global_belief.cols if split_by_cols else self.global_belief.rows
        unique_axis = np.unique(axis_values)
        axis_groups = [group.astype(np.int32) for group in np.array_split(unique_axis, self.num_drones)]

        home_regions: List[np.ndarray] = []
        for drone_idx, group in enumerate(axis_groups):
            if group.size == 0:
                fallback = np.arange(drone_idx, self.n_cells, self.num_drones, dtype=np.int32)
                home_regions.append(fallback)
                continue
            mask = np.isin(axis_values, group)
            home_regions.append(np.flatnonzero(mask).astype(np.int32))

        assist_regions: List[np.ndarray] = []
        for drone_idx, home_cells in enumerate(home_regions):
            if home_cells.size == 0:
                assist_regions.append(np.zeros(0, dtype=np.int32))
                continue

            home_axis_values = axis_values[home_cells]
            min_axis = int(np.min(home_axis_values))
            max_axis = int(np.max(home_axis_values))
            assist_axis = []
            if min_axis - 1 >= int(np.min(unique_axis)):
                assist_axis.append(min_axis - 1)
            if max_axis + 1 <= int(np.max(unique_axis)):
                assist_axis.append(max_axis + 1)

            if assist_axis:
                assist_mask = np.isin(axis_values, np.array(assist_axis, dtype=np.int32))
                assist_cells = np.setdiff1d(
                    np.flatnonzero(assist_mask).astype(np.int32),
                    home_cells,
                    assume_unique=False,
                )
            else:
                assist_cells = np.zeros(0, dtype=np.int32)
            assist_regions.append(assist_cells)

        return home_regions, assist_regions

    def _build_boustrophedon_route(self, home_cells: np.ndarray) -> np.ndarray:
        """Create a serpentine patrol route through one home subregion."""
        if home_cells.size == 0:
            return np.zeros(0, dtype=np.int32)

        rows = self.global_belief.rows[home_cells]
        cols = self.global_belief.cols[home_cells]
        route: List[int] = []
        for offset, row in enumerate(np.sort(np.unique(rows))):
            row_cells = home_cells[rows == row]
            row_cols = cols[rows == row]
            order = np.argsort(row_cols)
            if offset % 2 == 1:
                order = order[::-1]
            route.extend(int(cell) for cell in row_cells[order])

        route_array = np.asarray(route, dtype=np.int32)
        distances = np.linalg.norm(
            self.cell_centers_xy[route_array] - self.base_station[None, :],
            axis=1,
        )
        start_idx = int(np.argmin(distances))
        return np.concatenate([route_array[start_idx:], route_array[:start_idx]])

    def _build_threat_spawn_preference_field(self) -> np.ndarray:
        """
        Build a deterministic spawn-preference field for persistent threat patches.

        The existing ``suspicious_zones`` configuration becomes a low-churn way to
        express where hidden persistent threats are more likely to appear.
        """
        scores = np.ones(self.n_cells, dtype=np.float64)
        if not self.suspicious_zones:
            return scores

        centers = self.cell_centers_xy
        for zone in self.suspicious_zones:
            zone_center = np.array(zone.center, dtype=np.float64)
            distances = np.linalg.norm(centers - zone_center[None, :], axis=1)
            influence = np.clip(1.0 - distances / max(zone.radius, 1e-6), 0.0, 1.0)
            scores += influence * float(zone.score)
        return scores

    def _build_threat_patch_candidates(self) -> List[ThreatPatchCandidate]:
        """Enumerate contiguous 2x2 grid patches that can host one moving threat."""
        index_by_cell = {
            (int(row), int(col)): idx
            for idx, (row, col) in enumerate(zip(self.global_belief.rows, self.global_belief.cols))
        }
        candidates: List[ThreatPatchCandidate] = []
        min_row = int(np.min(self.global_belief.rows))
        max_row = int(np.max(self.global_belief.rows))
        min_col = int(np.min(self.global_belief.cols))
        max_col = int(np.max(self.global_belief.cols))

        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                cells: List[int] = []
                for row_offset in range(self.THREAT_PATCH_WIDTH):
                    for col_offset in range(self.THREAT_PATCH_WIDTH):
                        key = (row + row_offset, col + col_offset)
                        if key not in index_by_cell:
                            cells = []
                            break
                        cells.append(int(index_by_cell[key]))
                    if not cells:
                        break
                if len(cells) != self.THREAT_PATCH_WIDTH ** 2:
                    continue

                patch = np.asarray(cells, dtype=np.int32)
                centroid = np.mean(self.cell_centers_xy[patch], axis=0)
                base_distance = np.linalg.norm(centroid - self.base_station)
                base_penalty = 0.2 if base_distance < 2.0 * self.grid_resolution else 1.0
                weight = float(np.mean(self._threat_spawn_preference[patch]) * base_penalty)
                candidates.append(
                    ThreatPatchCandidate(
                        cell_indices=patch,
                        centroid_xy=np.asarray(centroid, dtype=np.float64),
                        weight=max(weight, 1e-6),
                    )
                )

        if self.enable_persistent_threats and not candidates:
            raise ValueError("BeliefCoverageEnv requires at least one 2x2 threat patch candidate.")
        return candidates

    def _active_threat_exists(self) -> bool:
        return bool(self._active_threat_cells.size > 0)

    def _active_threat_centroid(self) -> np.ndarray | None:
        if not self._active_threat_exists():
            return None
        if np.all(np.isfinite(self._active_threat_position)):
            return self._active_threat_position.copy()
        return np.mean(self.cell_centers_xy[self._active_threat_cells], axis=0)

    def _threat_base_reach_radius(self) -> float:
        return float(self.THREAT_BASE_REACH_RADIUS_FACTOR * self.grid_resolution)

    def _estimate_threat_base_eta_steps(self, centroid_xy: np.ndarray) -> int:
        distance = float(np.linalg.norm(np.asarray(centroid_xy, dtype=np.float64) - self.base_station))
        step_reach = max(self.persistent_threat_speed * self.dt, 1e-6)
        return max(1, int(np.ceil(distance / step_reach)))

    def _select_threat_patch_for_position(self, position_xy: np.ndarray) -> tuple[int, np.ndarray]:
        if self._threat_candidate_centroids.shape[0] == 0:
            return -1, np.zeros(0, dtype=np.int32)
        distances = np.linalg.norm(
            self._threat_candidate_centroids - np.asarray(position_xy, dtype=np.float64)[None, :],
            axis=1,
        )
        candidate_index = int(np.argmin(distances))
        return candidate_index, self._threat_patch_candidates[candidate_index].cell_indices.copy()

    def _compute_threat_lateral_sign(
        self,
        centroid_xy: np.ndarray,
        *,
        candidate_index: int,
        patch: np.ndarray,
    ) -> float:
        relative = np.asarray(centroid_xy, dtype=np.float64) - self.base_station
        signal = float(relative[0] - relative[1])
        if abs(signal) < 1e-6:
            seed = int(candidate_index if candidate_index >= 0 else np.sum(np.asarray(patch, dtype=np.int32)))
            signal = 1.0 if seed % 2 == 0 else -1.0
        return 1.0 if signal >= 0.0 else -1.0

    def _set_active_threat_cells(
        self,
        patch: np.ndarray,
        *,
        candidate_index: int,
        position_xy: np.ndarray,
    ) -> None:
        patch = np.unique(np.asarray(patch, dtype=np.int32))
        self._truth_persistent_threat.fill(0.0)
        if patch.size > 0:
            self._truth_persistent_threat[patch] = 1.0
        self._truth_risk_cue = self._truth_persistent_threat.copy()
        self._active_threat_cells = patch.copy()
        self._active_threat_patch_index = int(candidate_index)
        self._active_threat_position = np.asarray(position_xy, dtype=np.float64).copy()
        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )

    def _active_threat_reached_base_region(self) -> bool:
        if not self._active_threat_exists():
            return False
        reach_radius = self._threat_base_reach_radius()
        patch_distances = np.linalg.norm(
            self.cell_centers_xy[self._active_threat_cells] - self.base_station[None, :],
            axis=1,
        )
        if np.any(patch_distances <= reach_radius):
            return True
        centroid = self._active_threat_centroid()
        return bool(
            centroid is not None
            and float(np.linalg.norm(centroid - self.base_station)) <= reach_radius
        )

    def _advance_moving_threat(self) -> Dict[str, float | bool]:
        """
        Advance the hidden 2x2 threat patch toward the base with a mild lateral bias.

        The threat moves on a continuous centroid trajectory for interceptor
        retargeting and diagnostic plots, while the hidden world state remains a
        2x2 patch snapped to the nearest valid candidate on the grid.
        """
        if not self._active_threat_exists():
            return {
                "threat_moved": False,
                "threat_step_distance": 0.0,
                "physical_base_reached": False,
            }

        current_position = self._active_threat_centroid()
        if current_position is None:
            return {
                "threat_moved": False,
                "threat_step_distance": 0.0,
                "physical_base_reached": False,
            }

        to_base = self.base_station - current_position
        distance_to_base = float(np.linalg.norm(to_base))
        reach_radius = self._threat_base_reach_radius()
        if distance_to_base <= reach_radius:
            self._physical_base_reached_this_step = True
            self._mission_failed = True
            self._mission_fail_reason = "persistent_threat_reached_base"
            return {
                "threat_moved": False,
                "threat_step_distance": 0.0,
                "physical_base_reached": True,
            }

        base_direction = to_base / max(distance_to_base, 1e-9)
        lateral_direction = np.array([-base_direction[1], base_direction[0]], dtype=np.float64)
        lateral_scale = self.THREAT_LATERAL_BIAS * np.clip(
            distance_to_base / max(6.0 * self.grid_resolution, 1e-6),
            0.0,
            1.0,
        )
        direction = base_direction + self._active_threat_lateral_sign * lateral_scale * lateral_direction
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-9:
            direction /= direction_norm
        step_distance = min(self.persistent_threat_speed * self.dt, distance_to_base)
        new_position = current_position + direction * step_distance
        candidate_index, patch = self._select_threat_patch_for_position(new_position)
        self._active_threat_velocity = (new_position - current_position) / max(self.dt, 1e-9)
        self._set_active_threat_cells(
            patch,
            candidate_index=candidate_index,
            position_xy=new_position,
        )
        self._threat_base_eta_steps = self._estimate_threat_base_eta_steps(new_position)
        self._threat_trace.append(new_position.copy())

        physical_base_reached = self._active_threat_reached_base_region()
        if physical_base_reached:
            self._physical_base_reached_this_step = True
            self._mission_failed = True
            self._mission_fail_reason = "persistent_threat_reached_base"

        return {
            "threat_moved": True,
            "threat_step_distance": float(step_distance),
            "physical_base_reached": bool(physical_base_reached),
        }

    def _reset_threat_cycle_state(self) -> None:
        """Reset persistent-threat state for a new episode."""
        self._truth_persistent_threat.fill(0.0)
        self._truth_risk_cue = self._truth_persistent_threat.copy()
        self._threat_persistence_score.fill(0.0)
        self._confirmed_threat_mask.fill(False)
        self._active_threat_cells = np.zeros(0, dtype=np.int32)
        self._active_threat_patch_index = -1
        self._active_threat_position[:] = np.nan
        self._active_threat_velocity[:] = 0.0
        self._active_threat_confirmation_level = 0.0
        self._active_threat_lateral_sign = 1.0
        self._threat_confirmed = False
        self._threat_cycles_spawned = 0
        self._threat_cycles_completed = 0
        self._threat_removed_this_step = False
        self._threat_respawned_this_step = False
        self._interceptor_dispatched_this_step = False
        self._physical_base_reached_this_step = False
        self._central_command_notified = False
        self._tracking_bias_drones.fill(False)
        self._tracking_target_cells.fill(-1)
        self._mission_failed = False
        self._mission_fail_reason = ""
        self._threat_base_eta_steps = -1
        self._threat_base_timeout_steps = -1
        self._threat_patch_history = []
        self._threat_trace = []
        self._interceptor_active = False
        self._interceptor_position = self.base_station.copy()
        self._interceptor_velocity[:] = 0.0
        self._interceptor_target[:] = np.nan
        self._interceptor_trace = [self.base_station.copy()]
        self._interceptor_dispatch_count = 0

        if self.enable_persistent_threats and self.max_threat_cycles > 0:
            self._spawn_next_threat_patch()
            self._threat_respawned_this_step = False

    def _choose_next_threat_patch_candidate(self) -> int:
        if not self._threat_patch_candidates:
            return -1
        candidate_weights = np.array(
            [candidate.weight for candidate in self._threat_patch_candidates],
            dtype=np.float64,
        )
        if self._active_threat_patch_index >= 0:
            candidate_weights[self._active_threat_patch_index] = 0.0
        available_unseen = [
            idx
            for idx in range(len(candidate_weights))
            if idx != self._active_threat_patch_index and idx not in set(self._threat_patch_history)
        ]
        if available_unseen:
            mask = np.zeros_like(candidate_weights)
            mask[np.asarray(available_unseen, dtype=np.int32)] = 1.0
            candidate_weights *= mask
        else:
            for candidate_idx in self._threat_patch_history:
                if 0 <= int(candidate_idx) < len(candidate_weights):
                    candidate_weights[int(candidate_idx)] *= 0.25
        if not np.any(candidate_weights > 0.0):
            candidate_weights.fill(1.0)
        probabilities = candidate_weights / np.sum(candidate_weights)
        return int(self.np_random.choice(len(self._threat_patch_candidates), p=probabilities))

    def _activate_threat_patch(
        self,
        cell_indices: np.ndarray,
        *,
        candidate_index: int = -1,
        count_as_spawn: bool = True,
        base_eta_steps: Optional[int] = None,
        position_xy: Optional[np.ndarray] = None,
    ) -> None:
        patch = np.unique(np.asarray(cell_indices, dtype=np.int32))
        centroid_xy = (
            np.asarray(position_xy, dtype=np.float64)
            if position_xy is not None
            else (
                np.mean(self.cell_centers_xy[patch], axis=0)
                if patch.size > 0
                else self.base_station.copy()
            )
        )
        chosen_candidate_index = int(candidate_index)
        if chosen_candidate_index < 0 and patch.size > 0:
            chosen_candidate_index, _ = self._select_threat_patch_for_position(centroid_xy)
        self._set_active_threat_cells(
            patch,
            candidate_index=chosen_candidate_index,
            position_xy=centroid_xy,
        )
        self._threat_persistence_score.fill(0.0)
        self._active_threat_confirmation_level = 0.0
        self._confirmed_threat_mask.fill(False)
        self._threat_confirmed = False
        self._central_command_notified = False
        self._tracking_bias_drones.fill(False)
        self._tracking_target_cells.fill(-1)
        self._active_threat_lateral_sign = self._compute_threat_lateral_sign(
            centroid_xy,
            candidate_index=chosen_candidate_index,
            patch=patch,
        )
        self._active_threat_velocity[:] = 0.0
        if self._threat_trace:
            self._threat_trace.append(np.array([np.nan, np.nan], dtype=np.float64))
            self._threat_trace.append(centroid_xy.copy())
        else:
            self._threat_trace = [centroid_xy.copy()]
        self._interceptor_active = False
        self._interceptor_position = self.base_station.copy()
        self._interceptor_velocity[:] = 0.0
        self._interceptor_target[:] = np.nan
        self._interceptor_trace = [self.base_station.copy()]
        self._threat_base_eta_steps = (
            int(base_eta_steps)
            if base_eta_steps is not None
            else (
                self._estimate_threat_base_eta_steps(centroid_xy)
                if patch.size > 0
                else -1
            )
        )
        self._threat_base_timeout_steps = self._threat_base_eta_steps
        if count_as_spawn and patch.size > 0:
            self._threat_cycles_spawned += 1
            if chosen_candidate_index >= 0:
                self._threat_patch_history.append(int(chosen_candidate_index))

    def _spawn_next_threat_patch(self) -> bool:
        """Spawn the next persistent-threat patch if cycles remain."""
        if not self.enable_persistent_threats or self.max_threat_cycles <= 0:
            return False
        if self._threat_cycles_spawned >= self.max_threat_cycles:
            return False

        candidate_index = self._choose_next_threat_patch_candidate()
        if candidate_index < 0:
            return False
        candidate = self._threat_patch_candidates[candidate_index]
        self._activate_threat_patch(
            candidate.cell_indices,
            candidate_index=candidate_index,
            count_as_spawn=True,
        )
        self._threat_respawned_this_step = True
        return True

    def _reset_patrol_state(self) -> None:
        """Reset deterministic patrol progress and detour state."""
        self._patrol_route_indices.fill(0)
        self._patrol_detour_targets.fill(-1)
        self._patrol_detour_remaining.fill(0)
        self._patrol_detour_counts.fill(0)
        self._time_away_from_home_steps.fill(0)
        for drone_idx, route in enumerate(self._home_patrol_routes):
            if route.size == 0:
                continue
            distances = np.linalg.norm(
                self.cell_centers_xy[route] - self.base_station[None, :],
                axis=1,
            )
            self._patrol_route_indices[drone_idx] = int(np.argmin(distances))

    def _clear_patrol_detour(self, drone_idx: int) -> None:
        self._patrol_detour_targets[drone_idx] = -1
        self._patrol_detour_remaining[drone_idx] = 0

    def _record_detour_start(self, drone_idx: int) -> None:
        self._patrol_detour_counts[drone_idx] += 1

    def _current_risk_scores(self) -> np.ndarray:
        """
        Return the Phase 1 patrol-risk proxy.

        This keeps the existing uncertainty/anomaly storage intact but treats
        the monitored quantity operationally as conservative patrol risk.
        """
        return self.get_belief_state().combined_global_risk_belief

    def get_belief_risk_scores(self) -> np.ndarray:
        """Return the fused Phase 1 belief-risk score for each cell."""
        return self._current_risk_scores().copy()

    def get_patrol_risk_belief_scores(self) -> np.ndarray:
        """Return the fused patrol-risk belief component."""
        return self.global_belief.uncertainty.copy()

    def get_threat_belief_scores(self) -> np.ndarray:
        """Return the fused latent-threat belief component."""
        return self.global_belief.anomaly_score.copy()

    def get_threat_persistence_scores(self) -> np.ndarray:
        """Return the current persistence-confirmation score field."""
        return self._threat_persistence_score.copy()

    def get_confirmed_threat_mask(self) -> np.ndarray:
        """Return a float mask over currently confirmed persistent-threat cells."""
        return self._confirmed_threat_mask.astype(np.float64).copy()

    def get_active_threat_mask(self) -> np.ndarray:
        """Return a float mask over the currently active hidden threat patch."""
        active_mask = np.zeros(self.n_cells, dtype=np.float64)
        if self._active_threat_exists():
            active_mask[self._active_threat_cells] = 1.0
        return active_mask

    def get_interceptor_trace(self) -> np.ndarray:
        """Return the interceptor trajectory accumulated so far."""
        return np.asarray(self._interceptor_trace, dtype=np.float64).copy()

    def get_threat_trace(self) -> np.ndarray:
        """Return the moving threat-centroid trajectory accumulated so far."""
        return np.asarray(self._threat_trace, dtype=np.float64).copy()

    def get_truth_risk_scores(self) -> np.ndarray:
        """Return the backend truth-risk score for each cell."""
        return self.get_hidden_world_state().combined_risk_state.copy()

    def get_truth_monitoring_risk(self) -> np.ndarray:
        """Return the backend neglect-monitoring risk before cue fusion."""
        return self._truth_monitoring_risk.copy()

    def get_truth_persistent_threat_state(self) -> np.ndarray:
        """Return the hidden persistent-threat world state."""
        return self._truth_persistent_threat.copy()

    def get_hidden_world_state(self) -> PatrolHiddenWorldState:
        """Return the current hidden world-state view for POMDP-style analysis."""
        return PatrolHiddenWorldState(
            patrol_risk_state=self._truth_monitoring_risk.copy(),
            persistent_threat_state=self._truth_persistent_threat.copy(),
            drone_positions=self._drone_states[:, :2].copy(),
            drone_yaws=self._drone_states[:, 9].copy(),
        )

    def get_belief_state(self) -> PatrolBeliefState:
        """Return the current fused/local belief-state view."""
        return PatrolBeliefState.from_belief_grids(
            self.global_belief,
            self.local_beliefs,
            global_threat_confirmation_belief=self._threat_persistence_score,
        )

    def rasterize_cell_values(
        self,
        cell_values: np.ndarray,
        *,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Map per-cell values onto a regular row/column grid for plotting."""
        values = np.asarray(cell_values, dtype=np.float64).reshape(-1)
        if values.size != self.n_cells:
            raise ValueError(f"Expected {self.n_cells} cell values, got {values.size}")

        grid = np.full(
            (len(self._row_values), len(self._col_values)),
            fill_value,
            dtype=np.float64,
        )
        for idx, value in enumerate(values):
            row = self._row_lookup[int(self.global_belief.rows[idx])]
            col = self._col_lookup[int(self.global_belief.cols[idx])]
            grid[row, col] = float(value)
        return grid

    def get_home_strip_boundaries(self) -> Dict[str, np.ndarray | str]:
        """Return axis-aligned home-strip boundaries for diagnostic plotting."""
        split_by_cols = self.area_size[0] >= self.area_size[1]
        axis_values = self.global_belief.cols if split_by_cols else self.global_belief.rows
        boundaries: List[float] = []
        for home_cells in self._home_cell_indices[:-1]:
            if home_cells.size == 0:
                continue
            max_axis = int(np.max(axis_values[home_cells]))
            boundaries.append((max_axis + 1) * self.grid_resolution)
        return {
            "axis": "x" if split_by_cols else "y",
            "positions": np.asarray(boundaries, dtype=np.float64),
        }

    def get_home_cell_indices(self, drone_idx: int) -> np.ndarray:
        """Return a copy of one drone's home subregion cells."""
        return self._home_cell_indices[int(drone_idx)].copy()

    def get_assist_cell_indices(self, drone_idx: int) -> np.ndarray:
        """Return a copy of one drone's local assist-band cells."""
        return self._assist_cell_indices[int(drone_idx)].copy()

    def _peek_home_patrol_target(self, drone_idx: int) -> int:
        """Inspect the current home patrol target without mutating route state."""
        route = self._home_patrol_routes[drone_idx]
        if route.size == 0:
            return 0
        route_index = int(self._patrol_route_indices[drone_idx]) % int(route.size)
        return int(route[route_index])

    def _place_drones_in_formation(self) -> None:
        spacing = self.collision_radius * self.FORMATION_SPACING_FACTOR
        split_by_cols = self.area_size[0] >= self.area_size[1]
        home_axis_centers = np.array(
            [
                float(np.mean(self.cell_centers_xy[home_cells, 0 if split_by_cols else 1]))
                if home_cells.size
                else float(self.base_station[0 if split_by_cols else 1])
                for home_cells in self._home_cell_indices
            ],
            dtype=np.float64,
        )
        launch_order = np.argsort(home_axis_centers)
        primary_offsets = (np.arange(self.num_drones, dtype=np.float64) - 0.5 * (self.num_drones - 1)) * spacing
        secondary_offsets = np.array(
            [(-0.5 if idx % 2 == 0 else 0.5) * 0.45 * spacing for idx in range(self.num_drones)],
            dtype=np.float64,
        )

        for rank, drone_idx in enumerate(launch_order):
            if split_by_cols:
                x_offset = primary_offsets[rank]
                y_offset = secondary_offsets[rank]
            else:
                x_offset = secondary_offsets[rank]
                y_offset = primary_offsets[rank]
            position = np.array(
                [
                    self.base_station[0] + x_offset,
                    self.base_station[1] + y_offset,
                    self.fixed_altitude,
                ],
                dtype=np.float64,
            )
            self._simulator.set_drone_initial_state(int(drone_idx), position)

    def _sync_states_from_sim(self) -> None:
        for i in range(self.num_drones):
            drone = self._simulator.drones[i]
            pos = drone.position
            vel = drone.velocity
            acc = drone.acceleration
            w, qx, qy, qz = drone.q
            yaw = float(
                np.arctan2(
                    2.0 * (w * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
            )
            yaw_rate = float(drone.angular_velocity[2])
            self._drone_states[i] = np.array(
                [
                    pos[0], pos[1], pos[2],
                    vel[0], vel[1], vel[2],
                    acc[0], acc[1], acc[2],
                    yaw, yaw_rate,
                ],
                dtype=np.float64,
            )

    def _align_heading_to_targets(self, selected_cells: np.ndarray) -> None:
        """
        Keep the surveillance heading pointed toward the active target cell.

        The current simulator backend does not include a dedicated yaw-tracking
        controller for the Phase 1 cell-selection path, so we explicitly align
        the sensing heading here to make the forward-FOV observation semantics
        match the chosen high-level target.
        """
        for drone_idx, cell_idx in enumerate(np.asarray(selected_cells, dtype=np.int32)):
            target_xy = self.cell_centers_xy[int(cell_idx)]
            position_xy = self._drone_states[drone_idx, :2]
            delta = target_xy - position_xy
            if np.linalg.norm(delta) <= 1e-6:
                continue

            desired_yaw = float(np.arctan2(delta[1], delta[0]))
            previous_yaw = float(self._drone_states[drone_idx, 9])
            yaw_rate = float(_wrap_angle(desired_yaw - previous_yaw) / max(self.dt, 1e-6))

            drone = self._simulator.drones[drone_idx]
            drone.q = np.array(
                [np.cos(desired_yaw * 0.5), 0.0, 0.0, np.sin(desired_yaw * 0.5)],
                dtype=np.float64,
            )
            drone.angular_velocity = np.array([0.0, 0.0, yaw_rate], dtype=np.float64)

            self._drone_states[drone_idx, 9] = desired_yaw
            self._drone_states[drone_idx, 10] = yaw_rate

    def _apply_planar_tracking(self, selected_cells: np.ndarray) -> None:
        """
        Apply explicit planar waypoint tracking for the patrol-risk baseline.

        The current simulator backend models hover, battery, and yaw state well
        but does not provide a lateral attitude controller for this discrete
        cell-selection path. We keep the simulator backend active and add a
        simple world-frame tracker here so Phase 1 patrol produces meaningful
        map coverage motion.
        """
        for drone_idx, cell_idx in enumerate(np.asarray(selected_cells, dtype=np.int32)):
            drone = self._simulator.drones[drone_idx]
            target_xy = self.cell_centers_xy[int(cell_idx)]
            position_xy = drone.position[:2].copy()
            delta = target_xy - position_xy
            distance = float(np.linalg.norm(delta))

            if distance <= 1e-6:
                desired_velocity = np.zeros(2, dtype=np.float64)
            else:
                unit_delta = delta / distance
                if distance <= self.WP_ACCEPT_RADIUS:
                    desired_speed = min(self.CRUISE_SPEED, distance / max(self.dt, 1e-6))
                else:
                    desired_speed = self.CRUISE_SPEED
                desired_velocity = unit_delta * desired_speed

            current_velocity = drone.velocity[:2].copy()
            accel_xy = (desired_velocity - current_velocity) * self.PLANAR_TRACKING_GAIN
            accel_norm = float(np.linalg.norm(accel_xy))
            if accel_norm > self.accel_max:
                accel_xy *= self.accel_max / max(accel_norm, 1e-6)

            new_velocity = current_velocity + accel_xy * self.dt
            vel_norm = float(np.linalg.norm(new_velocity))
            if vel_norm > self.CRUISE_SPEED:
                new_velocity *= self.CRUISE_SPEED / max(vel_norm, 1e-6)

            step_delta = new_velocity * self.dt
            if distance <= self.WP_ACCEPT_RADIUS and np.linalg.norm(step_delta) >= distance:
                next_position = target_xy
                new_velocity[:] = 0.0
                accel_xy[:] = 0.0
            else:
                next_position = position_xy + step_delta

            drone.position[:2] = next_position
            drone.position[2] = self.fixed_altitude
            drone.velocity[:2] = new_velocity
            drone.velocity[2] = 0.0
            drone.acceleration[:2] = accel_xy
            drone.acceleration[2] = 0.0

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        cells = np.asarray(action, dtype=np.int64).reshape(-1)
        if cells.size != self.num_drones:
            raise ValueError(f"Expected {self.num_drones} cell actions, got {cells.size}")
        return np.clip(cells, 0, self.n_cells - 1).astype(np.int32)

    def _select_initial_targets(self) -> np.ndarray:
        selected = np.zeros(self.num_drones, dtype=np.int32)
        for drone_idx in range(self.num_drones):
            route = self._home_patrol_routes[drone_idx]
            selected[drone_idx] = int(route[self._patrol_route_indices[drone_idx]]) if route.size else 0
        return selected

    def _select_home_patrol_target(self, drone_idx: int, risk_scores: np.ndarray) -> int:
        route = self._home_patrol_routes[drone_idx]
        if route.size == 0:
            return 0

        route_index = int(self._patrol_route_indices[drone_idx]) % int(route.size)
        current_target = int(route[route_index])
        target_distance = float(
            np.linalg.norm(
                self.cell_centers_xy[current_target] - self._drone_states[drone_idx, :2]
            )
        )
        if (
            risk_scores[current_target] <= self.PATROL_HOME_CLEAR_RISK
            or target_distance <= self.WP_ACCEPT_RADIUS
        ):
            route_index = (route_index + 1) % int(route.size)
            self._patrol_route_indices[drone_idx] = route_index
            current_target = int(route[route_index])
        return current_target

    def _select_neighbor_assist_target(self, drone_idx: int, risk_scores: np.ndarray) -> int:
        assist_cells = self._assist_cell_indices[drone_idx]
        if assist_cells.size == 0:
            return -1

        position = self._drone_states[drone_idx, :2]
        distances = np.linalg.norm(self.cell_centers_xy[assist_cells] - position[None, :], axis=1)
        assist_scores = risk_scores[assist_cells] - 0.15 * np.clip(
            distances / max(np.linalg.norm(self.area_size), 1e-6),
            0.0,
            1.0,
        )
        return int(assist_cells[int(np.argmax(assist_scores))])

    def _positions_in_home_region(self) -> np.ndarray:
        """Return a binary mask for whether each drone is inside its own home strip."""
        in_home = np.zeros(self.num_drones, dtype=np.int32)
        split_by_cols = self.area_size[0] >= self.area_size[1]
        axis_values = self.global_belief.cols if split_by_cols else self.global_belief.rows

        for drone_idx, home_cells in enumerate(self._home_cell_indices):
            if home_cells.size == 0:
                continue
            home_axis = axis_values[home_cells]
            lower = float(np.min(home_axis) * self.grid_resolution)
            upper = float((np.max(home_axis) + 1) * self.grid_resolution)
            position_axis = float(self._drone_states[drone_idx, 0 if split_by_cols else 1])
            in_home[drone_idx] = int(lower <= position_axis < upper)
        return in_home

    def _update_target_references(self, selected_cells: np.ndarray) -> None:
        for drone_idx, cell_idx in enumerate(selected_cells):
            target = self.cell_centers[int(cell_idx)]
            self._references[drone_idx] = self._build_trajectory_to_waypoint(
                self._drone_states[drone_idx],
                target,
            )

    def _build_trajectory_to_waypoint(
        self,
        state: np.ndarray,
        waypoint: np.ndarray,
    ) -> np.ndarray:
        ref = np.zeros((self.horizon + 1, STATE_DIM), dtype=np.float64)
        position = state[:3].copy()
        direction = waypoint[:3] - position
        distance = float(np.linalg.norm(direction))

        if distance < 0.5:
            for k in range(self.horizon + 1):
                ref[k, :3] = position
            return ref

        unit_direction = direction / distance
        horizon_time = self.horizon * self.dt
        speed = self.CRUISE_SPEED if distance >= self.CRUISE_SPEED * horizon_time else distance / max(horizon_time, self.dt)
        target_velocity = unit_direction * speed
        current = position.copy()

        for k in range(self.horizon + 1):
            ref[k, :3] = current
            ref[k, 3:6] = target_velocity
            next_position = current + target_velocity * self.dt
            if np.linalg.norm(next_position - position) < distance:
                current = next_position
            else:
                current = waypoint[:3].copy()

        return ref

    def _accel_to_motor_cmds(self, controls: np.ndarray) -> np.ndarray:
        motor_cmds = np.zeros((self.num_drones, 4), dtype=np.float32)
        hover_norm = 0.5
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

    def _battery_levels(self) -> np.ndarray:
        levels = np.zeros(self.num_drones, dtype=np.float64)
        for i, drone in enumerate(self._simulator.drones):
            levels[i] = max(drone.battery_energy / drone.config.battery_capacity, 0.0)
        return levels

    def _distance_to_base_norm(self) -> np.ndarray:
        distances = np.linalg.norm(self._drone_states[:, :2] - self.base_station[None, :], axis=1)
        return np.clip(distances / max(np.linalg.norm(self.area_size), 1e-6), 0.0, 1.0)

    def _distance_to_selected_targets(self, selected_cells: np.ndarray) -> np.ndarray:
        targets = self.cell_centers_xy[selected_cells]
        distances = np.linalg.norm(self._drone_states[:, :2] - targets, axis=1)
        return np.clip(distances / max(np.linalg.norm(self.area_size), 1e-6), 0.0, 1.0)

    def _compute_connectivity(self) -> Dict[str, np.ndarray | int | float]:
        positions = self._drone_states[:, :2]
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        adjacency = (distances <= self.communication_range).astype(np.int32)
        np.fill_diagonal(adjacency, 0)

        component_ids = -np.ones(self.num_drones, dtype=np.int32)
        component_count = 0
        for node in range(self.num_drones):
            if component_ids[node] >= 0:
                continue
            queue = [node]
            component_ids[node] = component_count
            while queue:
                current = queue.pop()
                neighbours = np.flatnonzero(adjacency[current])
                for neighbour in neighbours:
                    if component_ids[neighbour] < 0:
                        component_ids[neighbour] = component_count
                        queue.append(int(neighbour))
            component_count += 1

        neighbor_counts = np.sum(adjacency, axis=1)
        connectivity_score = 1.0 / max(component_count, 1)
        return {
            "adjacency": adjacency,
            "component_ids": component_ids,
            "component_count": int(component_count),
            "neighbor_counts": neighbor_counts.astype(np.float64),
            "connectivity_score": float(connectivity_score),
        }

    def _project_action_cells(self, selected_cells: np.ndarray) -> np.ndarray:
        projected = selected_cells.copy()
        positions = self._drone_states[:, :2]
        for drone_idx, cell_idx in enumerate(selected_cells):
            goal = self.cell_centers_xy[int(cell_idx)]
            if self._goal_preserves_link(drone_idx, goal, positions):
                continue

            goal_distances = np.linalg.norm(self.cell_centers_xy - goal[None, :], axis=1)
            for candidate in np.argsort(goal_distances):
                candidate_goal = self.cell_centers_xy[int(candidate)]
                if self._goal_preserves_link(drone_idx, candidate_goal, positions):
                    projected[drone_idx] = int(candidate)
                    break
        return projected

    def _goal_preserves_link(
        self,
        drone_idx: int,
        goal_xy: np.ndarray,
        positions: np.ndarray,
    ) -> bool:
        if self.num_drones <= 1:
            return True
        others = np.delete(positions, drone_idx, axis=0)
        distances = np.linalg.norm(others - goal_xy[None, :], axis=1)
        return bool(np.any(distances <= self.communication_range))

    def _build_truth_risk_cue_field(self) -> np.ndarray:
        """
        Backward-compatible alias for older helpers that expected a cue field.

        The current persistent-threat loop uses suspicious zones as spawn
        preferences rather than as a static world-state field.
        """
        return self._build_threat_spawn_preference_field()

    def _compute_threat_belief_evidence(
        self,
        cell_indices: np.ndarray,
        qualities: np.ndarray,
    ) -> np.ndarray:
        """Project hidden persistent-threat state into local threat-belief evidence."""
        evidence = self.observation_model.project_threat_belief_evidence(
            cell_indices,
            qualities,
            self._truth_persistent_threat,
        )
        if evidence.size == 0:
            return evidence
        return np.clip(0.15 + 0.85 * evidence, 0.0, 1.0)

    def _truth_risk_counts(self) -> dict[str, int]:
        """Return threshold counts over the backend truth-risk field."""
        return {
            "gt_0_1": int(np.sum(self._truth_risk_score > self.global_belief.config.report_threshold)),
            "gt_0_4": int(np.sum(self._truth_risk_score > self.global_belief.config.revisit_threshold)),
            "gt_0_7": int(np.sum(self._truth_risk_score > self.global_belief.config.alert_threshold)),
        }

    def _update_truth_risk(
        self,
        *,
        step: int,
        grow: bool,
    ) -> Dict[str, object]:
        """Update the hidden world state using the explicit transition model."""
        threat_motion_metrics: Dict[str, float | bool] = {
            "threat_moved": False,
            "threat_step_distance": 0.0,
            "physical_base_reached": False,
        }
        if grow:
            self._truth_monitoring_risk = self.transition_model.advance_patrol_risk_state(
                self._truth_monitoring_risk
            )
            self._truth_persistent_threat = self.transition_model.advance_persistent_threat_state(
                self._truth_persistent_threat
            )
            threat_motion_metrics = self._advance_moving_threat()

        best_quality = np.zeros(self.n_cells, dtype=np.float64)
        observed_mask = np.zeros(self.n_cells, dtype=bool)
        for drone_idx in range(self.num_drones):
            cell_idx, qualities, _, _ = self.truth_observation_model.observe_cells(
                self._drone_states[drone_idx, :2],
                float(self._drone_states[drone_idx, 9]),
                self.cell_centers_xy,
                rng=None,
            )
            if cell_idx.size == 0:
                continue
            best_quality[cell_idx] = np.maximum(best_quality[cell_idx], qualities)
            observed_mask[cell_idx] = True

        observed_idx = np.flatnonzero(observed_mask)
        if observed_idx.size > 0:
            self._truth_monitoring_risk[observed_idx] = np.maximum(
                self.global_belief.config.uncertainty_min,
                1.0 - best_quality[observed_idx],
            )
            self._truth_last_observed_step[observed_idx] = int(step)

        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )
        return {
            "truth_observed_cell_count": int(observed_idx.size),
            **threat_motion_metrics,
        }

    def _update_beliefs(
        self,
        *,
        step: int,
        connectivity: Dict[str, np.ndarray | int | float],
        grow: bool,
        apply_global_sync: bool,
    ) -> Dict[str, object]:
        if grow:
            for local in self.local_beliefs:
                local.advance()

        observed_counts = np.zeros(self.num_drones, dtype=np.int32)
        shared_cell_count = 0
        global_sync_applied = False
        observed_mask = np.zeros(self.n_cells, dtype=bool)
        best_threat_evidence = np.zeros(self.n_cells, dtype=np.float64)

        for drone_idx, local in enumerate(self.local_beliefs):
            position = self._drone_states[drone_idx, :2]
            yaw = float(self._drone_states[drone_idx, 9])
            cell_idx, qualities, _, _ = self.observation_model.observe_cells(
                position,
                yaw,
                self.cell_centers_xy,
                rng=self.np_random,
            )
            observed_counts[drone_idx] = int(cell_idx.size)
            threat_evidence = self._compute_threat_belief_evidence(cell_idx, qualities)
            if cell_idx.size > 0:
                observed_mask[cell_idx] = True
                best_threat_evidence[cell_idx] = np.maximum(best_threat_evidence[cell_idx], threat_evidence)
            self.belief_updater.apply_local_observation(
                local,
                cell_idx,
                qualities,
                threat_evidence,
                step=step,
            )

        if self.enable_neighbor_sharing:
            shared_cell_count = self.belief_updater.share_neighbor_updates(
                self.local_beliefs,
                np.asarray(connectivity["adjacency"], dtype=np.int32),
                step=step,
            )

        self.belief_updater.rebuild_global_belief(
            self.global_belief,
            self.local_beliefs,
            step=step,
        )

        if apply_global_sync and int(connectivity["component_count"]) == 1:
            self.belief_updater.apply_global_sync(
                self.global_belief,
                self.local_beliefs,
                step=step,
            )
            global_sync_applied = True

        for local in self.local_beliefs:
            local.clear_recent_changes()

        return {
            "observed_counts": observed_counts,
            "observed_mask": observed_mask,
            "best_threat_evidence": best_threat_evidence,
            "shared_cell_count": int(shared_cell_count),
            "global_sync_applied": bool(global_sync_applied),
        }

    def _update_threat_confirmation(
        self,
        *,
        observed_mask: np.ndarray,
        best_threat_evidence: np.ndarray,
    ) -> Dict[str, float | bool]:
        """
        Update confirmation belief for the active moving threat patch.

        This is intentionally simple and interpretable: repeated elevated
        evidence on the same hidden patch pushes a persistence score upward.
        The score is carried with the active patch as it moves, then written
        back onto the current patch cells for plotting and analysis.
        """
        self._active_threat_confirmation_level *= self.THREAT_CONFIRMATION_DECAY
        self._threat_persistence_score.fill(0.0)
        self._confirmed_threat_mask.fill(False)

        if not self._active_threat_exists():
            self._threat_confirmed = False
            self._active_threat_confirmation_level = 0.0
            return {
                "threat_suspected": False,
                "threat_confirmed": False,
                "active_threat_confirmation_score": 0.0,
            }

        active_cells = self._active_threat_cells
        observed_active = active_cells[np.asarray(observed_mask[active_cells], dtype=bool)]
        if observed_active.size > 0:
            observed_evidence = float(np.mean(best_threat_evidence[observed_active]))
            self._active_threat_confirmation_level = float(
                np.clip(
                    self._active_threat_confirmation_level
                    + self.THREAT_CONFIRMATION_GAIN * observed_evidence,
                    0.0,
                    1.0,
                )
            )

        self._threat_persistence_score[active_cells] = self._active_threat_confirmation_level
        active_patch_score = float(self._active_threat_confirmation_level)
        active_belief = float(np.mean(self.global_belief.anomaly_score[active_cells]))
        threat_suspected = bool(
            active_belief >= self.THREAT_SUSPECT_THRESHOLD or active_patch_score > 0.0
        )
        if active_patch_score >= self.THREAT_CONFIRMATION_THRESHOLD:
            self._threat_confirmed = True
        if self._threat_confirmed:
            self._confirmed_threat_mask[active_cells] = True
            self._central_command_notified = True

        return {
            "threat_suspected": threat_suspected,
            "threat_confirmed": bool(self._threat_confirmed),
            "active_threat_confirmation_score": active_patch_score,
        }

    def _choose_tracking_bias_drones(self) -> np.ndarray:
        """Pick a small nearby subset of drones to add a light tracking bias."""
        selected = np.zeros(self.num_drones, dtype=bool)
        self._tracking_target_cells.fill(-1)
        if not self._threat_confirmed or not self._active_threat_exists():
            return selected

        patch_centroid = self._active_threat_centroid()
        if patch_centroid is None:
            return selected

        tracker_count = min(self.THREAT_MAX_TRACKERS, max(1, self.num_drones // 3))
        rankings: List[Tuple[int, float, float, int]] = []
        for drone_idx in range(self.num_drones):
            local_cells = np.union1d(
                self._home_cell_indices[drone_idx],
                self._assist_cell_indices[drone_idx],
            )
            local_overlap = int(np.intersect1d(local_cells, self._active_threat_cells).size > 0)
            region_distance = (
                float(np.min(np.linalg.norm(self.cell_centers_xy[local_cells] - patch_centroid[None, :], axis=1)))
                if local_cells.size > 0
                else float("inf")
            )
            position_distance = float(np.linalg.norm(self._drone_states[drone_idx, :2] - patch_centroid))
            rankings.append((drone_idx, -float(local_overlap), position_distance, region_distance))

        for drone_idx, *_ in sorted(rankings, key=lambda item: (item[1], item[2], item[3]))[:tracker_count]:
            selected[int(drone_idx)] = True
            threat_cells = self._active_threat_cells
            distances = np.linalg.norm(
                self.cell_centers_xy[threat_cells] - self._drone_states[int(drone_idx), :2][None, :],
                axis=1,
            )
            self._tracking_target_cells[int(drone_idx)] = int(threat_cells[int(np.argmin(distances))])
        return selected

    def _dispatch_interceptor(self) -> bool:
        """Dispatch a simple straight-line interceptor from the base."""
        if self._interceptor_active or not self._threat_confirmed or not self._active_threat_exists():
            return False
        target_xy = self._active_threat_centroid()
        if target_xy is None:
            return False
        delta = target_xy - self.base_station
        distance = float(np.linalg.norm(delta))
        direction = np.zeros(2, dtype=np.float64) if distance <= 1e-9 else delta / distance
        self._interceptor_active = True
        self._interceptor_position = self.base_station.copy()
        self._interceptor_velocity = direction * self.INTERCEPTOR_SPEED
        self._interceptor_target = target_xy.copy()
        self._interceptor_trace = [self.base_station.copy()]
        self._interceptor_dispatch_count += 1
        return True

    def _clear_threat_from_beliefs(self, cell_indices: np.ndarray) -> None:
        patch = np.asarray(cell_indices, dtype=np.int32)
        if patch.size == 0:
            return
        self.global_belief.anomaly_score[patch] = 0.0
        for local in self.local_beliefs:
            local.anomaly_score[patch] = 0.0
        self._threat_persistence_score[patch] = 0.0
        self._confirmed_threat_mask[patch] = False

    def _resolve_threat_intercept(self) -> None:
        """Clear the current moving threat after a successful interceptor hit."""
        removed_cells = self._active_threat_cells.copy()
        self._clear_threat_from_beliefs(removed_cells)
        self._truth_persistent_threat.fill(0.0)
        self._truth_risk_cue = self._truth_persistent_threat.copy()
        self._active_threat_cells = np.zeros(0, dtype=np.int32)
        self._active_threat_patch_index = -1
        self._active_threat_position[:] = np.nan
        self._active_threat_velocity[:] = 0.0
        self._active_threat_confirmation_level = 0.0
        self._threat_confirmed = False
        self._central_command_notified = False
        self._tracking_bias_drones.fill(False)
        self._tracking_target_cells.fill(-1)
        self._interceptor_active = False
        self._interceptor_velocity[:] = 0.0
        self._interceptor_target[:] = np.nan
        self._threat_removed_this_step = True
        self._threat_cycles_completed += 1
        self._threat_base_eta_steps = -1
        self._threat_base_timeout_steps = -1
        self._truth_risk_score = self.transition_model.compose_hidden_risk_state(
            self._truth_monitoring_risk,
            self._truth_persistent_threat,
        )
        self._spawn_next_threat_patch()

    def _advance_interceptor_and_threat(self) -> Dict[str, bool | float]:
        """
        Advance the retargeting interceptor and resolve countdown-based failures.

        The moving threat itself advances earlier in the hidden-state transition
        update so the drones observe the current patch location before the
        interceptor retargets.
        """
        dispatched = self._dispatch_interceptor()
        removal = False

        if self._interceptor_active and self._active_threat_exists():
            target_xy = self._active_threat_centroid()
            if target_xy is not None:
                self._interceptor_target = target_xy.copy()
                delta = target_xy - self._interceptor_position
                distance = float(np.linalg.norm(delta))
                step_distance = self.INTERCEPTOR_SPEED * self.dt
                if distance <= step_distance:
                    self._interceptor_position = target_xy.copy()
                elif distance > 1e-9:
                    self._interceptor_position += (delta / distance) * step_distance
                self._interceptor_trace.append(self._interceptor_position.copy())

                hit_radius = self.INTERCEPTOR_HIT_RADIUS_FACTOR * self.grid_resolution
                patch_distances = np.linalg.norm(
                    self.cell_centers_xy[self._active_threat_cells] - self._interceptor_position[None, :],
                    axis=1,
                )
                if np.any(patch_distances <= hit_radius):
                    self._resolve_threat_intercept()
                    removal = True

        if self._active_threat_exists() and not removal and not self._mission_failed:
            self._threat_base_eta_steps = self._estimate_threat_base_eta_steps(
                self._active_threat_centroid()
            )
            self._threat_base_timeout_steps -= 1
            if self._threat_base_timeout_steps <= 0:
                self._mission_failed = True
                self._mission_fail_reason = "persistent_threat_timeout_reached_base"

        return {
            "interceptor_dispatched": bool(dispatched),
            "threat_removed": bool(removal),
            "mission_failed": bool(self._mission_failed),
        }

    def _update_threat_response(
        self,
        *,
        belief_metrics: Dict[str, object],
        connectivity: Dict[str, np.ndarray | int | float],
        step: int,
    ) -> Dict[str, object]:
        del connectivity, step

        confirmation_metrics = self._update_threat_confirmation(
            observed_mask=np.asarray(belief_metrics["observed_mask"], dtype=bool),
            best_threat_evidence=np.asarray(belief_metrics["best_threat_evidence"], dtype=np.float64),
        )
        self._tracking_bias_drones = self._choose_tracking_bias_drones()
        interceptor_metrics = self._advance_interceptor_and_threat()
        self._interceptor_dispatched_this_step = bool(interceptor_metrics["interceptor_dispatched"])

        active_patch_score = (
            float(np.mean(self._threat_persistence_score[self._active_threat_cells]))
            if self._active_threat_exists()
            else 0.0
        )
        return {
            **confirmation_metrics,
            **interceptor_metrics,
            "active_threat_confirmation_score": active_patch_score,
            "tracking_bias_count": int(np.sum(self._tracking_bias_drones)),
        }

    def _build_observation(self, connectivity: Dict[str, np.ndarray | int | float]) -> np.ndarray:
        local_uncertainty = np.stack([local.uncertainty for local in self.local_beliefs], axis=0)
        local_anomaly = np.stack([local.anomaly_score for local in self.local_beliefs], axis=0)
        local_age = np.stack(
            [
                np.clip(local.get_age(self._step_count) / max(self.mission_duration, 1), 0.0, 1.0)
                for local in self.local_beliefs
            ],
            axis=0,
        )
        battery_levels = self._battery_levels()
        dist_to_base = self._distance_to_base_norm()
        neighbor_counts = np.asarray(connectivity["neighbor_counts"], dtype=np.float64)
        neighbor_count_norm = np.clip(neighbor_counts / max(self.num_drones - 1, 1), 0.0, 1.0)
        drone_state = np.column_stack(
            [
                self._drone_states[:, 0],
                self._drone_states[:, 1],
                np.cos(self._drone_states[:, 9]),
                np.sin(self._drone_states[:, 9]),
                battery_levels,
                dist_to_base,
                neighbor_count_norm,
            ]
        )

        risk_counts = {
            "gt_0_1": int(np.sum(self._current_risk_scores() > self.global_belief.config.report_threshold)),
            "gt_0_4": int(np.sum(self._current_risk_scores() > self.global_belief.config.revisit_threshold)),
            "gt_0_7": int(np.sum(self._current_risk_scores() > self.global_belief.config.alert_threshold)),
        }
        global_summary = np.array(
            [
                self.global_belief.mean_uncertainty(),
                self.global_belief.max_uncertainty(),
                self._neglect_pressure(),
                float(connectivity["connectivity_score"]),
                risk_counts["gt_0_1"] / max(self.n_cells, 1),
                risk_counts["gt_0_4"] / max(self.n_cells, 1),
                risk_counts["gt_0_7"] / max(self.n_cells, 1),
                float(np.mean(battery_levels)),
            ],
            dtype=np.float64,
        )

        observation_dict = {
            "local_uncertainty": local_uncertainty.astype(np.float32),
            "local_anomaly": local_anomaly.astype(np.float32),
            "local_age": local_age.astype(np.float32),
            "drone_state": drone_state.astype(np.float32),
            "global_summary": global_summary.astype(np.float32),
        }
        self._last_structured_obs = observation_dict

        return np.concatenate(
            [
                observation_dict["local_uncertainty"].ravel(),
                observation_dict["local_anomaly"].ravel(),
                observation_dict["local_age"].ravel(),
                observation_dict["drone_state"].ravel(),
                observation_dict["global_summary"].ravel(),
            ]
        ).astype(np.float32)

    def _neglect_pressure(self) -> float:
        beta = float(self.reward_shaper.weights.neglect_beta)
        return float(
            np.mean(np.expm1(beta * self.global_belief.uncertainty))
            / max(np.expm1(beta), 1e-6)
        )

    def _heading_error_to_targets(self, selected_cells: np.ndarray) -> np.ndarray:
        """Return absolute heading error to each selected target cell."""
        target_xy = self.cell_centers_xy[np.asarray(selected_cells, dtype=np.int32)]
        desired = np.arctan2(
            target_xy[:, 1] - self._drone_states[:, 1],
            target_xy[:, 0] - self._drone_states[:, 0],
        )
        return np.abs(_wrap_angle(desired - self._drone_states[:, 9]))

    def _build_info(
        self,
        *,
        connectivity: Dict[str, np.ndarray | int | float],
        reward: float,
        reward_components: Dict[str, float],
        solve_times: np.ndarray,
        observed_counts: Optional[np.ndarray] = None,
        requested_target_cells: Optional[np.ndarray] = None,
        projection_override_count: int = 0,
        shared_cell_count: int = 0,
        global_sync_applied: bool = False,
        truth_metrics: Optional[Dict[str, object]] = None,
        threat_metrics: Optional[Dict[str, object]] = None,
    ) -> Dict:
        anomaly_counts = self.global_belief.anomaly_counts()
        risk_scores = self._current_risk_scores()
        belief_state = self.get_belief_state()
        hidden_world_state = self.get_hidden_world_state()
        truth_metrics = truth_metrics or {}
        threat_metrics = threat_metrics or {}
        risk_counts = {
            "gt_0_1": int(np.sum(risk_scores > self.global_belief.config.report_threshold)),
            "gt_0_4": int(np.sum(risk_scores > self.global_belief.config.revisit_threshold)),
            "gt_0_7": int(np.sum(risk_scores > self.global_belief.config.alert_threshold)),
        }
        selected_cells = self._current_target_cells.copy()
        distance_to_targets = self._distance_to_selected_targets(selected_cells)
        heading_errors = self._heading_error_to_targets(selected_cells)
        home_target_cells = np.array(
            [self._peek_home_patrol_target(i) for i in range(self.num_drones)],
            dtype=np.int32,
        )
        selected_in_home = np.array(
            [
                int(selected_cells[i] in set(self._home_cell_indices[i].tolist()))
                for i in range(self.num_drones)
            ],
            dtype=np.int32,
        )
        selected_in_assist = np.array(
            [
                int(selected_cells[i] in set(self._assist_cell_indices[i].tolist()))
                for i in range(self.num_drones)
            ],
            dtype=np.int32,
        )
        in_home_region = self._positions_in_home_region()
        home_risk_max = np.array(
            [
                float(np.max(risk_scores[home_cells])) if home_cells.size else 0.0
                for home_cells in self._home_cell_indices
            ],
            dtype=np.float64,
        )
        assist_risk_max = np.array(
            [
                float(np.max(risk_scores[assist_cells])) if assist_cells.size else 0.0
                for assist_cells in self._assist_cell_indices
            ],
            dtype=np.float64,
        )
        home_coverage_fraction = np.array(
            [
                float(np.mean(self.global_belief.last_observed_step[home_cells] >= 0))
                if home_cells.size
                else 0.0
                for home_cells in self._home_cell_indices
            ],
            dtype=np.float64,
        )
        home_low_risk_fraction = np.array(
            [
                float(np.mean(risk_scores[home_cells] <= self.PATROL_HOME_CLEAR_RISK))
                if home_cells.size
                else 0.0
                for home_cells in self._home_cell_indices
            ],
            dtype=np.float64,
        )
        home_neglected_fraction = np.array(
            [
                float(np.mean(risk_scores[home_cells] >= self.PATROL_ASSIST_TRIGGER_RISK))
                if home_cells.size
                else 0.0
                for home_cells in self._home_cell_indices
            ],
            dtype=np.float64,
        )
        low_soc_penalties = self.reward_shaper.low_soc_penalty(
            self._battery_levels(),
            self._distance_to_base_norm(),
            threshold=self.reward_shaper.weights.low_soc_threshold,
            cap=self.reward_shaper.weights.low_soc_cap,
        )
        active_threat_mask = self.get_active_threat_mask()
        confirmed_threat_mask = self.get_confirmed_threat_mask()
        suspected_threat_mask = np.zeros(self.n_cells, dtype=np.float64)
        if self._active_threat_exists() and not self._threat_confirmed:
            suspected_threat_mask[self._active_threat_cells] = 1.0
        active_threat_confirmation_score = (
            float(threat_metrics.get("active_threat_confirmation_score", 0.0))
            if threat_metrics
            else 0.0
        )
        interceptor_distance = (
            float(np.linalg.norm(self._interceptor_target - self._interceptor_position))
            if self._interceptor_active and np.all(np.isfinite(self._interceptor_target))
            else 0.0
        )
        active_threat_position = self._active_threat_centroid()
        if active_threat_position is None:
            active_threat_position = np.full(2, np.nan, dtype=np.float64)
        info = {
            "reward": float(reward),
            "reward_components": dict(reward_components),
            "observation_dict": self.get_structured_observation(),
            "mean_uncertainty": self.global_belief.mean_uncertainty(),
            "total_uncertainty": self.global_belief.total_uncertainty(),
            "neglect_pressure": self._neglect_pressure(),
            "anomaly_counts": anomaly_counts,
            "risk_counts": risk_counts,
            "mean_patrol_risk_belief": float(np.mean(belief_state.global_patrol_risk_belief)),
            "mean_threat_belief": float(np.mean(belief_state.global_threat_belief)),
            "mean_anomaly_score": float(np.mean(self.global_belief.anomaly_score)),
            "mean_risk_score": float(np.mean(risk_scores)),
            "mean_persistent_threat_state": float(np.mean(hidden_world_state.persistent_threat_state)),
            "mean_truth_risk_score": float(np.mean(self._truth_risk_score)),
            "mean_truth_monitoring_risk": float(np.mean(self._truth_monitoring_risk)),
            "truth_risk_counts": self._truth_risk_counts(),
            "risk_mismatch_mean": float(np.mean(np.abs(risk_scores - self._truth_risk_score))),
            "risk_mismatch_max": float(np.max(np.abs(risk_scores - self._truth_risk_score))),
            "low_risk_fraction": float(np.mean(risk_scores <= self.PATROL_HOME_CLEAR_RISK)),
            "high_risk_fraction": float(np.mean(risk_scores >= self.PATROL_ASSIST_TRIGGER_RISK)),
            "never_observed_fraction": float(np.mean(self.global_belief.last_observed_step < 0)),
            "active_threat": bool(self._active_threat_exists()),
            "threat_suspected": bool(threat_metrics.get("threat_suspected", False)),
            "threat_confirmed": bool(self._threat_confirmed),
            "central_command_notified": bool(self._central_command_notified),
            "active_threat_cells": self._active_threat_cells.copy(),
            "active_threat_mask": active_threat_mask.astype(np.float32),
            "suspected_threat_mask": suspected_threat_mask.astype(np.float32),
            "confirmed_threat_mask": confirmed_threat_mask.astype(np.float32),
            "threat_persistence_scores": self._threat_persistence_score.astype(np.float32).copy(),
            "active_threat_confirmation_score": float(active_threat_confirmation_score),
            "threat_confirmation_threshold": float(self.THREAT_CONFIRMATION_THRESHOLD),
            "mean_threat_persistence_score": float(np.mean(self._threat_persistence_score)),
            "tracking_bias_drones": self._tracking_bias_drones.astype(np.int32).copy(),
            "tracking_target_cells": self._tracking_target_cells.copy(),
            "threat_cycle_index": int(self._threat_cycles_spawned),
            "threat_cycles_completed": int(self._threat_cycles_completed),
            "threat_cycles_remaining": int(max(self.max_threat_cycles - self._threat_cycles_spawned, 0)),
            "threat_removed_this_step": bool(self._threat_removed_this_step),
            "threat_respawned_this_step": bool(self._threat_respawned_this_step),
            "threat_moved_this_step": bool(truth_metrics.get("threat_moved", False)),
            "threat_step_distance": float(truth_metrics.get("threat_step_distance", 0.0)),
            "threat_base_eta_steps": int(self._threat_base_eta_steps),
            "threat_base_timeout_steps": int(self._threat_base_timeout_steps),
            "threat_speed_case": self.persistent_threat_speed_case,
            "threat_speed": float(self.persistent_threat_speed),
            "physical_base_reached": bool(self._physical_base_reached_this_step),
            "mission_failed": bool(self._mission_failed),
            "mission_fail_reason": self._mission_fail_reason or None,
            "threat_state": {
                "active": bool(self._active_threat_exists()),
                "position_xy": active_threat_position.astype(np.float32).copy(),
                "velocity_xy": self._active_threat_velocity.astype(np.float32).copy(),
                "speed": float(self.persistent_threat_speed),
                "speed_case": self.persistent_threat_speed_case,
            },
            "threat_trace": self.get_threat_trace().astype(np.float32),
            "interceptor_state": {
                "active": bool(self._interceptor_active),
                "position_xy": self._interceptor_position.astype(np.float32).copy(),
                "target_xy": self._interceptor_target.astype(np.float32).copy(),
                "velocity_xy": self._interceptor_velocity.astype(np.float32).copy(),
                "distance_to_target": float(interceptor_distance),
                "dispatch_count": int(self._interceptor_dispatch_count),
                "dispatched_this_step": bool(self._interceptor_dispatched_this_step),
            },
            "interceptor_trace": self.get_interceptor_trace().astype(np.float32),
            "connectivity_state": {
                "component_count": int(connectivity["component_count"]),
                "connectivity_score": float(connectivity["connectivity_score"]),
            },
            "requested_target_cells": (
                np.asarray(requested_target_cells, dtype=np.int32).copy()
                if requested_target_cells is not None
                else selected_cells.copy()
            ),
            "policy_name": self.policy_name,
            "selected_target_cells": selected_cells,
            "home_target_cells": home_target_cells,
            "selected_in_home": selected_in_home,
            "selected_in_assist": selected_in_assist,
            "in_home_region": in_home_region,
            "patrol_detouring": (self._patrol_detour_targets >= 0).astype(np.int32),
            "patrol_detour_counts": self._patrol_detour_counts.copy(),
            "time_away_from_home_steps": self._time_away_from_home_steps.copy(),
            "patrol_route_indices": self._patrol_route_indices.copy(),
            "home_cell_counts": np.array([len(cells) for cells in self._home_cell_indices], dtype=np.int32),
            "home_risk_max": home_risk_max.astype(np.float32),
            "assist_risk_max": assist_risk_max.astype(np.float32),
            "home_coverage_fraction": home_coverage_fraction.astype(np.float32),
            "home_low_risk_fraction": home_low_risk_fraction.astype(np.float32),
            "home_neglected_fraction": home_neglected_fraction.astype(np.float32),
            "projection_override_count": int(projection_override_count),
            "shared_cell_count": int(shared_cell_count),
            "global_sync_applied": bool(global_sync_applied),
            "distance_to_targets_norm": distance_to_targets.astype(np.float32),
            "heading_error_rad": heading_errors.astype(np.float32),
            "battery_penalties": low_soc_penalties.astype(np.float32),
            "mean_solve_time_ms": float(np.mean(solve_times) * 1e3) if solve_times.size else 0.0,
        }
        if observed_counts is not None:
            info["observed_counts"] = np.asarray(observed_counts, dtype=np.int32)
        self._last_reward_components = dict(reward_components)
        self._last_connectivity_state = dict(info["connectivity_state"])
        return info

    def _check_collision(self) -> bool:
        threshold = 0.5 * self.collision_radius
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                distance = np.linalg.norm(self._drone_states[i, :3] - self._drone_states[j, :3])
                if distance < threshold:
                    return True
        return False


def make_belief_coverage_env(**kwargs) -> BeliefCoverageEnv:
    """Convenience factory for the belief-coverage Phase 1 environment."""
    return BeliefCoverageEnv(**kwargs)
