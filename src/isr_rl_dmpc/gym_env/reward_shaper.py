"""
Multi-objective reward shaper for ISR-RL-DMPC learning.

Combines 5 reward components:
1. Coverage (r_cov): Grid surveillance efficiency
2. Energy (r_eng): Battery consumption penalty
3. Safety (r_safe): Collision and geofence enforcement
4. Threat (r_threat): Engagement outcome assessment
5. Learning (r_learn): TD-error gradient signal

Scenario presets
----------------
Use :func:`get_scenario_preset` to obtain a :class:`RewardWeights` tuned for
one of the three ISR missions (``"area_surveillance"``, ``"threat_response"``,
``"search_and_track"``).  Legacy presets (``"recon"``, ``"intel"``,
``"target_pursuit"``) are kept for backwards compatibility.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Tunable weights for reward components."""
    w_coverage: float = 1.0  # Coverage incentive
    w_energy: float = 0.5  # Energy efficiency
    w_safety: float = 10.0  # Safety critical
    w_threat: float = 2.0  # Threat response
    w_learning: float = 0.1  # Learning signal


@dataclass
class BeliefCoverageRewardWeights:
    """Weights for belief-based Phase 1 coverage rewards."""

    w_uncertainty_reduction: float = 4.0
    w_neglect: float = 2.5
    w_distance: float = 0.05
    w_energy: float = 0.75
    w_low_soc: float = 1.0
    w_connectivity: float = 1.5
    neglect_beta: float = 2.5
    low_soc_threshold: float = 0.2
    low_soc_cap: float = 1.0


# ── Scenario-specific presets ──────────────────────────────────────────────
#
# Weight rationale per scenario:
#
#   area_surveillance — Wide-area persistent coverage; no targets to engage.
#     * Coverage is the PRIMARY objective → highest w_coverage.
#     * Energy conservation matters for long (~17 min) endurance missions.
#     * Threat weight = 0 because num_targets = 0.
#
#   threat_response — Detect & intercept hostile targets quickly.
#     * Threat engagement dominates → highest w_threat.
#     * Coverage is background (maintain situational awareness of perimeter).
#     * Energy penalty relaxed so aggressive intercept manoeuvres are allowed.
#
#   search_and_track — Locate then continuously track mobile targets.
#     * Balanced coverage (search phase) + threat (track phase).
#     * Moderate energy conservation (long endurance needed).
#     * Safety always critical regardless of mission.
#
SCENARIO_REWARD_PRESETS: Dict[str, RewardWeights] = {
    "area_surveillance": RewardWeights(
        w_coverage=50.0,   # Primary: maximise coverage fraction
        w_energy=1.5,      # Penalise inefficient flight on long missions
        w_safety=10.0,     # Always critical
        w_threat=0.0,      # No targets in this scenario
        w_learning=0.1,
    ),
    "threat_response": RewardWeights(
        w_coverage=5.0,    # Background situational awareness
        w_energy=0.5,      # Aggressive manoeuvres are acceptable
        w_safety=10.0,
        w_threat=30.0,     # Primary: rapid detection & classification
        w_learning=0.1,
    ),
    "search_and_track": RewardWeights(
        w_coverage=20.0,   # Search phase: important to cover area
        w_energy=1.5,      # Long endurance needed
        w_safety=10.0,
        w_threat=15.0,     # Track phase: maintain contact with targets
        w_learning=0.1,
    ),
}

# ── Legacy task presets (kept for backwards compatibility) ─────────────────
# Coverage reward is delta-based (max ~0.01/step), energy is ~[-0.01,0],
# safety gives +0.1 per safe step, threat up to ±500 per detection.
# Weights are scaled so that no single component dominates.
TASK_REWARD_PRESETS = {
    "recon": RewardWeights(
        w_coverage=50.0,   # Primary: maximise area coverage
        w_energy=1.0,      # Moderate energy conservation
        w_safety=10.0,     # Safety always critical
        w_threat=0.5,      # Low priority – recon only
        w_learning=0.1,
    ),
    "intel": RewardWeights(
        w_coverage=20.0,   # Still important for search
        w_energy=2.0,      # Longer missions need energy mgmt
        w_safety=10.0,
        w_threat=5.0,      # Moderate – classify targets
        w_learning=0.1,
    ),
    "target_pursuit": RewardWeights(
        w_coverage=5.0,    # Background coverage
        w_energy=0.5,      # Aggressive manoeuvres allowed
        w_safety=10.0,
        w_threat=20.0,     # Primary: engage/track threats
        w_learning=0.1,
    ),
}


def get_scenario_preset(scenario: str) -> RewardWeights:
    """Return the :class:`RewardWeights` preset for *scenario*.

    Checks :data:`SCENARIO_REWARD_PRESETS` first, then
    :data:`TASK_REWARD_PRESETS`, and finally returns the default
    :class:`RewardWeights` if the name is not found.

    Args:
        scenario: Scenario name, e.g. ``"area_surveillance"``.

    Returns:
        Matching :class:`RewardWeights` instance (never ``None``).
    """
    if scenario in SCENARIO_REWARD_PRESETS:
        return SCENARIO_REWARD_PRESETS[scenario]
    if scenario in TASK_REWARD_PRESETS:
        return TASK_REWARD_PRESETS[scenario]
    return RewardWeights()


class BeliefCoverageRewardShaper:
    """Reward helper for Phase 1 belief-based coverage."""

    def __init__(
        self,
        num_drones: int,
        weights: Optional[BeliefCoverageRewardWeights] = None,
    ) -> None:
        self.num_drones = int(num_drones)
        self.weights = weights or BeliefCoverageRewardWeights()
        self.prev_total_uncertainty: float = 0.0
        self.prev_battery_levels = np.ones(self.num_drones, dtype=np.float64)

    def reset(
        self,
        total_uncertainty: float,
        battery_levels: Optional[np.ndarray] = None,
    ) -> None:
        self.prev_total_uncertainty = float(total_uncertainty)
        if battery_levels is not None:
            self.prev_battery_levels = np.asarray(battery_levels, dtype=np.float64).copy()
        else:
            self.prev_battery_levels = np.ones(self.num_drones, dtype=np.float64)

    def compute(
        self,
        *,
        total_uncertainty: float,
        uncertainty_values: np.ndarray,
        distance_norm: np.ndarray,
        battery_levels: np.ndarray,
        dist_to_base_norm: np.ndarray,
        connected_components: int,
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute the aggregate belief-coverage reward and its components.
        """
        uncertainty_values = np.asarray(uncertainty_values, dtype=np.float64)
        distance_norm = np.asarray(distance_norm, dtype=np.float64)
        battery_levels = np.asarray(battery_levels, dtype=np.float64)
        dist_to_base_norm = np.asarray(dist_to_base_norm, dtype=np.float64)

        uncertainty_reduction = self.prev_total_uncertainty - float(total_uncertainty)
        beta = float(self.weights.neglect_beta)
        neglect_pressure = float(
            np.mean(np.expm1(beta * np.clip(uncertainty_values, 0.0, 1.0)))
            / max(np.expm1(beta), 1e-6)
        )

        distance_penalty = float(np.mean(np.clip(distance_norm, 0.0, 1.0)))
        battery_drop = np.clip(self.prev_battery_levels - battery_levels, 0.0, 1.0)
        energy_penalty = float(np.mean(battery_drop))
        low_soc_penalties = self.low_soc_penalty(
            battery_levels,
            dist_to_base_norm,
            threshold=self.weights.low_soc_threshold,
            cap=self.weights.low_soc_cap,
        )
        connectivity_penalty = float(
            max(connected_components - 1, 0) / max(self.num_drones - 1, 1)
        )

        reward = (
            self.weights.w_uncertainty_reduction * uncertainty_reduction
            - self.weights.w_neglect * neglect_pressure
            - self.weights.w_distance * distance_penalty
            - self.weights.w_energy * energy_penalty
            - self.weights.w_low_soc * float(np.mean(low_soc_penalties))
            - self.weights.w_connectivity * connectivity_penalty
        )

        self.prev_total_uncertainty = float(total_uncertainty)
        self.prev_battery_levels = battery_levels.copy()

        return float(reward), {
            "uncertainty_reduction": float(uncertainty_reduction),
            "neglect_pressure": float(neglect_pressure),
            "distance_penalty": float(distance_penalty),
            "energy_penalty": float(energy_penalty),
            "low_soc_penalty": float(np.mean(low_soc_penalties)),
            "connectivity_penalty": float(connectivity_penalty),
        }

    @staticmethod
    def low_soc_penalty(
        battery_levels: np.ndarray,
        dist_to_base_norm: np.ndarray,
        *,
        threshold: float,
        cap: float,
    ) -> np.ndarray:
        """Stable capped low-SOC penalty with distance-to-base scaling."""
        soc = np.clip(np.asarray(battery_levels, dtype=np.float64), 0.0, 1.0)
        base_dist = np.clip(np.asarray(dist_to_base_norm, dtype=np.float64), 0.0, 1.0)
        deficit = np.maximum(0.0, threshold - soc) / max(threshold, 1e-6)
        penalty = cap * (deficit ** 2) * (0.5 + 0.5 * base_dist)
        return np.clip(penalty, 0.0, cap)


class RewardShaper:
    """
    Multi-objective reward function with component tracking.
    
    Manages 5-component reward aggregation with configurable weights,
    component statistics, and learning signal integration.
    """

    def __init__(
        self,
        num_drones: int,
        max_targets: int,
        grid_cells: int,
        weights: Optional[RewardWeights] = None,
        enable_component_stats: bool = True
    ):
        """
        Initialize reward shaper.

        Args:
            num_drones: Number of drones
            max_targets: Maximum targets
            grid_cells: Total grid cells in mission area
            weights: Reward weight configuration
            enable_component_stats: Track individual component statistics
        """
        self.num_drones = num_drones
        self.max_targets = max_targets
        self.grid_cells = grid_cells
        
        self.weights = weights or RewardWeights()
        self.enable_component_stats = enable_component_stats
        
        # Statistics tracking
        if enable_component_stats:
            self.stats = {
                'r_cov': [],
                'r_eng': [],
                'r_safe': [],
                'r_threat': [],
                'r_learn': [],
                'r_total': []
            }
        
        # Previous state for computing deltas
        # prev_coverage: fraction of grid covered (0–1)
        # prev_energy_sum: sum of normalised battery levels across all drones (0–num_drones)
        self.prev_coverage = 0.0
        self.prev_energy_sum = float(self.num_drones)  # start fully charged

    def compute_reward(
        self,
        drone_states: np.ndarray,
        target_states: np.ndarray,
        mission_state: np.ndarray,
        motor_commands: np.ndarray,
        coverage_map: np.ndarray,
        step: int,
        td_error: Optional[float] = None
    ) -> float:
        """
        Compute aggregate reward from all components.

        Args:
            drone_states: (num_drones, 18) drone state matrix
            target_states: (max_targets, 12) target state matrix
            mission_state: (grid_cells + 4,) mission state
            motor_commands: (num_drones, 4) motor PWM commands
            coverage_map: (grid_cells,) coverage grid
            step: Current step number
            td_error: Temporal difference error (optional, for learning signal)

        Returns:
            Aggregate reward scalar
        """
        # Compute individual components
        r_cov = self._coverage_reward(coverage_map)
        r_eng = self._energy_reward(drone_states, motor_commands)
        r_safe = self._safety_reward(drone_states, target_states)
        r_threat = self._threat_reward(target_states)
        r_learn = self._learning_reward(td_error, step)
        
        # Weighted aggregation
        r_total = (
            self.weights.w_coverage * r_cov +
            self.weights.w_energy * r_eng +
            self.weights.w_safety * r_safe +
            self.weights.w_threat * r_threat +
            self.weights.w_learning * r_learn
        )
        
        # Track statistics
        if self.enable_component_stats:
            self.stats['r_cov'].append(r_cov)
            self.stats['r_eng'].append(r_eng)
            self.stats['r_safe'].append(r_safe)
            self.stats['r_threat'].append(r_threat)
            self.stats['r_learn'].append(r_learn)
            self.stats['r_total'].append(r_total)
        
        # Update previous state
        self.prev_coverage = np.mean(coverage_map)
        self.prev_energy_sum = float(np.sum(drone_states[:, 13]))  # normalised battery sum
        
        return r_total

    def _coverage_reward(self, coverage_map: np.ndarray) -> float:
        """
        Compute coverage reward (area surveillance incentive).

        Rewards systematic grid coverage. Combines incremental gain with
        absolute coverage level to keep the signal meaningful throughout
        the episode.

        Args:
            coverage_map: (grid_cells,) binary coverage grid

        Returns:
            Coverage reward in [-0.1, 1.0]
        """
        current_coverage = np.mean(coverage_map)
        delta_coverage = current_coverage - self.prev_coverage
        
        # Incremental reward for new coverage (scaled to be meaningful)
        r_inc = max(0.0, delta_coverage) * 10.0
        
        # Small ongoing reward proportional to total coverage achieved
        r_abs = current_coverage * 0.1
        
        r_cov = r_inc + r_abs
        return float(np.clip(r_cov, -0.1, 1.0))

    def _energy_reward(
        self,
        drone_states: np.ndarray,
        motor_commands: np.ndarray
    ) -> float:
        """
        Compute energy efficiency penalty.

        Penalizes power consumption. Goal: minimize total energy usage while
        maintaining mission objectives.

        Args:
            drone_states: (num_drones, 18) state matrix
            motor_commands: (num_drones, 4) motor commands

        Returns:
            Energy penalty in [-0.01, 0]
        """
        # Column 13 is normalised battery level (0 = empty, 1 = full).
        # Typical hover consumption ≈ 8e-6 per drone per 0.02 s step
        # (64 W × 0.02 s / 266 400 J battery capacity, rounded up slightly).
        # Scale so that r_eng ≈ -0.01 when all drones consume at hover rate.
        _HOVER_NORM_RATE = 8e-6  # normalised battery consumed at hover per step
        battery_level = drone_states[:, 13]
        current_energy_sum = float(np.sum(battery_level))

        # Energy consumed this step (positive = consumed)
        delta_energy = self.prev_energy_sum - current_energy_sum

        # Normalise: -1.0 when consuming at rated hover power, 0 at rest
        denominator = max(self.num_drones * _HOVER_NORM_RATE, 1e-9)
        r_eng = -delta_energy / denominator * 0.01
        r_eng = float(np.clip(r_eng, -0.01, 0.0))

        return r_eng

    def _safety_reward(
        self,
        drone_states: np.ndarray,
        target_states: np.ndarray
    ) -> float:
        """
        Compute safety reward (collision/geofence enforcement).

        Hard constraints: severe penalties for violations, small positive
        reward for safe operation.

        Args:
            drone_states: (num_drones, 18) state matrix
            target_states: (max_targets, 12) state matrix

        Returns:
            Safety reward in [-1.0, 0.1]
        """
        r_safe = 0.0
        
        # Check drone-drone collisions (min separation = 1m)
        min_separation = 1.0
        for i in range(self.num_drones):
            pos_i = drone_states[i, :3]
            active_i = drone_states[i, 15]  # Active flag
            
            if active_i < 0.5:  # Inactive = collision occurred
                r_safe -= 1.0
                return float(r_safe)
            
            for j in range(i + 1, self.num_drones):
                pos_j = drone_states[j, :3]
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < min_separation and distance > 0:
                    r_safe -= 1.0  # Collision
                    return float(r_safe)
        
        # Check geofence violations (bounds = ±2000m x,y, 0-500m z)
        geofence = {
            'x_bound': 2000.0,
            'y_bound': 2000.0,
            'z_min': 0.0,
            'z_max': 500.0
        }
        
        for i in range(self.num_drones):
            x, y, z = drone_states[i, :3]
            
            if (abs(x) > geofence['x_bound'] or
                abs(y) > geofence['y_bound'] or
                z < geofence['z_min'] or
                z > geofence['z_max']):
                
                r_safe -= 0.5  # Out of bounds
                return float(r_safe)
        
        # Small positive reward for safe operation
        r_safe += 0.1
        
        return float(np.clip(r_safe, -1.0, 0.1))

    def _threat_reward(self, target_states: np.ndarray) -> float:
        """
        Compute threat engagement reward.

        Rewards correct threat assessment and engagement. Scaled so that
        the per-step contribution stays within a reasonable range.

        Args:
            target_states: (max_targets, 12) state matrix

        Returns:
            Threat reward in [-1.0, 1.0]
        """
        r_threat = 0.0
        detected_count = 0
        
        # Target state columns:
        # 0-2: position, 3-5: velocity, 6: threat_level, 7: priority,
        # 8: type_id, 9: confidence, 10: is_detected, 11: reserved
        
        for i in range(len(target_states)):
            is_detected = target_states[i, 10]
            target_type = int(target_states[i, 8])
            
            if is_detected < 0.5:  # Not detected
                continue
            
            detected_count += 1
            # Reward for detecting hostile targets
            if target_type == 2:  # Hostile (enum value)
                r_threat += 1.0  # Positive reward for detection
            elif target_type == 1:  # Friendly
                r_threat -= 1.0  # Negative reward (shouldn't engage)
            else:  # Unknown/Neutral
                r_threat += 0.2  # Small reward for investigation
        
        # Normalise by number of detections to keep magnitude bounded
        if detected_count > 0:
            r_threat /= detected_count
        
        return float(np.clip(r_threat, -1.0, 1.0))

    def _learning_reward(
        self,
        td_error: Optional[float],
        step: int
    ) -> float:
        """
        Compute learning signal reward.

        Provides gradient signal for learning-based DMPC. Penalizes large
        TD-errors indicating value function disagreement.

        Args:
            td_error: Temporal difference error from value network
            step: Current step number

        Returns:
            Learning reward in [-0.1, 0]
        """
        if td_error is None:
            # Default: small penalty for time
            return -0.001
        
        # Penalty proportional to TD-error magnitude
        # Max penalty: -0.1 (when |TD-error| > 0.2)
        r_learn = -0.5 * min(abs(td_error), 0.2)
        r_learn = np.clip(r_learn, -0.1, 0.0)
        
        return float(r_learn)

    def get_component_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all reward components.

        Returns:
            Dict with 'mean', 'std', 'min', 'max' for each component
        """
        if not self.enable_component_stats:
            return {}
        
        stats = {}
        for key, values in self.stats.items():
            if len(values) > 0:
                values_array = np.array(values)
                stats[key] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': len(values)
                }
        
        return stats

    def reset_stats(self) -> None:
        """Reset component statistics and previous-state accumulators."""
        if self.enable_component_stats:
            for key in self.stats:
                self.stats[key] = []
        self.prev_coverage = 0.0
        self.prev_energy_sum = float(self.num_drones)

    def update_weights(self, **kwargs) -> None:
        """
        Update reward weights dynamically.

        Args:
            **kwargs: Weight updates (e.g., w_coverage=2.0)
        """
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)

    def print_summary(self) -> None:
        """Print reward component summary."""
        print("\n=== Reward Shaper Summary ===")
        print(f"Configuration:")
        print(f"  Num Drones: {self.num_drones}")
        print(f"  Max Targets: {self.max_targets}")
        print(f"  Grid Cells: {self.grid_cells}")
        print(f"\nWeights:")
        print(f"  w_coverage: {self.weights.w_coverage:.2f}")
        print(f"  w_energy: {self.weights.w_energy:.2f}")
        print(f"  w_safety: {self.weights.w_safety:.2f}")
        print(f"  w_threat: {self.weights.w_threat:.2f}")
        print(f"  w_learning: {self.weights.w_learning:.2f}")
        
        if self.enable_component_stats:
            stats = self.get_component_stats()
            print(f"\nComponent Statistics (after {len(self.stats['r_total'])} steps):")
            for key, stat in stats.items():
                print(f"  {key:8s}: mean={stat['mean']:7.4f}, "
                      f"std={stat['std']:7.4f}, "
                      f"range=[{stat['min']:7.4f}, {stat['max']:7.4f}]")
