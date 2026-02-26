"""
Multi-objective reward shaper for ISR-RL-DMPC learning.

Combines 5 reward components:
1. Coverage (r_cov): Grid surveillance efficiency
2. Energy (r_eng): Battery consumption penalty
3. Safety (r_safe): Collision and geofence enforcement
4. Threat (r_threat): Engagement outcome assessment
5. Learning (r_learn): TD-error gradient signal

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
        self.prev_coverage = 0.0
        self.prev_energy_sum = 0.0

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
        self.prev_energy_sum = np.sum(drone_states[:, 13])  # Battery energy column
        
        return r_total

    def _coverage_reward(self, coverage_map: np.ndarray) -> float:
        """
        Compute coverage reward (area surveillance incentive).

        Rewards systematic grid coverage. Delta-based: rewards incremental
        improvement in coverage ratio.

        Args:
            coverage_map: (grid_cells,) binary coverage grid

        Returns:
            Coverage reward in [0, 0.01]
        """
        current_coverage = np.mean(coverage_map)
        delta_coverage = current_coverage - self.prev_coverage
        
        # Reward for new coverage
        # Max: 0.01 per step (full coverage in 100 steps)
        r_cov = max(0.0, delta_coverage) * 0.01
        
        return r_cov

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
        # Get battery energy from states (column 13)
        battery_energy = drone_states[:, 13]
        current_energy_sum = np.sum(battery_energy)
        
        # Energy consumed this step
        delta_energy = self.prev_energy_sum - current_energy_sum
        
        # Penalty: scaled to max -0.01 per step
        # Assume max consumption = 1000 J per step
        r_eng = -0.05 * (delta_energy / 1000.0)
        r_eng = np.clip(r_eng, -0.01, 0.0)
        
        return float(r_eng)

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
            Safety reward in [-1000, 0.1]
        """
        r_safe = 0.0
        
        # Check drone-drone collisions (min separation = 1m)
        min_separation = 1.0
        for i in range(self.num_drones):
            pos_i = drone_states[i, :3]
            active_i = drone_states[i, 15]  # Active flag
            
            if active_i < 0.5:  # Inactive = collision occurred
                r_safe -= 1000.0
                return float(r_safe)
            
            for j in range(i + 1, self.num_drones):
                pos_j = drone_states[j, :3]
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < min_separation and distance > 0:
                    r_safe -= 1000.0  # Collision
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
                
                r_safe -= 100.0  # Out of bounds
                return float(r_safe)
        
        # Small positive reward for safe operation
        r_safe += 0.1
        
        return float(np.clip(r_safe, -1000.0, 0.1))

    def _threat_reward(self, target_states: np.ndarray) -> float:
        """
        Compute threat engagement reward.

        Rewards correct threat assessment and engagement. Penalties for
        misclassification or inappropriate action.

        Args:
            target_states: (max_targets, 12) state matrix

        Returns:
            Threat reward in [-500, 500]
        """
        r_threat = 0.0
        
        # Target state columns:
        # 0-2: position, 3-5: velocity, 6: threat_level, 7: priority,
        # 8: type_id, 9: confidence, 10: is_detected, 11: reserved
        
        for i in range(len(target_states)):
            is_detected = target_states[i, 10]
            target_type = int(target_states[i, 8])
            threat_level = target_states[i, 6]
            
            if is_detected < 0.5:  # Not detected
                continue
            
            # Reward for detecting hostile targets
            if target_type == 2:  # Hostile (enum value)
                r_threat += 500.0  # Positive reward for detection
            elif target_type == 1:  # Friendly
                r_threat -= 500.0  # Negative reward (shouldn't engage)
            else:  # Unknown/Neutral
                r_threat += 100.0  # Small reward for investigation
        
        return float(np.clip(r_threat, -500.0, 500.0))

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
        """Reset component statistics."""
        if self.enable_component_stats:
            for key in self.stats:
                self.stats[key] = []

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
