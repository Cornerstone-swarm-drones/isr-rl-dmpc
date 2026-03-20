"""
Gymnasium-compatible environment for ISR-RL-DMPC autonomous swarm system.

Integrates 9 modules: mission planner, formation controller, sensor fusion,
classifier, threat assessor, task allocator, learning-based DMPC, attitude
controller, and learning module.

Classes:
    ISRGridEnv: Main Gymnasium environment with Gymnasium API compliance
    VectorEnv: Vectorized environment wrapper for parallel training
"""

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Minimal fallback so non-training workflows (e.g. evaluation scripts)
    # can run without installing gym/gymnasium.
    import numpy as _np
    from dataclasses import dataclass as _dataclass

    class _Space:
        def sample(self):
            raise NotImplementedError

    class _Box(_Space):
        def __init__(self, low, high, shape, dtype=_np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            low = self.low
            high = self.high

            # Handle infinite bounds by falling back to a normal-ish sample.
            low_finite = _np.isfinite(low).all() if hasattr(low, "shape") else _np.isfinite(low)
            high_finite = _np.isfinite(high).all() if hasattr(high, "shape") else _np.isfinite(high)

            rng = _np.random.default_rng()
            if low_finite and high_finite:
                return rng.uniform(low, high, size=self.shape).astype(self.dtype)
            return rng.standard_normal(size=self.shape).astype(self.dtype)

    class _Dict(_Space):
        def __init__(self, spaces_map):
            self.spaces = dict(spaces_map)

    class _GymEnv:
        metadata: dict = {}

        def reset(self, seed=None, options=None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

    class _GymModule:
        Env = _GymEnv

    gym = _GymModule()  # type: ignore

    class spaces:  # type: ignore
        Box = _Box
        Dict = _Dict
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

from isr_rl_dmpc.gym_env.simulator import (
    EnvironmentSimulator, DroneConfig, TargetConfig, EnvironmentConfig
)
from isr_rl_dmpc.gym_env.reward_shaper import RewardShaper
from isr_rl_dmpc.core import DroneState, TargetState, MissionState, StateManager


@dataclass
class MissionConfig:
    """Configuration for mission objectives."""
    num_drones: int = 10
    max_targets: int = 5
    grid_size: Tuple[int, int] = (20, 20)  # Grid cells
    mission_duration: int = 500  # steps (10 seconds at 50 Hz)
    coverage_target: float = 0.9  # 90% coverage
    threat_detection_target: float = 0.95  # 95% threat detection


class ISRGridEnv(gym.Env):
    """
    Gymnasium environment for ISR swarm control.
    
    Observation space:
    - Dict with keys: 'swarm', 'targets', 'environment', 'adjacency'
    - swarm: (N, 18) - drone states
    - targets: (M, 12) - target states
    - environment: (K+4,) - grid + mission state
    - adjacency: (N, N) - formation adjacency matrix
    
    Action space:
    - Box(0, 1, shape=(N, 4)) - motor PWM commands for N drones
    
    Reward:
    - Scalar value combining 5 components (coverage, energy, safety, threat, learning)
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50,
    }

    def __init__(
        self,
        num_drones: int = 10,
        max_targets: int = 5,
        grid_size: Tuple[int, int] = (20, 20),
        mission_duration: int = 500,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ISR environment.

        Args:
            num_drones: Number of drones in swarm
            max_targets: Maximum targets
            grid_size: Mission grid (cells)
            mission_duration: Episode length (steps)
            render_mode: 'human' or 'rgb_array'
            **kwargs: Additional config overrides
        """
        self.num_drones = num_drones
        self.max_targets = max_targets
        self.grid_size = grid_size
        self.grid_cells = grid_size[0] * grid_size[1]
        self.mission_duration = mission_duration
        self.render_mode = render_mode
        
        # Config
        self.mission_config = MissionConfig(
            num_drones=num_drones,
            max_targets=max_targets,
            grid_size=grid_size,
            mission_duration=mission_duration
        )
        
        # Initialize simulator (import in actual implementation)
        try:
            from isr_rl_dmpc.gym_env.simulator import (
                EnvironmentSimulator, DroneConfig, TargetConfig, EnvironmentConfig
            )
            self.simulator = EnvironmentSimulator(
                num_drones=num_drones,
                max_targets=max_targets,
                drone_config=DroneConfig(),
                target_config=TargetConfig(),
                env_config=EnvironmentConfig(grid_resolution=100.0)
            )
        except ImportError:
            self.simulator = None
            print("[ISRGridEnv] Simulator import failed - running in stub mode")
        
        # Initialize reward shaper (import in actual implementation)
        try:
            from isr_rl_dmpc.gym_env.reward_shaper import RewardShaper
            self.reward_shaper = RewardShaper(
                num_drones=num_drones,
                max_targets=max_targets,
                grid_cells=self.grid_cells
            )
        except ImportError:
            self.reward_shaper = None
            print("[ISRGridEnv] RewardShaper import failed - running in stub mode")
        
        # State tracking
        self.step_count = 0
        self.coverage_map = np.zeros(self.grid_cells)  # 1D array of grid cells
        self.detected_targets = set()
        self.mission_reward_history = []

        # Dashboard/evaluation-friendly derived metrics
        self._energy_usage_last: float = 0.0
        self._prev_avg_battery_energy: float = 0.0
        self._targets_tracked_last: int = 0

        # Optional module-signals wiring (does not affect physics/actions).
        self._last_module_signals: Dict[str, Any] = {}
        self._module_signals_enabled: bool = False
        self._task_allocator_mod = None
        self._threat_assessor_mod = None
        try:
            from isr_rl_dmpc.modules.task_allocator import TaskAllocator
            from isr_rl_dmpc.modules.threat_assessor import ThreatAssessor

            self._task_allocator_mod = TaskAllocator(num_drones=self.num_drones)
            self._threat_assessor_mod = ThreatAssessor()
            self._module_signals_enabled = True
        except (ImportError, ModuleNotFoundError):
            # Optional: keep environment usable without ML modules.
            self._module_signals_enabled = False
        
        # Define observation space
        # Dict space with 4 sub-spaces
        self.observation_space = spaces.Dict({
            'swarm': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_drones, 18),  # 18D per drone
                dtype=np.float32
            ),
            'targets': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_targets, 12),  # 12D per target
                dtype=np.float32
            ),
            'environment': spaces.Box(
                low=0,
                high=1,
                shape=(self.grid_cells + 4,),  # Grid + mission state
                dtype=np.float32
            ),
            'adjacency': spaces.Box(
                low=0,
                high=1,
                shape=(num_drones, num_drones),  # Formation adjacency
                dtype=np.float32
            )
        })
        
        # Define action space
        # Continuous motor commands [0, 1] for 4 motors per drone
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_drones, 4),
            dtype=np.float32
        )
        
        # RNG
        self.np_random = None
        self.seed()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional reset options

        Returns:
            observation, info dict
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset state tracking
        self.step_count = 0
        self.coverage_map = np.zeros(self.grid_cells)
        self.detected_targets = set()
        self.mission_reward_history = []

        # Reset derived metrics baseline
        self._energy_usage_last = 0.0
        self._prev_avg_battery_energy = 0.0
        self._targets_tracked_last = 0

        self._last_module_signals = {}
        
        # Reset reward shaper state
        if self.reward_shaper is not None:
            self.reward_shaper.prev_coverage = 0.0
            self.reward_shaper.prev_energy_sum = 0.0
        
        # Reset simulator
        if self.simulator is not None:
            self.simulator.reset()
            
            # Initialize drone positions (grid formation)
            positions = self._generate_initial_positions()
            for drone_id, pos in enumerate(positions):
                self.simulator.set_drone_initial_state(drone_id, pos)
            
            # Initialize targets randomly
            for target_id in range(min(self.max_targets, 3)):  # Start with 3 targets
                target_pos = self.np_random.uniform(
                    low=-500, high=500, size=3
                )
                target_pos[2] = self.np_random.uniform(50, 200)  # Altitude
                
                from isr_rl_dmpc.gym_env.simulator import TargetType
                target_type = self.np_random.choice([
                    TargetType.FRIENDLY,
                    TargetType.HOSTILE,
                    TargetType.NEUTRAL
                ])
                self.simulator.add_target(target_pos, target_type)

            # Baseline battery energy for energy usage (Joules)
            sim_stats = self.simulator.get_statistics()
            self._prev_avg_battery_energy = float(sim_stats["avg_battery_energy"])

            # At reset time targets are newly added but not yet detected
            self._targets_tracked_last = int(
                sum(1 for t in self.simulator.targets[: self.simulator.num_targets] if t.is_detected)
            )
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step of environment.
        Args:
            action: Motor commands (num_drones, 4)
        Returns:
            observation, reward, terminated, info
        """
        # Validate action
        action = np.clip(action, 0.0, 1.0)
        
        # Increment step count
        self.step_count += 1
        
        # Step simulator
        if self.simulator is not None:
            self.simulator.step(action)

            # Update derived metrics for dashboard/evaluation
            sim_stats = self.simulator.get_statistics()
            current_avg_battery = float(sim_stats["avg_battery_energy"])
            self._energy_usage_last = float(self._prev_avg_battery_energy - current_avg_battery)
            self._prev_avg_battery_energy = current_avg_battery
            self._targets_tracked_last = int(
                sum(
                    1
                    for t in self.simulator.targets[: self.simulator.num_targets]
                    if t.is_detected
                )
            )
        
        # Update coverage map (deterministic based on drone positions)
        self._update_coverage()

        # Optional module signals (allocation/threat/etc.) for dashboard/evaluation.
        self._update_module_signals()
        
        # Compute reward
        reward = self._compute_reward(action)
        self.mission_reward_history.append(reward)
        
        # Check termination
        terminated = self.step_count >= self.mission_duration
        
        # Check if all drones dead or all objectives met
        if self.simulator is not None:
            active_drones = sum(1 for d in self.simulator.drones if d.is_active)
            if active_drones == 0:
                terminated = True
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, False, info  # (obs, reward, terminated, truncated, info)

    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        Returns:
            RGB array if render_mode='rgb_array', else None
        """
        if self.render_mode == 'rgb_array':
            return self._render_rgb_array()
        elif self.render_mode == 'human':
            self._render_human()
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
        if self.simulator is not None:
            self.simulator.rng = self.np_random
        return [seed] if seed is not None else []

    def _generate_initial_positions(self) -> np.ndarray:
        """
        Generate initial drone positions in grid formation.

        Drones are spread across the mission area for effective coverage.

        Returns:
            Array of positions (num_drones, 3)
        """
        grid_side = int(np.ceil(np.sqrt(self.num_drones)))
        grid_x, grid_y = self.grid_size
        grid_res = 100.0  # m per cell
        # Space drones evenly across the grid area
        spacing_x = (grid_x * grid_res) / (grid_side + 1)
        spacing_y = (grid_y * grid_res) / (grid_side + 1)
        
        positions = []
        for i in range(self.num_drones):
            row = i // grid_side
            col = i % grid_side
            x = (col + 1) * spacing_x
            y = (row + 1) * spacing_y
            z = 50.0  # altitude
            positions.append(np.array([x, y, z]))
        
        return np.array(positions)

    def _update_coverage(self) -> None:
        """Update coverage map based on drone positions and sensor footprint."""
        if self.simulator is None:
            return
        
        grid_x, grid_y = self.grid_size
        grid_res = 100.0  # m per cell (from simulator config)
        sensor_radius_cells = 5  # sensor coverage radius in grid cells (500m at 100m/cell)
        
        for drone in self.simulator.drones:
            if drone.is_active:
                # Map drone position to grid cell
                cx = int(np.clip(drone.position[0] / grid_res, 0, grid_x - 1))
                cy = int(np.clip(drone.position[1] / grid_res, 0, grid_y - 1))
                # Mark cells within sensor radius
                for dx in range(-sensor_radius_cells, sensor_radius_cells + 1):
                    for dy in range(-sensor_radius_cells, sensor_radius_cells + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < grid_x and 0 <= ny < grid_y:
                            cell_idx = ny * grid_x + nx
                            self.coverage_map[cell_idx] = 1.0

    def _update_module_signals(self) -> None:
        """
        Compute lightweight module-derived signals for dashboards/evaluation.

        This is intentionally incremental: signals are produced where the
        corresponding modules are importable, but no module output is fed back
        into physics/control yet.
        """
        if not self._module_signals_enabled or self.simulator is None:
            return

        try:
            drone_states = self.simulator.get_drone_states()
            target_states = self.simulator.get_target_states()
        except Exception:
            return

        # Collect detected targets (skip padded zero rows).
        detected_idxs: List[int] = []
        for i in range(self.max_targets):
            if i >= target_states.shape[0]:
                break
            pos = target_states[i, :3]
            if not np.any(pos != 0.0):
                continue
            if target_states[i, 10] > 0.5:
                detected_idxs.append(i)

        if not detected_idxs:
            self._last_module_signals = {
                "targets_detected": 0,
                "task_allocation": {},
                "threat_assessment": {},
            }
            return

        # --- Task allocation signals (TaskAllocator) ---
        try:
            from isr_rl_dmpc.modules.task_allocator import (
                DroneCapability,
                ISRTask,
                TaskType,
            )
            from isr_rl_dmpc.gym_env.simulator import TargetType
        except Exception:
            return

        drone_positions = drone_states[:, :3]
        battery_norm = drone_states[:, 13]  # already normalized [0,1] by simulator

        drones: Dict[int, DroneCapability] = {}
        for i in range(self.num_drones):
            drones[i] = DroneCapability(
                drone_id=i,
                position=drone_positions[i].astype(np.float32),
                fuel_remaining=float(battery_norm[i] * 100.0),
                sensors=["radar", "optical", "rf", "acoustic"],
                current_load=0.0,
                max_speed=float(self.simulator.drone_config.max_linear_velocity),
                endurance=1e9,
                communication_range=500.0,
            )

        tasks: List[ISRTask] = []
        for tid in detected_idxs:
            tpos = target_states[tid, :3].astype(np.float32)
            type_id = int(round(float(target_states[tid, 8])))
            is_hostile = type_id == int(TargetType.HOSTILE.value)

            task_type = TaskType.TRACK if is_hostile else TaskType.CLASSIFY
            priority = 1.0 if is_hostile else 0.4

            tasks.append(
                ISRTask(
                    task_id=int(tid),
                    task_type=task_type,
                    target_position=tpos,
                    priority=float(priority),
                    required_sensors=["radar", "optical"],
                    estimated_duration=0.0,
                )
            )

        allocations = self._task_allocator_mod.allocate_tasks(tasks=tasks, drones=drones)
        allocation_metrics = self._task_allocator_mod.get_allocation_metrics()

        allocation_quality_norm = 0.0
        avg_cost = float(allocation_metrics.get("average_cost", 0.0)) if allocation_metrics else 0.0
        allocation_quality_norm = float(1.0 / (1.0 + avg_cost))

        # --- Threat assessment signals (ThreatAssessor) ---
        threat_scores: List[float] = []
        threat_per_target: List[Dict[str, Any]] = []

        own_position = np.mean(drone_positions, axis=0)
        current_time = float(self.step_count)

        for tid in detected_idxs:
            tpos = target_states[tid, :3].astype(float)
            vel = target_states[tid, 3:6].astype(float)
            rf_confidence = float(target_states[tid, 9])
            type_id = int(round(float(target_states[tid, 8])))

            if type_id == int(TargetType.HOSTILE.value):
                classification = "hostile"
                rf_strength = -35.0
                modulation = "psk"
            elif type_id == int(TargetType.FRIENDLY.value):
                classification = "friendly"
                rf_strength = -95.0
                modulation = "am"
            else:
                classification = "unknown"
                rf_strength = -70.0
                modulation = "am"

            assessment = self._threat_assessor_mod.assess_target(
                target_data={
                    "target_id": int(tid),
                    "position": tpos,
                    "velocity": vel,
                    "rf_strength": rf_strength,
                    "modulation": modulation,
                    "classification": classification,
                    "classification_confidence": rf_confidence,
                },
                own_position=own_position,
                current_time=current_time,
            )

            threat_scores.append(float(assessment.threat_score))
            threat_per_target.append(
                {
                    "target_id": int(tid),
                    "threat_level": str(assessment.threat_level.value),
                    "threat_score": float(assessment.threat_score),
                    "confidence": float(assessment.confidence),
                }
            )

        avg_threat_score = float(np.mean(threat_scores)) if threat_scores else 0.0

        self._last_module_signals = {
            "targets_detected": int(len(detected_idxs)),
            "task_allocation": {
                "average_cost": float(allocation_metrics.get("average_cost", 0.0)) if allocation_metrics else 0.0,
                "allocation_quality_norm": allocation_quality_norm,
                "assignment_coverage_norm": float(len(allocations) / max(len(tasks), 1)),
            },
            "threat_assessment": {
                "avg_threat_score": avg_threat_score,
                "max_threat_score": float(np.max(threat_scores)) if threat_scores else 0.0,
                "per_target": threat_per_target,
            },
        }

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute multi-objective reward.

        Args:
            action: Motor commands

        Returns:
            Scalar reward value (clipped to [-10, 10] for training stability)
        """
        if self.reward_shaper is None:
            # Stub: return small positive reward for progress
            return 0.01 * (1.0 - self.step_count / self.mission_duration)
        
        # Get states
        drone_states = self.simulator.get_drone_states()
        target_states = self.simulator.get_target_states()
        mission_state = np.concatenate([
            self.coverage_map,
            [self.step_count / self.mission_duration, 0.0, 0.0, 0.0]
        ])
        
        # Compute reward components
        reward = self.reward_shaper.compute_reward(
            drone_states=drone_states,
            target_states=target_states,
            mission_state=mission_state,
            motor_commands=action,
            coverage_map=self.coverage_map,
            step=self.step_count
        )
        
        # Clip reward to bounded range for training stability
        return float(np.clip(reward, -10.0, 10.0))

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.

        Returns:
            Dict with 'swarm', 'targets', 'environment', 'adjacency'
        """
        if self.simulator is None:
            # Return zero observation
            return {
                'swarm': np.zeros((self.num_drones, 18), dtype=np.float32),
                'targets': np.zeros((self.max_targets, 12), dtype=np.float32),
                'environment': np.zeros((self.grid_cells + 4,), dtype=np.float32),
                'adjacency': np.eye(self.num_drones, dtype=np.float32)
            }
        
        # Get states from simulator
        drone_states = self.simulator.get_drone_states()
        target_states = self.simulator.get_target_states()
        
        # Build environment state (coverage + mission state)
        mission_state = np.array([
            self.step_count / self.mission_duration,  # Progress
            np.mean(self.coverage_map),  # Coverage ratio
            0.0,  # Threat level (placeholder)
            0.0   # Energy efficiency (placeholder)
        ])
        environment = np.concatenate([self.coverage_map, mission_state])
        
        # Build adjacency matrix (formation topology)
        # Simple: all drones can see within 500m
        adjacency = np.zeros((self.num_drones, self.num_drones))
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i != j:
                    dist = np.linalg.norm(
                        drone_states[i, :3] - drone_states[j, :3]
                    )
                    if dist < 500.0 and dist > 0:
                        adjacency[i, j] = 1.0 / dist  # Inverse distance weight
        
        return {
            'swarm': drone_states.astype(np.float32),
            'targets': target_states.astype(np.float32),
            'environment': environment.astype(np.float32),
            'adjacency': adjacency.astype(np.float32)
        }

    def _get_info(self) -> Dict:
        """
        Get step info.

        Returns:
            Info dict
        """
        if self.simulator is None:
            return {
                'step': self.step_count,
                'coverage': 0.0,
                'active_drones': 0,
                'collisions': 0,
                'geofence_violations': 0
            }
        
        stats = self.simulator.get_statistics()
        
        return {
            'step': self.step_count,
            'coverage': float(np.mean(self.coverage_map)),
            'active_drones': stats['active_drones'],
            'total_drones': stats['total_drones'],
            'collisions': stats['collision_count'],
            'geofence_violations': stats['geofence_violations'],
            'avg_battery': stats['avg_battery_energy'],
            # Energy used since previous step (Joules). Positive = battery decreased.
            'energy_usage': self._energy_usage_last,
            'wind': stats['current_wind'],
            # Number of targets that are currently detected (per simulator truth).
            'targets_tracked': self._targets_tracked_last,
            # Incremental module-derived signals (does not affect physics).
            'module_signals': self._last_module_signals
        }

    def _render_human(self) -> None:
        """Render to human display (stub)."""
        if self.simulator is None:
            return
        
        stats = self.simulator.get_statistics()
        print(f"Step: {self.step_count:4d} | "
              f"Drones: {stats['active_drones']:2d}/{stats['total_drones']:2d} | "
              f"Coverage: {np.mean(self.coverage_map):.2%} | "
              f"Collisions: {stats['collision_count']:3d} | "
              f"Battery: {stats['avg_battery_energy']:.0f} J")

    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array (stub)."""
        # Placeholder: return 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw coverage map as heatmap (scaled)
        if len(self.coverage_map) > 0:
            coverage_ratio = np.mean(self.coverage_map)
            img[:, :, 0] = int(coverage_ratio * 255)  # Red channel
        
        return img

    def set_mission_config(self, **kwargs) -> None:
        """Update mission configuration."""
        for key, value in kwargs.items():
            if hasattr(self.mission_config, key):
                setattr(self.mission_config, key, value)


def make_env(env_id: str, num_envs: int = 1, **kwargs):
    """
    Factory function to create vectorized environments.

    Args:
        env_id: Environment ID (e.g., 'ISRGridEnv-v0')
        num_envs: Number of parallel environments
        **kwargs: Environment kwargs

    Returns:
        Single or vectorized environment
    """
    if env_id == 'ISRGridEnv-v0':
        if num_envs == 1:
            return ISRGridEnv(**kwargs)
        else:
            return VectorEnv([ISRGridEnv(**kwargs) for _ in range(num_envs)])
    else:
        raise ValueError(f"Unknown environment: {env_id}")


class VectorEnv:
    """
    Vectorized environment wrapper for parallel training.
    
    Maintains multiple independent environments and provides
    efficient batch operations for RL training.
    """

    def __init__(self, envs: List[ISRGridEnv]):
        """
        Initialize vectorized environment.

        Args:
            envs: List of environment instances
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.single_action_space = envs[0].action_space
        self.single_observation_space = envs[0].observation_space
        
        # For batch operations
        self.observations = None
        self.rewards = None
        self.dones = None

    def reset(self):
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        self.observations = self._stack_observations(observations)
        return self.observations

    def step(self, actions):
        """
        Step all environments with given actions.

        Args:
            actions: Batch of actions (num_envs, ...) matching action space

        Returns:
            observations, rewards, dones, infos
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)
        
        self.observations = self._stack_observations(observations)
        self.rewards = np.array(rewards)
        self.dones = np.array(dones)
        
        return self.observations, self.rewards, self.dones, infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

    def _stack_observations(self, obs_list):
        """Stack dict observations from multiple envs."""
        stacked = {}
        for key in obs_list[0].keys():
            stacked[key] = np.array([o[key] for o in obs_list])
        return stacked

    def render(self, mode='human'):
        """Render first environment."""
        if self.envs:
            return self.envs[0].render()
