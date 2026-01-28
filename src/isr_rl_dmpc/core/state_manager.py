"""
State Manager for ISR system.

Provides centralized, thread-safe management of drone states, target states,
and mission state for the entire swarm. Integrates with logging utilities
for performance tracking and metrics.
"""

from threading import RLock
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime

from .data_structures import DroneState, TargetState, MissionState


class StateManager:
    """
    Centralized state aggregation for the entire swarm.

    Manages N drones, M targets, and 1 mission state with thread-safe access.
    Provides global state vectorization for learning algorithms and enables
    performance profiling through integration with logging utilities.

    Thread Safety:
        - Uses RLock for recursive locking during concurrent sensor updates
        - Safe for multi-threaded sensor fusion and control loops
    """

    def __init__(self, n_drones: int = 10):
        """
        Initialize state manager.

        Args:
            n_drones (int): Number of drones in the swarm
        """
        self.n_drones = n_drones
        self.drone_states: List[Optional[DroneState]] = [None] * n_drones
        self.target_states: Dict[str, TargetState] = {}
        self.mission_state: Optional[MissionState] = None

        # Thread-safe state access
        self._lock = RLock()
        self._update_count = 0
        self._last_state_hash = 0

    def update_drone_state(self, drone_id: int, state: DroneState) -> None:
        """
        Thread-safe update of individual drone state.

        Args:
            drone_id (int): Index of drone [0, n_drones)
            state (DroneState): Updated drone state

        Raises:
            ValueError: If drone_id out of valid range
        """
        if not (0 <= drone_id < self.n_drones):
            raise ValueError(f"drone_id={drone_id} out of range [0, {self.n_drones})")

        with self._lock:
            self.drone_states[drone_id] = state.copy()
            self._update_count += 1

    def update_target_state(self, target_id: str, state: TargetState) -> None:
        """
        Thread-safe update of target state.

        Args:
            target_id (str): Unique identifier for target
            state (TargetState): Updated target state
        """
        with self._lock:
            self.target_states[target_id] = state.copy()
            self._update_count += 1

    def update_mission_state(self, state: MissionState) -> None:
        """
        Thread-safe update of mission state.

        Args:
            state (MissionState): Updated mission state
        """
        with self._lock:
            self.mission_state = state.copy()
            self._update_count += 1

    def get_drone_state(self, drone_id: int) -> Optional[DroneState]:
        """
        Get state of a specified drone.

        Args:
            drone_id (int): Index of drone

        Returns:
            Copy of DroneState or None if not initialized
        """
        with self._lock:
            if 0 <= drone_id < self.n_drones and self.drone_states[drone_id] is not None:
                return self.drone_states[drone_id].copy()
            return None

    def get_target_state(self, target_id: str) -> Optional[TargetState]:
        """
        Get state of a specified target.

        Args:
            target_id (str): Target identifier

        Returns:
            Copy of TargetState or None if not found
        """
        with self._lock:
            if target_id in self.target_states:
                return self.target_states[target_id].copy()
            return None

    def get_mission_state(self) -> Optional[MissionState]:
        """
        Get current mission state.

        Returns:
            Copy of MissionState or None if not initialized
        """
        with self._lock:
            return self.mission_state.copy() if self.mission_state else None

    def get_global_state_vector(self) -> np.ndarray:
        """
        Create global state vector for RL agent.

        Concatenates drone states (18D each), target states (12D each), and
        mission metadata into a single flat vector for learning algorithms.

        Returns:
            Concatenated state vector of dimension 18N + 12M + K + 4
            where N=drones, M=targets, K=coverage cells

        Example:
            state_vec = state_manager.get_global_state_vector()
            # state_vec.shape = (18*10 + 12*5 + 100 + 4,) for 10 drones, 5 targets, 100 cells
        """
        with self._lock:
            state_components = []

            # Drone States (18 dims each)
            for drone_state in self.drone_states:
                if drone_state is not None:
                    state_components.append(drone_state.to_vector())
                else:
                    state_components.append(np.zeros(18))

            # Target States (12 dims each)
            for target_state in self.target_states.values():
                state_components.append(target_state.to_vector())

            # Mission State (coverage cells + 4 metadata)
            if self.mission_state is not None:
                mission_vec = np.concatenate([
                    self.mission_state.coverage_matrix.astype(float),
                    np.array([
                        self.mission_state.elapsed_time,
                        self.mission_state.mission_duration,
                        self.mission_state.coverage_efficiency,
                        self.mission_state.get_coverage_percentage() / 100.0,
                    ])
                ])
                state_components.append(mission_vec)

            return np.concatenate(state_components) if state_components else np.array([])

    def get_swarm_positions(self) -> np.ndarray:
        """
        Get all drone positions as Nx3 matrix.

        Returns:
            Nx3 array where each row is [x, y, z] position in meters
        """
        with self._lock:
            positions = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    positions.append(drone_state.position)
                else:
                    positions.append(np.zeros(3))
            return np.array(positions)

    def get_swarm_velocities(self) -> np.ndarray:
        """
        Get all drone velocities as Nx3 matrix.

        Returns:
            Nx3 array where each row is [vx, vy, vz] velocity in m/s
        """
        with self._lock:
            velocities = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    velocities.append(drone_state.velocity)
                else:
                    velocities.append(np.zeros(3))
            return np.array(velocities)

    def get_swarm_quaternions(self) -> np.ndarray:
        """
        Get all drone quaternions as Nx4 matrix.

        Returns:
            Nx4 array where each row is [qw, qx, qy, qz] quaternion
        """
        with self._lock:
            quaternions = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    quaternions.append(drone_state.quaternion)
                else:
                    quaternions.append(np.array([1., 0., 0., 0.]))
            return np.array(quaternions)

    def get_swarm_battery_levels(self) -> np.ndarray:
        """
        Get all drone battery levels as N-vector.

        Returns:
            N-vector of battery energy in Wh
        """
        with self._lock:
            batteries = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    batteries.append(drone_state.battery_energy)
                else:
                    batteries.append(0.0)
            return np.array(batteries)

    def get_swarm_health(self) -> np.ndarray:
        """
        Get all drone health levels as N-vector.

        Returns:
            N-vector of health [0, 1] for each drone
        """
        with self._lock:
            healths = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    healths.append(drone_state.health)
                else:
                    healths.append(0.0)
            return np.array(healths)

    def get_target_positions(self) -> Dict[str, np.ndarray]:
        """
        Get all target positions.

        Returns:
            Dictionary mapping target_id -> position (3D array)
        """
        with self._lock:
            return {tid: ts.position.copy() for tid, ts in self.target_states.items()}

    def get_drone_target_distances(self) -> np.ndarray:
        """
        Compute distance matrix between drones and targets.

        Returns:
            NxM matrix where (i, j) is distance from drone i to target j (meters)
        """
        with self._lock:
            positions = self.get_swarm_positions()
            n_drones = len(positions)
            m_targets = len(self.target_states)

            if m_targets == 0:
                return np.zeros((n_drones, 0))

            distances = np.zeros((n_drones, m_targets))
            for j, target_state in enumerate(self.target_states.values()):
                for i in range(n_drones):
                    if self.drone_states[i] is not None:
                        distances[i, j] = np.linalg.norm(
                            self.drone_states[i].position - target_state.position
                        )
            return distances

    def get_inter_drone_distances(self) -> np.ndarray:
        """
        Compute pairwise distances between drones.

        Returns:
            NxN symmetric distance matrix with zero diagonal (meters)
        """
        with self._lock:
            positions = self.get_swarm_positions()
            n = len(positions)
            distances = np.zeros((n, n))

            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances[i, j], distances[j, i] = dist, dist

            return distances

    def get_inter_drone_adjacency(self, communication_radius: float) -> np.ndarray:
        """
        Get adjacency matrix for drones within communication range.

        Args:
            communication_radius (float): Maximum communication distance (meters)

        Returns:
            NxN binary adjacency matrix (1 if within range, 0 otherwise)
        """
        distances = self.get_inter_drone_distances()
        return (distances <= communication_radius).astype(int)

    def get_swarm_state(self) -> List[DroneState]:
        """
        Return all active drone states.

        Returns:
            List of DroneState copies (excludes None entries)
        """
        with self._lock:
            return [state.copy() for state in self.drone_states if state is not None]

    def get_all_targets(self) -> Dict[str, TargetState]:
        """
        Return all tracked targets.

        Returns:
            Dictionary mapping target_id -> TargetState copy
        """
        with self._lock:
            return {tid: ts.copy() for tid, ts in self.target_states.items()}

    def save_to_file(self, filepath: str) -> None:
        """
        Save complete state to JSON file.

        Args:
            filepath (str): Path to output JSON file
        """
        with self._lock:
            data = {
                "timestamp": datetime.now().isoformat(),
                "n_drones": self.n_drones,
                "drone_states": [s.to_dict() if s is not None else None for s in self.drone_states],
                "target_states": {tid: ts.to_dict() for tid, ts in self.target_states.items()},
                "mission_state": self.mission_state.to_dict() if self.mission_state else None,
                "update_count": self._update_count,
            }

            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """
        Load state from JSON file.

        Args:
            filepath (str): Path to input JSON file
        """
        with self._lock:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.n_drones = data["n_drones"]
            self.drone_states = [
                DroneState.from_dict(s) if s is not None else None
                for s in data["drone_states"]
            ]
            self.target_states = {
                tid: TargetState.from_dict(ts)
                for tid, ts in data["target_states"].items()
            }
            self.mission_state = (
                MissionState.from_dict(data["mission_state"])
                if data["mission_state"] is not None else None
            )
            self._update_count = data.get("update_count", 0)

    def clear(self) -> None:
        """Clear all states and reset counters."""
        with self._lock:
            self.drone_states = [None] * self.n_drones
            self.target_states = {}
            self.mission_state = None
            self._update_count = 0

    def get_statistics(self) -> Dict:
        """
        Get summary statistics about current state.

        Returns:
            Dictionary with counts and metrics
        """
        with self._lock:
            positions = self.get_swarm_positions()
            batteries = self.get_swarm_battery_levels()
            healths = self.get_swarm_health()

            active_drones = sum(1 for s in self.drone_states if s is not None)
            valid_positions = positions[~np.any(np.isnan(positions), axis=1)]

            stats = {
                "active_drones": active_drones,
                "total_drones": self.n_drones,
                "num_targets": len(self.target_states),
                "mission_active": self.mission_state is not None,
                "state_updates": self._update_count,
            }

            if len(valid_positions) > 0:
                stats["avg_battery"] = float(np.mean(batteries[batteries > 0]))
                stats["min_battery"] = float(np.min(batteries))
                stats["avg_health"] = float(np.mean(healths))
                stats["min_health"] = float(np.min(healths))
                stats["swarm_spread"] = float(np.std(valid_positions))

            if self.mission_state is not None:
                stats["mission_progress"] = self.mission_state.get_mission_progress()
                stats["coverage_percentage"] = self.mission_state.get_coverage_percentage()

            return stats

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            active_drones = sum(1 for s in self.drone_states if s is not None)
            return (
                f"StateManager(drones={active_drones}/{self.n_drones}, "
                f"targets={len(self.target_states)}, "
                f"mission={'active' if self.mission_state else 'inactive'}, "
                f"updates={self._update_count})"
            )

    def __str__(self) -> str:
        """Detailed string representation."""
        stats = self.get_statistics()
        lines = ["StateManager Summary:"]
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
