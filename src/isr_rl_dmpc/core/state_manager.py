"""
State Manager for ISR system.

Provides centralized, thread-safe management of drone states, 
target states, and mission state for the entire swarm.
"""

from typing import List, Dict, Optional
import numpy as np
from threading import RLock
import json
from pathlib import Path
from datetime import datetime

from .data_structures import DroneState, TargetState, MissionState


class StateManager:
    """
    Centralized state aggregation for the entire swarm.

    Manages N drones, M targets, and 1 mission state with thread-safe access.
    Provides global state vectorization for learning algorithms.
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

        # To allow safe concurrent updates from multiple sensor threads while the main control loop reads data.
        self._lock = RLock()
        self._update_count = 0

    def update_drone_state(self, drone_id: int, state: DroneState):
        """
        Thread-safe update of individual drone state.
        
        Args:
            drone_id (int): Index of drone [0, n_drones)
            state (DroneState): Updated drone state
        """
        if not (0 <= drone_id < self.n_drones):
            raise ValueError(f"drone_id={drone_id} out of range [0, {self.n_drones}]")
        
        with self._lock:
            self.drone_states[drone_id] = state.copy()
            self._update_count += 1

    def update_target_state(self, target_id: str, state: TargetState):
        """
        Thread-safe update of target state.
        
        Args:
            target_id (str): Unique identifier for target
            state (TargetState): Updated target state
        """
        with self._lock:
            self.target_states[target_id] = state.copy()
            self._update_count += 1

    def update_mission_state(self, state: MissionState):
        """Update mission state."""
        with self._lock:
            self.mission_state = state.copy()
            self._update_count += 1
    
    def get_drone_state(self, drone_id: int) -> Optional[DroneState]:
        """Get state of a specified drone."""
        with self._lock:
            if (0 <= drone_id < self.n_drones):
                return self.drone_states[drone_id].copy() if self.drone_states[drone_id] else None
            
            return None
    
    def get_target_state(self, target_id: str) -> Optional[TargetState]:
        """Get state of a specified target."""
        with self._lock:
            if target_id in self.target_states:
                return self.target_states[target_id].copy()
            
            return None
    
    def get_mission_state(self) -> Optional[MissionState]:
        """Get current mission state."""
        with self._lock:
            return self.mission_state.copy() if self.mission_state else None
        
    def get_global_state_vector(self) -> np.ndarray:
        """
        Create global state vector for RL agent.

        Returns:
            Concatenated state vector of dimension 18N + 12M + K + 4
            where N=drones, M=targets, K=coverage cells
        """
        with self._lock:
            state_components = []

            # Drone States
            for drone_state in self.drone_states:
                if drone_state is not None:
                    state_components.append(drone_state.to_vector())
                else:
                    state_components.append(np.zeros(18))
            
            # Target States
            for target_state in self.target_states.values():
                state_components.append(target_state.to_vector())

            # Mission State (coverage + 4 metadata)
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
        """Get all drone positions as Nx3 matrix."""
        with self._lock:
            positions = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    positions.append(drone_state.position)
                else:
                    positions.append(np.zeros(3))
            return np.array(positions)

    def get_swarm_velocities(self) -> np.ndarray:
        """Get all drone velocities as Nx3 matrix."""
        with self._lock:
            velocities = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    velocities.append(drone_state.velocity)
                else:
                    velocities.append(np.zeros(3))
            return np.array(velocities)
        
    def get_swarm_battery_levels(self) -> np.ndarray:
        """Get all drone battery levels as N-vector."""
        with self._lock:
            batteries = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    batteries.append(drone_state.battery_energy)
                else:
                    batteries.append(0.0)
            return np.array(batteries)
    
    def get_swarm_health(self) -> np.ndarray:
        """Get all drone health levels as N-vector."""
        with self._lock:
            healths = []
            for drone_state in self.drone_states:
                if drone_state is not None:
                    healths.append(drone_state.health)
                else:
                    healths.append(0.0)
            return np.array(healths)
        
    def get_target_positions(self) -> Dict[str, np.ndarray]:
        """Get all target positions."""
        with self._lock:
            return {tid: ts.position.copy() for tid, ts in self.target_states.items()}
        
    def get_drone_target_distances(self) -> np.ndarray:
        """
        Compute distance matrix between drones and targets.

        Returns:
            NxM matrix where (i, j)th entry is the distance from drone i to target j
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
                        distances[i, j] = np.linalg.norm(self.drone_states[i].position - target_state.position)

            return distances
        
    def get_inter_drone_distances(self) -> np.ndarray:
        """
        Compute pairwise distances betweeen drones.

        Returns:
            NxN distance matrix (symmetric, zero diagonal)
        """
        positions = self.get_swarm_positions()
        n = len(positions)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j], distances[j, i] = dist, dist

        return distances
    
    def get_swarm_state(self) -> List[DroneState]:
        """Return all drone states."""
        with self._lock:
            return [state.copy() for state in self.drone_states if state is not None]
        
    def get_all_targets(self) -> Dict[str, TargetState]:
        """Return all tracked targets."""
        with self._lock:
            return {tid: ts.copy() for tid, ts in self.target_states.items()}
        
    def save_to_file(self, filepath: str):
        """Save complete state to JSON file."""
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

    def load_from_file(self, filepath: str):
        """Load state from JSON file."""
        with self._lock:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.n_drones = data["n_drones"]
            self.drone_states = [DroneState.from_dict(s) if s is not None else None for s in data["drone_states"]]
            self.target_states = {tid: TargetState.from_dict(ts) for tid, ts in data["target_states"].items()}
            self.mission_state = (MissionState.from_dict(data["mission_state"]) if data["mission_state"] is not None else None)
            self._update_count = data.get("update_count", 0)
                
    def clear(self):
        """Clear all states."""
        with self._lock:
            self.drone_states = [None] * self.n_drones
            self.target_states = {}
            self.mission_state = None
            self._update_count = 0

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