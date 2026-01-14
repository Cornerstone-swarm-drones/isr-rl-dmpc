"""
Module 2: Formation Controller - Consensus Based Multi-Agent Control

Implements distributed formation control using consensus algorithms and 
graph based topology management for coordinated swarm behavior.
"""

from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from ..core import DroneState, StateManager
from ..utils import PerformanceLogger


class FormationType(Enum):
    """Supported formation geometries"""
    LINE = "line" # Linear formation
    WEDGE = "wedge" # V-shaped formation
    COLUMN = "column" # # Single file
    CIRCULAR = "circular" # Circular perimeter
    GRID = "grid" # Grid pattern
    SPHERE = "sphere" # 3D spherical

@dataclass
class FormationConfig:
    """Formation conrol configuration"""
    type: FormationType = FormationType.WEDGE
    scale: float = 50.0 # Formation size (m)
    spacing: float = 10.0 # Inter-drone spacing (m)
    convergence_threshold: float = 0.5 # Position error threshold (m)
    velocity_damping: float = 0.1 # Velocity damping (0-1)
    rotation_rate: float = 0.0 # Formation rotation (rad/s)
    z_spacing: float = 2.0 # Vertical spacing (m)

@dataclass
class ConsensusState:
    """Consensus variables for a drone"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    neighbor_count: int = 0
    converged: bool = False

class FormationGeometry:
    """Generate desired formation positions"""

    def __init__(self, config: FormationConfig):
        """
        Initialize formation geometry
        Arguments:
            config (FormationConfig): Formation Configuration
        """
        self.config = config

    def generate_desired_positions(self, n_drones: int, center: np.ndarray, heading: float = 0.0) -> Dict[int, np.ndarray]:
        """
        Generate desired positions for drones in formation
        Arguments:
            n_drones (int): Number of drones
            center (np.ndarray): Formation center position (3D)
            heading (float): Formation heading angle (rad)
        Returns:
            Dict mapping drone_id -> desired 3D position
        """
        if self.config.type == FormationType.LINE:
            return self._line_formation(n_drones, center, heading)
        elif self.config.type == FormationType.WEDGE:
            return self._wedge_formation(n_drones, center, heading)
        elif self.config.type == FormationType.COLUMN:
            return self._column_formation(n_drones, center, heading)
        elif self.config.type == FormationType.CIRCULAR:
            return self._circular_formation(n_drones, center, heading)
        elif self.config.type == FormationType.GRID:
            return self._grid_formation(n_drones, center, heading)
        elif self.config.type == FormationType.SPHERE:
            return self._sphere_formation(n_drones, center)
        else:
            raise ValueError(f"Unknown formation type: {self.config.type}")
        
    def _line_formation(self, n_drones: int, center: np.ndarray, heading: float) -> Dict[int, np.ndarray]:
        """Linear formation along heading direction"""
        positions = {}
        for i in range(n_drones):
            offset = (i - (n_drones-1) / 2) * self.config.spacing
            # Rotation matrix for heading
            x = offset * np.cos(heading)
            y = offset * np.sin(heading)
            z = (i - (n_drones - 1) / 2) * self.config.z_spacing

            positions[i] = center + np.array([x, y, z])
        return positions
    
    def _wedge_formation(self, n_drones: int, center: np.ndarray, heading: float) -> Dict[int, np.ndarray]:
        """Wedge (V-shaped) formation with leader at front"""
        positions = {}
        lead_drone = 0
        positions[lead_drone] = center.copy()

        # Arrange remaining drones in V pattern
        remaining = n_drones - 1
        for i in range(1, n_drones):
            # angle increases with position
            angle = (i/n_drones) * np.pi # +- 30 deg
            # side alternation (left, right, left, right, ...)
            side = 1 if i%2 == 1 else -1

            # Position relative to center
            r = self.config.spacing * (i//2 + 1)
            x = r*np.cos(heading + side*angle)
            y = r*np.sin(heading + side*angle)
            z = (i%2) * self.config.z_spacing

            positions[i] = center + np.array([x, y, z])

        return positions
    
    def _column_formation(self, n_drones: int, center: np.ndarray, heading: float) -> Dict[int, np.ndarray]:
        """Single-file column formation."""
        positions = {}
        for i in range(n_drones):
            offset = i * self.config.spacing
            x = offset * np.cos(heading)
            y = offset * np.sin(heading)
            positions[i] = center + np.array([x, y, 0.0])
        return positions

    def _circular_formation(self, n_drones: int, center: np.ndarray, heading: float) -> Dict[int, np.ndarray]:
        """Circular formation around center point."""
        positions = {}
        radius = self.config.scale / 2
        for i in range(n_drones):
            angle = 2 * np.pi * i / n_drones + heading
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = center + np.array([x, y, 0.0])
        return positions

    def _grid_formation(self, n_drones: int, center: np.ndarray, heading: float) -> Dict[int, np.ndarray]:
        """Grid/matrix formation."""
        positions = {}
        grid_size = int(np.ceil(np.sqrt(n_drones)))
        for i in range(n_drones):
            row = i // grid_size
            col = i % grid_size

            x = (col - grid_size / 2) * self.config.spacing
            y = (row - grid_size / 2) * self.config.spacing
            # Apply heading rotation
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            x_rot = x * cos_h - y * sin_h
            y_rot = x * sin_h + y * cos_h

            positions[i] = center + np.array([x_rot, y_rot, 0.0])
        return positions

    def _sphere_formation(self, n_drones: int, center: np.ndarray) -> Dict[int, np.ndarray]:
        """3D spherical formation using Fibonacci sphere algorithm."""
        positions = {}
        radius = self.config.scale / 2
        for i in range(n_drones):
            # Fibonacci sphere distribution
            phi = np.arccos(1 - 2 * i / n_drones)
            theta = np.sqrt(n_drones * np.pi) * phi

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z_offset = radius * np.cos(phi)

            positions[i] = center + np.array([x, y, z_offset])
        return positions


class ConsensusController:
    """Distributed consensus-based formation control"""

    def __init__(self, dt: float = 0.02, comm_range: float = 100.0):
        """
        Initialize consensus controller
        Arguments:
            dt (float): Control time step (s)
            comm_range (float): Communication range (m)
        """
        self.dt = dt
        self.comm_range = comm_range
        self.consensus_states: Dict[int, ConsensusState] = {}
        self.perf_logger = PerformanceLogger('formation_control')

    def compute_formation_control(
            self, state_manager: StateManager,
            formation_config: FormationConfig,
            formation_center: np.ndarray,
            formation_heading: float = 0.0
    ) -> Dict[int, np.ndarray]:
        """
        Compute formation control input for all drones

        Arguments:
            state_manager (StateManager): Global swarm state
            formation_config (FormationConfig): Desired formation config
            formation_center (np.ndarray): Formation center position (3D)
            formation_heading (float): Formation heading (rad)
        Returns:
            Dict mapping drone_id -> contol acceleration (3D)
        """
        with self.perf_logger.start_timer('total_control'):
            # Generate desired formation
            geo = FormationGeometry(formation_config)
            desired_positions = geo.generate_desired_positions(state_manager.n_drones, formation_center, formation_heading)

            # Adjacency (communication topology)
            adjacency = state_manager.get_inter_drone_adjacency(self.comm_range)

            # contol for each drone
            controls = {}
            with self.perf_logger.start_timer('individual_controls'):
                for drone_id in range(state_manager.n_drones):
                    drone_state = state_manager.get_drone_state(drone_id)
                    desired_pos = desired_positions.get(drone_id)

                    # Get neighbors from adjacency
                    neighbors = np.where(adjacency[drone_id, :] > 0)[0]

                    control = self._compute_drone_control(drone_state, desired_pos, state_manager, neighbors, formation_config)
                    controls[drone_id] = control
            return controls
        
    def _compute_drone_controls(
            self, drone_state: DroneState, 
            desired_position: np.ndarray, 
            state_manager: StateManager, 
            neighbors: np.ndarray, 
            config: FormationConfig
    ) -> np.ndarray:
        """
        Compute control acceleration for single drone using 
        proportional-derivative with consensus correction.
        Arguments:
            drone_state (DroneState): Current drone state
            desired_position (np.ndarray): Desired 3D position
            state_manager (StateManager): Global state access
            neighbors (np.ndarray): Array of neighbor drone IDs
            config (FormationConfig): Formation configuration
        Returns:
            Control acceleration (3D)
        """
        # Position error
        pos_error = desired_position - drone_state.position
        pos_error_norm = np.linalg.norm(pos_error)

        # Proportional term (gain based on error magnitude)
        kp = 2.0 # Position gain
        if pos_error_norm > 0.1:
            up = kp*pos_error
        else:
            up = 2.0 * pos_error

        # Derivative term (velocity damping)
        kd = config.velocity_damping
        ud = -kd*drone_state.velocity

        # Consensus correction from neighbors
        u_consensus = np.zeros(3)
        if len(neighbors) > 0:
            # Average neighbor positions
            neighbors_avg_pos = np.zeros(3)
            for n_id in neighbors:
                n_state = state_manager.get_drone_state(n_id)
                neighbors_avg_pos += n_state.position

            neighbors_avg_pos /= len(neighbors)

            # Consensus term (pull towards neighbor average)
            k_consensus = 0.1
            u_consensus = k_consensus * (neighbors_avg_pos - drone_state.position)

        # Combine control terms
        u_total = up + ud + u_consensus

        # Saturate control (max acceleration)
        max_acceleration = 5.0 # m/s^2
        u_norm = np.linalg.norm(u_total)
        if u_norm > max_acceleration:
            u_total = (max_acceleration / u_norm) * u_total

        return u_total

    def check_formation_convergence(
            self, state_manager: StateManager,
            desired_positions: Dict[int, np.ndarray],
            threshold: float = 0.5,
    ) -> bool:
        """
        Check if formation has converged
        Arguments:
            state_manager (StateManager): Global state
            desired_positions (Dict): Desired position for each drone
            threshold (float): Convergence thresold (m)
        Returns:
            True if all drones within threshold of desired positions
        """
        for drone_id, desired_pos in desired_positions.items():
            drone_state = state_manager.get_drone_state(drone_id)
            error = np.linalg.norm(drone_state.position - desired_pos)

            if error > threshold:
                return False
        return True
    
    def get_formation_metrics(self, state_manager: StateManager, desired_positions: Dict[int, np.ndarray]) -> Dict:
        """
        Compute formation quality metrics
        Arguments:
            state_manager (StateManager): Global state
            desired_positions (Dict): Desired position for each drone
        Returns:
            Dict with formation metrics
        """
        errors = []
        for drone_id, desired_pos in desired_positions.items():
            drone_state = state_manager.get_drone_state(drone_id)
            error = np.linalg.norm(drone_state.position - desired_pos)
            errors.append(error)
        errors = np.array(errors)

        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'std_error': np.std(errors),
            'rmse': np.sqrt(np.mean(errors ** 2))
        }
    

class FormationController:
    """
    Main formation controller orchestrating multi-agent coordination
    Workflow:
        1. Define desired formation geometry
        2. Compute consensus-based control laws
        3. Monitor formation convergence
        4. Track topology changes
    """

    def __init__(self, dt: float = 0.02, comm_range: float = 100.0):
        """
        Initialize formation controller
        Arguments:
            dt (float): Control time step (s)
            comm_range (float): Interdrone communication range (m)
        """
        self.dt = dt
        self.comm_range = comm_range
        self.consensus = ConsensusController(dt, comm_range)
        self.formation_config = FormationConfig(type=FormationType.WEDGE)
        self.formation_center = np.array([0.0, 0.0, 50.0])
        self.formation_heading = 0.0

    def update_formation(
            self, state_manager: StateManager,
            formation_center: Optional[np.ndarray] = None,
            formation_heading: Optional[float] = None,
            formation_type: Optional[FormationType] = None
    ) -> Dict[int, np.ndarray]:
        """
        Update formation center and compute control

        Arguments:
            state_manager (StateManager): Global swarm state
            formation_center (Optional): New formation center
            formation_heading (Optional): New formation heading (rad)
            formation_type (Optional): New formation type

        Returns: 
            Dict mapping drone_id -> control acceleration
        """
        # Update formation params
        if formation_center is not None:
            self.formation_center = formation_center.copy()
        if formation_heading is not None:
            self.formation_heading = formation_heading
        if formation_type is not None:
            self.formation_config.type = formation_type

        # Compute formation control
        controls = self.consensus.compute_formation_control(
            state_manager, self.formation_config, self.formation_center, self.formation_heading
        )
        return controls
    
    def set_formation_config(self, config: FormationConfig):
        """Set formation configuration"""
        self.formation_config = config

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FormationController("
            f"type={self.formation_config.type.value}, "
            f"spacing={self.formation_config.spacing}m, "
            f"scale={self.formation_config.scale}m)"
        )
