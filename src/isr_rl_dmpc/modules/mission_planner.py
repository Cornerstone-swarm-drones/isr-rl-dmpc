"""
Module 1: Mission PLanner - Grid Based Decomposition and Waypoint Generation

Implements cpverage path planning through Grid-based decomposition and
waypoint generation for multi-drone ISR missions.
"""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

from ..core import MissionState
from ..utils import GeometryOps, MissionLogger


@dataclass
class GridCell:
    """Represents a single grid cell for coverage tracking."""
    cell_id: int
    vertices: np.ndarray # Nx2 array of polygon vertices
    center: np.ndarray # 2D centroid
    area: float # Cell area (m^2)
    priority: float # Coverage priority (0-1, higher = more important)
    covered: bool = False # Coverage status

    def __post_init__(self):
        """Validate and coompute cell properties."""
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 2
        assert len(self.vertices) >= 3

        if self.center is None: # compute centroid if not provided
            self.center = np.mean(self.vertices, axis=0)
        
        # Compute area
        self.area = GeometryOps.polygon_area(self.vertices)


class GridDecomposer:
    """Decompose mission area into coverage cells using grid methods. """

    def __init__(self, area_boundary: np.ndarray, grid_resolution: float = 20.0):
        """
        Initialize grid decomposer.

        Arguments:
            area_boundary (np.ndarray): Nx2 polygon vertices of mission area
            grid_resolution (float): Grid cell size (m)
        """
        self.area_boundary = area_boundary
        self.grid_resolution = grid_resolution
        self.cells: List[GridCell] = []
        self._logger = None

    def decompose_grid(self, n_drones: int=10) -> List[GridCell]:
        """
        Decompose mission area using uniform grid.

        Arguments:
            n_drones (int): Number of drones

        Returns:
            List of GridCell objects covering the area
        """
        # area bounds
        minx,miny = np.min(self.area_boundary, axis=0)
        maxx,maxy = np.max(self.area_boundary, axis=0)

        # create grid
        x=np.arange(minx, maxx + self.grid_resolution, self.grid_resolution)
        y=np.arange(miny, maxy + self.grid_resolution, self.grid_resolution)

        cell_id = 0
        self.cells = []

        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                vertices = np.array([[x[i], y[j]], [x[i+1], y[j]],
                    [x[i+1], y[j+1]], [x[i], y[j+1]],
                ])
                # check for overlap with mission area
                center=np.mean(vertices, axis=0)
                if GeometryOps.point_in_polygon(center, self.area_boundary):
                    # Compute prio based on distance from boundary
                    prio = self._compute_cell_prio(center)

                    cell = GridCell(cell_id, vertices, center, self.grid_resolution**2, prio)
                    self.cells.append(cell)
                    cell_id += 1

        return self.cells
    
    def _compute_cell_prio(self, point: np.ndarray) -> float:
        """
        Compute prio for a point. Higher prio at mission center, lower at edges.

        Arguments:
            point (np.ndarray): 2D point

        Returns:
            Priority Score (0-1)
        """
        # Center of mission area
        area_center = np.mean(self.area_boundary, axis=0)
        # Distance from area center
        dist_center = np.linalg.norm(point - area_center)
        # Area size scale
        area_size = np.linalg.norm(
            np.max(self.area_boundary, axis=0) - np.min(self.area_boundary, axis=0)
        )

        # Priority decreases with distance from center
        prio = np.exp(-dist_center / (area_size / 10.0))
        return np.clip(prio, 0.1, 1.0) # Normalized Prio
    
    def get_cells_by_priority(self) -> List[GridCell]:
        """Get cells sorted by coverage prio (decreasing)"""
        return sorted(self.cells, key=lambda c: c.priority, reverse=True)
    

class WaypointGenerator:
    """Generate waypoints for drone navigation through coverage cells."""

    def __init__(self, altitude: float = 50.0, hover_time: float = 5.0):
        """
        Initialize waypoint generator.
        Arguments:
            altitude (float): Mission altitude (m)
            hover_time (float): Hover time per waypoint (s)
        """
        self.altitude = altitude
        self.hover_time = hover_time

    def generate_path(self, cells: List[GridCell], start_position: np.ndarray, opt_strat: str = 'nearest') -> np.ndarray:
        """
        Generate waypoint sequence through cells.
        Arguments:
            cells (List[GridCell]): List of cells to visit
            start_position (np.ndarray: Starting 2D position
            opt_strat (str): Path optimization method ('nearest', 'swwep', 'spiral')

        Returns:
            Nx3 array of waypoints [x, y, z]
        """
        if not cells:
            return np.array([ [start_position[0], start_position[1], self.altitude]])
        
        if opt_strat == 'nearest':
            cell_seq = self._nearest_neighbor_path(cells, start_position)
        elif opt_strat == 'sweep':
            cell_seq = self._sweep_path(cells)
        elif opt_strat == 'spiral':
            cell_seq = self._spiral_path(cells)
        else:
            cell_seq = cells

        # Convert cell centers to 3D waypoints
        waypoints = np.array([
            [cell.center[0], cell.center[1], self.altitude] for cell in cell_seq
        ])
        return waypoints
    
    def _nearest_neighbor_path(self, cells: List[GridCell], start: np.ndarray) -> List[GridCell]:
        """Greedy nearest-neighbor path through cells."""
        remaining = list(cells)
        path=[]
        current = start

        while remaining:
            # Find nearest cell
            dist = [np.linalg.norm(c.center - current) for c in remaining]
            nearest_idx = np.argmin(dist)
            path.append(remaining[nearest_idx])
            current = remaining[nearest_idx].center
            remaining.pop(nearest_idx)

        return path
    
    def _sweep_path(self, cells: List[GridCell]) -> List[GridCell]:
        """Sweep path: sort cells by y-coordinate, then x within rows"""
        # Group cells by y-coordinate bands
        sorted_cells = sorted(cells, key=lambda c: (c.center[1], c.center[0]))

        return sorted_cells
    
    def _spiral_path(self, cells: List[GridCell]) -> List[GridCell]:
        """Outward Spiral path from the center"""
        # Compute centroid of all cells
        center = np.mean([c.center for c in cells], axis=0)

        # sort by distance from center
        sorted_cells = sorted(cells, key=lambda c: np.linalg.norm(c.center - center))
        return sorted_cells
    
    def estimate_mission_time(self, waypoints: np.ndarray, drone_speed: float = 5.0) -> float:
        """
        Estimate total mission time.

        Arguments:
            waypoints (np.ndarray): Nx3 waypoint array
            drone_speed (float): Drone cruise speed (m/s)

        Returns:
            Estimated mission time (s)
        """
        # Hover time applies per waypoint, even if there is only one
        hover_time = len(waypoints) * self.hover_time

        if len(waypoints) < 2:
            # No travel, only hover + buffer
            return hover_time * 1.2
        
        # Travel time
        path_lenght = 0.0
        for i in range(len(waypoints) - 1):
            dist = np.linalg.norm(waypoints[i+1] - waypoints[i])
            path_lenght += dist

        travel_time = path_lenght/ (drone_speed + 1e-6)
        total_time = (travel_time + hover_time) *1.2 # 20% buffer for turns and vertival movements

        return total_time
    

class MissionPlanner:
    """
    Main mission planner for grid decomposition and waypoint generation.

    Workflow:
        1. Decompose mission area into grid cells
        2. Assign cells to drones based on position and capability
        3. Generate optimal waypoint paths for each drone
        4. Update mission state with coverage tracking
    """

    def __init__(self, grid_resolution: float = 20.0, altitude: float = 50.0):
        """
        Initialize mission planner.

        Arguments:
            grid_reolution (float): Grid cell size (m)
            altitude (float): Mission altitude (m)
        """
        self.grid_resolution = grid_resolution
        self.altitude = altitude
        self.decomposer: Optional[GridDecomposer] = None
        self.waypoint_gen = WaypointGenerator(altitude=altitude)
        self.cell_assignments: Dict[int, List[int]] = {} # drone_id -> cell_ids
        self._logger = MissionLogger('mission_planner')

    def plan_mission(self, mission_state: MissionState, drone_positions: np.ndarray, n_drones: int = 10) -> Dict[int, np.ndarray]:
        """
        Plan complete mission with cell decomposition and waypoint assignement.

        Arguments:
            mission_state (MissionState): Current mission state
            drone_positions (np.ndarray): Nx3 current drone positons
            n_drones (int): Number of drones

        Returns:
            Dict mapping drone_id -> waypoint array (Nx3)
        """
        # 1. Decompose area into grid cells
        self._logger.debug(f"Decomposing mission are with reolution {self.grid_resolution}m")
        self.decomposer = GridDecomposer(mission_state.area_boundary, self.grid_resolution)
        cells = self.decomposer.decompose_grid(n_drones)
        self._logger.info(f"Grid decomposition created {len(cells)} cells.")

        # 2. Assign cells to drones based on position
        self.cell_assignments = self._assign_cells_to_drones(cells, drone_positions)

        # 3. Generate waypoints for each drone
        drone_waypoints = {}
        for drone_id, cell_ids in self.cell_assignments.items():
            if drone_id < len(drone_positions):
                assigned_cells = [cells[cid] for cid in cell_ids]
                start_pos = drone_positions[drone_id, :2]

                waypoints = self.waypoint_gen.generate_path(assigned_cells, start_pos, opt_strat='nearest')
                drone_waypoints[drone_id] = waypoints
            
                # Add waypoints to mission state
                for wp in waypoints:
                    mission_state.add_waypoint(wp)

                self._logger.debug(
                    f"Drone {drone_id}: {len(waypoints)} waypoints "
                    f"(mission time: {self.waypoint_gen.estimate_mission_time(waypoints):.1f}s)"
                )
        # 4. Update mission state with coverage matrix
        self.update_coverage(mission_state, self.cell_assignments.values())
        
        self._logger.mission_started('ISR_Mission', n_drones, mission_state.area_boundary.shape[0])

        return drone_waypoints
    
    def _assign_cells_to_drones(self, cells: List[GridCell], drone_positions: np.ndarray) -> Dict[int, List[int]]:
        """
        Assign cells to drones using spatial partioning.

        Arguments:
            cells (List[GridCells]): List of grid cells
            drone_positions (np.ndarray): Nx3 drone postions

        Returns:
            Dict mapping drone_id -> list of cell_ids 
        """
        assignments: Dict[int, List[int]] = {}
        n_drones = len(drone_positions)

        # Sort cells by prio
        prio_sorted = self.decomposer.get_cells_by_priority()
        
        # Assign cells to nearest drone (greedy)
        for cell in prio_sorted:
            dists = [ np.linalg.norm(cell.center - drone_positions[i, :2] for i in range(n_drones))]
            nearest_drone = np.argmin(dists)

            if nearest_drone not in assignments:
                assignments[nearest_drone].append(cell.cell_id)

        return assignments
    
    def _update_coverage_matrix(self, mission_state:MissionState) -> MissionState:
        """Update empty mission state coverage matrix"""
        if self.decomposer is None:
            return

        # Create coverage matrix (initially all uncovered)
        n_cells = len(self.decomposer.cells)
        mission_state.coverage_matrix = np.zeros(n_cells, dtype=bool)
        return mission_state

    def update_coverage(self, mission_state: MissionState, cell_ids: List[int]) -> None:
        """
        Update coverage status for completed cells.

        Args:
            mission_state (MissionState): Mission state to update
            cell_ids (List[int]): IDs of cells to mark as covered
        """
        mission_state = self._update_coverage_matrix(mission_state)
        for cell_id in cell_ids:
            mission_state.mark_cell_covered(cell_id)
  
    def get_next_uncovered_cell(self, mission_state: MissionState, drone_id: int) -> Optional[GridCell]:
        """
        Get next uncovered cell for a drone.
        Args:
            mission_state (MissionState): Current mission state
            drone_id (int): Drone ID
        Returns:
            Next uncovered GridCell or None
        """
        if self.decomposer is None or drone_id not in self.cell_assignments:
            return None

        cell_ids = self.cell_assignments[drone_id]
        for cell_id in cell_ids:
            if not mission_state.coverage_matrix[cell_id]:
                return self.decomposer.cells[cell_id]

        return None

    def get_coverage_statistics(self, mission_state: MissionState) -> Dict:
        """
        Get mission coverage statistics.
        Args:
            mission_state (MissionState): Current mission state
        Returns:
            Dictionary with statistics
        """
        if self.decomposer is None:
            return {}

        total_cells = len(self.decomposer.cells)
        covered_cells = np.sum(mission_state.coverage_matrix)
        return {
            'total_cells': total_cells,
            'covered_cells': int(covered_cells),
            'uncovered_cells': int(total_cells - covered_cells),
            'coverage_percentage': mission_state.get_coverage_percentage(),
            'remaining_time': mission_state.mission_duration - mission_state.elapsed_time
        }

    def __repr__(self) -> str:
        """String representation."""
        cell_count = len(self.decomposer.cells) if self.decomposer else 0
        return (
            f"MissionPlanner("
            f"grid_res={self.grid_resolution}m, "
            f"altitude={self.altitude}m, "
            f"cells={cell_count})"
        )
