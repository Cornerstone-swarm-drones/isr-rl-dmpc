"""
Module 6: Task Allocator - Multi-Agent Task Assignment and Optimization

Uses Hungarian algorithm and optimization to assign ISR tasks
(detection, classification, tracking) to drone swarm for maximum
mission effectiveness.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import itertools
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Types of ISR tasks."""
    DETECT = "detect" # Find targets
    CLASSIFY = "classify" # Identify target type
    TRACK = "track" # Maintain target lock
    COVER = "cover" # Surveillance area
    TRANSIT = "transit" # Move to position


class TaskStatus(Enum):
    """Task execution status."""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ISRTask:
    """Single ISR task to be assigned."""
    task_id: int
    task_type: TaskType
    target_position: np.ndarray # [x, y, z]
    priority: float # (0, 1), higher is more important
    status: TaskStatus = TaskStatus.UNASSIGNED
    assigned_drone: Optional[int] = None
    estimated_duration: float = 0.0 # (sec)
    required_sensors: List[str] = None
    
    def __post_init__(self):
        if self.required_sensors is None:
            self.required_sensors = []


@dataclass
class DroneCapability:
    """Drone sensor and capability profile."""
    drone_id: int
    position: np.ndarray # [x, y, z]
    fuel_remaining: float # (0, 100)%
    sensors: List[str] # Available sensors
    current_load: float # Current task load (0, 1)
    max_speed: float # m/s
    endurance: float # max flight time (s)
    communication_range: float # meters


class HungarianAssignment:
    """
    Hungarian algorithm for optimal assignment.
    
    Algorithm Steps:
        1. Copy input and handle rectangular matrices via padding
        2. Initialize dual variables (u, v) for row/column potentials
        3. Initialize permutation array p and predecessor array way
        4. For each task i, find minimum cost augmenting path
        5. Update potentials and assignments using alternating path
        6. Return assignment for original (non-padded) tasks
    """
    
    def __init__(self):
        """Initialize Hungarian algorithm."""
        self.cost_matrix = None
        self.assignment = None
    
    def solve(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Solve assignment problem using Hungarian algorithm.
        
        The algorithm finds the minimum-cost perfect matching in a bipartite graph
        represented as a cost matrix. Handles both square and rectangular matrices.
        
        Args:
            cost_matrix (np.ndarray): Cost matrix of shape (n_tasks, n_drones)
                - Lower cost = better/preferred assignment
                - Can be rectangular (n_tasks != n_drones)
        
        Returns:
            np.ndarray: Assignment array of length n_tasks where element i 
                    indicates the assigned drone index for task i
        """
        
        # Step 1: Copy and prepare cost matrix
        cost = cost_matrix.copy().astype(float)
        n_tasks, n_drones = cost.shape
        
        # Handle rectangular matrices by padding with dummy high-cost entries
        if n_tasks != n_drones:
            if n_tasks > n_drones:
                # More tasks than drones: add dummy drones with infeasible (high) costs
                padding = np.full((n_tasks, n_tasks - n_drones), 1e6)
                cost = np.hstack([cost, padding])
            else:
                # More drones than tasks: add dummy tasks with infeasible (high) costs
                padding = np.full((n_drones - n_tasks, n_drones), 1e6)
                cost = np.vstack([cost, padding])
        
        n = cost.shape[0]
        
        # Step 2: Initialize potentials and matching
        u = np.zeros(n + 1)
        v = np.zeros(n + 1)
        p = np.zeros(n + 1, dtype=int)
        way = np.zeros(n + 1, dtype=int)
        
        # Step 3: Main Hungarian algorithm loop
        for i in range(1, n+1):
            p[0] = i
            j0 = 0
            minv = np.full(n + 1, 1e9)
            used = np.zeros(n + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = 1e9
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            # Augment matching
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Build assignment: p[j] = i (1-based); we want for each task i its column j
        ans = np.full(n, -1, dtype=int)
        for j in range(1, n + 1):
            i = p[j] - 1
            if 0 <= i < n:
                ans[i] = j - 1

        # Return only for original tasks; assignment values are drone indices
        self.assignment = ans[:n_tasks]
        return self.assignment


class TaskAllocator:
    """
    Allocates ISR tasks to drone swarm.
    
    Uses optimal assignment to maximize mission effectiveness
    subject to drone capabilities and constraints.
    """
    
    def __init__(self, num_drones: int):
        """Initialize task allocator."""
        self.num_drones = num_drones
        self.hungarian = HungarianAssignment()
        self.task_queue: List[ISRTask] = []
        self.assignments: Dict[int, ISRTask] = {}  # drone_id -> assigned task
        self.task_history: List[Dict] = []
    
    def allocate_tasks(self, tasks: List[ISRTask], drones: Dict[int, DroneCapability]) -> Dict[int, ISRTask]:
        """
        Allocate tasks to drones optimally.
        Arguments:
            tasks (List[ISRTask]): Unassigned tasks
            drones (Dict[int, DroneCapability]): Available drones
        
        Returns:
            Dict mapping drone_id to assigned task
        """
        if not tasks or not drones:
            return {}
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(tasks, drones)
        
        # Solve assignment problem
        task_indices = np.arange(len(tasks))
        drone_ids = sorted(drones.keys())
        
        if len(tasks) <= len(drones):
            # More or equal drones to tasks: perfect assignment possible
            assignments_array = self.hungarian.solve(cost_matrix)
            
            assignments = {}
            for task_idx, drone_idx in enumerate(assignments_array):
                if drone_idx < len(drone_ids):
                    drone_id = drone_ids[drone_idx]
                    task = tasks[task_idx]
                    
                    # Update task
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_drone = drone_id
                    
                    assignments[drone_id] = task
                    self.assignments[drone_id] = task
                    
                    # Record in history
                    self.task_history.append({
                        'task_id': task.task_id,
                        'drone_id': drone_id,
                        'cost': cost_matrix[task_idx, drone_idx],
                        'task_type': task.task_type.value
                    })
        else:
            # More tasks than drones: greedy assignment
            assignments = self._greedy_assign(tasks, drones, cost_matrix)
        
        return assignments
    
    def _build_cost_matrix(self, tasks: List[ISRTask], drones: Dict[int, DroneCapability]) -> np.ndarray:
        """
        Build cost matrix for assignment problem.
        Lower cost = better assignment.
        Arguments:
            tasks (List[ISRTask]): Tasks to assign
            drones (Dict[int, DroneCapability]): Available drones
        Returns:
            Cost matrix (n_tasks, n_drones)
        """
        n_tasks = len(tasks)
        n_drones = len(drones)
        
        cost = np.zeros((n_tasks, n_drones))
        drone_ids = sorted(drones.keys())
        
        for task_idx, task in enumerate(tasks):
            for drone_idx, drone_id in enumerate(drone_ids):
                drone = drones[drone_id]
                # Compute assignment cost (lower is better)
                cost[task_idx, drone_idx] = self._compute_assignment_cost(task, drone)
        return cost
    
    def _compute_assignment_cost(self, task: ISRTask, drone: DroneCapability) -> float:
        """
        Compute cost of assigning task to drone.
        Arguments:
            task (ISRTask): Task to assign
            drone (DroneCapability): Drone to assign to
        Returns:
            Assignment cost (0 = perfect, infinite = impossible)
        """
        cost = 0.0
        
        # Distance cost (closer is better)
        distance = np.linalg.norm(task.target_position - drone.position)
        distance_cost = distance / 1000  # Normalize to ~0-2
        cost += 0.3 * distance_cost
        
        # Fuel cost (more fuel is better)
        fuel_cost = (100 - drone.fuel_remaining) / 100
        cost += 0.2 * fuel_cost
        
        # Task load cost (lower load is better)
        load_cost = drone.current_load
        cost += 0.2 * load_cost
        
        # Sensor capability cost
        sensor_cost = 0.0
        if task.required_sensors:
            available = sum(1 for s in task.required_sensors if s in drone.sensors)
            sensor_cost = 1.0 - (available / len(task.required_sensors))
        cost += 0.15 * sensor_cost
        
        # Priority bonus (higher priority tasks get lower cost)
        priority_bonus = task.priority * 0.15
        cost -= priority_bonus
        
        # Feasibility check
        # Time to reach + task duration must fit in endurance
        travel_time = distance / drone.max_speed
        total_time = travel_time + task.estimated_duration
        
        if total_time > drone.endurance:
            # Infeasible assignment
            cost = 1e6
        
        return max(cost, 0)  # Ensure non-negative
    
    def _greedy_assign(
            self, tasks: List[ISRTask], drones: Dict[int, DroneCapability], cost_matrix: np.ndarray
    ) -> Dict[int, ISRTask]:
        """
        Greedy assignment when more tasks than drones.
        Arguments:
            tasks: Tasks to assign
            drones: Available drones
            cost_matrix: Pre-computed costs
        Returns:
            Assignments
        """
        assignments = {}
        drone_ids = sorted(drones.keys())
        assigned_tasks = set()
        
        # Sort tasks by priority (descending)
        task_priority = [(i, tasks[i].priority) for i in range(len(tasks))]
        task_priority.sort(key=lambda x: x[1], reverse=True)
        
        for task_idx, _ in task_priority:
            # Find best drone for this task
            best_drone_idx = -1
            best_cost = 1e9
            
            for drone_idx in range(len(drone_ids)):
                if drone_idx not in assigned_tasks:
                    if cost_matrix[task_idx, drone_idx] < best_cost:
                        best_cost = cost_matrix[task_idx, drone_idx]
                        best_drone_idx = drone_idx
            
            if best_drone_idx >= 0:
                drone_id = drone_ids[best_drone_idx]
                task = tasks[task_idx]
                
                task.status = TaskStatus.ASSIGNED
                task.assigned_drone = drone_id
                
                assignments[drone_id] = task
                self.assignments[drone_id] = task
                assigned_tasks.add(best_drone_idx)
                
                self.task_history.append({
                    'task_id': task.task_id,
                    'drone_id': drone_id,
                    'cost': best_cost,
                    'task_type': task.task_type.value
                })
        
        return assignments
    
    def update_task_status(self, task_id: int, status: TaskStatus) -> bool:
        """
        Update task status.
        Arguments:
            task_id (int): Task ID
            status (TaskStatus): New status
        Returns:
            True if successful
        """
        for task in self.task_queue:
            if task.task_id == task_id:
                task.status = status
                return True
        
        for task in self.assignments.values():
            if task.task_id == task_id:
                task.status = status
                return True
        
        return False
    
    def reassign_task(self, task_id: int, drones: Dict[int, DroneCapability]) -> bool:
        """
        Reassign a task to different drone.
        Args:
            task_id (int): Task to reassign
            drones (Dict[int, DroneCapability]): Available drones
        Returns:
            True if successful
        """
        # Find current assignment
        current_drone = None
        task = None
        
        for d_id, t in self.assignments.items():
            if t.task_id == task_id:
                current_drone = d_id
                task = t
                break
        
        if task is None:
            return False
        
        # Find best alternative
        best_cost = 1e9
        best_drone = None
        
        for drone_id, drone in drones.items():
            if drone_id != current_drone:
                cost = self._compute_assignment_cost(task, drone)
                if cost < best_cost:
                    best_cost = cost
                    best_drone = drone_id
        
        if best_drone is not None:
            # Update assignment
            del self.assignments[current_drone]
            task.assigned_drone = best_drone
            self.assignments[best_drone] = task
            return True
        
        return False
    
    def get_allocation_metrics(self) -> Dict:
        """
        Get metrics about task allocation quality.
        Returns: Dictionary with allocation metrics
        """
        if not self.task_history:
            return {}
        
        costs = [h['cost'] for h in self.task_history]
        
        return {
            'total_assignments': len(self.task_history),
            'average_cost': np.mean(costs),
            'best_cost': np.min(costs),
            'worst_cost': np.max(costs),
            'assignments_per_task_type': self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count assignments by task type."""
        counts = {}
        for h in self.task_history:
            task_type = h['task_type']
            counts[task_type] = counts.get(task_type, 0) + 1
        return counts
    
    def get_drone_load(self, drone_id: int) -> float:
        """
        Get current task load for drone.
        Returns: Task load (0-1)
        """
        if drone_id in self.assignments:
            task = self.assignments[drone_id]
            if task.status == TaskStatus.IN_PROGRESS:
                return task.estimated_duration / 3600  # Normalized to hours
        return 0.0
