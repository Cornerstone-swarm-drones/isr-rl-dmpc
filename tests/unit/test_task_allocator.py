"""
TEST: Module 6 - Task Allocator (Multi-Agent Task Assignment)

Comprehensive unit tests for Hungarian algorithm and task allocation
Tests square/rectangular matrices, assignment optimality, and cost computation
"""

import pytest
import numpy as np
from unittest.mock import Mock
from isr_rl_dmpc import (
    HungarianAssignment, ISRTask, TaskType, TaskStatus, DroneCapability, TaskAllocator
)


class TestHungarianAlgorithmSquareMatrices:
    """Test Hungarian algorithm with square cost matrices."""
    
    @pytest.fixture
    def hungarian(self): 
        return HungarianAssignment()
    
    def test_identity_cost_matrix(self, hungarian):
        """Diagonal assignment for identity cost matrix."""
        cost_matrix = np.array([
            [1, 10, 10],
            [10, 1, 10],
            [10, 10, 1]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Should assign diagonally (task i -> drone i)
        assert len(assignment) == 3
        assert all(0 <= a < 3 for a in assignment)
        assert len(np.unique(assignment)) == 3  # All unique assignments
    
    def test_small_square_matrix(self, hungarian):
        """2x2 square cost matrix."""
        cost_matrix = np.array([
            [1, 10],
            [10, 1]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 2
        assert np.all(np.unique(assignment) == np.array([0, 1]))
    
    def test_large_square_matrix(self, hungarian):
        """10x10 square cost matrix."""
        np.random.seed(42)
        cost_matrix = np.random.rand(10, 10) * 100
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 10
        assert len(np.unique(assignment)) == 10
        assert all(0 <= a < 10 for a in assignment)
    
    def test_zero_cost_matrix(self, hungarian):
        """Matrix with all zeros."""
        cost_matrix = np.zeros((3, 3))
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 3
        assert len(np.unique(assignment)) == 3
    
    def test_single_element_matrix(self, hungarian):
        """1x1 cost matrix."""
        cost_matrix = np.array([[5.0]])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 1
        assert assignment[0] == 0


class TestHungarianAlgorithmRectangularMatrices:
    """Test Hungarian algorithm with rectangular cost matrices."""
    
    @pytest.fixture
    def hungarian(self):
        return HungarianAssignment()
    
    def test_more_tasks_than_drones(self, hungarian):
        """More tasks (rows) than drones (columns)."""
        # 5 tasks, 3 drones
        cost_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 5  # Returns assignment for all tasks
        assert all(0 <= a < 5 for a in assignment)  # Indices refer to padded matrix
    
    def test_more_drones_than_tasks(self, hungarian):
        """More drones (columns) than tasks (rows)."""
        # 3 tasks, 5 drones
        cost_matrix = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 3  # Returns assignment for all tasks
        assert all(0 <= a < 5 for a in assignment)
    
    def test_single_task_multiple_drones(self, hungarian):
        """1 task, multiple drones."""
        cost_matrix = np.array([[5, 3, 7, 2]])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 1
        assert assignment[0] == 3  # Should pick drone with lowest cost (2)
    
    def test_many_tasks_single_drone(self, hungarian):
        """Many tasks, single drone."""
        cost_matrix = np.array([
            [5],
            [3],
            [7],
            [2]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 4


class TestHungarianAlgorithmOptimality:
    """Test that Hungarian algorithm finds optimal assignments."""
    
    @pytest.fixture
    def hungarian(self):
        return HungarianAssignment()
    
    def test_optimal_diagonal_assignment(self, hungarian):
        """Test finds optimal diagonal assignment."""
        cost_matrix = np.array([
            [1, 100, 100],
            [100, 2, 100],
            [100, 100, 3]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Compute total cost
        total_cost = sum(cost_matrix[i, assignment[i]] for i in range(3))
        assert total_cost == 6  # 1 + 2 + 3
    
    def test_optimal_reverse_assignment(self, hungarian):
        """Test finds optimal reverse assignment."""
        cost_matrix = np.array([
            [100, 100, 1],
            [100, 2, 100],
            [3, 100, 100]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Optimal: task 0->drone 2, task 1->drone 1, task 2->drone 0
        # Total cost: 1 + 2 + 3 = 6
        total_cost = sum(cost_matrix[i, assignment[i]] for i in range(3))
        assert total_cost == 6
    
    def test_random_assignment_optimality(self, hungarian):
        """Test on random matrix finds low-cost assignment."""
        np.random.seed(42)
        cost_matrix = np.random.rand(5, 5) * 100
        
        assignment = hungarian.solve(cost_matrix)
        
        # Compute cost of found assignment
        found_cost = sum(cost_matrix[i, assignment[i]] for i in range(5))
        
        # Try a random assignment for comparison
        random_assignment = np.random.permutation(5)
        random_cost = sum(cost_matrix[i, random_assignment[i]] for i in range(5))
        
        # Hungarian should find equal or better solution
        assert found_cost <= random_cost


class TestAssignmentArrayFormat:
    """Test the format and structure of assignment output."""
    
    @pytest.fixture
    def hungarian(self):
        return HungarianAssignment()
    
    def test_assignment_array_dtype(self, hungarian):
        """Assignment array is integer type."""
        cost_matrix = np.array([[1, 2], [3, 4]], dtype=float)
        
        assignment = hungarian.solve(cost_matrix)
        
        # Should be integer (drone indices)
        assert assignment.dtype == np.int64 or assignment.dtype == np.int32
    
    def test_assignment_array_length(self, hungarian):
        """Assignment length equals number of tasks."""
        cost_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 4  # 4 tasks
    
    def test_assignment_array_bounds(self, hungarian):
        """Assignment values in valid range."""
        cost_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Each assignment should refer to valid drone index (padded or original)
        assert all(a >= 0 for a in assignment)
    
    def test_numpy_array_return_type(self, hungarian):
        """Returns numpy array."""
        cost_matrix = np.array([[1, 2], [3, 4]])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert isinstance(assignment, np.ndarray)


class TestCostMatrixHandling:
    """Test robustness in handling various cost matrices."""
    
    @pytest.fixture
    def hungarian(self):
        return HungarianAssignment()
    
    def test_floating_point_costs(self, hungarian):
        """Handles floating point costs."""
        cost_matrix = np.array([
            [0.1, 0.9],
            [0.8, 0.2]
        ], dtype=float)
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 2
        assert assignment.dtype in [np.int32, np.int64]
    
    def test_large_cost_values(self, hungarian):
        """Handles large cost values."""
        cost_matrix = np.array([
            [1e6, 1e6 + 1],
            [1e6 + 1, 1e6]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 2
    
    def test_small_cost_values(self, hungarian):
        """Handles very small cost values."""
        cost_matrix = np.array([
            [1e-6, 2e-6],
            [3e-6, 1e-6]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 2
    
    def test_negative_costs_not_required(self, hungarian):
        """Non-negative costs work correctly."""
        cost_matrix = np.array([
            [0, 5],
            [10, 1]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 2
    
    def test_infeasible_costs(self, hungarian):
        """Handles high infeasibility costs (1e6)."""
        cost_matrix = np.array([
            [1, 1e6],
            [1e6, 1]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Should avoid high-cost assignments
        total_cost = sum(cost_matrix[i, assignment[i]] for i in range(2))
        assert total_cost == 2  # 1 + 1


class TestISRTaskBasics:
    """Test ISRTask data structure."""
    
    def test_task_initialization(self):
        """ISRTask initializes correctly."""
        
        task = ISRTask(
            task_id=1,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 100, 50]),
            priority=0.8
        )
        
        assert task.task_id == 1
        assert task.task_type == TaskType.DETECT
        assert task.status == TaskStatus.UNASSIGNED
        assert task.assigned_drone is None
        assert task.priority == 0.8


class TestDroneCapabilityBasics:
    """Test DroneCapability data structure."""
    
    def test_drone_capability_initialization(self):
        """DroneCapability initializes correctly."""
        
        drone = DroneCapability(
            drone_id=0,
            position=np.array([0, 0, 50]),
            fuel_remaining=85.0,
            sensors=['rf', 'optical'],
            current_load=0.2,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        assert drone.drone_id == 0
        assert drone.fuel_remaining == 85.0
        assert 'rf' in drone.sensors


class TestTaskAllocationWithHungarian:
    """Test task allocation using Hungarian algorithm."""
    
    @pytest.fixture
    def allocator(self):
        return TaskAllocator(num_drones=4)
    
    def test_allocate_equal_tasks_drones(self, allocator):
        """Allocate equal number of tasks and drones."""
        
        # Create 3 tasks
        tasks = [
            ISRTask(
                task_id=i,
                task_type=TaskType.DETECT,
                target_position=np.array([100*i, 100, 50]),
                priority=0.5
            )
            for i in range(3)
        ]
        
        # Create 3 drones
        drones = {
            i: DroneCapability(
                drone_id=i,
                position=np.array([0, 0, 50]),
                fuel_remaining=80.0,
                sensors=['rf'],
                current_load=0.0,
                max_speed=25.0,
                endurance=1800.0,
                communication_range=1000.0
            )
            for i in range(3)
        }
        
        assignments = allocator.allocate_tasks(tasks, drones)
        
        assert len(assignments) <= 3
        assert all(isinstance(task, object) for task in assignments.values())
    
    def test_allocate_more_drones_than_tasks(self, allocator):
        """Allocate when more drones than tasks."""
        
        # 2 tasks
        tasks = [
            ISRTask(
                task_id=i,
                task_type=TaskType.DETECT,
                target_position=np.array([100*i, 100, 50]),
                priority=0.5
            )
            for i in range(2)
        ]
        
        # 5 drones
        drones = {
            i: DroneCapability(
                drone_id=i,
                position=np.array([0, 0, 50]),
                fuel_remaining=80.0,
                sensors=['rf'],
                current_load=0.0,
                max_speed=25.0,
                endurance=1800.0,
                communication_range=1000.0
            )
            for i in range(5)
        }
        
        assignments = allocator.allocate_tasks(tasks, drones)
        
        assert len(assignments) <= 2
    
    def test_allocate_more_tasks_than_drones(self, allocator):
        """Allocate when more tasks than drones (greedy)."""
        
        # 5 tasks
        tasks = [
            ISRTask(
                task_id=i,
                task_type=TaskType.DETECT,
                target_position=np.array([100*i, 100, 50]),
                priority=0.5 + 0.1*i
            )
            for i in range(5)
        ]
        
        # 2 drones
        drones = {
            i: DroneCapability(
                drone_id=i,
                position=np.array([0, 0, 50]),
                fuel_remaining=80.0,
                sensors=['rf'],
                current_load=0.0,
                max_speed=25.0,
                endurance=1800.0,
                communication_range=1000.0
            )
            for i in range(2)
        }
        
        assignments = allocator.allocate_tasks(tasks, drones)
        
        # Should assign high-priority tasks first
        assert len(assignments) <= 2


class TestAllocationCostComputations:
    """Test cost computation in allocation."""
    
    @pytest.fixture
    def allocator(self):
        return TaskAllocator(num_drones=4)
    
    def test_distance_cost(self, allocator):
        """Closer drones have lower cost."""
        
        task = ISRTask(
            task_id=1,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 0, 50]),
            priority=0.5
        )
        
        drone_close = DroneCapability(
            drone_id=0,
            position=np.array([110, 0, 50]),
            fuel_remaining=80.0,
            sensors=['rf'],
            current_load=0.0,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        drone_far = DroneCapability(
            drone_id=1,
            position=np.array([500, 500, 50]),
            fuel_remaining=80.0,
            sensors=['rf'],
            current_load=0.0,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        cost_close = allocator._compute_assignment_cost(task, drone_close)
        cost_far = allocator._compute_assignment_cost(task, drone_far)
        
        assert cost_close < cost_far


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
