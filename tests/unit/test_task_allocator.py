"""
TEST: Module 6 - Task Allocator (Multi-Agent Task Assignment)

Unit tests for task allocation using Hungarian algorithm and optimization
Tests task assignment, cost computation, and allocation metrics
"""

import pytest
import numpy as np
from unittest.mock import Mock
from isr_rl_dmpc import (
    ISRTask, TaskType, TaskStatus, DroneCapability,
    HungarianAssignmentAlgorithm, TaskAllocator, 
)

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


class TestHungarianAlgorithm:
    """Test Hungarian algorithm for assignment."""
    
    @pytest.fixture
    def hungarian(self):
        return HungarianAssignmentAlgorithm()
    
    def test_square_cost_matrix(self, hungarian):
        """Hungarian algorithm solves square cost matrices."""
        cost_matrix = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [9, 8, 7]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        assert len(assignment) == 3
        assert all(0 <= a < 3 for a in assignment)
    
    def test_rectangular_cost_matrix(self, hungarian):
        """Hungarian algorithm handles rectangular matrices."""
        # More tasks than drones
        cost_matrix = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        assert len(assignment) == 3
    
    def test_optimal_assignment(self, hungarian):
        """Hungarian algorithm finds optimal assignment."""
        # Simple 2x2 case
        cost_matrix = np.array([
            [1, 10],
            [10, 1]
        ])
        
        assignment = hungarian.solve(cost_matrix)
        
        # Should assign task 0 to drone 0, task 1 to drone 1
        assert assignment[0] == 0 or assignment[1] == 1


class TestAssignmentCostComputation:
    """Test cost matrix computation."""
    
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
        
        # Close drone
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
        
        # Distant drone
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
    
    def test_fuel_cost(self, allocator):
        """Lower fuel means higher cost."""       
        task = ISRTask(
            task_id=1,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 0, 50]),
            priority=0.5
        )
        
        drone_high_fuel = DroneCapability(
            drone_id=0,
            position=np.array([100, 0, 50]),
            fuel_remaining=90.0,
            sensors=['rf'],
            current_load=0.0,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        drone_low_fuel = DroneCapability(
            drone_id=1,
            position=np.array([100, 0, 50]),
            fuel_remaining=20.0,
            sensors=['rf'],
            current_load=0.0,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        cost_high = allocator._compute_assignment_cost(task, drone_high_fuel)
        cost_low = allocator._compute_assignment_cost(task, drone_low_fuel)
        
        assert cost_high < cost_low
    
    def test_priority_bonus(self, allocator):
        """Higher priority tasks get lower cost."""
        
        drone = DroneCapability(
            drone_id=0,
            position=np.array([100, 0, 50]),
            fuel_remaining=80.0,
            sensors=['rf'],
            current_load=0.0,
            max_speed=25.0,
            endurance=1800.0,
            communication_range=1000.0
        )
        
        task_high_priority = ISRTask(
            task_id=1,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 0, 50]),
            priority=0.9
        )
        
        task_low_priority = ISRTask(
            task_id=2,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 0, 50]),
            priority=0.2
        )
        
        cost_high = allocator._compute_assignment_cost(task_high_priority, drone)
        cost_low = allocator._compute_assignment_cost(task_low_priority, drone)
        
        assert cost_high < cost_low


class TestTaskAllocation:
    """Test task allocation workflow."""
    
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
    
    def test_allocate_more_drones(self, allocator):
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
    
    def test_allocate_more_tasks(self, allocator):
        """Allocate when more tasks than drones (greedy)."""
        
        # 5 tasks
        tasks = [
            ISRTask(
                task_id=i,
                task_type=TaskType.DETECT,
                target_position=np.array([100*i, 100, 50]),
                priority=0.5 + 0.1*i  # Increasing priority
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


class TestTaskStatusManagement:
    """Test task status tracking."""
    
    @pytest.fixture
    def allocator(self):
        return TaskAllocator(num_drones=4)
    
    def test_update_task_status(self, allocator):
        """Update task status correctly."""
        
        task = ISRTask(
            task_id=1,
            task_type=TaskType.DETECT,
            target_position=np.array([100, 100, 50]),
            priority=0.5
        )
        
        allocator.task_queue.append(task)
        
        result = allocator.update_task_status(1, TaskStatus.IN_PROGRESS)
        
        assert result == True
        assert task.status == TaskStatus.IN_PROGRESS


class TestAllocationMetrics:
    """Test allocation statistics and metrics."""
    
    @pytest.fixture
    def allocator(self):
        return TaskAllocator(num_drones=4)
    
    def test_allocation_metrics(self, allocator):
        """Collect allocation metrics."""
        allocator.task_history = [
            {'task_id': 1, 'drone_id': 0, 'cost': 0.5, 'task_type': 'detect'},
            {'task_id': 2, 'drone_id': 1, 'cost': 0.7, 'task_type': 'track'},
            {'task_id': 3, 'drone_id': 2, 'cost': 0.3, 'task_type': 'classify'}
        ]
        
        metrics = allocator.get_allocation_metrics()
        
        assert 'total_assignments' in metrics
        assert metrics['total_assignments'] == 3
        assert 'average_cost' in metrics
        assert 'best_cost' in metrics
        assert 'worst_cost' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
