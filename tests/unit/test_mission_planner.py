"""
TEST: Module 1 - Mission Planner (Grid Decomposition & Waypoint Generation)

Focused unit tests for grid-based mission planning
"""

import pytest
import numpy as np
from unittest.mock import Mock


class TestGridDecomposerBasics:
    """Grid decomposition fundamental behavior."""
    
    @pytest.fixture
    def rectangular_area(self):
        """100x100m rectangular mission area."""
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    
    def test_grid_creates_correct_cell_count(self, rectangular_area):
        """20m grid on 100x100 area creates 5x5 grid (25 cells)."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        cells = decomposer.decompose_grid(n_drones=4)
        
        assert len(cells) == 25
    
    def test_cell_centers_inside_area_bounds(self, rectangular_area):
        """All cell centers must be within mission area."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        cells = decomposer.decompose_grid(n_drones=4)
        
        minx, miny = np.min(rectangular_area, axis=0)
        maxx, maxy = np.max(rectangular_area, axis=0)
        
        for cell in cells:
            assert minx <= cell.center[0] <= maxx
            assert miny <= cell.center[1] <= maxy
    
    def test_cells_have_positive_area(self, rectangular_area):
        """Grid cells have positive area."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        cells = decomposer.decompose_grid(n_drones=4)
        
        for cell in cells:
            assert cell.area > 0
    
    def test_priority_values_normalized(self, rectangular_area):
        """Cell priorities bounded in [0.1, 1.0]."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        cells = decomposer.decompose_grid(n_drones=4)
        
        for cell in cells:
            assert 0.1 <= cell.priority <= 1.0


class TestGridDecomposerPriority:
    """Priority computation for cells."""
    
    @pytest.fixture
    def rectangular_area(self):
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    
    def test_cells_sorted_by_priority_descending(self, rectangular_area):
        """get_cells_by_priority returns sorted descending."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        decomposer.decompose_grid(n_drones=4)
        
        sorted_cells = decomposer.get_cells_by_priority()
        priorities = [c.priority for c in sorted_cells]
        
        # Should be non-increasing
        assert all(priorities[i] >= priorities[i+1] for i in range(len(priorities)-1))
    
    def test_center_cells_higher_priority_than_edges(self, rectangular_area):
        """Cells near center have higher priority than edge cells."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=20.0)
        cells = decomposer.decompose_grid(n_drones=4)
        
        center_point = np.array([50.0, 50.0])
        edge_point = np.array([0.0, 0.0])
        
        center_prio = decomposer._compute_cell_prio(center_point)
        edge_prio = decomposer._compute_cell_prio(edge_point)
        
        assert center_prio > edge_prio


class TestWaypointGeneratorBasics:
    """Waypoint path generation."""
    
    @pytest.fixture
    def sample_cells(self, rectangular_area):
        """Create 4 simple grid cells."""
        from isr_rl_dmpc import GridCell
        return [
            GridCell(0, np.array([[0, 0], [20, 0], [20, 20], [0, 20]]), None, 400, 0.8),
            GridCell(1, np.array([[20, 0], [40, 0], [40, 20], [20, 20]]), None, 400, 0.7),
            GridCell(2, np.array([[0, 20], [20, 20], [20, 40], [0, 40]]), None, 400, 0.6),
            GridCell(3, np.array([[20, 20], [40, 20], [40, 40], [20, 40]]), None, 400, 0.5),
        ]
    
    @pytest.fixture
    def rectangular_area(self):
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    
    def test_waypoint_path_has_altitude(self, sample_cells):
        """Generated waypoints include altitude dimension."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0)
        
        path = gen.generate_path(sample_cells, np.array([0, 0]), opt_strat='nearest')
        
        assert path.shape[1] == 3  # [x, y, z]
        assert np.all(path[:, 2] == 50.0)  # All at altitude
    
    def test_waypoint_path_visits_all_cells(self, sample_cells):
        """Path includes waypoint for each cell."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0)
        
        path = gen.generate_path(sample_cells, np.array([0, 0]), opt_strat='nearest')
        
        assert len(path) == len(sample_cells)
    
    @pytest.mark.parametrize("strategy", ["nearest", "sweep", "spiral"])
    def test_all_path_strategies_work(self, sample_cells, strategy):
        """All path optimization strategies produce valid paths."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0)
        
        path = gen.generate_path(sample_cells, np.array([0, 0]), opt_strat=strategy)
        
        assert path.shape == (len(sample_cells), 3)


class TestMissionTime:
    """Mission time estimation."""
    
    def test_mission_time_positive(self):
        """Mission time is always positive."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0, hover_time=5.0)
        
        waypoints = np.array([[0, 0, 50], [50, 0, 50], [100, 100, 50]])
        time = gen.estimate_mission_time(waypoints, drone_speed=5.0)
        
        assert time > 0
    
    def test_mission_time_scales_with_distance(self):
        """Longer paths take more time."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0, hover_time=5.0)
        
        short_path = np.array([[0, 0, 50], [10, 0, 50]])
        long_path = np.array([[0, 0, 50], [100, 100, 50]])
        
        time_short = gen.estimate_mission_time(short_path, drone_speed=5.0)
        time_long = gen.estimate_mission_time(long_path, drone_speed=5.0)
        
        assert time_long > time_short
    
    def test_mission_time_includes_hover(self):
        """Mission time includes hover time at waypoints."""
        from isr_rl_dmpc import WaypointGenerator
        gen = WaypointGenerator(altitude=50.0, hover_time=10.0)
        
        waypoints = np.array([[0, 0, 50]])  # Single waypoint, no travel
        time = gen.estimate_mission_time(waypoints, drone_speed=5.0)
        
        # Should include at least hover time
        assert time >= 10.0


class TestResolutionParameterization:
    """Test with different grid resolutions."""
    
    @pytest.fixture
    def rectangular_area(self):
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    
    @pytest.mark.parametrize("resolution", [10.0, 20.0, 50.0])
    def test_different_resolutions_create_correct_cells(self, rectangular_area, resolution):
        """Grid resolution affects cell count correctly."""
        from isr_rl_dmpc import GridDecomposer
        decomposer = GridDecomposer(rectangular_area, grid_resolution=resolution)
        cells = decomposer.decompose_grid(n_drones=4)
        
        expected = int((100 / resolution) ** 2)
        assert len(cells) == expected
    
    def test_finer_resolution_creates_more_cells(self, rectangular_area):
        """Finer resolution (smaller cells) creates more cells."""
        from isr_rl_dmpc import GridDecomposer
        decomp_coarse = GridDecomposer(rectangular_area, grid_resolution=20.0)
        decomp_fine = GridDecomposer(rectangular_area, grid_resolution=10.0)
        
        cells_coarse = decomp_coarse.decompose_grid(n_drones=4)
        cells_fine = decomp_fine.decompose_grid(n_drones=4)
        
        assert len(cells_fine) > len(cells_coarse)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
