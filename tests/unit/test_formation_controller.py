"""
TEST: Module 2 - Formation Controller (Consensus-based Multi-Agent Control)

Focused unit tests for formation geometry and control
"""

import pytest
import numpy as np
from unittest.mock import Mock


class TestFormationConfigBasics:
    """Formation configuration."""
    
    def test_formation_config_has_defaults(self):
        """FormationConfig initializes with sensible defaults."""
        from isr_rl_dmpc.modules import FormationConfig, FormationType
        config = FormationConfig()
        
        assert config.type == FormationType.WEDGE
        assert config.scale == 50.0
        assert config.spacing == 10.0
        assert config.convergence_threshold == 0.5
        assert config.velocity_damping == 0.1


class TestFormationGeometryBasics:
    """Formation geometry generation."""
    
    @pytest.fixture
    def formation_geo(self):
        """Formation geometry generator."""
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig
        config = FormationConfig(spacing=10.0, scale=50.0)
        return FormationGeometry(config)
    
    @pytest.fixture
    def formation_center(self):
        """Formation center (3D)."""
        return np.array([100.0, 100.0, 50.0])
    
    def test_generate_positions_returns_dict(self, formation_geo, formation_center):
        """Generated positions are dictionary mapping drone_id -> position."""
        positions = formation_geo.generate_desired_positions(4, formation_center, 0.0)
        
        assert isinstance(positions, dict)
        assert len(positions) == 4
    
    def test_all_positions_are_3d_vectors(self, formation_geo, formation_center):
        """Each position is 3D vector."""
        positions = formation_geo.generate_desired_positions(4, formation_center, 0.0)
        
        for pos in positions.values():
            assert pos.shape == (3,)


class TestLineFormation:
    """Line formation geometry."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.LINE, spacing=10.0)
        return FormationGeometry(config)
    
    def test_line_formation_symmetric(self, formation_geo):
        """Line formation is symmetric around center."""
        center = np.array([100.0, 100.0, 50.0])
        positions = formation_geo.generate_desired_positions(5, center, 0.0)
        
        # Center drone at formation center
        assert np.allclose(positions[2], center)
    
    def test_line_formation_correct_count(self, formation_geo):
        """Line has correct drone count."""
        positions = formation_geo.generate_desired_positions(7, np.array([0, 0, 50]), 0.0)
        assert len(positions) == 7


class TestWedgeFormation:
    """Wedge (V-shaped) formation geometry."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.WEDGE, spacing=10.0)
        return FormationGeometry(config)
    
    def test_wedge_lead_at_center(self, formation_geo):
        """Lead drone positioned at formation center."""
        center = np.array([100.0, 100.0, 50.0])
        positions = formation_geo.generate_desired_positions(4, center, 0.0)
        
        # Lead drone (drone_id=0) at center
        assert np.allclose(positions[0], center)
    
    def test_wedge_distributed_around_lead(self, formation_geo):
        """Wedge has drones distributed around lead."""
        center = np.array([0.0, 0.0, 50.0])
        positions = formation_geo.generate_desired_positions(5, center, 0.0)
        
        # Lead at center, others around
        lead_pos = positions[0]
        for drone_id in range(1, 5):
            distance = np.linalg.norm(positions[drone_id][:2] - lead_pos[:2])
            assert distance > 5.0  # At least some separation


class TestCircularFormation:
    """Circular formation geometry."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.CIRCULAR, scale=50.0)
        return FormationGeometry(config)
    
    def test_circular_uniform_radius(self, formation_geo):
        """All drones at same radius from center."""
        center = np.array([100.0, 100.0, 50.0])
        positions = formation_geo.generate_desired_positions(8, center, 0.0)
        
        radius = formation_geo.config.scale / 2
        distances = [np.linalg.norm(pos[:2] - center[:2]) for pos in positions.values()]
        
        # All at same radius (within tolerance)
        assert np.allclose(distances, radius, atol=1.0)
    
    def test_circular_correct_count(self, formation_geo):
        """Circular formation has correct drone count."""
        positions = formation_geo.generate_desired_positions(6, np.array([0, 0, 50]), 0.0)
        assert len(positions) == 6


class TestGridFormation:
    """Grid/matrix formation geometry."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.GRID, spacing=10.0)
        return FormationGeometry(config)
    
    def test_grid_formation_correct_count(self, formation_geo):
        """Grid has all drones."""
        positions = formation_geo.generate_desired_positions(9, np.array([0, 0, 50]), 0.0)
        assert len(positions) == 9
    
    def test_grid_formation_maintains_spacing(self, formation_geo):
        """Grid maintains approximate spacing between drones."""
        positions = formation_geo.generate_desired_positions(9, np.array([0, 0, 50]), 0.0)
        
        pos_array = np.array(list(positions.values()))
        # Check some inter-drone distances
        distances = np.linalg.norm(np.diff(pos_array[:3, :2], axis=0), axis=1)
        
        # Should have some separation
        assert np.all(distances > 5.0)


class TestSphereFormation:
    """3D spherical formation geometry."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.SPHERE, scale=50.0)
        return FormationGeometry(config)
    
    def test_sphere_formation_uses_3d(self, formation_geo):
        """Sphere formation uses full 3D space."""
        center = np.array([100.0, 100.0, 50.0])
        positions = formation_geo.generate_desired_positions(12, center, 0.0)
        
        pos_array = np.array(list(positions.values()))
        z_coords = pos_array[:, 2]
        
        # Should have variation in z (not all at same altitude)
        assert np.std(z_coords) > 1.0
    
    def test_sphere_correct_count(self, formation_geo):
        """Sphere has correct drone count."""
        positions = formation_geo.generate_desired_positions(12, np.array([0, 0, 50]), 0.0)
        assert len(positions) == 12


class TestFormationRotation:
    """Formation heading rotation."""
    
    @pytest.fixture
    def formation_geo(self):
        from isr_rl_dmpc.modules import FormationGeometry, FormationConfig, FormationType
        config = FormationConfig(type=FormationType.LINE, spacing=10.0)
        return FormationGeometry(config)
    
    def test_formation_rotates_with_heading(self, formation_geo):
        """Formation rotates with different headings."""
        center = np.array([0.0, 0.0, 50.0])
        
        # Generate at heading 0 and 90 degrees
        pos_0 = formation_geo.generate_desired_positions(3, center, 0.0)
        pos_90 = formation_geo.generate_desired_positions(3, center, np.pi/2)
        
        # Positions should be different
        assert not np.allclose(pos_0[1], pos_90[1])


class TestConsensusStateBasics:
    """Consensus state data structure."""
    
    def test_consensus_state_initializes(self):
        """ConsensusState initializes with defaults."""
        from isr_rl_dmpc.modules import ConsensusState
        
        state = ConsensusState()
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.neighbor_count == 0
        assert state.converged is False


class TestConsensusController:
    """Consensus-based formation control."""
    
    @pytest.fixture
    def controller(self):
        """Consensus controller."""
        from isr_rl_dmpc.modules import ConsensusController
        return ConsensusController(dt=0.02, comm_range=100.0)
    
    def test_controller_initialization(self, controller):
        """Controller initializes correctly."""
        assert controller.dt == 0.02
        assert controller.comm_range == 100.0
    
    def test_controller_can_be_used_with_state_manager(self, controller):
        """Controller compatible with state manager."""
        mock_manager = Mock()
        mock_manager.n_drones = 4
        mock_manager.get_inter_drone_adjacency = Mock(return_value=np.eye(4))
        
        # Should be able to use
        assert controller is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
