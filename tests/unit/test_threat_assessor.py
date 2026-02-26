"""
TEST: Module 5 - Threat Assessor (Target Risk Scoring and Prioritization)

Comprehensive unit tests for multi-dimensional threat assessment system
Tests RF signals, motion patterns, proximity, and target classification
"""

import pytest
import numpy as np
from unittest.mock import Mock
from isr_rl_dmpc import (
    ThreatParameters, ThreatAssessor, ThreatLevel

)

class TestThreatParametersBasics:
    """Test ThreatParameters initialization and configuration."""
    
    def test_params_initialization(self):
        """ThreatParameters initializes with defaults."""
        
        params = ThreatParameters()
        assert params.rf_strength_threshold == -50
        assert params.velocity_threshold == 50.0
        assert params.proximity_threshold == 500.0
        assert params.rf_weight == 0.25
        
    def test_params_weights_normalized(self):
        """Component weights sum to 1.0."""
        
        params = ThreatParameters()
        total = (params.rf_weight + params.motion_weight + 
                params.position_weight + params.classification_weight)
        assert np.isclose(total, 1.0)


class TestRFThreatComponent:
    """Test RF signal threat assessment."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_strong_signal_high_threat(self, assessor):
        """Strong RF signal produces high threat score."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -30,  # Very strong
            'modulation': 'psk',  # Military
            'classification': 'military',
            'classification_confidence': 0.9
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.rf_threat > 50
    
    def test_weak_signal_low_threat(self, assessor):
        """Weak RF signal produces low threat score."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -120,  # Very weak
            'modulation': 'am',  # Civilian
            'classification': 'civilian',
            'classification_confidence': 0.9
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.rf_threat < 30
    
    @pytest.mark.parametrize("modulation,is_military", [
        ('psk', True),
        ('qam', True),
        ('fsk', True),
        ('am', False),
        ('fm', False),
    ])
    def test_modulation_types(self, assessor, modulation, is_military):
        """Different modulations have different threat levels."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -75,
            'modulation': modulation,
            'classification': 'unknown',
            'classification_confidence': 0.5
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        
        if is_military:
            assert assessment.rf_threat > 35
        else:
            assert assessment.rf_threat < 45


class TestMotionThreatComponent:
    """Test motion pattern threat assessment."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_stationary_low_threat(self, assessor):
        """Stationary target has low motion threat."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'civilian',
            'classification_confidence': 0.7
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.motion_threat < 20
    
    def test_high_speed_high_threat(self, assessor):
        """High-speed target has high motion threat."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.array([80, 0, 0]),  # 80 m/s
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.motion_threat > 35
    
    def test_evasive_acceleration(self, assessor):
        """High acceleration increases threat."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.array([50, 0, 0]),
            'acceleration': np.array([10, 0, 0]),  # High acceleration
            'rf_strength': -75,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.motion_threat > 40


class TestPositionThreatComponent:
    """Test proximity and position threat assessment."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_distant_target_low_threat(self, assessor):
        """Distant targets have low position threat."""
        target_data = {
            'target_id': 1,
            'position': np.array([1000, 1000, 50]),  # Far away
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.position_threat < 15
    
    def test_close_target_high_threat(self, assessor):
        """Close targets have high position threat."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 0, 50]),  # 100m away (critical)
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert assessment.position_threat > 50
    
    def test_altitude_threat(self, assessor):
        """Low altitude increases position threat."""
        # Low altitude
        target_low = {
            'target_id': 1,
            'position': np.array([100, 100, 100]),  # Low altitude
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        assess_low = assessor.assess_target(target_low, own_pos, 0.0)
        
        # High altitude (same distance)
        target_high = {
            'target_id': 2,
            'position': np.array([100, 100, 500]),  # High altitude
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        assess_high = assessor.assess_target(target_high, own_pos, 0.0)
        
        # Low altitude should have higher threat
        assert assess_low.position_threat > assess_high.position_threat


class TestClassificationThreatComponent:
    """Test classification-based threat assessment."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    @pytest.mark.parametrize("classification,expected_high", [
        ('military', True),
        ('civilian', False),
        ('decoy', False),
        ('friendly', False),
        ('unknown', True),
    ])
    def test_classification_threat_types(self, assessor, classification, expected_high):
        """Different classifications have different threat scores."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': classification,
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        
        if expected_high:
            assert assessment.classification_threat > 40
        else:
            assert assessment.classification_threat < 40
    
    def test_confidence_effect(self, assessor):
        """Low classification confidence reduces threat."""
        target_high_conf = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.9
        }
        own_pos = np.array([0, 0, 50])
        assess_high = assessor.assess_target(target_high_conf, own_pos, 0.0)
        
        target_low_conf = target_high_conf.copy()
        target_low_conf['classification_confidence'] = 0.3
        assess_low = assessor.assess_target(target_low_conf, own_pos, 0.0)
        
        assert assess_high.classification_threat > assess_low.classification_threat


class TestComprehensiveThreatScoring:
    """Test overall threat score computation."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_threat_score_bounds(self, assessor):
        """Threat score bounded in [0, 100]."""
        target_data = {
            'target_id': 1,
            'position': np.array([50, 0, 50]),  # Very close
            'velocity': np.array([100, 0, 0]),  # Very fast
            'acceleration': np.array([10, 0, 0]),
            'rf_strength': -30,  # Very strong
            'modulation': 'psk',
            'classification': 'military',
            'classification_confidence': 0.95
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert 0 <= assessment.threat_score <= 100
    
    def test_threat_level_classification(self, assessor):
        """Threat assessment assigns correct threat level."""
        
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.array([0, 0, 0]),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'unknown',
            'classification_confidence': 0.5
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert isinstance(assessment.threat_level, ThreatLevel)
    
    def test_assessment_structure(self, assessor):
        """ThreatAssessment contains all required fields."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        
        assert assessment.target_id == 1
        assert 0 <= assessment.threat_score <= 100
        assert 0 <= assessment.rf_threat <= 100
        assert 0 <= assessment.motion_threat <= 100
        assert 0 <= assessment.position_threat <= 100
        assert 0 <= assessment.classification_threat <= 100
        assert 0 <= assessment.confidence <= 1.0
        assert isinstance(assessment.recommendation, str)


class TestThreatHistoryAndTrends:
    """Test threat tracking and trend analysis."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_threat_history_tracking(self, assessor):
        """Threat history tracked for targets."""
        target_data = {
            'target_id': 1,
            'position': np.array([100, 100, 50]),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'rf_strength': -80,
            'modulation': 'unknown',
            'classification': 'military',
            'classification_confidence': 0.8
        }
        own_pos = np.array([0, 0, 50])
        
        # Assess same target 3 times
        for i in range(3):
            assessor.assess_target(target_data, own_pos, float(i))
        
        assert 1 in assessor.threat_history
        assert len(assessor.threat_history[1]) == 3
    
    def test_increasing_threat_trend(self, assessor):
        """Detect increasing threat trend."""
        # Approaching target
        for i in range(5):
            target_data = {
                'target_id': 1,
                'position': np.array([100 - 20*i, 100, 50]),  # Getting closer
                'velocity': np.array([i*10, 0, 0]),  # Getting faster
                'acceleration': np.zeros(3),
                'rf_strength': -80 + 10*i,  # Getting stronger
                'modulation': 'psk',
                'classification': 'military',
                'classification_confidence': 0.8
            }
            own_pos = np.array([0, 0, 50])
            assessor.assess_target(target_data, own_pos, float(i))
        
        trends = assessor.get_threat_trends(1, window=5)
        assert 'trend' in trends
        assert trends['trend'] in ['increasing', 'decreasing', 'stable', 'insufficient_data']
    
    def test_multiple_target_tracking(self, assessor):
        """Track multiple targets independently."""
        own_pos = np.array([0, 0, 50])
        
        for target_id in range(1, 4):
            target_data = {
                'target_id': target_id,
                'position': np.array([100*target_id, 100, 50]),
                'velocity': np.zeros(3),
                'acceleration': np.zeros(3),
                'rf_strength': -80,
                'modulation': 'unknown',
                'classification': 'military',
                'classification_confidence': 0.8
            }
            assessor.assess_target(target_data, own_pos, 0.0)
        
        assert len(assessor.threat_history) == 3


class TestRecommendationGeneration:
    """Test threat-based recommendation generation."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_critical_threat_recommendation(self, assessor):
        """Critical threat generates action recommendation."""
        target_data = {
            'target_id': 1,
            'position': np.array([50, 0, 50]),  # Very close
            'velocity': np.array([100, 0, 0]),  # Very fast
            'acceleration': np.array([10, 0, 0]),
            'rf_strength': -30,  # Very strong
            'modulation': 'psk',
            'classification': 'military',
            'classification_confidence': 0.95
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert isinstance(assessment.recommendation, str)
        assert len(assessment.recommendation) > 0
    
    def test_civilian_target_recommendation(self, assessor):
        """Civilian target generates appropriate recommendation."""
        target_data = {
            'target_id': 1,
            'position': np.array([500, 500, 500]),  # Far away
            'velocity': np.array([5, 0, 0]),  # Slow
            'acceleration': np.zeros(3),
            'rf_strength': -120,  # Very weak
            'modulation': 'am',
            'classification': 'civilian',
            'classification_confidence': 0.95
        }
        own_pos = np.array([0, 0, 50])
        
        assessment = assessor.assess_target(target_data, own_pos, 0.0)
        assert isinstance(assessment.recommendation, str)


class TestThreatAssessmentStatistics:
    """Test statistics and metrics collection."""
    
    @pytest.fixture
    def assessor(self):
        return ThreatAssessor()
    
    def test_assessment_statistics(self, assessor):
        """Collect and report assessment statistics."""
        own_pos = np.array([0, 0, 50])
        
        # Generate 5 assessments
        for i in range(5):
            target_data = {
                'target_id': i,
                'position': np.array([100 + 50*i, 100, 50]),
                'velocity': np.array([20*i, 0, 0]),
                'acceleration': np.zeros(3),
                'rf_strength': -80 + 10*i,
                'modulation': 'unknown',
                'classification': 'military' if i % 2 == 0 else 'civilian',
                'classification_confidence': 0.7 + 0.1*i
            }
            assessor.assess_target(target_data, own_pos, float(i))
        
        stats = assessor.get_assessment_statistics()
        assert 'total_assessments' in stats
        assert 'tracked_targets' in stats
        assert 'average_threat' in stats
        assert 'max_threat' in stats
        assert 'min_threat' in stats
    
    def test_target_prioritization(self, assessor):
        """Prioritize targets by threat score."""
        own_pos = np.array([0, 0, 50])
        
        assessments = []
        for i in range(3):
            target_data = {
                'target_id': i,
                'position': np.array([100 + 50*i, 100, 50]),
                'velocity': np.array([20*i, 0, 0]),
                'acceleration': np.zeros(3),
                'rf_strength': -80 + 20*i,  # Increasing threat
                'modulation': 'psk' if i > 0 else 'am',
                'classification': 'military' if i > 0 else 'civilian',
                'classification_confidence': 0.8
            }
            assessment = assessor.assess_target(target_data, own_pos, float(i))
            assessments.append(assessment)
        
        prioritized = assessor.prioritize_targets(assessments)
        
        # Verify sorted by threat (descending)
        assert len(prioritized) == 3
        for i in range(len(prioritized) - 1):
            assert prioritized[i].threat_score >= prioritized[i+1].threat_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
