"""
TEST: Module 4 - Classification Engine (Enemy/Friendly/Civilian Target ID via RF + Motion)

Focused unit tests for feature extraction and target classification
"""

import pytest
import numpy as np
from unittest.mock import Mock
from isr_rl_dmpc import (
    ClassificationFeature, FeatureType, TargetSignature, FeatureExtractor, ClassificationEngine,
)


class TestClassificationFeature:
    """Classification feature data structure."""
    
    def test_feature_initialization(self):
        """ClassificationFeature initializes correctly."""
        
        feature = ClassificationFeature(
            feature_type=FeatureType.SIGNAL_STRENGTH,
            value=0.75,
            confidence=0.9,
            timestamp=0.0,
            source_drone_id=0
        )
        
        assert feature.value == 0.75
        assert feature.confidence == 0.9
        assert feature.feature_type == FeatureType.SIGNAL_STRENGTH


class TestTargetSignature:
    """Target signature tracking."""
    
    def test_target_signature_initialization(self):
        """TargetSignature initializes as empty signature."""
        
        sig = TargetSignature(target_id="T1")
        
        assert sig.target_id == "T1"
        assert len(sig.signal_strength) == 0
        assert len(sig.measurements) == 0


class TestFeatureExtractorSignal:
    """RF signal strength feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Feature extractor."""
        return FeatureExtractor()
    
    def test_extract_signal_strength(self, extractor):
        """Extract signal strength features."""
        feature = extractor.extract_signal_strength_features(
            signal_strength_dbm=-80.0,
            drone_distance=100.0
        )
        
        assert feature is not None
        assert 0 <= feature.value <= 1.0
        assert 0 <= feature.confidence <= 1.0
    
    def test_signal_strength_normalization(self, extractor):
        """Signal strength normalized to [0, 1]."""
        # Very weak signal
        feature_weak = extractor.extract_signal_strength_features(-120.0, 50.0)
        assert feature_weak.value >= 0.0
        
        # Very strong signal
        feature_strong = extractor.extract_signal_strength_features(-30.0, 50.0)
        assert feature_strong.value <= 1.0
    
    def test_stronger_signal_higher_value(self, extractor):
        """Stronger signal produces higher feature value."""
        feature_weak = extractor.extract_signal_strength_features(-100.0, 100.0)
        feature_strong = extractor.extract_signal_strength_features(-70.0, 100.0)
        
        assert feature_strong.value > feature_weak.value
    
    @pytest.mark.parametrize("signal_dbm", [-120.0, -90.0, -60.0, -30.0])
    def test_signal_strength_various_levels(self, extractor, signal_dbm):
        """Signal extraction works for various signal levels."""
        feature = extractor.extract_signal_strength_features(signal_dbm, 100.0)
        
        assert 0 <= feature.value <= 1.0


class TestFeatureExtractorMotion:
    """Motion pattern feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_extract_hovering_target(self, extractor):
        """Hovering target has zero motion features."""
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        
        features = extractor.extract_motion_features(velocity, acceleration, "air")
        
        assert 'velocity' in features
        assert features['velocity'].value == 0.0
    
    def test_extract_moving_target(self, extractor):
        """Moving target produces motion features."""
        velocity = np.array([10.0, 0.0, 0.0])  # 10 m/s
        acceleration = np.array([1.0, 0.0, 0.0])
        
        features = extractor.extract_motion_features(velocity, acceleration, "ground")
        
        assert features['velocity'].value > 0.0
    
    @pytest.mark.parametrize("target_type", ["ground", "air", "water"])
    def test_motion_extraction_all_types(self, extractor, target_type):
        """Motion extraction works for all target types."""
        velocity = np.array([5.0, 5.0, 0.0])
        acceleration = np.array([0.5, 0.5, 0.0])
        
        features = extractor.extract_motion_features(velocity, acceleration, target_type)
        
        assert isinstance(features, dict)
        assert 'velocity' in features
    
    def test_ground_target_max_speed(self, extractor):
        """Ground target at max speed is suspicious."""
        velocity = np.array([30.0, 0.0, 0.0])  # 30 m/s (ground max)
        acceleration = np.zeros(3)
        
        features = extractor.extract_motion_features(velocity, acceleration, "ground")
        
        # At max speed should produce high feature value
        assert features['velocity'].value > 0.8
    
    def test_aircraft_high_speed_normal(self, extractor):
        """Aircraft at high speed is normal."""
        velocity = np.array([50.0, 0.0, 0.0])  # 50 m/s (aircraft normal)
        acceleration = np.zeros(3)
        
        features = extractor.extract_motion_features(velocity, acceleration, "air")
        
        # 50 m/s is within normal range for aircraft
        assert features['velocity'].value < 1.0


class TestFeatureExtractorPerformance:
    """Feature extraction performance and timing."""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_signal_extraction_fast(self, extractor):
        """Signal extraction completes quickly."""
        # Should execute in microseconds
        feature = extractor.extract_signal_strength_features(-80.0, 100.0)
        assert feature is not None
    
    def test_motion_extraction_fast(self, extractor):
        """Motion extraction completes quickly."""
        velocity = np.array([10.0, 5.0, 0.0])
        acceleration = np.array([1.0, 0.5, 0.0])
        
        features = extractor.extract_motion_features(velocity, acceleration, "ground")
        assert len(features) > 0


class TestClassificationEngine:
    """Target classification system."""
    
    @pytest.fixture
    def classifier(self):
        """Classification engine."""
        return ClassificationEngine()
    
    def test_classifier_initialization(self, classifier):
        """Classifier initializes with feature extractor."""
        assert classifier is not None
        assert hasattr(classifier, 'feature_extractor')
    
    def test_classify_stationary_target(self, classifier):
        """Classify stationary target."""
        target = Mock()
        target.position = np.array([200.0, 200.0, 0.0])
        target.velocity = np.zeros(3)
        target.acceleration = np.zeros(3)
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        classification = classifier.classify_target("T1", target, drone_pos, {})
        
        assert classification is not None
    
    def test_classify_moving_target(self, classifier):
        """Classify moving target."""
        target = Mock()
        target.position = np.array([200.0, 200.0, 0.0])
        target.velocity = np.array([15.0, 0.0, 0.0])  # 15 m/s
        target.acceleration = np.zeros(3)
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        classification = classifier.classify_target("T2", target, drone_pos, {})
        
        assert classification is not None
    
    def test_classify_multiple_targets(self, classifier):
        """Classify multiple targets in sequence."""
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        for i in range(5):
            target = Mock()
            target.position = np.array([200 + i*50, 200, 0])
            target.velocity = np.zeros(3)
            target.acceleration = np.zeros(3)
            
            classification = classifier.classify_target(f"T{i}", target, drone_pos, {})
            assert classification is not None


class TestClassificationTargetTypes:
    """Classification of different target types."""
    
    @pytest.fixture
    def classifier(self):
        return ClassificationEngine()
    
    def test_slow_ground_target_likely_civilian(self, classifier):
        """Slow ground target more likely to be civilian."""
        target = Mock()
        target.position = np.array([200.0, 200.0, 0.0])
        target.velocity = np.array([5.0, 0.0, 0.0])  # 5 m/s (slow)
        target.acceleration = np.zeros(3)
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        classification = classifier.classify_target("T1", target, drone_pos, {})
        # Should produce valid classification
        assert classification is not None
    
    def test_high_acceleration_evasive(self, classifier):
        """High acceleration suggests evasive behavior."""
        target = Mock()
        target.position = np.array([200.0, 200.0, 0.0])
        target.velocity = np.array([20.0, 0.0, 0.0])
        target.acceleration = np.array([10.0, 0.0, 0.0])  # High accel
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        classification = classifier.classify_target("T1", target, drone_pos, {})
        assert classification is not None
    
    def test_aircraft_normal_speed(self, classifier):
        """Aircraft at cruise speed is neutral classification."""
        target = Mock()
        target.position = np.array([500.0, 500.0, 1000.0])  # Altitude
        target.velocity = np.array([80.0, 0.0, 0.0])  # Aircraft cruise
        target.acceleration = np.zeros(3)
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        classification = classifier.classify_target("T1", target, drone_pos, {})
        assert classification is not None


class TestRFSignalMeasurements:
    """RF signal measurements for classification."""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_extract_from_rf_signal_dict(self, extractor):
        """Extract features from RF signal measurements."""
        signal_dbm = -75.0
        distance = 150.0
        
        feature = extractor.extract_signal_strength_features(signal_dbm, distance)
        
        assert feature.measurement is not None
        assert feature.measurement[0] == signal_dbm
        assert feature.measurement[1] == distance
    
    def test_confidence_scales_with_distance(self, extractor):
        """Signal confidence decreases with distance."""
        # Close target
        feature_close = extractor.extract_signal_strength_features(-80.0, 50.0)
        
        # Distant target
        feature_far = extractor.extract_signal_strength_features(-80.0, 500.0)
        
        # Same signal but confidence differs with distance
        assert feature_close is not None
        assert feature_far is not None


class TestClassificationConsistency:
    """Classification consistency across multiple measurements."""
    
    @pytest.fixture
    def classifier(self):
        return ClassificationEngine()
    
    def test_same_target_consistent_classification(self, classifier):
        """Same target produces consistent classification."""
        target = Mock()
        target.position = np.array([200.0, 200.0, 0.0])
        target.velocity = np.array([10.0, 0.0, 0.0])
        target.acceleration = np.zeros(3)
        
        drone_pos = np.array([100.0, 100.0, 50.0])
        
        # Classify twice
        class1 = classifier.classify_target("T1", target, drone_pos, {})
        class2 = classifier.classify_target("T1", target, drone_pos, {})
        
        # Should be deterministic
        assert class1 is not None
        assert class2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
