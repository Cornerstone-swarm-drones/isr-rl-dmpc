"""
Module 4: Classification Engine - Enemy/Friendly Target Identification

Implements multi-feature classification for target identification using
signal strength, motion patterns, and visual features with uncertainty quantification.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..core import TargetState, DroneState
from ..utils import UnitConversions, BearingDistance, PerformanceLogger, NumericalOps


class TargetClassification(Enum):
    """Target classification categories."""
    UNKNOWN = "unknown"        # Unknown/unclassified
    CIVILIAN = "civilian"      # Civilian target
    MILITARY = "military"      # Military target
    FRIENDLY = "friendly"      # Friendly force
    DECOY = "decoy"           # Decoy/false target
    DEBRIS = "debris"         # Non-target debris


class FeatureType(Enum):
    """Feature types for classification."""
    SIGNAL_STRENGTH = "signal"      # RF signal strength
    MOTION_PATTERN = "motion"       # Velocity/acceleration pattern
    VISUAL_FEATURES = "visual"      # Visual appearance features
    THERMAL_FEATURES = "thermal"    # Thermal signature
    RADAR_SIGNATURE = "radar"       # Radar RCS
    SIZE_ESTIMATE = "size"          # Estimated target size


@dataclass
class ClassificationFeature:
    """Single classification feature."""
    feature_type: FeatureType
    value: float                  # Feature value (normalized 0-1 where possible)
    confidence: float             # Feature confidence (0-1)
    timestamp: float              # Measurement time
    source_drone_id: int          # Which drone measured this
    measurement: np.ndarray = field(default_factory=lambda: np.array([]))  # Raw measurement


@dataclass
class TargetSignature:
    """Collected signature for a target."""
    target_id: str
    signal_strength: List[float] = field(default_factory=list)      # dBm history
    velocity_magnitude: List[float] = field(default_factory=list)   # Speed history
    acceleration_magnitude: List[float] = field(default_factory=list) # Accel history
    visual_features: List[np.ndarray] = field(default_factory=list) # Visual descriptors
    measurements: List[ClassificationFeature] = field(default_factory=list)
    classification_history: List[Tuple[float, TargetClassification]] = field(default_factory=list)


class FeatureExtractor:
    """Extract classification features from drone and target measurements."""

    def __init__(self):
        """Initialize feature extractor."""
        self.perf_logger = PerformanceLogger('feature_extraction')

    def extract_signal_strength_features(self, signal_strength_dbm: float,
                                        drone_distance: float) -> ClassificationFeature:
        """
        Extract signal strength features.

        Args:
            signal_strength_dbm (float): Signal strength in dBm
            drone_distance (float): Distance to target (meters)

        Returns:
            ClassificationFeature for signal strength
        """
        # Normalize signal strength (-120 dBm = 0, -30 dBm = 1)
        normalized = np.clip((signal_strength_dbm + 120) / 90, 0, 1)

        # Free-space path loss model suggests exponential decay
        # Strong signals at close range suggest active emitter
        signal_quality = normalized

        return ClassificationFeature(
            feature_type=FeatureType.SIGNAL_STRENGTH,
            value=signal_quality,
            confidence=min(1.0, drone_distance / 100),  # More confident at longer ranges
            timestamp=0.0,
            source_drone_id=0,
            measurement=np.array([signal_strength_dbm, drone_distance])
        )

    def extract_motion_features(self, velocity: np.ndarray, acceleration: np.ndarray,
                               target_type: str = "ground") -> Dict[str, ClassificationFeature]:
        """
        Extract motion pattern features.

        Args:
            velocity (np.ndarray): Target velocity (3D, m/s)
            acceleration (np.ndarray): Target acceleration (3D, m/s^2)
            target_type (str): Type of target ('ground', 'air', 'water')

        Returns:
            Dictionary of motion features
        """
        features = {}

        # Velocity magnitude
        v_mag = np.linalg.norm(velocity)

        # Expected ranges for different target types
        if target_type == 'ground':
            max_speed = 30.0  # Ground vehicle max ~30 m/s (~108 km/h)
            max_accel = 5.0
        elif target_type == 'air':
            max_speed = 100.0  # Aircraft max ~100 m/s
            max_accel = 20.0
        else:  # water
            max_speed = 15.0
            max_accel = 2.0

        # Normalize velocity (suspicious if exceeds normal range)
        v_normalized = np.clip(v_mag / max_speed, 0, 1)

        features['velocity'] = ClassificationFeature(
            feature_type=FeatureType.MOTION_PATTERN,
            value=v_normalized,
            confidence=0.8,
            timestamp=0.0,
            source_drone_id=0,
            measurement=velocity
        )

        # Acceleration magnitude
        a_mag = np.linalg.norm(acceleration)
        a_normalized = np.clip(a_mag / max_accel, 0, 1)

        # High acceleration suggests military/threat
        features['acceleration'] = ClassificationFeature(
            feature_type=FeatureType.MOTION_PATTERN,
            value=a_normalized,
            confidence=0.7,
            timestamp=0.0,
            source_drone_id=0,
            measurement=acceleration
        )

        # Jerk (rate of change of acceleration) - higher suggests evasion
        # Would need acceleration history for proper computation

        return features

    def extract_size_estimate(self, visual_detection: np.ndarray,
                             drone_distance: float) -> ClassificationFeature:
        """
        Extract size estimate from visual detection.

        Args:
            visual_detection (np.ndarray): Visual bounding box or features
            drone_distance (float): Distance to target (meters)

        Returns:
            ClassificationFeature for size
        """
        # Visual detection typically: [bbox_width, bbox_height] in pixels
        if len(visual_detection) < 2:
            visual_detection = np.array([0, 0])

        # Estimate physical size assuming ~50 degree FOV and standard camera
        pixel_size = np.mean(visual_detection)
        
        # Rough angle-to-meters conversion
        angle_radians = pixel_size / 1000  # Simplified mapping
        estimated_size = 2 * drone_distance * np.tan(angle_radians)

        # Classify by size (military vs civilian thresholds)
        # Small: <2m (person, small UGV)
        # Medium: 2-8m (car, truck)
        # Large: >8m (military vehicle, aircraft)
        size_normalized = np.clip(estimated_size / 10, 0, 1)

        return ClassificationFeature(
            feature_type=FeatureType.SIZE_ESTIMATE,
            value=size_normalized,
            confidence=0.6,  # Size estimation from visual is less reliable
            timestamp=0.0,
            source_drone_id=0,
            measurement=np.array([estimated_size])
        )


class BayesianClassifier:
    """Bayesian multi-feature classifier for target identification."""

    def __init__(self):
        """Initialize classifier with prior probabilities."""
        # Prior probabilities for each class
        self.priors: Dict[TargetClassification, float] = {
            TargetClassification.UNKNOWN: 0.3,
            TargetClassification.CIVILIAN: 0.4,
            TargetClassification.MILITARY: 0.15,
            TargetClassification.FRIENDLY: 0.1,
            TargetClassification.DECOY: 0.03,
            TargetClassification.DEBRIS: 0.02
        }

        # Likelihood matrices (feature value -> class probability)
        self.likelihoods: Dict[FeatureType, Dict] = self._init_likelihoods()

    def _init_likelihoods(self) -> Dict[FeatureType, Dict]:
        """Initialize feature likelihoods for each class."""
        return {
            FeatureType.SIGNAL_STRENGTH: {
                TargetClassification.MILITARY: self._likelihood_military_signal,
                TargetClassification.CIVILIAN: self._likelihood_civilian_signal,
                TargetClassification.FRIENDLY: self._likelihood_friendly_signal,
                TargetClassification.UNKNOWN: self._likelihood_unknown_signal
            },
            FeatureType.MOTION_PATTERN: {
                TargetClassification.MILITARY: self._likelihood_military_motion,
                TargetClassification.CIVILIAN: self._likelihood_civilian_motion,
                TargetClassification.FRIENDLY: self._likelihood_friendly_motion,
                TargetClassification.UNKNOWN: self._likelihood_unknown_motion
            },
            FeatureType.SIZE_ESTIMATE: {
                TargetClassification.MILITARY: self._likelihood_military_size,
                TargetClassification.CIVILIAN: self._likelihood_civilian_size,
                TargetClassification.FRIENDLY: self._likelihood_friendly_size,
                TargetClassification.UNKNOWN: self._likelihood_unknown_size
            }
        }

    # Likelihood functions for signal strength feature
    def _likelihood_military_signal(self, feature_value: float) -> float:
        """Military targets: strong encrypted signals."""
        return 0.8 * np.exp(-0.5 * ((feature_value - 0.7) / 0.2) ** 2)

    def _likelihood_civilian_signal(self, feature_value: float) -> float:
        """Civilian: weak or no military-type signals."""
        return 0.9 * np.exp(-0.5 * ((feature_value - 0.3) / 0.3) ** 2)

    def _likelihood_friendly_signal(self, feature_value: float) -> float:
        """Friendly: known signature pattern."""
        return 0.7 * (1 - np.abs(feature_value - 0.5))

    def _likelihood_unknown_signal(self, feature_value: float) -> float:
        """Unknown: uniform."""
        return 0.5

    # Likelihood functions for motion pattern
    def _likelihood_military_motion(self, feature_value: float) -> float:
        """Military: high acceleration (evasive maneuvers)."""
        return 0.85 * np.exp(-0.5 * ((feature_value - 0.8) / 0.15) ** 2)

    def _likelihood_civilian_motion(self, feature_value: float) -> float:
        """Civilian: moderate speeds and accelerations."""
        return 0.8 * np.exp(-0.5 * ((feature_value - 0.4) / 0.2) ** 2)

    def _likelihood_friendly_motion(self, feature_value: float) -> float:
        """Friendly: coordinated, predictable patterns."""
        return 0.75 * np.exp(-0.5 * ((feature_value - 0.35) / 0.15) ** 2)

    def _likelihood_unknown_motion(self, feature_value: float) -> float:
        """Unknown: uniform."""
        return 0.5

    # Likelihood functions for size
    def _likelihood_military_size(self, feature_value: float) -> float:
        """Military: large vehicles/equipment."""
        return 0.8 * np.exp(-0.5 * ((feature_value - 0.7) / 0.2) ** 2)

    def _likelihood_civilian_size(self, feature_value: float) -> float:
        """Civilian: varied sizes."""
        return 0.7 * np.exp(-0.5 * ((feature_value - 0.5) / 0.25) ** 2)

    def _likelihood_friendly_size(self, feature_value: float) -> float:
        """Friendly: known platform sizes."""
        return 0.7 * (1 - np.abs(feature_value - 0.6))

    def _likelihood_unknown_size(self, feature_value: float) -> float:
        """Unknown: uniform."""
        return 0.5

    def classify(self, features: List[ClassificationFeature]) -> Tuple[TargetClassification, Dict[TargetClassification, float]]:
        """
        Classify target using Bayesian inference on features.

        Args:
            features (List[ClassificationFeature]): Extracted features

        Returns:
            Tuple of (best_classification, class_posteriors_dict)
        """
        # Initialize posteriors with priors
        posteriors = self.priors.copy()

        # Process each feature
        for feature in features:
            if feature.confidence < 0.1:  # Skip very uncertain features
                continue

            feature_likelihoods = self.likelihoods.get(feature.feature_type)
            if feature_likelihoods is None:
                continue

            # Update posteriors using Bayes rule
            for target_class in TargetClassification:
                if target_class == TargetClassification.DEBRIS:
                    continue  # Special handling for debris

                likelihood_fn = feature_likelihoods.get(target_class)
                if likelihood_fn is None:
                    continue

                # P(C|F) ∝ P(F|C) * P(C)
                likelihood = likelihood_fn(feature.value)
                posteriors[target_class] *= (likelihood * feature.confidence)

        # Normalize posteriors to sum to 1
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        else:
            posteriors = self.priors.copy()

        # Find best classification
        best_class = max(posteriors, key=posteriors.get)

        return best_class, posteriors

    def update_confidence(self, current_class: TargetClassification,
                         new_posterior: float,
                         measurement_count: int) -> float:
        """
        Compute classification confidence considering history.

        Args:
            current_class (TargetClassification): Current classification
            new_posterior (float): New posterior probability
            measurement_count (int): Number of measurements so far

        Returns:
            Updated confidence score (0-1)
        """
        # Confidence grows with consistent measurements
        base_confidence = new_posterior

        # Weight increases with more measurements (asymptotes at ~95% confidence)
        measurement_factor = 1.0 - np.exp(-0.1 * measurement_count)

        confidence = base_confidence * (0.5 + 0.5 * measurement_factor)

        return np.clip(confidence, 0, 1)


class ClassificationEngine:
    """
    Main classification engine for multi-feature target identification.

    Workflow:
        1. Extract features from sensor measurements
        2. Combine features using Bayesian classifier
        3. Maintain signature history for each target
        4. Track classification confidence over time
        5. Provide identification summary for ISR operations
    """

    def __init__(self):
        """Initialize classification engine."""
        self.feature_extractor = FeatureExtractor()
        self.classifier = BayesianClassifier()
        self.target_signatures: Dict[str, TargetSignature] = {}
        self._logger = PerformanceLogger('classification')

    def classify_target(self, target_id: str,
                       target_state: TargetState,
                       drone_position: np.ndarray,
                       measurements: Dict) -> Tuple[TargetClassification, Dict]:
        """
        Classify a target using available measurements.

        Args:
            target_id (str): Target identifier
            target_state (TargetState): Current target state
            drone_position (np.ndarray): Position of observing drone
            measurements (Dict): Available measurements {'signal_strength': dBm, 'visual': array, ...}

        Returns:
            Tuple of (classification, classification_scores)
        """
        with self._logger.start_timer('classify_target'):
            # Initialize or retrieve signature
            if target_id not in self.target_signatures:
                self.target_signatures[target_id] = TargetSignature(target_id=target_id)

            signature = self.target_signatures[target_id]

            # Extract features
            features = []

            # Signal strength feature
            if 'signal_strength' in measurements:
                distance = np.linalg.norm(drone_position - target_state.position)
                signal_feature = self.feature_extractor.extract_signal_strength_features(
                    measurements['signal_strength'], distance
                )
                features.append(signal_feature)
                signature.signal_strength.append(measurements['signal_strength'])

            # Motion features
            motion_features = self.feature_extractor.extract_motion_features(
                target_state.velocity, target_state.acceleration
            )
            features.extend(motion_features.values())
            signature.velocity_magnitude.append(np.linalg.norm(target_state.velocity))
            signature.acceleration_magnitude.append(np.linalg.norm(target_state.acceleration))

            # Size estimate
            if 'visual_detection' in measurements:
                size_feature = self.feature_extractor.extract_size_estimate(
                    measurements['visual_detection'], 
                    np.linalg.norm(drone_position - target_state.position)
                )
                features.append(size_feature)

            # Classify using Bayesian approach
            classification, posteriors = self.classifier.classify(features)

            # Update confidence
            confidence = self.classifier.update_confidence(
                classification, posteriors[classification],
                len(signature.classification_history)
            )

            # Record in history
            signature.classification_history.append((confidence, classification))
            signature.measurements.extend(features)

        return classification, posteriors

    def get_classification_summary(self, target_id: str) -> Dict:
        """
        Get classification summary for a target.

        Args:
            target_id (str): Target identifier

        Returns:
            Dictionary with classification summary
        """
        if target_id not in self.target_signatures:
            return {}

        signature = self.target_signatures[target_id]

        if not signature.classification_history:
            return {}

        # Most recent classification
        recent_confidence, recent_class = signature.classification_history[-1]

        # Classification trend (consensus over last 10 measurements)
        recent_classes = [c for _, c in signature.classification_history[-10:]]
        class_counts = {cls: recent_classes.count(cls) for cls in TargetClassification}
        consensus_class = max(class_counts, key=class_counts.get)
        consensus_strength = class_counts[consensus_class] / len(recent_classes)

        # Statistics
        avg_signal = np.mean(signature.signal_strength[-10:]) if signature.signal_strength else 0
        avg_speed = np.mean(signature.velocity_magnitude[-10:]) if signature.velocity_magnitude else 0
        avg_accel = np.mean(signature.acceleration_magnitude[-10:]) if signature.acceleration_magnitude else 0

        return {
            'target_id': target_id,
            'current_classification': recent_class.value,
            'current_confidence': recent_confidence,
            'consensus_classification': consensus_class.value,
            'consensus_strength': consensus_strength,
            'measurement_count': len(signature.measurements),
            'avg_signal_strength_dbm': avg_signal,
            'avg_speed_ms': avg_speed,
            'avg_acceleration_ms2': avg_accel
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClassificationEngine("
            f"targets_tracked={len(self.target_signatures)}, "
            f"classifier={self.classifier.__class__.__name__})"
        )
