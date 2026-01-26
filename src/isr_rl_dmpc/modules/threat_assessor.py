"""
Module 5: Threat Assessor - Target Risk Scoring and Prioritization

Computes threat scores for detected targets using multi-dimensional threat
assessment combining RF signatures, motion patterns, and positions.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Threat classification levels"""
    NEGLIGIBLE = 0 # No threat
    LOW = 1 # Minor concern
    MODERATE = 2 # Warrant attention
    HIGH = 3 # Significant threat
    CRITICAL = 4 # Immediate action required

@dataclass
class ThreatAssessment:
    """Complete threat assessment for a target"""
    target_id: int
    threat_level: ThreatLevel
    threat_score: float # 0-100
    rf_threat: float # RF signal threat component
    motion_threat: float # Motion pattern threat component
    position_threat: float # Proximity threat component
    classification_threat: float # Classification-based threat
    confidence: float # Assessment confidence (0, 1)
    assessment_time: float
    recommendation: str

class ThreatParameters:
    """Configurable threat assessment parameters"""

    def __init__(self):
        """Initialize default threat parameters"""
        # RF signal threat weights
        self.rf_strength_threshold = -50  # dBm
        self.rf_modulation_military = 0.7  # Military mod correlates to threat
        self.rf_power = 0.3

        # Motion threat weights
        self.velocity_threshold = 50.0     # m/s
        self.acceleration_threshold = 5.0  # m/s^2
        self.motion_power = 0.25

        # Position threat weights
        self.proximity_threshold = 500.0   # m (critical distance)
        self.altitude_power = 0.15

        # Classification weights
        self.military_threat_score = 0.9
        self.civilian_threat_score = 0.1
        self.decoy_threat_score = 0.05
        self.unknown_threat_score = 0.5

        # Overall weights
        self.rf_weight = 0.25
        self.motion_weight = 0.20
        self.position_weight = 0.30
        self.classification_weight = 0.25

class ThreatAssessor:
    """
    Computes threat scores for targets.
    Multi-dimensional threat assessment including:
        - RF signal characteristics
        - Motion patterns
        - Proximity and position
        - Classification results
    """

    def __init__(self, params: Optional[ThreatParameters] = None):
        """
        Initialize threat assessor.
        Arguments:
            params (Optional[ThreatParaneters]): Threat parameters
        """
        self.params = params or ThreatParameters()
        self.threat_history = {} # target_id -> list of threat scores
        self.assessment_count = 0

    def assess_target(self, target_data: Dict, own_position: np.ndarray, current_time: float) -> ThreatAssessment:
        """
        Compute threat assessment for target
        Arguments:
            target_data (Dict): Target information
                - position: [x, y, z]
                - velocity: [vx, vy, vz]
                - rf_strength: signal power (dBm)
                - modulation: modulation type
                - classification: class label
                - confidence: classification confidence
            own_position (np.ndarray): Own drone position
            current_time (float): Current timestep
        Returns:
            ThreatAssessment with detailed breakdown
        """
        target_id = target_data.get('target_id', -1)
        
        # Component threat scores
        rf_threat = self._assess_rf_threat(target_data)
        motion_threat = self._assess_motion_threat(target_data)
        position_threat = self._assess_position_threat(target_data['position'], own_position)
        classification_threat = self._assess_classification_threat(target_data)

        # Weighted combination
        threat_score = (
            self.params.rf_weight * rf_threat + 
            self.params.motion_weight * motion_threat + 
            self.params.position_weight * position_threat +
            self.params.classification_weight * classification_threat
        )

        threat_score = np.clip(threat_score*100, 0, 100) # Normalize to (0, 100)
        threat_level = self._classify_threat_level(threat_score)
        confidence = self._compute_confidence(target_data)
        recommendation = self._generate_recommendation(threat_level, threat_score)

        # Track history
        if target_id not in self.threat_history:
            self.threat_history[target_id] = []
        self.threat_history[target_id].append(threat_score)
        self.assessment_count += 1
        return ThreatAssessment(
            target_id, threat_level, threat_score, rf_threat*100,
            motion_threat*100, position_threat*100, classification_threat*100,
            confidence, current_time, recommendation,
        )
    
    def _assess_rf_threat(self, target_data: Dict) -> float:
        """
        Assess threat from RF signal characteristics
        Returns: RF threat score (0, 1)
        """
        rf_strength = target_data.get('rf_strength', -100)
        modulation = target_data.get('modulation', 'unknown')

        signal_threat = 0.0 # Signal strength threat
        if rf_strength > self.params.rf_strength_threshold:
            # Strong signal is more threatning
            norm_strength = (rf_strength - (-100)) / (-self.params.rf_strength_threshold + 100)
        
        # Modulation threat (military modulations are more threatning)
        modulation_threat = 0.0
        if modulation.lower() in ['psk', 'qam', 'fsk']: # Military-typical
            modulation_threat = self.params.rf_modulation_military
        elif modulation.lower() in ['am', 'fm']: # Civilian-typical
            modulation_threat = 0.2
        else: # Unkown is moderately threatning
            modulation_threat = 0.5
        
        # Combine components
        rf_threat = (signal_threat * self.params.rf_power + modulation_threat * (1 - self.params.rf_power))

        return np.clip(rf_threat, 0, 1)
    
    def _assess_motion_threat(self, target_data: Dict) -> float:
        """
        Assess threat from motion patterns
        Returns: Motion threat score (0, 1)
        """
        velocity = target_data.get('velocity', np.zeros(3))
        acceleration = target_data.get('acceleration', np.zeros(3))

        # Velocity threat
        speed = np.linalg.norm(velocity)
        velocity_threat = 0.0
        if speed > self.params.velocity_threshold: # High speed is threatning
            norm_speed = (speed - self.params.velocity_threshold) / (200 - self.params.velocity_threshold)
            velocity_threat = np.clip(norm_speed, 0, 1)

        # Acceleration threat
        accel_mag = np.linalg.norm(acceleration)
        accel_threat = 0.0
        if accel_mag > self.params.acceleration_threshold: # Evasive moves
            norm_accel = (accel_mag - self.params.acceleration_threshold) / (20 - self.params.acceleration_threshold)
            accel_threat = np.clip(norm_accel, 0, 1)

        # Combine
        motion_threat = (velocity_threat*0.6 + accel_threat*0.4)
        return np.clip(motion_threat, 0, 1)
    
    def _assess_position_threat(self, target_pos: np.ndarray, own_pos: np.ndarray) -> float:
        """
        Assess threat from position and proximity
        Returns: Position threat score (0, 1)
        """
        distance = np.linalg.norm(target_pos - own_pos)

        # Proximity threat
        if distance < self.params.proximity_threshold: # Critical proxomity
            threat = 1.0 - (distance/self.params.proximity_threshold)
            threat = np.clip(threat, 0, 1)
        else: # Distant target
            threat = 0.1 * np.exp(-distance/1000)
        
        # Altitude threat (low altitude is more threatning)
        altitude = target_pos[2]
        altitude_threat = 0.0
        if altitude < 300: # low altitude
            altitude_threat = 0.3 * (1- altitude / 300)
        position_threat = threat + altitude_threat * self.params.altitude_power

        return np.clip(position_threat, 0, 1)
    
    def _assess_classification_threat(self, target_data: Dict) -> float:
        """
        Assess threat based on target classification
        Returns: Classification threat score (0, 1)
        """
        classification = target_data.get('classification', 'unkown')
        confidence = target_data.get('classification_confidence', 0.5)

        # Base threat from classification
        threat_map = {
            'military': self.params.military_threat_score,
            'civilian': self.params.civilian_threat_score,
            'decoy': self.params.decoy_threat_score,
            'friendly': 0.0,
            'unknown': self.params.unknown_threat_score
        }
        
        base_threat = threat_map.get(classification.lower(), self.params.unknown_threat_score)

        # Weight by confidence
        classification_threat = base_threat * confidence
        return np.clip(classification_threat, 0, 1)
    
    def _classify_threat_level(self, score: float) -> ThreatLevel:
        """Classify threat score into discrete level"""
        if score < 10:
            return ThreatLevel.NEGLIGIBLE
        elif score < 30:
            return ThreatLevel.LOW
        elif score > 50:
            return ThreatLevel.MODERATE
        elif score < 75:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def _compute_confidence(self, target_data: Dict) -> float:
        """Compute confidence in assessment"""
        # Based on data quality
        confidence = 0.5 # Base confidence
        # Increase with classification confidence
        class_conf = target_data.get('classification_confidence', 0)
        confidence += class_conf * 0.3

        # Increase with detections
        detections = target_data.get('classification_confidence', 0)
        confidence += min(detections/5, 1.0) * 0.2

        return np.clip(confidence, 0, 1)
    
    def _generate_recommendation(self, level: ThreatLevel, score: float) -> str:
        """Generate action recommendation based on threat level."""
        if level == ThreatLevel.CRITICAL:
            return "IMMEDIATE ACTION: High-priority threat, engage defensive measures"
        elif level == ThreatLevel.HIGH:
            return "WARNING: Elevated threat, increase monitoring, prepare countermeasures"
        elif level == ThreatLevel.MODERATE:
            return "CAUTION: Moderate threat, maintain vigilance, continue classification"
        elif level == ThreatLevel.LOW:
            return "ADVISORY: Low-level threat detected, track for development"
        else:
            return "CLEAR: No significant threat, routine monitoring"
        

    def get_threat_trends(self, target_id: int, window: int = 5) -> Dict:
        """
        Get threat trend analysis for target
        Arguments:
            target_id (int): Target ID
            window (int): Time window for trend
        Returns:
            Dict with trend metrics
        """
        if target_id not in self.threat_history:
            return {}
        history = self.threat_history[target_id][-window:]

        if len(history) < 2:
            trend = "insufficient_data"
            trend_rate = 0
        else:
            # Trend Analysis
            threat_change = history[-1] - history[0]
            trend_rate = threat_change / len(history)
            if threat_change > 5:
                trend = "increasing"
            elif threat_change < -5:
                trend = "decreasing"
            else:
                trend = "stable"
        return {
            'current_threat': history[-1] if history else 0,
            'trend': trend,
            'trend_rate': trend_rate,
            'mean_threat': np.mean(history),
            'max_threat': np.max(history),
            'history': history
        }
    
    def prioritize_targets(self, assessments: List[ThreatAssessment]) -> List[ThreatAssessment]:
        """"
        Sort targets by priority (threat score descending).
        Arguments:
            assessments (List[ThreatAssessment]): List of assessments
        Returns:
            Sorted list (highest threat first)
        """
        return sorted(assessments, key=lambda x: x.threat_score, reverse=True)

    def get_assessment_statistics(self) -> Dict:
        """Get statistics about threat assessments."""
        if not self.threat_history:
            return {}
        all_scores = []
        for scores in self.threat_history.values():
            all_scores.extend(scores)
        return {
            'total_assessments': self.assessment_count,
            'tracked_targets': len(self.threat_history),
            'average_threat': np.mean(all_scores),
            'max_threat': np.max(all_scores),
            'min_threat': np.min(all_scores),
            'critical_count': len([s for s in all_scores if s > 75]),
            'high_count': len([s for s in all_scores if 50 <= s <= 75])
        }
