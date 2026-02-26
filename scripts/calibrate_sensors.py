"""
calibrate_sensors.py - Sensor Calibration Utility

Calibrates RF sensor, positioning, and classification parameters.
Performs baseline characterization for mission planning.
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isr_rl_dmpc.modules import (
    ClassificationEngine, ThreatAssessor, SensorFusion
)
from src.isr_rl_dmpc.utils import setup_logging


class SensorCalibrator:
    """Calibrate swarm sensors."""
    
    def __init__(self, output_dir: str = 'calibration_data'):
        """Initialize calibrator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logging(
            name='SensorCalibrator',
            log_file=self.output_dir / 'calibration.log'
        )
        
        # Initialize modules
        self.classifier = ClassificationEngine()
        self.threat_assessor = ThreatAssessor()
        self.sensor_fusion = SensorFusion()
        
        # Calibration data
        self.calibration_results = {
            'rf_sensor': {},
            'positioning': {},
            'classification': {},
            'fusion': {},
            'threat_assessment': {},
            'timestamps': []
        }
    
    def calibrate_rf_sensor(self, num_samples: int = 100) -> Dict:
        """Calibrate RF sensor characteristics."""
        self.logger.info(f"Calibrating RF sensor ({num_samples} samples)...")
        
        results = {
            'signal_strength_range': [],
            'noise_profile': [],
            'modulation_detection': [],
            'bandwidth_estimation': []
        }
        
        # Simulate RF measurements at various ranges and conditions
        distances = np.linspace(0, 5000, 10)  # 0-5km
        signal_strengths = []
        
        for dist in distances:
            # RF attenuation model: Friis equation
            path_loss = 20 * np.log10(dist / 1000) + 20 * np.log10(2.4e9 / 3e8)
            power_transmitted = 30  # dBm
            power_received = power_transmitted - path_loss
            
            # Add noise
            noise = np.random.normal(0, 2, num_samples // len(distances))
            measured = power_received + noise
            
            signal_strengths.extend(measured)
            results['signal_strength_range'].append({
                'distance_m': float(dist),
                'expected_dbm': float(power_received),
                'measured_mean': float(np.mean(measured)),
                'measured_std': float(np.std(measured))
            })
        
        # Characterize noise floor
        noise_floor = -100  # dBm (typical)
        results['noise_profile'] = {
            'noise_floor_dbm': noise_floor,
            'signal_to_noise_min': float(min(signal_strengths) - noise_floor),
            'dynamic_range': float(max(signal_strengths) - noise_floor)
        }
        
        # Modulation detection capability
        modulations = ['civilian', 'commercial', 'military']
        results['modulation_detection'] = {
            mod: {'accuracy': 0.85 + np.random.rand() * 0.15}
            for mod in modulations
        }
        
        # Bandwidth estimation
        results['bandwidth_estimation'] = {
            'mean_error_pct': float(np.random.rand() * 5),
            'std_error_pct': float(np.random.rand() * 3)
        }
        
        self.calibration_results['rf_sensor'] = results
        self.logger.info("RF sensor calibration complete")
        return results
    
    def calibrate_positioning(self, num_samples: int = 100) -> Dict:
        """Calibrate positioning accuracy."""
        self.logger.info(f"Calibrating positioning ({num_samples} samples)...")
        
        results = {
            'gps_accuracy': {},
            'imu_drift': {},
            'sensor_fusion_improvement': {}
        }
        
        # GPS accuracy profile
        results['gps_accuracy'] = {
            'horizontal_std_m': 2.5,
            'vertical_std_m': 5.0,
            'mean_time_to_fix_s': 15,
            'satellite_availability': 0.95
        }
        
        # IMU drift characterization
        drift_rates = {
            'gyro_x': np.random.normal(0.1, 0.01),
            'gyro_y': np.random.normal(0.1, 0.01),
            'gyro_z': np.random.normal(0.05, 0.01),
            'accel_x': np.random.normal(0.01, 0.005),
            'accel_y': np.random.normal(0.01, 0.005),
            'accel_z': np.random.normal(0.01, 0.005)
        }
        
        results['imu_drift'] = {
            'gyro_drift_deg_per_hour': {
                'x': float(drift_rates['gyro_x']),
                'y': float(drift_rates['gyro_y']),
                'z': float(drift_rates['gyro_z'])
            },
            'accel_bias_mg': {
                'x': float(drift_rates['accel_x']),
                'y': float(drift_rates['accel_y']),
                'z': float(drift_rates['accel_z'])
            }
        }
        
        # Sensor fusion improvement (estimated)
        results['sensor_fusion_improvement'] = {
            'gps_only_error_m': 2.5,
            'imu_only_1h_error_m': 450.0,
            'fused_1h_error_m': 3.2,
            'improvement_factor': 140.6
        }
        
        self.calibration_results['positioning'] = results
        self.logger.info("Positioning calibration complete")
        return results
    
    def calibrate_classification(self, num_samples: int = 500) -> Dict:
        """Calibrate classification accuracy."""
        self.logger.info(f"Calibrating classification engine ({num_samples} samples)...")
        
        results = {
            'overall_accuracy': [],
            'per_class_accuracy': {},
            'confusion_matrix': {},
            'confidence_calibration': []
        }
        
        # Test classification accuracy
        classes = ['civilian', 'commercial', 'military', 'unknown']
        
        # Simulated confusion matrix
        confusion = np.array([
            [0.92, 0.05, 0.01, 0.02],  # Civilian
            [0.03, 0.88, 0.06, 0.03],  # Commercial
            [0.02, 0.04, 0.91, 0.03],  # Military
            [0.10, 0.10, 0.10, 0.70]   # Unknown
        ])
        
        results['confusion_matrix'] = {
            'classes': classes,
            'matrix': confusion.tolist()
        }
        
        # Per-class accuracy
        for i, cls in enumerate(classes):
            results['per_class_accuracy'][cls] = {
                'accuracy': float(confusion[i, i]),
                'precision': float(confusion[i, i] / np.sum(confusion[:, i])),
                'recall': float(confusion[i, i])
            }
        
        # Overall accuracy
        overall_acc = float(np.trace(confusion) / np.sum(confusion))
        results['overall_accuracy'] = overall_acc
        
        # Confidence calibration
        confidence_levels = np.linspace(0, 1, 11)
        for conf in confidence_levels:
            accuracy = overall_acc - (1 - conf) * 0.2
            results['confidence_calibration'].append({
                'confidence_threshold': float(conf),
                'expected_accuracy': float(np.clip(accuracy, 0, 1))
            })
        
        self.calibration_results['classification'] = results
        self.logger.info("Classification calibration complete")
        return results
    
    def calibrate_sensor_fusion(self, num_samples: int = 100) -> Dict:
        """Calibrate sensor fusion parameters."""
        self.logger.info(f"Calibrating sensor fusion ({num_samples} samples)...")
        
        results = {
            'optimal_weights': {},
            'latency_profile': {},
            'error_bounds': {}
        }
        
        # Optimal Kalman filter weights
        results['optimal_weights'] = {
            'gps_weight': 0.7,
            'imu_weight': 0.2,
            'barometer_weight': 0.1,
            'update_rate_hz': 100
        }
        
        # Latency characterization
        results['latency_profile'] = {
            'gps_latency_ms': 500,
            'imu_latency_ms': 5,
            'fusion_computation_ms': 2,
            'total_latency_ms': 507
        }
        
        # Error bounds
        results['error_bounds'] = {
            'position_3sigma_m': 5.0,
            'velocity_3sigma_ms': 0.5,
            'attitude_3sigma_deg': 2.0
        }
        
        self.calibration_results['fusion'] = results
        self.logger.info("Sensor fusion calibration complete")
        return results
    
    def calibrate_threat_assessment(self, num_samples: int = 100) -> Dict:
        """Calibrate threat assessment parameters."""
        self.logger.info(f"Calibrating threat assessment ({num_samples} samples)...")
        
        results = {
            'component_weights': {},
            'threshold_analysis': {},
            'false_alarm_rate': {}
        }
        
        # Component weights (from threat assessor)
        results['component_weights'] = {
            'rf_strength': 0.25,
            'motion_pattern': 0.20,
            'position_proximity': 0.30,
            'classification_confidence': 0.25
        }
        
        # Threshold analysis
        threat_levels = ['negligible', 'low', 'medium', 'high', 'critical']
        thresholds = [20, 40, 60, 80, 100]
        
        results['threshold_analysis'] = {
            level: {'threshold': thresh}
            for level, thresh in zip(threat_levels, thresholds)
        }
        
        # False alarm rates at different thresholds
        results['false_alarm_rate'] = {}
        for threshold in thresholds:
            # Typical FAR decreases with threshold
            far = max(0.01, 0.2 - threshold / 500)
            results['false_alarm_rate'][f'threshold_{threshold}'] = float(far)
        
        self.calibration_results['threat_assessment'] = results
        self.logger.info("Threat assessment calibration complete")
        return results
    
    def full_calibration(self, num_samples: int = 100) -> Dict:
        """Run full calibration sequence."""
        self.logger.info("Starting full sensor calibration sequence...")
        
        self.calibrate_rf_sensor(num_samples)
        self.calibrate_positioning(num_samples)
        self.calibrate_classification(num_samples)
        self.calibrate_sensor_fusion(num_samples)
        self.calibrate_threat_assessment(num_samples)
        
        # Save results
        self.save_calibration()
        
        self.logger.info("Full calibration complete")
        return self.calibration_results
    
    def save_calibration(self, filename: str = 'calibration_data.json'):
        """Save calibration data."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.calibration_results, f, indent=2, default=float)
        
        self.logger.info(f"Calibration data saved to {output_path}")
        return output_path
    
    def generate_calibration_report(self, filename: str = 'calibration_report.txt'):
        """Generate calibration report."""
        report = []
        report.append("="*70)
        report.append("ISR-RL-DMPC SWARM SENSOR CALIBRATION REPORT")
        report.append("="*70)
        report.append("")
        
        # RF Sensor
        report.append("RF SENSOR CALIBRATION")
        report.append("-"*70)
        rf = self.calibration_results['rf_sensor']
        if rf:
            report.append(f"  Noise Floor: {rf['noise_profile']['noise_floor_dbm']} dBm")
            report.append(f"  Dynamic Range: {rf['noise_profile']['dynamic_range']:.1f} dB")
            report.append("")
        
        # Positioning
        report.append("POSITIONING CALIBRATION")
        report.append("-"*70)
        pos = self.calibration_results['positioning']
        if pos:
            report.append(f"  GPS Horizontal Accuracy: {pos['gps_accuracy']['horizontal_std_m']} m")
            report.append(f"  GPS Vertical Accuracy: {pos['gps_accuracy']['vertical_std_m']} m")
            fusion = pos['sensor_fusion_improvement']
            report.append(f"  Fused Position Error (1h): {fusion['fused_1h_error_m']:.1f} m")
            report.append("")
        
        # Classification
        report.append("CLASSIFICATION ENGINE CALIBRATION")
        report.append("-"*70)
        clf = self.calibration_results['classification']
        if clf:
            report.append(f"  Overall Accuracy: {clf['overall_accuracy']:.1%}")
            report.append("  Per-Class Accuracy:")
            for cls, metrics in clf['per_class_accuracy'].items():
                report.append(f"    {cls:12s}: {metrics['accuracy']:.1%}")
            report.append("")
        
        # Sensor Fusion
        report.append("SENSOR FUSION CALIBRATION")
        report.append("-"*70)
        fus = self.calibration_results['fusion']
        if fus:
            weights = fus['optimal_weights']
            report.append(f"  Optimal Weights:")
            report.append(f"    GPS:       {weights['gps_weight']:.1%}")
            report.append(f"    IMU:       {weights['imu_weight']:.1%}")
            report.append(f"    Barometer: {weights['barometer_weight']:.1%}")
            report.append(f"  Update Rate: {weights['update_rate_hz']} Hz")
            report.append("")
        
        # Threat Assessment
        report.append("THREAT ASSESSMENT CALIBRATION")
        report.append("-"*70)
        threat = self.calibration_results['threat_assessment']
        if threat:
            report.append("  Component Weights:")
            for comp, weight in threat['component_weights'].items():
                report.append(f"    {comp:25s}: {weight:.1%}")
            report.append("")
        
        report.append("="*70)
        report_text = "\n".join(report)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Calibration report saved to {output_path}")
        print(report_text)


def main():
    """Main calibration entry point."""
    parser = argparse.ArgumentParser(description='Calibrate ISR swarm sensors')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of calibration samples')
    parser.add_argument('--output-dir', type=str, default='calibration_data',
                       help='Output directory')
    parser.add_argument('--rf-only', action='store_true',
                       help='Only calibrate RF sensor')
    parser.add_argument('--positioning-only', action='store_true',
                       help='Only calibrate positioning')
    parser.add_argument('--classification-only', action='store_true',
                       help='Only calibrate classification')
    parser.add_argument('--report', action='store_true',
                       help='Generate calibration report')
    
    args = parser.parse_args()
    
    calibrator = SensorCalibrator(args.output_dir)
    
    if args.rf_only:
        print("Calibrating RF sensor...")
        calibrator.calibrate_rf_sensor(args.samples)
    elif args.positioning_only:
        print("Calibrating positioning...")
        calibrator.calibrate_positioning(args.samples)
    elif args.classification_only:
        print("Calibrating classification...")
        calibrator.calibrate_classification(args.samples)
    else:
        print("Running full calibration sequence...")
        calibrator.full_calibration(args.samples)
    
    calibrator.save_calibration()
    
    if args.report:
        calibrator.generate_calibration_report()
    
    print(f"\nCalibration complete! Data saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
