from .core import (
    DroneState,
    TargetState,
    MissionState,
    StateManager,
    DroneStateEstimator,
    TargetStateEstimator,
    SensorType,
    RadarMeasurement,
    OpticalMeasurement,
    RFMeasurement,
    AcousticMeasurement,
    TargetTrackingEKF,
)
from .config import Config, load_config, create_default_config_yaml
from .utils import *

# Modules 1–9 and agents/models. Guard against missing optional deps
# (cvxpy, scipy) so non-optimisation workflows remain importable.
try:  # pragma: no cover
    from .modules import (
        GridCell, GridDecomposer, WaypointGenerator, MissionPlanner,
        FormationType, FormationConfig, ConsensusState, FormationGeometry,
        ConsensusController, FormationController,
        SensorFusionManager,
        TargetClassification, FeatureType, ClassificationFeature,
        TargetSignature, FeatureExtractor, BayesianClassifier,
        ClassificationEngine,
        ThreatLevel, ThreatAssessment, ThreatParameters, ThreatAssessor,
        TaskType, TaskStatus, ISRTask, DroneCapability,
        HungarianAssignment, TaskAllocator,
        DMPCConfig, MPCSolver, DMPC, compute_lqr_terminal_cost,
        DroneParameters, GeometricController, AttitudeController,
        StepRecord, DMPCAnalytics,
    )
    from .agents import DMPCAgent
    from .models import ModelRegistry, load_checkpoint, save_checkpoint
except (ModuleNotFoundError, ImportError):
    # Keep package importable for environment/evaluation code.
    pass

__all__ = [
    # Core
    "DroneState", "TargetState", "MissionState", "StateManager",
    "DroneStateEstimator", "TargetStateEstimator", "SensorType",
    "RadarMeasurement", "OpticalMeasurement", "RFMeasurement",
    "AcousticMeasurement", "TargetTrackingEKF",

    # 1 Mission Planner
    "GridCell", "GridDecomposer", "WaypointGenerator", "MissionPlanner",

    # 2 Formation Controller
    "FormationType", "FormationConfig", "ConsensusState", "FormationGeometry",
    "ConsensusController", "FormationController",

    # 3 Sensor Fusion
    "SensorFusionManager",

    # 4 Classification Engine
    "TargetClassification", "FeatureType", "ClassificationFeature",
    "TargetSignature", "FeatureExtractor", "BayesianClassifier",
    "ClassificationEngine",

    # 5 Threat Assessor
    "ThreatLevel", "ThreatAssessment", "ThreatParameters", "ThreatAssessor",

    # 6 Task Allocator
    "TaskType", "TaskStatus", "ISRTask", "DroneCapability",
    "HungarianAssignment", "TaskAllocator",

    # 7 DMPC Controller (pure optimisation)
    "DMPCConfig", "MPCSolver", "DMPC", "compute_lqr_terminal_cost",

    # 8 Attitude Controller (geometric, fixed gains)
    "DroneParameters", "GeometricController", "AttitudeController",

    # 9 DMPC Analytics
    "StepRecord", "DMPCAnalytics",

    # Agents
    "DMPCAgent",

    # Models / persistence
    "ModelRegistry", "load_checkpoint", "save_checkpoint",

    # Config
    "Config", "load_config", "create_default_config_yaml",

    # Utils
    "UnitConversions", "AttitudeConversions", "PositionProjections",
    "BearingDistance", "TimeConversions", "ColoredFormatter",
    "JSONFormatter", "MetricsLogger", "MissionLogger",
    "PerformanceLogger", "Timer", "setup_logging", "get_logger",
    "configure_file_logging", "configure_rotating_file_logging",
    "setup_logger", "QuaternionOps", "MatrixOps",
    "CoordinateTransform", "GeometryOps", "NumericalOps",
    "TrajectoryVisualizer", "MissionVisualizer", "LearningVisualizer",
    "FormationVisualizer", "EnergyVisualizer", "StatisticsVisualizer",
]
