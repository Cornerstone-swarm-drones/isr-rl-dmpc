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

# NOTE:
# `isr_rl_dmpc` is sometimes imported for non-ML functionality (e.g. the Gym
# environment + evaluation scripts). Many ML/visualization submodules depend on
# heavy optional dependencies like `torch` and `mcap`.
#
# To keep those workflows usable without installing the full ML stack, we guard
# module/agent/model imports. If you need ML training/inference, install `torch`
# and ensure the optional deps are present.
try:  # pragma: no cover
    from .modules import (
        GridCell,
        GridDecomposer,
        WaypointGenerator,
        MissionPlanner,
    )
    from .modules import (
        FormationType,
        FormationConfig,
        ConsensusState,
        FormationGeometry,
        ConsensusController,
        FormationController,
    )
    from .modules import SensorFusionManager
    from .modules import (
        TargetClassification,
        FeatureType,
        ClassificationFeature,
        TargetSignature,
        FeatureExtractor,
        BayesianClassifier,
        ClassificationEngine,
    )
    from .modules import (
        ThreatLevel,
        ThreatAssessment,
        ThreatParameters,
        ThreatAssessor,
    )
    from .modules import (
        TaskType,
        TaskStatus,
        ISRTask,
        DroneCapability,
        HungarianAssignment,
        TaskAllocator,
    )
    from .modules import (
        DMPCConfig,
        CostWeightNetwork,
        DynamicsResidualNetwork,
        ValueNetworkMPC,
        MPCSolver,
        DMPC,
    )
    from .modules import (
        DroneParameters,
        GainAdaptationNetwork,
        GeometricController,
        AttitudeController,
    )
    from .modules import (
        Transition,
        ValueNetwork,
        PolicyNetwork,
        ExperienceBuffer,
        LearningModule,
    )
    from .agents import DMPCAgent, ActorCriticTrainer, TrainingConfig
    from .agents import PrioritizedReplayBuffer, TrajectoryBuffer
    from .models import CriticNetwork, RFClassifier, RFFingerprint
    from .models import ModelRegistry, load_checkpoint, save_checkpoint
except ModuleNotFoundError:
    # Keep package importable for environment/evaluation code.
    pass

__all__ = [
    # Core
    "DroneState",  "TargetState",  "MissionState",  "StateManager",
    "DroneStateEstimator", "TargetStateEstimator", "SensorType","RadarMeasurement",
    "OpticalMeasurement", "RFMeasurement", "AcousticMeasurement", "TargetTrackingEKF",
    
    # Modules
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

    # 7 DMPC Controller
    "DMPCConfig", "CostWeightNetwork", "DynamicsResidualNetwork", "ValueNetworkMPC", 
    "MPCSolver", "DMPC",

    # 8 Attitude Controller
    "DroneParameters", "GainAdaptationNetwork", "GeometricController", "AttitudeController",

    # 9 Learning Module
    "Transition", "ValueNetwork", "PolicyNetwork", "ExperienceBuffer", "LearningModule",

    # Agents
    "DMPCAgent",
    "ActorCriticTrainer",
    "TrainingConfig",
    "PrioritizedReplayBuffer",
    "TrajectoryBuffer",

    # Models
    "CriticNetwork",
    "RFClassifier",
    "RFFingerprint",
    "ModelRegistry",
    "load_checkpoint",
    "save_checkpoint",
    
    # Config
    "Config", "load_config", "create_default_config_yaml",

    # Utils
    "UnitConversions", "AttitudeConversions", "PositionProjections", "BearingDistance", 
    "TimeConversions", "ColoredFormatter", "JSONFormatter", "MetricsLogger",
    "MissionLogger", "PerformanceLogger", "Timer", "setup_logging", "get_logger",
    "configure_file_logging", "configure_rotating_file_logging", "setup_logger",
    "QuaternionOps", "MatrixOps",
    "CoordinateTransform", "GeometryOps", "NumericalOps", 'TrajectoryVisualizer',
    "MissionVisualizer", "LearningVisualizer", "FormationVisualizer", "EnergyVisualizer",
    "StatisticsVisualizer",

]