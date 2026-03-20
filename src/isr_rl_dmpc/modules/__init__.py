from .mission_planner import (
    GridCell, GridDecomposer, WaypointGenerator, MissionPlanner
)
from .formation_controller import (
    FormationType, FormationConfig, ConsensusState, 
    FormationGeometry, ConsensusController, FormationController
)
from .sensor_fusion import SensorFusionManager
from .classification_engine import (
    TargetClassification, FeatureType, ClassificationFeature, 
    TargetSignature, FeatureExtractor, BayesianClassifier,
    ClassificationEngine, 
)
from .threat_assessor import (
    ThreatLevel, ThreatAssessment, ThreatParameters, ThreatAssessor,
)
from .task_allocator import (
    TaskType, TaskStatus, ISRTask, DroneCapability, 
    HungarianAssignment, TaskAllocator, 
)

# Optional ML/optimization modules (depend on torch/cvxpy).
try:  # pragma: no cover
    from .dmpc_controller import (
        DMPCConfig,
        CostWeightNetwork,
        DynamicsResidualNetwork,
        ValueNetworkMPC,
        MPCSolver,
        DMPC,
    )
    from .attitude_controller import (
        DroneParameters,
        GainAdaptationNetwork,
        GeometricController,
        AttitudeController,
    )
    from .learning_module import (
        Transition,
        ValueNetwork,
        PolicyNetwork,
        ExperienceBuffer,
        LearningModule,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    DMPCConfig = None  # type: ignore[assignment]
    CostWeightNetwork = None  # type: ignore[assignment]
    DynamicsResidualNetwork = None  # type: ignore[assignment]
    ValueNetworkMPC = None  # type: ignore[assignment]
    MPCSolver = None  # type: ignore[assignment]
    DMPC = None  # type: ignore[assignment]

    DroneParameters = None  # type: ignore[assignment]
    GainAdaptationNetwork = None  # type: ignore[assignment]
    GeometricController = None  # type: ignore[assignment]
    AttitudeController = None  # type: ignore[assignment]

    Transition = None  # type: ignore[assignment]
    ValueNetwork = None  # type: ignore[assignment]
    PolicyNetwork = None  # type: ignore[assignment]
    ExperienceBuffer = None  # type: ignore[assignment]
    LearningModule = None  # type: ignore[assignment]


__all__ = [
    # 1 Mission Planner
    "GridCell", "GridDecomposer", "WaypointGenerator",
    "MissionPlanner",

    # 2 Formation Controller
    "FormationType", "FormationConfig", "ConsensusState",
    "FormationGeometry", "ConsensusController", "FormationController",

    # 3 Sensor Fusion
    "SensorFusionManager",

    # 4 Classification Engine
    "TargetClassification", "FeatureType", "ClassificationFeature", "TargetSignature", 
    "FeatureExtractor", "BayesianClassifier", "ClassificationEngine", 

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

]