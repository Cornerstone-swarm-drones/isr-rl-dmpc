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

# Optimisation modules (depend on cvxpy/scipy).
try:  # pragma: no cover
    from .dmpc_controller import (
        DMPCConfig,
        MPCSolver,
        DMPC,
        compute_lqr_terminal_cost,
    )
    from .attitude_controller import (
        DroneParameters,
        GeometricController,
        AttitudeController,
    )
    from .learning_module import (
        StepRecord,
        DMPCAnalytics,
    )
    from .admm_consensus import (
        ADMMConfig,
        ADMMConsensus,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    DMPCConfig = None  # type: ignore[assignment]
    MPCSolver = None  # type: ignore[assignment]
    DMPC = None  # type: ignore[assignment]
    compute_lqr_terminal_cost = None  # type: ignore[assignment]

    DroneParameters = None  # type: ignore[assignment]
    GeometricController = None  # type: ignore[assignment]
    AttitudeController = None  # type: ignore[assignment]

    StepRecord = None  # type: ignore[assignment]
    DMPCAnalytics = None  # type: ignore[assignment]

    ADMMConfig = None  # type: ignore[assignment]
    ADMMConsensus = None  # type: ignore[assignment]


__all__ = [
    # 1 Mission Planner
    "GridCell", "GridDecomposer", "WaypointGenerator", "MissionPlanner",

    # 2 Formation Controller
    "FormationType", "FormationConfig", "ConsensusState",
    "FormationGeometry", "ConsensusController", "FormationController",

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

    # 7 DMPC Controller (pure optimisation, no NN)
    "DMPCConfig", "MPCSolver", "DMPC", "compute_lqr_terminal_cost",

    # 8 Attitude Controller (geometric, fixed gains)
    "DroneParameters", "GeometricController", "AttitudeController",

    # 9 DMPC Analytics
    "StepRecord", "DMPCAnalytics",
]
