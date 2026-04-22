from .simulator import (
    EnvironmentSimulator, DronePhysics, TargetPhysics,
    WindModel, TargetType, DroneConfig, TargetConfig, EnvironmentConfig,
)
from .reward_shaper import (
    RewardShaper,
    RewardWeights,
    BeliefCoverageRewardShaper,
    BeliefCoverageRewardWeights,
)
from .sensor_simulator import SensorNoiseModel, SensorSimulator
from .sensor_model import ForwardFOVSensorModel, VisionSensorConfig
from .patrol_pomdp import (
    PatrolBeliefState,
    PatrolBeliefUpdater,
    PatrolHiddenWorldState,
    PatrolObservationModel,
    PatrolTransitionModel,
)

__all__ = [
    # Physics
    "EnvironmentSimulator",
    "DronePhysics",
    "TargetPhysics",
    "WindModel",
    "TargetType",
    "DroneConfig",
    "TargetConfig",
    "EnvironmentConfig",
    # Reward / sensors
    "RewardShaper",
    "RewardWeights",
    "BeliefCoverageRewardShaper",
    "BeliefCoverageRewardWeights",
    "SensorNoiseModel",
    "SensorSimulator",
    "ForwardFOVSensorModel",
    "VisionSensorConfig",
    "PatrolHiddenWorldState",
    "PatrolBeliefState",
    "PatrolTransitionModel",
    "PatrolObservationModel",
    "PatrolBeliefUpdater",
]

try:  # pragma: no cover - optional dependency guard
    from .isr_env import ISRGridEnv, VectorEnv, make_env as make_isr_env

    __all__.extend(
        [
            "ISRGridEnv",
            "VectorEnv",
            "make_isr_env",
        ]
    )
except (ModuleNotFoundError, ImportError):
    pass

try:  # pragma: no cover - optional dependency guard
    from .marl_env import MARLDMPCEnv, make_env

    __all__.extend(
        [
            "MARLDMPCEnv",
            "make_env",
        ]
    )
except (ModuleNotFoundError, ImportError):
    pass

try:  # pragma: no cover - optional dependency guard
    from .belief_coverage_env import BeliefCoverageEnv, make_belief_coverage_env

    __all__.extend(
        [
            "BeliefCoverageEnv",
            "make_belief_coverage_env",
        ]
    )
except (ModuleNotFoundError, ImportError):
    pass
