"""ISR-RL-DMPC Utils Module - Initialization."""

from .conversions import (
    UnitConversions, 
    AttitudeConversions, 
    PositionProjections, 
    BearingDistance, 
    TimeConversions
)

from .logging_utils import (
    ColoredFormatter,
    JSONFormatter,
    MetricsLogger,
    MissionLogger,
    PerformanceLogger,
    Timer,
    setup_logging,
    get_logger,
    configure_file_logging,
    configure_rotating_file_logging,
    setup_logger,
)

from .math_utils import  (
    QuaternionOps,
    MatrixOps,
    CoordinateTransform,
    GeometryOps,
    NumericalOps
)

from .visualization import (
    TrajectoryVisualizer,
    MissionVisualizer,
    LearningVisualizer,
    FormationVisualizer,
    EnergyVisualizer,
    StatisticsVisualizer,
)

from .foxglove_bridge import FoxgloveBridge, extract_targets_from_obs

# MCAP is optional because it requires the external `mcap` dependency.
# Most of the project (gym env + training + evaluation) should work without it.
try:
    from .mcap_logger import MCAPRecorder
except ModuleNotFoundError:  # pragma: no cover
    MCAPRecorder = None  # type: ignore[assignment]


__all__ = [
    # conversions
    "UnitConversions", 
    "AttitudeConversions", 
    "PositionProjections", 
    "BearingDistance", 
    "TimeConversions",
    # logging
    "ColoredFormatter",
    "JSONFormatter",
    "MetricsLogger",
    "MissionLogger",
    "PerformanceLogger",
    "Timer",
    "setup_logging",
    "get_logger",
    "configure_file_logging",
    "configure_rotating_file_logging",
    "setup_logger",
    # math
    "QuaternionOps",
    "MatrixOps",
    "CoordinateTransform",
    "GeometryOps",
    "NumericalOps",
    # Viz
    'TrajectoryVisualizer',
    "MissionVisualizer",
    "LearningVisualizer",
    "FormationVisualizer",
    "EnergyVisualizer",
    "StatisticsVisualizer",
    # Foxglove
    "FoxgloveBridge",
    "extract_targets_from_obs",
    "MCAPRecorder",
]
