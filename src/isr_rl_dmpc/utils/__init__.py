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

from .foxglove_bridge import FoxgloveBridge
from .mcap_logger import MCAPRecorder


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
    "MCAPRecorder",
]
