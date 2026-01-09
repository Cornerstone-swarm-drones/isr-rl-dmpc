"""
Configuration system for ISR-RL-DMPC.

Provides configuration classes for drone, sensor, mission, learning, and DMPC parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


@dataclass
class DroneConfig:
    """Drone physical and control parameters."""
    
    mass: float = 1.0  # kg
    inertia: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.1])  # kg⋅m²
    max_acceleration: float = 10.0  # m/s²
    max_velocity: float = 20.0  # m/s
    max_angular_velocity: float = 2.0  # rad/s
    max_yaw_rate: float = 2.0  # rad/s
    battery_capacity: float = 5000.0  # Wh
    
    def validate(self):
        """Validate drone parameters."""
        assert self.mass > 0, "mass must be positive"
        assert len(self.inertia) == 3, "inertia must be 3-element list"
        assert all(i > 0 for i in self.inertia), "inertia elements must be positive"
        assert self.max_acceleration > 0, "max_acceleration must be positive"
        assert self.max_velocity > 0, "max_velocity must be positive"
        assert self.battery_capacity > 0, "battery_capacity must be positive"


@dataclass
class SensorConfig:
    """Sensor and control frequency parameters."""
    
    control_frequency: float = 50.0  # Hz
    radar_range: float = 200.0  # m
    radar_update_rate: float = 5.0  # Hz
    optical_fov: float = 120.0  # degrees
    optical_update_rate: float = 30.0  # Hz
    rf_range: float = 100.0  # m
    rf_update_rate: float = 10.0  # Hz
    max_radar_targets: int = 50
    max_optical_targets: int = 20
    
    def validate(self):
        """Validate sensor parameters."""
        assert self.control_frequency > 0, "control_frequency must be positive"
        assert self.radar_range > 0, "radar_range must be positive"
        assert 0 < self.optical_fov <= 360, "optical_fov must be in (0, 360]"


@dataclass
class MissionConfig:
    """Mission and coverage parameters."""
    
    grid_cell_size: float = 10.0  # m
    coverage_radius: float = 5.0  # m
    communication_radius: float = 100.0  # m
    coverage_goal: float = 0.95  # [0, 1]
    min_swarm_separation: float = 2.0  # m
    max_swarm_spread: float = 200.0  # m
    
    def validate(self):
        """Validate mission parameters."""
        assert self.grid_cell_size > 0, "grid_cell_size must be positive"
        assert self.coverage_radius > 0, "coverage_radius must be positive"
        assert self.communication_radius > 0, "communication_radius must be positive"
        assert 0 <= self.coverage_goal <= 1, "coverage_goal must be in [0, 1]"
        assert self.min_swarm_separation > 0, "min_swarm_separation must be positive"


@dataclass
class LearningConfig:
    """Reinforcement learning hyperparameters."""
    
    discount_factor: float = 0.99
    learning_rate_critic: float = 1e-3
    learning_rate_actor: float = 1e-4
    batch_size: int = 32
    buffer_size: int = 1e5
    target_update_frequency: int = 1000
    
    # Reward weights
    weight_coverage: float = 10.0
    weight_energy: float = 5.0
    weight_collision: float = -100.0
    weight_target_engagement: float = 20.0
    weight_formation: float = 2.0
    
    def validate(self):
        """Validate learning parameters."""
        assert 0 < self.discount_factor < 1, "discount_factor must be in (0, 1)"
        assert self.learning_rate_critic > 0, "learning_rate_critic must be positive"
        assert self.learning_rate_actor > 0, "learning_rate_actor must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.buffer_size > 0, "buffer_size must be positive"


@dataclass
class DMPCConfig:
    """Distributed Model Predictive Control parameters."""
    
    prediction_horizon: int = 10  # steps
    control_horizon: int = 5  # steps
    receding_horizon_step: int = 1  # steps
    constraint_tightening: float = 0.95  # relaxation factor
    solver_max_iterations: int = 100
    solver_tolerance: float = 1e-4
    
    def validate(self):
        """Validate DMPC parameters."""
        assert self.prediction_horizon > 0, "prediction_horizon must be positive"
        assert self.control_horizon > 0, "control_horizon must be positive"
        assert self.control_horizon <= self.prediction_horizon, "control_horizon <= prediction_horizon"
        assert 0 < self.constraint_tightening <= 1, "constraint_tightening must be in (0, 1]"


@dataclass
class Config:
    """Master configuration class."""
    
    drone: DroneConfig = field(default_factory=DroneConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    dmpc: DMPCConfig = field(default_factory=DMPCConfig)
    
    def validate(self):
        """Validate all configuration sections."""
        self.drone.validate()
        self.sensor.validate()
        self.mission.validate()
        self.learning.validate()
        self.dmpc.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "drone": asdict(self.drone),
            "sensor": asdict(self.sensor),
            "mission": asdict(self.mission),
            "learning": asdict(self.learning),
            "dmpc": asdict(self.dmpc),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        config = cls()
        if "drone" in data:
            config.drone = DroneConfig(**data["drone"])
        if "sensor" in data:
            config.sensor = SensorConfig(**data["sensor"])
        if "mission" in data:
            config.mission = MissionConfig(**data["mission"])
        if "learning" in data:
            config.learning = LearningConfig(**data["learning"])
        if "dmpc" in data:
            config.dmpc = DMPCConfig(**data["dmpc"])
        return config
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def apply_overrides(self, overrides: Dict[str, Any]):
        """Apply command-line overrides to configuration."""
        if "drone" in overrides:
            self.drone = DroneConfig(**{**asdict(self.drone), **overrides["drone"]})
        if "sensor" in overrides:
            self.sensor = SensorConfig(**{**asdict(self.sensor), **overrides["sensor"]})
        if "mission" in overrides:
            self.mission = MissionConfig(**{**asdict(self.mission), **overrides["mission"]})
        if "learning" in overrides:
            self.learning = LearningConfig(**{**asdict(self.learning), **overrides["learning"]})
        if "dmpc" in overrides:
            self.dmpc = DMPCConfig(**{**asdict(self.dmpc), **overrides["dmpc"]})


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration with optional overrides.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: Dictionary of parameter overrides
        
    Returns:
        Loaded and validated configuration
    """
    # Load base configuration
    if config_path is None:
        config_path = "config/default_config.yaml"
    
    if Path(config_path).exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    # Apply overrides
    if overrides:
        config.apply_overrides(overrides)
    
    # Validate
    config.validate()
    
    return config


def create_default_config_yaml(filepath: str = "config/default_config.yaml"):
    """Create default configuration YAML file."""
    config = Config()
    config.to_yaml(filepath)