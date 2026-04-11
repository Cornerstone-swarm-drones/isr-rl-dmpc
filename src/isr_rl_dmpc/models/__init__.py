"""
ISR-RL-DMPC Models Module.

Neural-network model classes (CriticNetwork, RFClassifier) have been
removed.  This module now exports only checkpoint utilities used to
persist DMPC configuration data, plus the ``hector_quadrotor`` sub-package
that provides URDF and mesh assets for PyBullet simulation.
"""

from .pretrained_models import ModelRegistry, load_checkpoint, save_checkpoint
from .hector_quadrotor import get_urdf_path, get_models_dir

__all__ = [
    "ModelRegistry",
    "load_checkpoint",
    "save_checkpoint",
    "get_urdf_path",
    "get_models_dir",
]
