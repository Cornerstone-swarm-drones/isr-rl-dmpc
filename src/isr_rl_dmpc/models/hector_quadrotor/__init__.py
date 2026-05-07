"""
isr_rl_dmpc.models.hector_quadrotor
====================================
Provides path helpers for the ISR-DMPC swarm drone URDF model.

The URDF is a self-contained, mesh-free quadrotor built entirely from
PyBullet-native primitives (box body, cylinder arms, cylinder motor +
propeller discs).  No external STL or DAE files are required.
"""

from pathlib import Path


def get_urdf_path() -> str:
    """Return the absolute path to the drone URDF file.

    Returns
    -------
    str
        Absolute path to ``quadrotor.urdf`` inside this package directory.
    """
    return str(Path(__file__).parent / "quadrotor.urdf")


def get_models_dir() -> str:
    """Return the absolute path to the parent ``models/`` directory.

    Useful for setting PyBullet's additional search path so that any
    relative paths inside URDF files (such as mesh references) resolve
    correctly without needing to hardcode absolute paths.

    Returns
    -------
    str
        Absolute path to ``src/isr_rl_dmpc/models/``.
    """
    return str(Path(__file__).parents[1])
