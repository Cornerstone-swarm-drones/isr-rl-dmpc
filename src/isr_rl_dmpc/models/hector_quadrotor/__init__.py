"""
isr_rl_dmpc.models.hector_quadrotor
====================================
Helper utilities for the ISR-DMPC drone URDF model.

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
