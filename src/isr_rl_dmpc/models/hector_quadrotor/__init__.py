"""
hector_quadrotor — Canonical URDF and mesh assets for the ISR-DMPC drone.

Model source: hector_quadrotor (TU Darmstadt)
https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor

Provides ``get_urdf_path()`` so that any simulator (PyBullet, etc.) can
locate the URDF without hard-coded relative paths.

Usage
-----
>>> from isr_rl_dmpc.models.hector_quadrotor import get_urdf_path
>>> import pybullet as p
>>> drone_id = p.loadURDF(get_urdf_path(), basePosition=[0, 0, 1])
"""

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_URDF_PATH = _PACKAGE_DIR / "drone.urdf"
_MODELS_DIR = _PACKAGE_DIR.parent          # src/isr_rl_dmpc/models/


def get_urdf_path() -> str:
    """Return the absolute path to the hector_quadrotor URDF.

    The URDF references meshes via relative paths (``meshes/hector_quadrotor/...``)
    resolved from the ``src/isr_rl_dmpc/models/`` directory.  To make PyBullet
    resolve them correctly, callers should use::

        p.loadURDF(get_urdf_path(), ...)

    PyBullet resolves ``<mesh filename="..."/>`` paths relative to the URDF
    directory, so passing the absolute URDF path ensures the relative mesh
    references resolve to ``src/isr_rl_dmpc/models/meshes/hector_quadrotor/``.
    """
    return str(_URDF_PATH)


def get_models_dir() -> str:
    """Return the absolute path to the ``src/isr_rl_dmpc/models/`` directory."""
    return str(_MODELS_DIR)


__all__ = ["get_urdf_path", "get_models_dir"]
