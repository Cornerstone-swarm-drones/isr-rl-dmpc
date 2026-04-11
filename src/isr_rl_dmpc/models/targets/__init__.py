"""
targets — URDF assets for target entities in PyBullet simulation.

Each target type (hostile, neutral, friendly, unknown) has a corresponding
URDF that references an OBJ mesh in ``../meshes/``.

Mesh sources
------------
- cesium_air.obj     — Converted from CesiumGS Cesium_Air.glb (Apache 2.0)
- cesium_milk_truck.obj — Converted from CesiumGS CesiumMilkTruck.glb (Apache 2.0)
- box.obj            — Converted from CesiumGS Box.glb (Apache 2.0)

Usage
-----
>>> from isr_rl_dmpc.models.targets import get_target_urdf_path
>>> import pybullet as p
>>> tgt_id = p.loadURDF(get_target_urdf_path("hostile"), basePosition=[0, 0, 0])
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# Mapping of target type names to their URDF filenames
_TARGET_URDFS = {
    "hostile": _PACKAGE_DIR / "hostile_target.urdf",
    "neutral": _PACKAGE_DIR / "neutral_target.urdf",
    "friendly": _PACKAGE_DIR / "friendly_target.urdf",
    "unknown": _PACKAGE_DIR / "unknown_target.urdf",
}


def get_target_urdf_path(target_type: str) -> str:
    """Return the absolute path to the URDF for the given target type.

    Parameters
    ----------
    target_type : str
        One of ``"hostile"``, ``"neutral"``, ``"friendly"``, ``"unknown"``.

    Returns
    -------
    str
        Absolute filesystem path to the URDF file.

    Raises
    ------
    ValueError
        If *target_type* is not recognised.
    """
    key = target_type.lower()
    if key not in _TARGET_URDFS:
        raise ValueError(
            f"Unknown target type {target_type!r}. "
            f"Valid types: {sorted(_TARGET_URDFS)}"
        )
    return str(_TARGET_URDFS[key])


def available_target_types() -> list:
    """Return a sorted list of available target type names."""
    return sorted(_TARGET_URDFS)


__all__ = ["get_target_urdf_path", "available_target_types"]
