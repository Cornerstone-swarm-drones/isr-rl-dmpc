"""Configuration management API."""

import pathlib

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

router = APIRouter()


@router.get("/list")
async def list_configs():
    """List all YAML config files in config/."""
    if not CONFIG_DIR.is_dir():
        return {"files": []}
    files = sorted(
        f.name for f in CONFIG_DIR.iterdir() if f.suffix in (".yaml", ".yml")
    )
    return {"files": files}


@router.get("/defaults")
async def get_defaults():
    """Return the parsed default config."""
    try:
        from isr_rl_dmpc.config import load_config

        cfg = load_config()
        # Convert dataclass / object to dict
        if hasattr(cfg, "__dict__"):
            data = _to_serialisable(cfg.__dict__)
        else:
            data = _to_serialisable(cfg)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{filename}")
async def read_config(filename: str):
    """Read and return a YAML config file as JSON."""
    filepath = CONFIG_DIR / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Config file '{filename}' not found")
    try:
        with open(filepath, "r") as fh:
            data = yaml.safe_load(fh)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/{filename}")
async def update_config(filename: str, body: dict):
    """Update a YAML config file from JSON body."""
    filepath = CONFIG_DIR / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Config file '{filename}' not found")
    try:
        with open(filepath, "w") as fh:
            yaml.dump(body, fh, default_flow_style=False, sort_keys=False)
        return {"status": "ok", "filename": filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _to_serialisable(obj):
    """Recursively convert an object to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _to_serialisable(v) for k, v in obj.__dict__.items()}
    # numpy scalars
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
