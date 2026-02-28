"""Data access API – browse and read data files."""

from __future__ import annotations

import json
import pathlib

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

router = APIRouter()


def _list_files(directory: pathlib.Path) -> list[dict[str, str]]:
    """Return a list of file entries for a directory (non-recursive)."""
    if not directory.is_dir():
        return []
    entries = []
    for entry in sorted(directory.iterdir()):
        entries.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else 0,
        })
    return entries


@router.get("/training-logs")
async def list_training_logs():
    """List files in data/training_logs/."""
    return {"files": _list_files(DATA_DIR / "training_logs")}


@router.get("/trained-models")
async def list_trained_models():
    """List files in data/trained_models/."""
    return {"files": _list_files(DATA_DIR / "trained_models")}


@router.get("/mission-results")
async def list_mission_results():
    """List files in data/mission_results/."""
    return {"files": _list_files(DATA_DIR / "mission_results")}


@router.get("/recordings")
async def list_recordings():
    """List files in data/recordings/ if it exists."""
    return {"files": _list_files(DATA_DIR / "recordings")}


@router.get("/file/{path:path}")
async def read_file(path: str):
    """Read and return contents of a file from the data/ directory."""
    # Path traversal protection
    filepath = DATA_DIR / path
    resolved = filepath.resolve()
    if not str(resolved).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.")

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if filepath.suffix == ".json":
            with open(filepath, "r") as fh:
                data = json.load(fh)
            return data
        else:
            text = filepath.read_text(errors="replace")
            return PlainTextResponse(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
