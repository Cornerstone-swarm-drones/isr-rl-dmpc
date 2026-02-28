"""Training control API – manages training subprocesses."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING_LOGS_DIR = ROOT / "data" / "training_logs"
SCRIPTS_DIR = ROOT / "scripts"

router = APIRouter()


def _parse_metrics_csv(csv_path: pathlib.Path) -> list[dict[str, Any]]:
    """Parse a CSV metrics file into a list of dicts with numeric coercion."""
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


_training_state: dict[str, Any] = {
    "process": None,
    "run_id": None,
    "status": "idle",
    "start_time": None,
}


class TrainRequest(BaseModel):
    num_episodes: int = Field(default=100, ge=1)
    num_steps: int = Field(default=1000, ge=1)
    seed: int = 42
    device: str = "cpu"
    config_path: str | None = None
    task: str = "recon"


class SweepRequest(BaseModel):
    num_trials: int = Field(default=10, ge=1)
    device: str = "cpu"


def _check_process() -> None:
    """Update status if the process has finished."""
    proc = _training_state["process"]
    if proc is not None and proc.poll() is not None:
        _training_state["status"] = "completed"
        _training_state["process"] = None


@router.post("/start")
async def start_training(req: TrainRequest):
    """Launch a training run as a subprocess."""
    _check_process()
    if _training_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A training run is already active.")

    run_id = str(uuid.uuid4())[:8]
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_agent.py"),
        "--num-episodes", str(req.num_episodes),
        "--num-steps", str(req.num_steps),
        "--seed", str(req.seed),
        "--device", req.device,
        "--task", req.task,
    ]
    if req.config_path:
        cmd.extend(["--config", req.config_path])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {exc}")

    _training_state.update(
        process=proc,
        run_id=run_id,
        status="running",
        start_time=time.time(),
    )
    return {"run_id": run_id, "status": "running"}


@router.post("/stop")
async def stop_training():
    """Kill the active training process."""
    _check_process()
    proc = _training_state["process"]
    if proc is None:
        raise HTTPException(status_code=400, detail="No active training process.")

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    _training_state.update(process=None, status="idle")
    return {"status": "stopped"}


@router.get("/status")
async def training_status():
    """Return current training status."""
    _check_process()
    result: dict[str, Any] = {
        "status": _training_state["status"],
        "run_id": _training_state["run_id"],
    }
    if _training_state["start_time"] is not None:
        result["elapsed_seconds"] = round(time.time() - _training_state["start_time"], 2)
    return result


@router.post("/sweep")
async def start_sweep(req: SweepRequest):
    """Start a hyperparameter sweep subprocess."""
    _check_process()
    if _training_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A training run is already active.")

    run_id = f"sweep-{uuid.uuid4().hex[:8]}"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hyperparameter_search.py"),
        "--num-trials", str(req.num_trials),
        "--device", req.device,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start sweep: {exc}")

    _training_state.update(
        process=proc,
        run_id=run_id,
        status="running",
        start_time=time.time(),
    )
    return {"run_id": run_id, "status": "running"}


@router.get("/runs")
async def list_runs():
    """List training run directories from data/training_logs/."""
    if not TRAINING_LOGS_DIR.is_dir():
        return {"runs": []}
    runs = []
    for entry in sorted(TRAINING_LOGS_DIR.iterdir()):
        if entry.is_dir():
            runs.append({
                "run_id": entry.name,
                "path": str(entry),
                "timestamp": entry.name,
            })
    return {"runs": runs}


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    """Read the CSV metrics file from a run directory."""
    run_dir = TRAINING_LOGS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    csv_files = list(run_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="No metrics CSV found in run directory.")

    metrics_file = csv_files[0]
    try:
        rows = _parse_metrics_csv(metrics_file)
        return {"metrics": rows, "filename": metrics_file.name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/runs/{run_id}/stats")
async def get_run_stats(run_id: str):
    """Read training_stats.json from a run directory."""
    run_dir = TRAINING_LOGS_DIR / run_id
    stats_file = run_dir / "training_stats.json"
    if not stats_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"training_stats.json not found for run '{run_id}'.",
        )

    try:
        with open(stats_file, "r") as fh:
            data = json.load(fh)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/compare")
async def compare_runs(run_ids: str = Query(..., description="Comma-separated run IDs")):
    """Return metrics for multiple runs for comparison."""
    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    results: dict[str, Any] = {}

    for rid in ids:
        run_dir = TRAINING_LOGS_DIR / rid
        if not run_dir.is_dir():
            results[rid] = {"error": f"Run '{rid}' not found"}
            continue

        csv_files = list(run_dir.glob("*.csv"))
        if not csv_files:
            results[rid] = {"error": "No metrics CSV found"}
            continue

        try:
            results[rid] = {"metrics": _parse_metrics_csv(csv_files[0])}
        except Exception as exc:
            results[rid] = {"error": str(exc)}

    return {"runs": results}
