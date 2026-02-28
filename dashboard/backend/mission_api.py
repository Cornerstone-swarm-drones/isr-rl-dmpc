"""Mission control API – manages ISRGridEnv simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_sim_state: dict[str, Any] = {
    "env": None,
    "obs": None,
    "info": None,
    "step_count": 0,
    "done": False,
    "total_reward": 0.0,
}

MODULE_CHAIN = [
    "planner",
    "formation",
    "dmpc",
    "attitude",
    "sensors_ekf",
    "classification",
    "threat_alloc",
    "reward",
]


class ResetRequest(BaseModel):
    num_drones: int = Field(default=4, ge=1)
    max_targets: int = Field(default=10, ge=1)
    mission_duration: int = Field(default=200, ge=1)


class StepRequest(BaseModel):
    action: list[float] | None = None


def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _obs_summary(obs: Any, env: Any) -> dict:
    """Extract a compact summary from the observation."""
    summary: dict[str, Any] = {}
    if isinstance(obs, dict):
        summary["keys"] = list(obs.keys())
        if "coverage" in obs:
            summary["coverage"] = _numpy_to_python(obs["coverage"])
    elif isinstance(obs, np.ndarray):
        summary["shape"] = list(obs.shape)
        summary["mean"] = float(obs.mean())
    # Try to get coverage / active drones from env
    try:
        if hasattr(env, "coverage"):
            summary["coverage"] = float(env.coverage)
    except Exception:
        pass
    try:
        if hasattr(env, "num_drones"):
            summary["active_drones"] = int(env.num_drones)
    except Exception:
        pass
    return summary


@router.post("/reset")
async def reset_mission(req: ResetRequest):
    """Create a new ISRGridEnv, reset it, and return the initial observation summary."""
    try:
        from isr_rl_dmpc.gym_env import ISRGridEnv
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot import ISRGridEnv: {exc}")

    try:
        env = ISRGridEnv(
            num_drones=req.num_drones,
            max_targets=req.max_targets,
            mission_duration=req.mission_duration,
        )
        obs, info = env.reset()
        _sim_state.update(
            env=env,
            obs=obs,
            info=info,
            step_count=0,
            done=False,
            total_reward=0.0,
        )
        summary = _obs_summary(obs, env)
        summary["step_count"] = 0
        return _numpy_to_python(summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/step")
async def step_mission(req: StepRequest = StepRequest()):
    """Advance the simulation by one step."""
    env = _sim_state["env"]
    if env is None:
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")
    if _sim_state["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    try:
        if req.action is not None:
            action = np.array(req.action, dtype=np.float32)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        _sim_state.update(
            obs=obs,
            info=info,
            step_count=_sim_state["step_count"] + 1,
            done=done,
            total_reward=_sim_state["total_reward"] + float(reward),
        )
        summary = _obs_summary(obs, env)
        summary.update(
            reward=float(reward),
            done=done,
            step_count=_sim_state["step_count"],
            total_reward=_sim_state["total_reward"],
        )
        if info:
            summary["info"] = _numpy_to_python(info)
        return _numpy_to_python(summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def mission_status():
    """Return current simulation state summary."""
    if _sim_state["env"] is None:
        return {"active": False}

    env = _sim_state["env"]
    summary: dict[str, Any] = {
        "active": True,
        "step_count": _sim_state["step_count"],
        "done": _sim_state["done"],
        "total_reward": _sim_state["total_reward"],
    }
    try:
        if hasattr(env, "coverage"):
            summary["coverage"] = float(env.coverage)
    except Exception:
        pass
    try:
        if hasattr(env, "num_drones"):
            summary["active_drones"] = int(env.num_drones)
    except Exception:
        pass
    return _numpy_to_python(summary)


@router.get("/module-chain")
async def module_chain():
    """Return the module chain info."""
    return {"modules": MODULE_CHAIN}
