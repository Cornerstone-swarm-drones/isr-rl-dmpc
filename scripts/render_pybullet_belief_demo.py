#!/usr/bin/env python3
"""Render a presentation MP4 for the belief-coverage ISR demo.

The renderer is intentionally lightweight: the live ``BeliefCoverageEnv`` owns
the patrol, moving-threat, EKF, and interceptor logic; PyBullet is only used as
the 3-D presentation layer.  It loads the Hector quadrotor URDF for each drone
and replays the deterministic environment rollout into an MP4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "imageio/imageio-ffmpeg is required for MP4 writing. "
        "Install with: python -m pip install imageio imageio-ffmpeg"
    ) from exc

try:
    import pybullet as p
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "pybullet is required for this renderer. "
        "Install with a Python that has a PyBullet wheel, for example the local "
        "Anaconda Python used in this workspace."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("pyyaml is required to load mission_scenarios.yaml") from exc

from isr_rl_dmpc.gym_env import BeliefCoverageEnv
from isr_rl_dmpc.models.hector_quadrotor import get_urdf_path


DRONE_COLORS = [
    (0.15, 0.47, 0.74, 1.0),
    (0.91, 0.36, 0.20, 1.0),
    (0.16, 0.64, 0.53, 1.0),
    (0.58, 0.35, 0.65, 1.0),
    (0.96, 0.64, 0.32, 1.0),
    (0.15, 0.27, 0.33, 1.0),
]


@dataclass
class FrameState:
    step: int
    drone_xyz: np.ndarray
    drone_yaw: np.ndarray
    selected_cells: np.ndarray
    tracking_drones: np.ndarray
    pending_watch_drones: np.ndarray
    active_threat_cells: np.ndarray
    pending_threat_cells: np.ndarray
    threat_xy: np.ndarray
    pending_xy: np.ndarray
    track_xy: np.ndarray
    interceptor_xy: np.ndarray
    interceptor_active: bool
    threat_confirmed: bool
    base_confirmed: bool
    track_confidence: float
    threat_removed: bool
    threat_cycles_completed: int
    pending_available: bool
    pending_promoted: bool
    mission_failed: bool
    mean_patrol_risk: float
    neglect_pressure: float
    home_fraction: float
    tracking_fraction: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Phase 1-3 belief-coverage behavior as a PyBullet MP4."
    )
    parser.add_argument("--output", default=str(ROOT / "visualizations" / "presentation" / "slide13_video_pybullet_belief_isr.mp4"))
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-drones", type=int, default=6)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--world-scale", type=float, default=0.04)
    parser.add_argument("--drone-scale", type=float, default=7.0)
    parser.add_argument("--fixed-altitude-render", type=float, default=1.8)
    parser.add_argument("--threat-speed-case", choices=["slow", "medium", "fast"], default="fast")
    parser.add_argument("--max-threat-cycles", type=int, default=2)
    parser.add_argument("--pending-threat-delay-steps", type=int, default=12)
    parser.add_argument("--camera-preset", choices=["oblique", "top"], default="oblique")
    return parser.parse_args()


def _scenario_config() -> dict:
    path = ROOT / "config" / "mission_scenarios.yaml"
    with path.open("r", encoding="utf-8") as handle:
        scenarios = yaml.safe_load(handle) or {}
    return scenarios["area_surveillance"]


def _make_env(args: argparse.Namespace) -> BeliefCoverageEnv:
    cfg = _scenario_config()
    belief_cfg = dict(cfg.get("belief_coverage", {}))
    threat_cfg = dict(belief_cfg.get("persistent_threats", {}))
    return BeliefCoverageEnv(
        scenario="area_surveillance",
        num_drones=int(args.num_drones),
        mission_duration=int(cfg.get("max_duration", 1000)),
        area_size=tuple(cfg.get("area_size", [400.0, 400.0])),
        fixed_altitude=float(cfg.get("min_altitude", 30.0)),
        communication_range=float(cfg.get("communication_range", 250.0)),
        sensor_range=float(belief_cfg.get("sensor_range", 120.0)),
        growth_rate=float(belief_cfg.get("growth_rate", 0.03)),
        global_sync_steps=int(belief_cfg.get("global_sync_steps", 25)),
        base_station=tuple(belief_cfg.get("base_station", [200.0, 200.0])),
        suspicious_zones=belief_cfg.get("suspicious_zones", []),
        enable_neighbor_sharing=True,
        enable_goal_projection=True,
        enable_persistent_threats=bool(threat_cfg.get("enabled", True)),
        max_threat_cycles=int(args.max_threat_cycles),
        persistent_threat_speed_case=str(args.threat_speed_case),
        enable_sequential_pending_threats=True,
        pending_threat_delay_steps=int(args.pending_threat_delay_steps),
        interceptor_guidance_mode="ekf",
        response_policy="phase2",
        validation_dynamics="fast_planar",
        threat_belief_mode="limited_strict",
        base_sensor={"enabled": True, "delay_steps": 1},
    )


def _snapshot(env: BeliefCoverageEnv, info: dict, step: int) -> FrameState:
    shared_track = info["shared_track_state"]
    interceptor = info["interceptor_state"]
    return FrameState(
        step=int(step),
        drone_xyz=env._drone_states[:, :3].copy(),  # Presentation renderer reads env state only.
        drone_yaw=env._drone_states[:, 9].copy(),
        selected_cells=np.asarray(info["selected_target_cells"], dtype=np.int32).copy(),
        tracking_drones=np.asarray(info["tracking_bias_drones"], dtype=np.int32).copy(),
        pending_watch_drones=np.asarray(info["pending_watchlist_drones"], dtype=np.int32).copy(),
        active_threat_cells=np.asarray(info["active_threat_cells"], dtype=np.int32).copy(),
        pending_threat_cells=np.asarray(info["pending_threat_cells"], dtype=np.int32).copy(),
        threat_xy=np.asarray(info["threat_state"]["position_xy"], dtype=np.float64).copy(),
        pending_xy=np.asarray(info["threat_state"]["pending_position_xy"], dtype=np.float64).copy(),
        track_xy=np.asarray(shared_track["position_xy"], dtype=np.float64).copy(),
        interceptor_xy=np.asarray(interceptor["position_xy"], dtype=np.float64).copy(),
        interceptor_active=bool(interceptor["active"]),
        threat_confirmed=bool(info["threat_confirmed"]),
        base_confirmed=bool(info["base_threat_state"]["confirmed"]),
        track_confidence=float(shared_track["confidence"]),
        threat_removed=bool(info["threat_removed_this_step"]),
        threat_cycles_completed=int(info["threat_cycles_completed"]),
        pending_available=bool(info["pending_threat_available"]),
        pending_promoted=bool(info["pending_threat_promoted_this_step"]),
        mission_failed=bool(info["mission_failed"]),
        mean_patrol_risk=float(info["mean_patrol_risk_belief"]),
        neglect_pressure=float(info["neglect_pressure"]),
        home_fraction=float(np.mean(info["selected_in_home"])),
        tracking_fraction=float(np.mean(info["tracking_bias_drones"])),
    )


def _rollout(args: argparse.Namespace) -> tuple[BeliefCoverageEnv, list[FrameState], dict]:
    env = _make_env(args)
    _, info = env.reset(seed=args.seed)
    frames = [_snapshot(env, info, 0)]
    last_info = info
    for step in range(1, args.steps + 1):
        action = env.select_patrol_action()
        _, _, terminated, truncated, last_info = env.step(action)
        frames.append(_snapshot(env, last_info, step))
        if terminated or truncated:
            break
    return env, frames, last_info


def _world_xy(xy: Iterable[float], base: np.ndarray, scale: float) -> tuple[float, float]:
    arr = np.asarray(xy, dtype=np.float64)
    return float((arr[0] - base[0]) * scale), float((arr[1] - base[1]) * scale)


def _body_at_xy(
    xy: Iterable[float],
    *,
    base: np.ndarray,
    scale: float,
    z: float,
    yaw: float = 0.0,
) -> tuple[list[float], tuple[float, float, float, float]]:
    x, y = _world_xy(xy, base, scale)
    return [x, y, z], p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])


def _hide_body(body_id: int) -> None:
    p.resetBasePositionAndOrientation(body_id, [0.0, 0.0, -50.0], [0.0, 0.0, 0.0, 1.0])


def _create_box_body(half_extents: list[float], rgba: tuple[float, float, float, float]) -> int:
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual)


def _create_sphere_body(radius: float, rgba: tuple[float, float, float, float]) -> int:
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual)


def _load_drone_bodies(env: BeliefCoverageEnv, args: argparse.Namespace) -> list[int]:
    drone_ids: list[int] = []
    for i in range(env.num_drones):
        start_xy = env.base_station + np.array([(i - env.num_drones / 2.0) * 3.0, 0.0])
        pos, quat = _body_at_xy(
            start_xy,
            base=env.base_station,
            scale=args.world_scale,
            z=args.fixed_altitude_render,
        )
        body_id = p.loadURDF(
            get_urdf_path(),
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=True,
            globalScaling=float(args.drone_scale),
        )
        rgba = DRONE_COLORS[i % len(DRONE_COLORS)]
        for link_idx in [-1] + list(range(p.getNumJoints(body_id))):
            try:
                p.changeVisualShape(body_id, link_idx, rgbaColor=rgba)
            except Exception:
                pass
        drone_ids.append(body_id)
    return drone_ids


def _camera_matrices(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    if args.camera_preset == "top":
        eye = [0.0, -0.1, 24.0]
        target = [0.0, 0.0, 0.0]
    else:
        eye = [0.0, -15.5, 13.0]
        target = [0.0, 0.0, 0.0]
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
    projection = p.computeProjectionMatrixFOV(
        fov=50.0,
        aspect=float(args.width) / float(args.height),
        nearVal=0.1,
        farVal=100.0,
    )
    return view, projection


def _add_static_scene(env: BeliefCoverageEnv, args: argparse.Namespace) -> None:
    area = np.asarray(env.area_size, dtype=np.float64)
    ground_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[area[0] * args.world_scale / 2.0, area[1] * args.world_scale / 2.0, 0.015],
        rgbaColor=[0.78, 0.82, 0.78, 1.0],
    )
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=ground_visual, basePosition=[0.0, 0.0, -0.02])

    # Base/charging station.
    base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.55, length=0.12, rgbaColor=[0.0, 0.28, 0.9, 1.0])
    base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.55, height=0.12)
    p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_collision, baseVisualShapeIndex=base_visual, basePosition=[0.0, 0.0, 0.08])

    # Home-strip boundaries as low black rails.
    home_boundaries = env.get_home_strip_boundaries()
    for boundary in np.asarray(home_boundaries["positions"], dtype=np.float64):
        if home_boundaries["axis"] == "x":
            x, _ = _world_xy([boundary, env.base_station[1]], env.base_station, args.world_scale)
            pos = [x, 0.0, 0.04]
            ext = [0.025, area[1] * args.world_scale / 2.0, 0.025]
        else:
            _, y = _world_xy([env.base_station[0], boundary], env.base_station, args.world_scale)
            pos = [0.0, y, 0.04]
            ext = [area[0] * args.world_scale / 2.0, 0.025, 0.025]
        rail = _create_box_body(ext, (0.08, 0.08, 0.08, 0.55))
        p.resetBasePositionAndOrientation(rail, pos, [0.0, 0.0, 0.0, 1.0])

    # Outer boundary.
    hx = area[0] * args.world_scale / 2.0
    hy = area[1] * args.world_scale / 2.0
    for pos, ext in [
        ([0, hy, 0.06], [hx, 0.035, 0.035]),
        ([0, -hy, 0.06], [hx, 0.035, 0.035]),
        ([hx, 0, 0.06], [0.035, hy, 0.035]),
        ([-hx, 0, 0.06], [0.035, hy, 0.035]),
    ]:
        rail = _create_box_body(ext, (0.05, 0.05, 0.05, 0.8))
        p.resetBasePositionAndOrientation(rail, pos, [0.0, 0.0, 0.0, 1.0])


def _overlay_text(image: np.ndarray, frame: FrameState) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.rounded_rectangle((22, 18, 550, 150), radius=16, fill=(8, 16, 22, 188))
    draw.text((42, 32), "Belief ISR: Patrol + EKF + Interceptor", fill=(255, 255, 255, 255), font=title_font)
    line = (
        f"step {frame.step:03d} | threat eliminations {frame.threat_cycles_completed} | "
        f"track conf {frame.track_confidence:.2f}"
    )
    draw.text((42, 74), line, fill=(235, 245, 255, 255), font=body_font)
    line2 = (
        f"home patrol {frame.home_fraction:.2f} | tracking drones {frame.tracking_fraction:.2f} | "
        f"neglect {frame.neglect_pressure:.2f}"
    )
    draw.text((42, 104), line2, fill=(235, 245, 255, 255), font=body_font)

    status = []
    if frame.threat_confirmed:
        status.append("confirmed")
    if frame.base_confirmed:
        status.append("base aware")
    if frame.interceptor_active:
        status.append("interceptor active")
    if frame.pending_available:
        status.append("pending threat")
    if frame.pending_promoted:
        status.append("pending promoted")
    if frame.threat_removed:
        status.append("threat removed")
    if frame.mission_failed:
        status.append("MISSION FAIL")
    if status:
        draw.rounded_rectangle((22, 160, 520, 205), radius=12, fill=(215, 68, 54, 205))
        draw.text((42, 170), " | ".join(status), fill=(255, 255, 255, 255), font=body_font)
    return np.asarray(pil)


def _render_video(env: BeliefCoverageEnv, frames: list[FrameState], args: argparse.Namespace) -> dict:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not start PyBullet DIRECT renderer")

    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        _add_static_scene(env, args)
        drone_ids = _load_drone_bodies(env, args)

        cell_extent = float(env.grid_resolution) * args.world_scale * 0.46
        active_boxes = [_create_box_body([cell_extent, cell_extent, 0.08], (0.95, 0.05, 0.05, 0.92)) for _ in range(4)]
        pending_boxes = [_create_box_body([cell_extent, cell_extent, 0.055], (1.0, 0.56, 0.05, 0.78)) for _ in range(4)]
        interceptor_body = _create_sphere_body(0.25, (1.0, 0.9, 0.05, 1.0))
        track_body = _create_sphere_body(0.20, (0.70, 0.15, 1.0, 0.85))
        target_body = _create_sphere_body(0.14, (0.05, 0.05, 0.05, 0.95))

        view, projection = _camera_matrices(args)
        rendered_frames = 0
        stride = max(1, int(args.frame_stride))
        with imageio.get_writer(out_path, fps=int(args.fps), codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame in frames[::stride]:
                for i, body_id in enumerate(drone_ids):
                    xy = frame.drone_xyz[i, :2]
                    pos, quat = _body_at_xy(
                        xy,
                        base=env.base_station,
                        scale=args.world_scale,
                        z=args.fixed_altitude_render,
                        yaw=float(frame.drone_yaw[i]),
                    )
                    p.resetBasePositionAndOrientation(body_id, pos, quat)

                    # Show the current high-level cell target as a small black puck.
                    if i == 0 and frame.selected_cells.size > 0:
                        target_xy = env.cell_centers_xy[int(frame.selected_cells[i])]
                        target_pos, target_quat = _body_at_xy(
                            target_xy,
                            base=env.base_station,
                            scale=args.world_scale,
                            z=0.16,
                        )
                        p.resetBasePositionAndOrientation(target_body, target_pos, target_quat)

                for boxes, cells, z in [
                    (active_boxes, frame.active_threat_cells, 0.16),
                    (pending_boxes, frame.pending_threat_cells if frame.pending_available else np.zeros(0, dtype=np.int32), 0.12),
                ]:
                    for j, body_id in enumerate(boxes):
                        if j < int(cells.size):
                            xy = env.cell_centers_xy[int(cells[j])]
                            pos, quat = _body_at_xy(xy, base=env.base_station, scale=args.world_scale, z=z)
                            p.resetBasePositionAndOrientation(body_id, pos, quat)
                        else:
                            _hide_body(body_id)

                if frame.interceptor_active and np.all(np.isfinite(frame.interceptor_xy)):
                    pos, quat = _body_at_xy(
                        frame.interceptor_xy,
                        base=env.base_station,
                        scale=args.world_scale,
                        z=1.2,
                    )
                    p.resetBasePositionAndOrientation(interceptor_body, pos, quat)
                else:
                    _hide_body(interceptor_body)

                if np.all(np.isfinite(frame.track_xy)):
                    pos, quat = _body_at_xy(frame.track_xy, base=env.base_station, scale=args.world_scale, z=1.0)
                    p.resetBasePositionAndOrientation(track_body, pos, quat)
                else:
                    _hide_body(track_body)

                p.stepSimulation()
                _, _, rgba, _, _ = p.getCameraImage(
                    int(args.width),
                    int(args.height),
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=p.ER_TINY_RENDERER,
                )
                img = np.reshape(np.asarray(rgba, dtype=np.uint8), (int(args.height), int(args.width), 4))[:, :, :3]
                writer.append_data(_overlay_text(img, frame))
                rendered_frames += 1
    finally:
        p.disconnect()

    return {
        "output": str(out_path),
        "frames_written": rendered_frames,
        "fps": int(args.fps),
        "duration_seconds": float(rendered_frames) / float(args.fps),
    }


def main() -> None:
    args = _parse_args()
    env, frames, final_info = _rollout(args)
    try:
        render_summary = _render_video(env, frames, args)
    finally:
        env.close()

    summary = {
        **render_summary,
        "steps_simulated": len(frames) - 1,
        "seed": int(args.seed),
        "num_drones": int(args.num_drones),
        "threat_speed_case": str(args.threat_speed_case),
        "threat_cycles_completed": int(final_info["threat_cycles_completed"]),
        "mission_failed": bool(final_info["mission_failed"]),
        "threat_belief_mode": final_info["threat_belief_mode"],
        "response_policy": final_info["response_policy"],
        "validation_dynamics": final_info["validation_dynamics"],
        "interceptor_guidance_mode": final_info["interceptor_guidance_mode"],
        "base_confirm_step": int(final_info["base_threat_state"]["first_confirmation_step"]),
        "track_confidence_final": float(final_info["shared_track_state"]["confidence"]),
    }

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output).with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("PyBullet belief-coverage presentation render")
    print("=" * 72)
    print(f"Output MP4            : {summary['output']}")
    print(f"Summary JSON          : {summary_path}")
    print(f"Frames written        : {summary['frames_written']}")
    print(f"Duration [s]          : {summary['duration_seconds']:.2f}")
    print(f"Threat eliminations   : {summary['threat_cycles_completed']}")
    print(f"Mission failed        : {summary['mission_failed']}")


if __name__ == "__main__":
    main()
