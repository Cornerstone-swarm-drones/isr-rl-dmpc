#!/usr/bin/env python3
"""
Diagnostic visualization for the Phase 1 patrol-risk baseline.

This script runs the deterministic home-strip patrol policy and saves a
matplotlib figure that compares:

* backend truth risk
* fused belief risk
* moving persistent-threat diagnostics
* rollout-level patrol behavior over time
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    yaml = None

sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.gym_env import BeliefCoverageEnv


DRONE_COLORS = [
    "#2E86AB",
    "#E76F51",
    "#2A9D8F",
    "#8E5A9F",
    "#F4A261",
    "#264653",
]

DEFAULT_SCENARIOS = {
    "area_surveillance": {
        "area_size": [400.0, 400.0],
        "num_drones": 4,
        "max_duration": 1000.0,
        "communication_range": 250.0,
        "min_altitude": 30.0,
        "belief_coverage": {
            "base_station": [200.0, 200.0],
            "sensor_range": 120.0,
            "growth_rate": 0.03,
            "global_sync_steps": 25,
            "persistent_threats": {
                "enabled": True,
                "max_cycles": 3,
            },
            "suspicious_zones": [
                {"center": [120.0, 120.0], "radius": 45.0, "score": 0.55},
                {"center": [280.0, 260.0], "radius": 55.0, "score": 0.8},
            ],
        },
    }
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the Phase 1 patrol-risk baseline."
    )
    parser.add_argument("--scenario", default="area_surveillance")
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-drones", type=int, default=None)
    parser.add_argument(
        "--assist-spike-drone",
        type=int,
        default=None,
        help="Inject a local assist-band risk spike for this drone during rollout",
    )
    parser.add_argument(
        "--assist-spike-step",
        type=int,
        default=4,
        help="Step index at which to inject the optional assist-band risk spike",
    )
    parser.add_argument(
        "--assist-spike-level",
        type=float,
        default=0.95,
        help="Risk level used for the optional assist-band spike",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "visualizations" / "phase1_patrol_risk.png"),
        help="Path for the saved PNG diagnostic figure",
    )
    parser.add_argument("--show", action="store_true", help="Also open the figure window")
    parser.add_argument(
        "--disable-neighbor-sharing",
        action="store_true",
        help="Turn off local neighbor sharing during the rollout",
    )
    parser.add_argument(
        "--disable-goal-projection",
        action="store_true",
        help="Turn off connectivity-aware goal projection",
    )
    parser.add_argument(
        "--disable-persistent-threats",
        action="store_true",
        help="Turn off the moving persistent-threat loop for patrol-only diagnostics",
    )
    parser.add_argument(
        "--max-threat-cycles",
        type=int,
        default=None,
        help="Maximum number of moving threat cycles to run in one rollout",
    )
    parser.add_argument(
        "--force-threat-home-drone",
        type=int,
        default=None,
        help="Force the initial moving threat patch into this drone's home strip for diagnostics",
    )
    parser.add_argument(
        "--ignore-suspicious-zones",
        action="store_true",
        help="Ignore configured suspicious zones and use uniform threat-patch spawn preference",
    )
    parser.add_argument(
        "--threat-speed-case",
        choices=["slow", "medium", "fast"],
        default="medium",
        help="Named moving-threat speed case to visualize",
    )
    parser.add_argument(
        "--threat-speed",
        type=float,
        default=None,
        help="Override the moving-threat speed directly in m/s",
    )
    return parser.parse_args()


def _load_scenario(path: str, scenario_name: str) -> dict:
    if yaml is None:
        if scenario_name not in DEFAULT_SCENARIOS:
            available = ", ".join(sorted(DEFAULT_SCENARIOS)) or "<none>"
            raise KeyError(
                f"Scenario '{scenario_name}' not found in built-in defaults. Available: {available}"
            )
        return DEFAULT_SCENARIOS[scenario_name]

    with open(path, "r", encoding="utf-8") as handle:
        scenarios = yaml.safe_load(handle) or {}
    if scenario_name not in scenarios:
        available = ", ".join(sorted(scenarios)) or "<none>"
        raise KeyError(f"Scenario '{scenario_name}' not found. Available: {available}")
    return scenarios[scenario_name]


def _build_env_kwargs(scenario_cfg: dict, args: argparse.Namespace) -> dict:
    belief_cfg = dict(scenario_cfg.get("belief_coverage", {}))
    threat_cfg = dict(belief_cfg.get("persistent_threats", {}))
    return {
        "scenario": "area_surveillance",
        "num_drones": int(args.num_drones or scenario_cfg.get("num_drones", 4)),
        "mission_duration": int(scenario_cfg.get("max_duration", 1000)),
        "area_size": tuple(scenario_cfg.get("area_size", [400.0, 400.0])),
        "fixed_altitude": float(scenario_cfg.get("min_altitude", 30.0)),
        "communication_range": float(scenario_cfg.get("communication_range", 250.0)),
        "sensor_range": float(belief_cfg.get("sensor_range", 120.0)),
        "growth_rate": float(belief_cfg.get("growth_rate", 0.03)),
        "global_sync_steps": int(belief_cfg.get("global_sync_steps", 25)),
        "base_station": tuple(belief_cfg.get("base_station", [200.0, 200.0])),
        "suspicious_zones": [] if args.ignore_suspicious_zones else belief_cfg.get("suspicious_zones", []),
        "enable_neighbor_sharing": not args.disable_neighbor_sharing,
        "enable_goal_projection": not args.disable_goal_projection,
        "enable_persistent_threats": bool(
            threat_cfg.get("enabled", True) and not args.disable_persistent_threats
        ),
        "max_threat_cycles": int(
            args.max_threat_cycles
            if args.max_threat_cycles is not None
            else threat_cfg.get("max_cycles", 3)
        ),
        "persistent_threat_speed_case": str(args.threat_speed_case),
        "persistent_threat_speed": args.threat_speed,
    }


def _run_rollout(
    env: BeliefCoverageEnv,
    *,
    steps: int,
    seed: int,
    assist_spike_drone: int | None = None,
    assist_spike_step: int = 4,
    assist_spike_level: float = 0.95,
    force_threat_home_drone: int | None = None,
) -> tuple[dict, dict]:
    _, info = env.reset(seed=seed)
    if force_threat_home_drone is not None:
        _force_initial_threat_patch(env, drone_idx=int(force_threat_home_drone))
        info = env._build_info(
            connectivity=env._compute_connectivity(),
            reward=0.0,
            reward_components={},
            solve_times=np.zeros(env.num_drones, dtype=np.float64),
        )

    history = {
        "positions": [env._drone_states[:, :2].copy()],
        "selected_target_cells": [info["selected_target_cells"].copy()],
        "selected_in_home": [info["selected_in_home"].copy()],
        "selected_in_assist": [info["selected_in_assist"].copy()],
        "in_home_region": [info["in_home_region"].copy()],
        "patrol_detouring": [info["patrol_detouring"].copy()],
        "tracking_bias_drones": [info["tracking_bias_drones"].copy()],
        "mean_patrol_risk_belief": [float(info["mean_patrol_risk_belief"])],
        "mean_threat_belief": [float(info["mean_threat_belief"])],
        "mean_threat_persistence": [float(info["mean_threat_persistence_score"])],
        "active_threat_confirmation_score": [float(info["active_threat_confirmation_score"])],
        "low_risk_fraction": [float(info["low_risk_fraction"])],
        "never_observed_fraction": [float(info["never_observed_fraction"])],
        "home_fraction": [float(np.mean(info["selected_in_home"]))],
        "assist_fraction": [float(np.mean(info["selected_in_assist"]))],
        "in_home_fraction": [float(np.mean(info["in_home_region"]))],
        "detour_fraction": [float(np.mean(info["patrol_detouring"]))],
        "tracking_fraction": [float(np.mean(info["tracking_bias_drones"]))],
        "threat_active": [float(info["active_threat"])],
        "threat_confirmed": [float(info["threat_confirmed"])],
        "central_command_notified": [float(info["central_command_notified"])],
        "interceptor_dispatched": [float(info["interceptor_state"]["dispatched_this_step"])],
        "interceptor_active": [float(info["interceptor_state"]["active"])],
        "interceptor_distance": [float(info["interceptor_state"]["distance_to_target"])],
        "threat_position_xy": [info["threat_state"]["position_xy"].copy()],
        "threat_distance_to_base": [
            float(np.linalg.norm(info["threat_state"]["position_xy"] - env.base_station))
            if bool(info["active_threat"])
            else float("nan")
        ],
        "threat_cycle_index": [float(info["threat_cycle_index"])],
        "threat_cycles_completed": [float(info["threat_cycles_completed"])],
        "threat_removed": [float(info["threat_removed_this_step"])],
        "threat_respawned": [float(info["threat_respawned_this_step"])],
        "physical_base_reached": [float(info["physical_base_reached"])],
        "mission_failed": [float(info["mission_failed"])],
        "home_coverage_fraction": [info["home_coverage_fraction"].copy()],
        "home_low_risk_fraction": [info["home_low_risk_fraction"].copy()],
        "home_neglected_fraction": [info["home_neglected_fraction"].copy()],
        "assist_spike_target": None,
    }

    last_info = info
    for step_idx in range(steps):
        if assist_spike_drone is not None and step_idx == assist_spike_step:
            history["assist_spike_target"] = _inject_assist_spike(
                env,
                drone_idx=int(assist_spike_drone),
                level=float(assist_spike_level),
            )
        action = env.select_patrol_action()
        _, _, terminated, truncated, last_info = env.step(action)
        history["positions"].append(env._drone_states[:, :2].copy())
        history["selected_target_cells"].append(last_info["selected_target_cells"].copy())
        history["selected_in_home"].append(last_info["selected_in_home"].copy())
        history["selected_in_assist"].append(last_info["selected_in_assist"].copy())
        history["in_home_region"].append(last_info["in_home_region"].copy())
        history["patrol_detouring"].append(last_info["patrol_detouring"].copy())
        history["tracking_bias_drones"].append(last_info["tracking_bias_drones"].copy())
        history["mean_patrol_risk_belief"].append(float(last_info["mean_patrol_risk_belief"]))
        history["mean_threat_belief"].append(float(last_info["mean_threat_belief"]))
        history["mean_threat_persistence"].append(float(last_info["mean_threat_persistence_score"]))
        history["active_threat_confirmation_score"].append(float(last_info["active_threat_confirmation_score"]))
        history["low_risk_fraction"].append(float(last_info["low_risk_fraction"]))
        history["never_observed_fraction"].append(float(last_info["never_observed_fraction"]))
        history["home_fraction"].append(float(np.mean(last_info["selected_in_home"])))
        history["assist_fraction"].append(float(np.mean(last_info["selected_in_assist"])))
        history["in_home_fraction"].append(float(np.mean(last_info["in_home_region"])))
        history["detour_fraction"].append(float(np.mean(last_info["patrol_detouring"])))
        history["tracking_fraction"].append(float(np.mean(last_info["tracking_bias_drones"])))
        history["threat_active"].append(float(last_info["active_threat"]))
        history["threat_confirmed"].append(float(last_info["threat_confirmed"]))
        history["central_command_notified"].append(float(last_info["central_command_notified"]))
        history["interceptor_dispatched"].append(float(last_info["interceptor_state"]["dispatched_this_step"]))
        history["interceptor_active"].append(float(last_info["interceptor_state"]["active"]))
        history["interceptor_distance"].append(float(last_info["interceptor_state"]["distance_to_target"]))
        history["threat_position_xy"].append(last_info["threat_state"]["position_xy"].copy())
        history["threat_distance_to_base"].append(
            float(np.linalg.norm(last_info["threat_state"]["position_xy"] - env.base_station))
            if bool(last_info["active_threat"])
            else float("nan")
        )
        history["threat_cycle_index"].append(float(last_info["threat_cycle_index"]))
        history["threat_cycles_completed"].append(float(last_info["threat_cycles_completed"]))
        history["threat_removed"].append(float(last_info["threat_removed_this_step"]))
        history["threat_respawned"].append(float(last_info["threat_respawned_this_step"]))
        history["physical_base_reached"].append(float(last_info["physical_base_reached"]))
        history["mission_failed"].append(float(last_info["mission_failed"]))
        history["home_coverage_fraction"].append(last_info["home_coverage_fraction"].copy())
        history["home_low_risk_fraction"].append(last_info["home_low_risk_fraction"].copy())
        history["home_neglected_fraction"].append(last_info["home_neglected_fraction"].copy())
        if terminated or truncated:
            break

    return history, last_info


def _summarize_rollout(env: BeliefCoverageEnv, history: dict, final_info: dict) -> dict[str, np.ndarray | float]:
    steps = max(len(history["home_fraction"]) - 1, 1)
    selected_in_home = np.asarray(history["selected_in_home"], dtype=np.float64)
    selected_in_assist = np.asarray(history["selected_in_assist"], dtype=np.float64)
    in_home_region = np.asarray(history["in_home_region"], dtype=np.float64)
    tracking_bias = np.asarray(history["tracking_bias_drones"], dtype=np.float64)

    return {
        "per_drone_home_selection_fraction": np.mean(selected_in_home, axis=0),
        "per_drone_assist_selection_fraction": np.mean(selected_in_assist, axis=0),
        "per_drone_in_home_fraction": np.mean(in_home_region, axis=0),
        "per_drone_away_fraction": np.asarray(final_info["time_away_from_home_steps"], dtype=np.float64) / steps,
        "per_drone_tracking_fraction": np.mean(tracking_bias, axis=0),
        "per_drone_detour_counts": np.asarray(final_info["patrol_detour_counts"], dtype=np.float64),
        "home_coverage_fraction": np.asarray(final_info["home_coverage_fraction"], dtype=np.float64),
        "home_low_risk_fraction": np.asarray(final_info["home_low_risk_fraction"], dtype=np.float64),
        "home_neglected_fraction": np.asarray(final_info["home_neglected_fraction"], dtype=np.float64),
    }


def _first_event_step(history: dict, key: str) -> int | None:
    values = np.asarray(history[key], dtype=np.float64)
    indices = np.flatnonzero(values > 0.5)
    if indices.size == 0:
        return None
    return int(indices[0])


def _inject_assist_spike(env: BeliefCoverageEnv, drone_idx: int, level: float) -> int | None:
    """
    Inject a local assist-band spike after calming the owner's home strip.

    This is a diagnostic visualization aid only. It creates a situation where
    the conservative patrol policy is allowed to take a short local detour,
    making bounded assistance visible in the saved figure.
    """
    assist_cells = env.get_assist_cell_indices(drone_idx)
    if assist_cells.size == 0:
        return None

    target = int(assist_cells[0])
    home_cells = env.get_home_cell_indices(drone_idx)
    calm_level = min(0.2, env.PATROL_HOME_CLEAR_RISK * 0.5)
    for grid in [env.global_belief, *env.local_beliefs]:
        if home_cells.size > 0:
            grid.uncertainty[home_cells] = np.minimum(grid.uncertainty[home_cells], calm_level)
        grid.uncertainty[target] = max(float(grid.uncertainty[target]), float(level))
    if home_cells.size > 0:
        env._truth_monitoring_risk[home_cells] = np.minimum(
            env._truth_monitoring_risk[home_cells],
            calm_level,
        )
    env._truth_monitoring_risk[target] = max(float(env._truth_monitoring_risk[target]), float(level))
    env._truth_risk_score = np.maximum(env._truth_monitoring_risk, env._truth_risk_cue)
    return target


def _force_initial_threat_patch(env: BeliefCoverageEnv, drone_idx: int) -> None:
    """Force the current moving threat patch into one home strip for diagnostics."""
    home_cells = set(env.get_home_cell_indices(drone_idx).tolist())
    candidates = [
        candidate
        for candidate in env._threat_patch_candidates
        if set(candidate.cell_indices.tolist()).issubset(home_cells)
    ]
    if not candidates:
        return
    candidate = min(
        candidates,
        key=lambda item: float(np.linalg.norm(item.centroid_xy - env.base_station)),
    )
    env._activate_threat_patch(
        candidate.cell_indices,
        candidate_index=-1,
        count_as_spawn=False,
        base_eta_steps=env._estimate_threat_base_eta_steps(candidate.centroid_xy),
    )
    env._truth_risk_score = env.transition_model.compose_hidden_risk_state(
        env._truth_monitoring_risk,
        env._truth_persistent_threat,
    )


def _draw_home_boundaries(ax: plt.Axes, env: BeliefCoverageEnv) -> None:
    boundaries = env.get_home_strip_boundaries()
    for position in np.asarray(boundaries["positions"], dtype=np.float64):
        if boundaries["axis"] == "x":
            ax.axvline(position, color="white", linestyle="--", linewidth=1.0, alpha=0.85)
        else:
            ax.axhline(position, color="white", linestyle="--", linewidth=1.0, alpha=0.85)


def _draw_overlay(ax: plt.Axes, env: BeliefCoverageEnv, history: dict, final_info: dict) -> None:
    positions = np.asarray(history["positions"], dtype=np.float64)
    detouring = np.asarray(final_info["patrol_detouring"], dtype=np.int32)
    selected_in_assist = np.asarray(final_info["selected_in_assist"], dtype=np.int32)
    selected_cells = np.asarray(final_info["selected_target_cells"], dtype=np.int32)

    ax.scatter(
        [env.base_station[0]],
        [env.base_station[1]],
        marker="*",
        s=160,
        color="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )

    for drone_idx in range(env.num_drones):
        color = DRONE_COLORS[drone_idx % len(DRONE_COLORS)]
        trajectory = positions[:, drone_idx, :]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=1.7,
            alpha=0.95,
            zorder=4,
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=56,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )

        target_xy = env.cell_centers_xy[int(selected_cells[drone_idx])]
        marker = "X" if int(selected_in_assist[drone_idx]) else "s"
        size = 86 if int(detouring[drone_idx]) else 62
        ax.scatter(
            target_xy[0],
            target_xy[1],
            marker=marker,
            s=size,
            facecolors="none" if marker == "s" else color,
            edgecolors=color,
            linewidths=1.8,
            zorder=8,
        )

    active_threat_cells = np.asarray(final_info["active_threat_cells"], dtype=np.int32)
    if active_threat_cells.size > 0:
        threat_xy = env.cell_centers_xy[active_threat_cells]
        confirmed = bool(final_info["threat_confirmed"])
        ax.scatter(
            threat_xy[:, 0],
            threat_xy[:, 1],
            marker="D",
            s=74,
            facecolors="#D62828" if confirmed else "none",
            edgecolors="#D62828" if confirmed else "#F77F00",
            linewidths=1.6,
            zorder=9,
        )

    threat_trace = np.asarray(final_info["threat_trace"], dtype=np.float64)
    if threat_trace.ndim == 2 and threat_trace.shape[0] > 0:
        ax.plot(
            threat_trace[:, 0],
            threat_trace[:, 1],
            color="#D62828",
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            zorder=6,
        )
        last_threat = threat_trace[-1]
        if np.all(np.isfinite(last_threat)):
            ax.scatter(
                last_threat[0],
                last_threat[1],
                marker="X",
                s=84,
                color="#D62828",
                edgecolors="white",
                linewidths=0.6,
                zorder=10,
            )

    interceptor_trace = np.asarray(final_info["interceptor_trace"], dtype=np.float64)
    if interceptor_trace.ndim == 2 and interceptor_trace.shape[0] > 0:
        ax.plot(
            interceptor_trace[:, 0],
            interceptor_trace[:, 1],
            color="black",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.9,
            zorder=5,
        )
        ax.scatter(
            interceptor_trace[-1, 0],
            interceptor_trace[-1, 1],
            marker=">",
            s=80,
            color="black",
            edgecolors="white",
            linewidths=0.6,
            zorder=10,
        )

    _draw_home_boundaries(ax, env)
    ax.set_xlim(0.0, env.area_size[0])
    ax.set_ylim(0.0, env.area_size[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _draw_field(
    ax: plt.Axes,
    env: BeliefCoverageEnv,
    field: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    history: dict,
    final_info: dict,
) -> None:
    image = ax.imshow(
        field,
        origin="lower",
        extent=(0.0, env.area_size[0], 0.0, env.area_size[1]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    _draw_overlay(ax, env, history, final_info)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_rollout_diagnostics(
    env: BeliefCoverageEnv,
    history: dict,
    final_info: dict,
    *,
    output_path: Path,
    show: bool,
) -> None:
    summary = _summarize_rollout(env, history, final_info)
    hidden_patch_grid = env.rasterize_cell_values(env.get_active_threat_mask(), fill_value=0.0)
    patrol_risk_grid = env.rasterize_cell_values(env.get_patrol_risk_belief_scores(), fill_value=0.0)
    threat_belief_grid = env.rasterize_cell_values(env.get_threat_belief_scores(), fill_value=0.0)
    persistence_grid = env.rasterize_cell_values(env.get_threat_persistence_scores(), fill_value=0.0)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)

    _draw_field(
        axes[0, 0],
        env,
        hidden_patch_grid,
        title=f"Hidden Moving Threat Patch ({final_info['threat_speed_case']})",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        history=history,
        final_info=final_info,
    )
    _draw_field(
        axes[0, 1],
        env,
        patrol_risk_grid,
        title="Patrol-Risk Belief",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        history=history,
        final_info=final_info,
    )
    _draw_field(
        axes[0, 2],
        env,
        threat_belief_grid,
        title="Threat Belief",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        history=history,
        final_info=final_info,
    )
    _draw_field(
        axes[0, 3],
        env,
        persistence_grid,
        title="Threat Persistence Score",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        history=history,
        final_info=final_info,
    )

    timeline = axes[1, 0]
    steps = np.arange(len(history["mean_patrol_risk_belief"]))
    timeline.plot(steps, history["mean_patrol_risk_belief"], label="Patrol-risk belief", linewidth=2.0)
    timeline.plot(steps, history["mean_threat_belief"], label="Threat belief", linewidth=1.8)
    timeline.plot(steps, history["active_threat_confirmation_score"], label="Patch confirmation", linewidth=1.8)
    timeline.plot(steps, history["low_risk_fraction"], label="Low-risk fraction", linewidth=1.6)
    timeline.plot(steps, history["tracking_fraction"], label="Tracking fraction", linewidth=1.5, linestyle="--")
    timeline.plot(steps, history["interceptor_active"], label="Interceptor active", linewidth=1.4, linestyle=":")
    timeline.plot(
        steps,
        np.clip(
            np.asarray(history["threat_distance_to_base"], dtype=np.float64)
            / max(np.linalg.norm(env.area_size), 1e-6),
            0.0,
            1.0,
        ),
        label="Threat dist-to-base (norm)",
        linewidth=1.5,
        linestyle="-.",
    )
    timeline.set_ylim(0.0, 1.05)
    timeline.set_xlabel("Step")
    timeline.set_ylabel("Normalized value")
    timeline.set_title("Patrol / Threat Timeline")
    timeline.grid(alpha=0.25)
    timeline.legend(loc="upper right", fontsize=8)

    summary_text = (
        f"cycle={final_info['threat_cycle_index']} completed={final_info['threat_cycles_completed']}\n"
        f"confirmed={int(final_info['threat_confirmed'])} notified={int(final_info['central_command_notified'])}\n"
        f"trackers={int(np.sum(final_info['tracking_bias_drones']))}/{env.num_drones}\n"
        f"eta={final_info['threat_base_eta_steps']} timeout={final_info['threat_base_timeout_steps']}"
    )
    if final_info["mission_failed"]:
        summary_text += "\nMISSION FAIL"
    timeline.text(
        0.02,
        0.02,
        summary_text,
        transform=timeline.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    per_drone = axes[1, 1]
    drone_idx = np.arange(env.num_drones)
    width = 0.2
    per_drone.bar(
        drone_idx - 1.5 * width,
        summary["per_drone_home_selection_fraction"],
        width=width,
        label="Home target",
        color="#2A9D8F",
    )
    per_drone.bar(
        drone_idx - 0.5 * width,
        summary["per_drone_assist_selection_fraction"],
        width=width,
        label="Assist target",
        color="#E9C46A",
    )
    per_drone.bar(
        drone_idx + 0.5 * width,
        summary["per_drone_in_home_fraction"],
        width=width,
        label="In home region",
        color="#457B9D",
    )
    per_drone.bar(
        drone_idx + 1.5 * width,
        summary["per_drone_away_fraction"],
        width=width,
        label="Away fraction",
        color="#E76F51",
    )
    per_drone.plot(
        drone_idx,
        summary["per_drone_tracking_fraction"],
        color="black",
        linewidth=1.5,
        marker="o",
        label="Tracking bias",
    )
    per_drone.set_xticks(drone_idx)
    per_drone.set_xticklabels([f"D{i}" for i in drone_idx])
    per_drone.set_ylim(0.0, 1.05)
    per_drone.set_title("Per-Drone Coordination")
    per_drone.set_ylabel("Fraction")
    per_drone.grid(axis="y", alpha=0.25)
    per_drone.legend(loc="upper right", fontsize=8)

    for idx, detours in enumerate(np.asarray(summary["per_drone_detour_counts"], dtype=np.int32)):
        per_drone.text(
            idx,
            1.02,
            f"d={int(detours)}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    per_region = axes[1, 2]
    region_idx = np.arange(env.num_drones)
    per_region.bar(
        region_idx - width,
        summary["home_coverage_fraction"],
        width=width,
        label="Observed in strip",
        color="#2A9D8F",
    )
    per_region.bar(
        region_idx,
        summary["home_low_risk_fraction"],
        width=width,
        label="Low-risk cells",
        color="#457B9D",
    )
    per_region.bar(
        region_idx + width,
        summary["home_neglected_fraction"],
        width=width,
        label="Neglected cells",
        color="#E76F51",
    )
    per_region.set_xticks(region_idx)
    per_region.set_xticklabels([f"H{i}" for i in region_idx])
    per_region.set_ylim(0.0, 1.05)
    per_region.set_title("Home-Strip Outcomes")
    per_region.set_ylabel("Fraction")
    per_region.grid(axis="y", alpha=0.25)
    per_region.legend(loc="upper right", fontsize=8)

    lifecycle = axes[1, 3]
    lifecycle.plot(steps, history["threat_active"], label="Threat active", linewidth=1.8)
    lifecycle.plot(steps, history["threat_confirmed"], label="Threat confirmed", linewidth=1.8)
    lifecycle.plot(steps, history["central_command_notified"], label="Central notified", linewidth=1.6)
    lifecycle.plot(steps, history["interceptor_dispatched"], label="Interceptor dispatched", linewidth=1.6, linestyle=":")
    lifecycle.plot(steps, history["threat_cycles_completed"], label="Cycles completed", linewidth=1.6)
    lifecycle.plot(steps, history["threat_cycle_index"], label="Current cycle", linewidth=1.5)
    lifecycle.plot(steps, np.clip(np.asarray(history["interceptor_distance"]) / max(np.linalg.norm(env.area_size), 1e-6), 0.0, 1.0), label="Interceptor dist (norm)", linewidth=1.5)
    lifecycle.scatter(
        steps,
        history["threat_removed"],
        label="Threat removed",
        color="#2A9D8F",
        marker="x",
        s=28,
    )
    lifecycle.scatter(
        steps,
        history["threat_respawned"],
        label="Threat respawned",
        color="#E76F51",
        marker="o",
        s=22,
        facecolors="none",
    )
    lifecycle.plot(steps, history["physical_base_reached"], label="Physical base reach", linewidth=1.6, color="#F77F00")
    lifecycle.plot(steps, history["mission_failed"], label="Mission fail", linewidth=1.8, color="#D62828")
    lifecycle.set_ylim(-0.05, max(1.05, float(max(history["threat_cycle_index"]) + 0.5)))
    lifecycle.set_xlabel("Step")
    lifecycle.set_ylabel("State / count")
    lifecycle.set_title("Threat Cycle / Interceptor")
    lifecycle.grid(alpha=0.25)
    lifecycle.legend(loc="upper right", fontsize=8)

    if final_info["mission_failed"]:
        lifecycle.text(
            0.02,
            0.92,
            "Mission failed: threat reached base",
            transform=lifecycle.transAxes,
            fontsize=10,
            color="#D62828",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    scenario_cfg = _load_scenario(args.scenario_config, args.scenario)
    env_kwargs = _build_env_kwargs(scenario_cfg, args)
    output_path = Path(args.output)

    env = BeliefCoverageEnv(**env_kwargs)
    try:
        if args.assist_spike_drone is not None:
            history, final_info = _run_rollout(
                env,
                steps=args.steps,
                seed=args.seed,
                assist_spike_drone=args.assist_spike_drone,
                assist_spike_step=args.assist_spike_step,
                assist_spike_level=args.assist_spike_level,
                force_threat_home_drone=args.force_threat_home_drone,
            )
        else:
            history, final_info = _run_rollout(
                env,
                steps=args.steps,
                seed=args.seed,
                force_threat_home_drone=args.force_threat_home_drone,
            )
        _plot_rollout_diagnostics(
            env,
            history,
            final_info,
            output_path=output_path,
            show=args.show,
        )
    finally:
        env.close()

    print("=" * 72)
    print("BeliefCoverageEnv patrol-risk visualization")
    print("=" * 72)
    confirmation_step = _first_event_step(history, "threat_confirmed")
    interceptor_launch_step = _first_event_step(history, "interceptor_dispatched")
    removal_step = _first_event_step(history, "threat_removed")
    mission_fail_step = _first_event_step(history, "mission_failed")
    print(f"Output                : {output_path}")
    print(f"Steps                 : {len(history['mean_patrol_risk_belief']) - 1}")
    print(f"Threat speed case     : {final_info['threat_speed_case']}")
    print(f"Threat speed [m/s]    : {final_info['threat_speed']:.2f}")
    print(f"Mean patrol belief    : {final_info['mean_patrol_risk_belief']:.3f}")
    print(f"Mean threat belief    : {final_info['mean_threat_belief']:.3f}")
    print(f"Threat persistence    : {final_info['active_threat_confirmation_score']:.3f}")
    print(f"Confirm step          : {confirmation_step}")
    print(f"Interceptor launch    : {interceptor_launch_step}")
    print(f"Intercept step        : {removal_step}")
    print(f"Mission fail step     : {mission_fail_step}")
    print(f"Low-risk fraction     : {final_info['low_risk_fraction']:.3f}")
    print(f"Never observed        : {final_info['never_observed_fraction']:.3f}")
    print(f"Home selected         : {int(np.sum(final_info['selected_in_home']))}/{env.num_drones}")
    print(f"Assist selected       : {int(np.sum(final_info['selected_in_assist']))}/{env.num_drones}")
    print(f"Tracking drones       : {int(np.sum(final_info['tracking_bias_drones']))}/{env.num_drones}")
    print(f"Detouring drones      : {int(np.sum(final_info['patrol_detouring']))}")
    print(f"Threat confirmed      : {bool(final_info['threat_confirmed'])}")
    print(f"Threat cycle          : {final_info['threat_cycle_index']} completed={final_info['threat_cycles_completed']}")
    print(f"Threat removed step   : {bool(final_info['threat_removed_this_step'])}")
    print(f"Threat respawned step : {bool(final_info['threat_respawned_this_step'])}")
    print(f"Interceptor active    : {bool(final_info['interceptor_state']['active'])}")
    print(f"Base ETA steps        : {final_info['threat_base_eta_steps']}")
    print(f"Base timeout steps    : {final_info['threat_base_timeout_steps']}")
    print(f"Physical base reach   : {bool(final_info['physical_base_reached'])}")
    print(f"Mission failed        : {bool(final_info['mission_failed'])}")
    print(f"Max assist fraction   : {np.max(history['assist_fraction']):.2f}")
    print(f"Max tracking fraction : {np.max(history['tracking_fraction']):.2f}")
    print(f"Max detour fraction   : {np.max(history['detour_fraction']):.2f}")
    print(f"Per-drone home select : {np.round(_summarize_rollout(env, history, final_info)['per_drone_home_selection_fraction'], 3).tolist()}")
    print(f"Per-drone tracking    : {np.round(_summarize_rollout(env, history, final_info)['per_drone_tracking_fraction'], 3).tolist()}")
    print(f"Per-drone away frac   : {np.round(_summarize_rollout(env, history, final_info)['per_drone_away_fraction'], 3).tolist()}")
    print(f"Home coverage frac    : {np.round(final_info['home_coverage_fraction'], 3).tolist()}")
    if history["assist_spike_target"] is not None:
        print(f"Assist spike target   : {history['assist_spike_target']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
