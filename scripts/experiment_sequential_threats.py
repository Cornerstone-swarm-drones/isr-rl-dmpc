#!/usr/bin/env python3
"""Validate bounded sequential-threat handling on the Phase 2 base-defense path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import visualize_belief_coverage as viz


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded sequential-threat Phase 3 validation rollouts."
    )
    parser.add_argument("--scenario", default="area_surveillance")
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
    )
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--num-drones-cases", default="4,5,6")
    parser.add_argument("--speed-cases", default="medium,fast")
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--max-threat-cycles", type=int, default=2)
    parser.add_argument("--pending-threat-delay-steps", type=int, default=12)
    parser.add_argument("--threat-belief-mode", choices=["shared", "limited", "limited_strict"], default="limited_strict")
    parser.add_argument("--validation-dynamics", choices=["full", "fast_planar"], default="fast_planar")
    parser.add_argument("--interceptor-launch-confidence", type=float, default=0.55)
    parser.add_argument("--threat-comm-delay-per-hop-steps", type=int, default=2)
    parser.add_argument("--threat-measurement-noise-std", type=float, default=2.0)
    parser.add_argument("--threat-measurement-staleness-gain", type=float, default=0.4)
    parser.add_argument("--base-sensor-range", type=float, default=None)
    parser.add_argument("--base-sensor-noise-std", type=float, default=None)
    parser.add_argument("--base-sensor-delay-steps", type=int, default=1)
    parser.add_argument("--ignore-suspicious-zones", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "visualizations" / "phase3_sequential_threats"),
    )
    return parser.parse_args()


def _build_viz_args(args: argparse.Namespace, *, num_drones: int, speed_case: str) -> SimpleNamespace:
    return SimpleNamespace(
        num_drones=num_drones,
        disable_neighbor_sharing=False,
        disable_goal_projection=False,
        disable_persistent_threats=False,
        max_threat_cycles=args.max_threat_cycles,
        force_threat_home_drone=None,
        ignore_suspicious_zones=args.ignore_suspicious_zones,
        threat_speed_case=speed_case,
        threat_speed=None,
        enable_sequential_pending_threats=True,
        pending_threat_delay_steps=args.pending_threat_delay_steps,
        interceptor_guidance_mode="ekf",
        response_policy="phase2",
        interceptor_launch_confidence=args.interceptor_launch_confidence,
        threat_comm_delay_per_hop_steps=args.threat_comm_delay_per_hop_steps,
        threat_measurement_noise_std=args.threat_measurement_noise_std,
        threat_measurement_staleness_gain=args.threat_measurement_staleness_gain,
        base_sensor_range=args.base_sensor_range,
        base_sensor_noise_std=args.base_sensor_noise_std,
        base_sensor_delay_steps=args.base_sensor_delay_steps,
        validation_dynamics=args.validation_dynamics,
        threat_belief_mode=args.threat_belief_mode,
        assist_spike_drone=None,
        assist_spike_step=4,
        assist_spike_level=0.95,
    )


def _event_steps(history: dict, key: str) -> list[int]:
    values = np.asarray(history.get(key, []), dtype=np.float64)
    return [int(idx) for idx in np.flatnonzero(values > 0.5)]


def _rising_edge_steps(history: dict, key: str) -> list[int]:
    values = np.asarray(history.get(key, []), dtype=np.float64)
    if values.size == 0:
        return []
    previous = np.concatenate([[0.0], values[:-1]])
    return [int(idx) for idx in np.flatnonzero((values > 0.5) & (previous <= 0.5))]


def _single_run(
    scenario_cfg: dict,
    args: argparse.Namespace,
    *,
    num_drones: int,
    speed_case: str,
    seed: int,
) -> dict[str, object]:
    local_args = _build_viz_args(args, num_drones=num_drones, speed_case=speed_case)
    env = viz.BeliefCoverageEnv(**viz._build_env_kwargs(scenario_cfg, local_args))
    try:
        start = time.perf_counter()
        history, final_info = viz._run_rollout(env, steps=args.steps, seed=seed)
        runtime = time.perf_counter() - start

        removal_steps = _event_steps(history, "threat_removed")
        launch_steps = _event_steps(history, "interceptor_dispatched")
        confirmation_steps = _rising_edge_steps(history, "threat_confirmed")
        base_confirmation_steps = _rising_edge_steps(history, "base_confirmed")
        pending_appeared_steps = _event_steps(history, "pending_threat_appeared")
        pending_promoted_steps = _event_steps(history, "pending_threat_promoted")
        pending_cue_steps = _event_steps(history, "pending_threat_cue_applied")
        pending_prior_steps = _event_steps(history, "pending_threat_prior_applied")
        pending_observed_steps = _event_steps(history, "pending_threat_observed")
        track_errors = np.asarray(history["track_error"], dtype=np.float64)
        track_errors = track_errors[np.isfinite(track_errors)]
        second_promotion_step = pending_promoted_steps[0] if pending_promoted_steps else None
        second_confirmation_step = confirmation_steps[1] if len(confirmation_steps) > 1 else None
        second_base_confirmation_step = (
            base_confirmation_steps[1] if len(base_confirmation_steps) > 1 else None
        )
        second_launch_step = launch_steps[1] if len(launch_steps) > 1 else None
        second_intercept_step = removal_steps[1] if len(removal_steps) > 1 else None

        return {
            "num_drones": int(num_drones),
            "speed_case": speed_case,
            "seed": int(seed),
            "validation_dynamics": str(final_info["validation_dynamics"]),
            "threat_belief_mode": str(final_info["threat_belief_mode"]),
            "max_threat_cycles": int(args.max_threat_cycles),
            "pending_threat_delay_steps": int(args.pending_threat_delay_steps),
            "threats_completed": int(final_info["threat_cycles_completed"]),
            "threats_spawned": int(final_info["threat_cycle_index"]),
            "handled_both_threats": bool(final_info["threat_cycles_completed"] >= 2),
            "second_threat_present": bool(
                len(pending_appeared_steps) > 0
                or int(final_info["threat_cycle_index"]) >= 2
                or int(final_info["threat_cycles_completed"]) >= 2
            ),
            "mission_failed": bool(final_info["mission_failed"]),
            "mission_fail_step": (viz._first_event_step(history, "mission_failed")),
            "removal_steps": removal_steps,
            "launch_steps": launch_steps,
            "confirmation_steps": confirmation_steps,
            "base_confirmation_steps": base_confirmation_steps,
            "pending_appeared_steps": pending_appeared_steps,
            "pending_promoted_steps": pending_promoted_steps,
            "pending_cue_steps": pending_cue_steps,
            "pending_prior_steps": pending_prior_steps,
            "pending_observed_steps": pending_observed_steps,
            "first_launch_step": launch_steps[0] if launch_steps else None,
            "second_promotion_step": second_promotion_step,
            "second_confirmation_step": second_confirmation_step,
            "second_base_confirmation_step": second_base_confirmation_step,
            "second_launch_step": second_launch_step,
            "first_intercept_step": removal_steps[0] if removal_steps else None,
            "second_intercept_step": second_intercept_step,
            "second_promotion_to_confirmation_steps": (
                int(second_confirmation_step - second_promotion_step)
                if second_confirmation_step is not None and second_promotion_step is not None
                else None
            ),
            "second_promotion_to_launch_steps": (
                int(second_launch_step - second_promotion_step)
                if second_launch_step is not None and second_promotion_step is not None
                else None
            ),
            "second_promotion_to_intercept_steps": (
                int(second_intercept_step - second_promotion_step)
                if second_intercept_step is not None and second_promotion_step is not None
                else None
            ),
            "mean_home_fraction": float(np.mean(history["home_fraction"])),
            "mean_tracking_fraction": float(np.mean(history["tracking_fraction"])),
            "mean_pending_watchlist_fraction": float(np.mean(history["pending_watchlist_fraction"])),
            "max_pending_preconfirmation": float(np.max(history["pending_threat_preconfirmation"])),
            "mean_assist_fraction": float(np.mean(history["assist_fraction"])),
            "mean_neglect_pressure": float(np.mean(history["neglect_pressure"])),
            "mean_track_confidence": float(np.mean(history["track_confidence"])),
            "mean_track_error": float(np.mean(track_errors)) if track_errors.size > 0 else float("nan"),
            "final_never_observed_fraction": float(final_info["never_observed_fraction"]),
            "runtime_seconds": float(runtime),
        }
    finally:
        env.close()


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["num_drones"]), str(row["speed_case"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (num_drones, speed_case), group in sorted(grouped.items()):
        first_intercepts = [row["first_intercept_step"] for row in group if row["first_intercept_step"] is not None]
        second_intercepts = [row["second_intercept_step"] for row in group if row["second_intercept_step"] is not None]
        second_confirm_latencies = [
            row["second_promotion_to_confirmation_steps"]
            for row in group
            if row["second_promotion_to_confirmation_steps"] is not None
        ]
        second_launch_latencies = [
            row["second_promotion_to_launch_steps"]
            for row in group
            if row["second_promotion_to_launch_steps"] is not None
        ]
        second_intercept_latencies = [
            row["second_promotion_to_intercept_steps"]
            for row in group
            if row["second_promotion_to_intercept_steps"] is not None
        ]
        summary.append(
            {
                "num_drones": num_drones,
                "speed_case": speed_case,
                "n_runs": len(group),
                "mission_fail_rate": float(np.mean([float(row["mission_failed"]) for row in group])),
                "second_threat_present_rate": float(np.mean([float(row["second_threat_present"]) for row in group])),
                "handled_both_rate": float(np.mean([float(row["handled_both_threats"]) for row in group])),
                "mean_threats_completed": float(np.mean([row["threats_completed"] for row in group])),
                "mean_first_intercept_step": float(np.mean(first_intercepts)) if first_intercepts else float("nan"),
                "mean_second_intercept_step": float(np.mean(second_intercepts)) if second_intercepts else float("nan"),
                "mean_second_promotion_to_confirmation_steps": (
                    float(np.mean(second_confirm_latencies)) if second_confirm_latencies else float("nan")
                ),
                "mean_second_promotion_to_launch_steps": (
                    float(np.mean(second_launch_latencies)) if second_launch_latencies else float("nan")
                ),
                "mean_second_promotion_to_intercept_steps": (
                    float(np.mean(second_intercept_latencies)) if second_intercept_latencies else float("nan")
                ),
                "mean_home_fraction": float(np.mean([row["mean_home_fraction"] for row in group])),
                "mean_tracking_fraction": float(np.mean([row["mean_tracking_fraction"] for row in group])),
                "mean_pending_watchlist_fraction": float(
                    np.mean([row["mean_pending_watchlist_fraction"] for row in group])
                ),
                "mean_max_pending_preconfirmation": float(
                    np.mean([row["max_pending_preconfirmation"] for row in group])
                ),
                "mean_neglect_pressure": float(np.mean([row["mean_neglect_pressure"] for row in group])),
                "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in group])),
            }
        )
    return summary


def _json_safe(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    args = _parse_args()
    scenario_cfg = viz._load_scenario(args.scenario_config, args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for num_drones in _parse_csv_ints(args.num_drones_cases):
        for speed_case in _parse_csv_strs(args.speed_cases):
            for seed in _parse_csv_ints(args.seeds):
                row = _single_run(
                    scenario_cfg,
                    args,
                    num_drones=num_drones,
                    speed_case=speed_case,
                    seed=seed,
                )
                rows.append(row)
                print(
                    f"{num_drones}d {speed_case:>6} seed={seed:<3} "
                    f"completed={row['threats_completed']} failed={row['mission_failed']} "
                    f"removals={row['removal_steps']} launches={row['launch_steps']} "
                    f"pending={row['pending_appeared_steps']}"
                )

    summary = _aggregate(rows)
    rows_csv = output_dir / "sequential_threat_runs.csv"
    summary_csv = output_dir / "sequential_threat_summary.csv"
    metrics_json = output_dir / "sequential_threat_metrics.json"

    run_fieldnames = list(rows[0].keys()) if rows else []
    with rows_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})

    summary_fieldnames = list(summary[0].keys()) if summary else []
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    payload = {
        "config": {
            "steps": int(args.steps),
            "num_drones_cases": _parse_csv_ints(args.num_drones_cases),
            "speed_cases": _parse_csv_strs(args.speed_cases),
            "seeds": _parse_csv_ints(args.seeds),
            "max_threat_cycles": int(args.max_threat_cycles),
            "pending_threat_delay_steps": int(args.pending_threat_delay_steps),
            "validation_dynamics": args.validation_dynamics,
            "threat_belief_mode": args.threat_belief_mode,
        },
        "runs": rows,
        "summary": summary,
    }
    metrics_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print("=" * 72)
    print("Sequential-threat validation complete")
    print(f"Runs written    : {rows_csv}")
    print(f"Summary written : {summary_csv}")
    print(f"JSON written    : {metrics_json}")
    for item in summary:
        print(
            f"{item['num_drones']}d {item['speed_case']}: "
            f"fail={item['mission_fail_rate']:.2f} "
            f"second_present={item['second_threat_present_rate']:.2f} "
            f"handled_both={item['handled_both_rate']:.2f} "
            f"home={item['mean_home_fraction']:.2f} "
            f"neglect={item['mean_neglect_pressure']:.3f}"
        )


if __name__ == "__main__":
    main()
