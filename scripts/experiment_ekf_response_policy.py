#!/usr/bin/env python3
"""
Compare EKF baseline vs improved response policy on moving-threat patrol runs.
"""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import visualize_belief_coverage as viz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EKF baseline vs improved response policy.")
    parser.add_argument("--scenario", default="area_surveillance")
    parser.add_argument("--scenario-config", default=str(ROOT / "config" / "mission_scenarios.yaml"))
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument(
        "--num-drones-cases",
        default=None,
        help="Comma-separated drone counts to validate; overrides --num-drones when set",
    )
    parser.add_argument("--max-threat-cycles", type=int, default=1)
    parser.add_argument("--speed-cases", default="fast")
    parser.add_argument("--seeds", default="7,11,19,23,29")
    parser.add_argument("--ignore-suspicious-zones", action="store_true")
    parser.add_argument("--threat-comm-delay-per-hop-steps", type=int, default=2)
    parser.add_argument("--threat-measurement-noise-std", type=float, default=2.0)
    parser.add_argument("--threat-measurement-staleness-gain", type=float, default=0.4)
    parser.add_argument("--interceptor-launch-confidence", type=float, default=0.55)
    parser.add_argument("--policies", default="improved,phase2")
    parser.add_argument("--base-sensor-range", type=float, default=None)
    parser.add_argument("--base-sensor-noise-std", type=float, default=None)
    parser.add_argument("--base-sensor-delay-steps", type=int, default=1)
    parser.add_argument(
        "--validation-dynamics",
        choices=["full", "fast_planar"],
        default="full",
        help="Use full DMPC-backed dynamics or low-cost deterministic planar validation dynamics.",
    )
    parser.add_argument(
        "--threat-belief-mode",
        choices=["shared", "limited"],
        default="shared",
        help="Use current shared central threat belief or delayed limited-belief relay semantics.",
    )
    parser.add_argument(
        "--skip-diagnostic-plots",
        action="store_true",
        help="Only write CSV/JSON metrics; skip representative rollout PNGs for faster validation.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "visualizations" / "ekf_response_policy"))
    return parser.parse_args()


def _build_args(
    *,
    args: argparse.Namespace,
    num_drones: int,
    speed_case: str,
    response_policy: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_drones=num_drones,
        disable_neighbor_sharing=False,
        disable_goal_projection=False,
        disable_persistent_threats=False,
        max_threat_cycles=args.max_threat_cycles,
        ignore_suspicious_zones=args.ignore_suspicious_zones,
        threat_speed_case=speed_case,
        threat_speed=None,
        interceptor_guidance_mode="ekf",
        response_policy=response_policy,
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
        force_threat_home_drone=None,
    )


def _first_event_step(history: dict, key: str) -> int | None:
    return viz._first_event_step(history, key)


def _first_nonnegative_history_value(history: dict, key: str) -> int | None:
    values = np.asarray(history.get(key, []), dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0.0)]
    return int(values[0]) if values.size else None


def _single_run(
    scenario_cfg: dict,
    *,
    num_drones: int,
    speed_case: str,
    response_policy: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    local_args = _build_args(
        args=args,
        num_drones=num_drones,
        speed_case=speed_case,
        response_policy=response_policy,
    )
    env = viz.BeliefCoverageEnv(**viz._build_env_kwargs(scenario_cfg, local_args))
    try:
        start_time = time.perf_counter()
        history, final_info = viz._run_rollout(env, steps=args.steps, seed=seed)
        runtime_seconds = time.perf_counter() - start_time
        track_errors = np.asarray(history["track_error"], dtype=np.float64)
        track_errors = track_errors[np.isfinite(track_errors)]
        etas = np.asarray(history["threat_estimated_time_to_base"], dtype=np.float64)
        etas = etas[np.isfinite(etas)]
        first_observed_step = _first_nonnegative_history_value(history, "first_threat_observed_step")
        first_base_confirmation_step = _first_nonnegative_history_value(history, "base_first_confirmation_step")
        first_base_track_step = _first_nonnegative_history_value(history, "base_first_track_update_step")
        base_confirmation_lag = (
            int(first_base_confirmation_step - first_observed_step)
            if first_base_confirmation_step is not None and first_observed_step is not None
            else -1
        )
        base_track_lag = (
            int(first_base_track_step - first_observed_step)
            if first_base_track_step is not None and first_observed_step is not None
            else -1
        )
        return {
            "num_drones": int(num_drones),
            "speed_case": speed_case,
            "response_policy": response_policy,
            "validation_dynamics": str(final_info["validation_dynamics"]),
            "threat_belief_mode": str(final_info["threat_belief_mode"]),
            "seed": int(seed),
            "history": history,
            "final_info": final_info,
            "confirmation_step": _first_event_step(history, "threat_confirmed"),
            "base_confirmation_step": _first_event_step(history, "base_confirmed"),
            "launch_step": _first_event_step(history, "interceptor_dispatched"),
            "intercept_step": _first_event_step(history, "threat_removed"),
            "mission_fail_step": _first_event_step(history, "mission_failed"),
            "intercept_success": bool(_first_event_step(history, "threat_removed") is not None),
            "mission_failed": bool(final_info["mission_failed"]),
            "mean_track_confidence": float(np.mean(history["track_confidence"])),
            "mean_track_error": float(np.mean(track_errors)) if track_errors.size > 0 else float("nan"),
            "mean_urgency": float(np.mean(history["threat_urgency_score"])),
            "mean_eta": float(np.mean(etas)) if etas.size > 0 else float("inf"),
            "mean_home_fraction": float(np.mean(history["home_fraction"])),
            "mean_tracking_fraction": float(np.mean(history["tracking_fraction"])),
            "mean_assist_fraction": float(np.mean(history["assist_fraction"])),
            "mean_neglect_pressure": float(np.mean(history["neglect_pressure"])),
            "mean_base_sensor_detected": float(np.mean(history["base_sensor_detected"])),
            "mean_base_sensor_quality": float(np.mean(history["base_sensor_quality"])),
            "mean_local_confirmed_fraction": float(np.mean(history["local_confirmed_fraction"])),
            "mean_base_confirmation_level": float(np.mean(history["base_confirmation_level"])),
            "base_confirmation_lag_steps": int(base_confirmation_lag),
            "base_track_lag_steps": int(base_track_lag),
            "base_confirmation_messages_received": int(
                np.sum(history["base_confirmation_received"])
            ),
            "final_base_sensor_detections": int(final_info["base_sensor_state"]["total_detections"]),
            "final_base_sensor_updates": int(final_info["base_sensor_state"]["total_updates"]),
            "final_never_observed_fraction": float(final_info["never_observed_fraction"]),
            "final_low_risk_fraction": float(final_info["low_risk_fraction"]),
            "runtime_seconds": float(runtime_seconds),
        }
    finally:
        env.close()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    def _mean_step(key: str) -> float:
        vals = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "num_drones": float(rows[0]["num_drones"]) if rows else float("nan"),
        "n_runs": float(len(rows)),
        "base_compromise_rate": float(np.mean([float(row["mission_failed"]) for row in rows])),
        "intercept_success_rate": float(np.mean([float(row["intercept_success"]) for row in rows])),
        "mean_confirmation_step": _mean_step("confirmation_step"),
        "mean_base_confirmation_step": _mean_step("base_confirmation_step"),
        "mean_launch_step": _mean_step("launch_step"),
        "mean_intercept_step": _mean_step("intercept_step"),
        "mean_track_confidence": float(np.mean([row["mean_track_confidence"] for row in rows])),
        "mean_track_error": float(np.nanmean([row["mean_track_error"] for row in rows])),
        "mean_urgency": float(np.mean([row["mean_urgency"] for row in rows])),
        "mean_eta": float(np.mean([row["mean_eta"] for row in rows])),
        "mean_home_fraction": float(np.mean([row["mean_home_fraction"] for row in rows])),
        "mean_tracking_fraction": float(np.mean([row["mean_tracking_fraction"] for row in rows])),
        "mean_assist_fraction": float(np.mean([row["mean_assist_fraction"] for row in rows])),
        "mean_neglect_pressure": float(np.mean([row["mean_neglect_pressure"] for row in rows])),
        "mean_base_sensor_detected": float(np.mean([row["mean_base_sensor_detected"] for row in rows])),
        "mean_base_sensor_quality": float(np.mean([row["mean_base_sensor_quality"] for row in rows])),
        "mean_local_confirmed_fraction": float(np.mean([row["mean_local_confirmed_fraction"] for row in rows])),
        "mean_base_confirmation_level": float(np.mean([row["mean_base_confirmation_level"] for row in rows])),
        "mean_base_confirmation_lag_steps": float(np.mean([row["base_confirmation_lag_steps"] for row in rows])),
        "mean_base_track_lag_steps": float(np.mean([row["base_track_lag_steps"] for row in rows])),
        "mean_base_confirmation_messages_received": float(
            np.mean([row["base_confirmation_messages_received"] for row in rows])
        ),
        "mean_final_base_sensor_detections": float(np.mean([row["final_base_sensor_detections"] for row in rows])),
        "mean_final_base_sensor_updates": float(np.mean([row["final_base_sensor_updates"] for row in rows])),
        "mean_final_never_observed_fraction": float(np.mean([row["final_never_observed_fraction"] for row in rows])),
        "mean_final_low_risk_fraction": float(np.mean([row["final_low_risk_fraction"] for row in rows])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
    }


def _scalar_run_row(row: dict[str, object]) -> dict[str, object]:
    """Drop bulky rollout history while keeping teammate-facing run metrics."""
    keys = [
        "num_drones",
        "speed_case",
        "response_policy",
        "validation_dynamics",
        "threat_belief_mode",
        "seed",
        "confirmation_step",
        "base_confirmation_step",
        "launch_step",
        "intercept_step",
        "mission_fail_step",
        "intercept_success",
        "mission_failed",
        "mean_track_confidence",
        "mean_track_error",
        "mean_urgency",
        "mean_eta",
        "mean_home_fraction",
        "mean_tracking_fraction",
        "mean_assist_fraction",
        "mean_neglect_pressure",
        "mean_base_sensor_detected",
        "mean_base_sensor_quality",
        "mean_local_confirmed_fraction",
        "mean_base_confirmation_level",
        "base_confirmation_lag_steps",
        "base_track_lag_steps",
        "base_confirmation_messages_received",
        "final_base_sensor_detections",
        "final_base_sensor_updates",
        "final_never_observed_fraction",
        "final_low_risk_fraction",
        "runtime_seconds",
    ]
    return {key: row.get(key) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    drone_cases: list[int],
    speed_cases: list[str],
    seeds: list[int],
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    run_rows = [_scalar_run_row(row) for row in rows]
    summary_path = output_dir / "response_policy_validation_summary.csv"
    run_path = output_dir / "response_policy_validation_runs.csv"
    json_path = output_dir / "response_policy_validation_metrics.json"
    _write_csv(summary_path, summaries)
    _write_csv(run_path, run_rows)
    payload = {
        "config": {
            "steps": int(args.steps),
            "num_drones_cases": drone_cases,
            "speed_cases": speed_cases,
            "seeds": seeds,
            "max_threat_cycles": int(args.max_threat_cycles),
            "interceptor_guidance_mode": "ekf",
            "response_policies": [token.strip() for token in args.policies.split(",") if token.strip()],
            "interceptor_launch_confidence": float(args.interceptor_launch_confidence),
            "threat_comm_delay_per_hop_steps": int(args.threat_comm_delay_per_hop_steps),
            "threat_measurement_noise_std": float(args.threat_measurement_noise_std),
            "threat_measurement_staleness_gain": float(args.threat_measurement_staleness_gain),
            "base_sensor_range": args.base_sensor_range,
            "base_sensor_noise_std": args.base_sensor_noise_std,
            "base_sensor_delay_steps": int(args.base_sensor_delay_steps),
            "validation_dynamics": str(args.validation_dynamics),
            "threat_belief_mode": str(args.threat_belief_mode),
        },
        "summaries": summaries,
        "runs": run_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary_path, run_path, json_path


def _plot_comparison(
    grouped: dict[tuple[str, str], dict[str, float]],
    speed_cases: list[str],
    policies: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    width = 0.8 / max(len(policies), 1)
    x = np.arange(len(speed_cases), dtype=np.float64)

    def _values(metric: str, policy: str) -> np.ndarray:
        vals = []
        for speed in speed_cases:
            summary = grouped.get((policy, speed))
            vals.append(float(summary[metric]) if summary else np.nan)
        return np.asarray(vals, dtype=np.float64)

    panels = [
        ("base_compromise_rate", "Base Compromise Rate", (0.0, 1.0)),
        ("intercept_success_rate", "Intercept Success Rate", (0.0, 1.0)),
        ("mean_launch_step", "Mean Launch Step", None),
        ("mean_intercept_step", "Mean Intercept Step", None),
        ("mean_neglect_pressure", "Mean Neglect Pressure", None),
        ("mean_home_fraction", "Mean Home Fraction", (0.0, 1.0)),
    ]

    for axis, (metric, title, ylim) in zip(axes.flat, panels):
        for policy_idx, policy in enumerate(policies):
            offset = (policy_idx - (len(policies) - 1) / 2.0) * width
            axis.bar(x + offset, _values(metric, policy), width=width, label=policy)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(speed_cases)
        axis.grid(axis="y", alpha=0.25)
        if ylim is not None:
            axis.set_ylim(*ylim)

    axes[0, 0].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    speed_cases = [token.strip() for token in args.speed_cases.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    policies = [token.strip() for token in args.policies.split(",") if token.strip()]
    drone_cases = (
        [int(token.strip()) for token in args.num_drones_cases.split(",") if token.strip()]
        if args.num_drones_cases
        else [int(args.num_drones)]
    )
    scenario_cfg = viz._load_scenario(args.scenario_config, args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for num_drones in drone_cases:
        for speed_case in speed_cases:
            for response_policy in policies:
                for seed in seeds:
                    rows.append(
                        _single_run(
                            scenario_cfg,
                            num_drones=num_drones,
                            speed_case=speed_case,
                            response_policy=response_policy,
                            seed=seed,
                            args=args,
                        )
                    )

                if not args.skip_diagnostic_plots:
                    representative = next(
                        row
                        for row in rows
                        if row["num_drones"] == num_drones
                        and row["speed_case"] == speed_case
                        and row["response_policy"] == response_policy
                    )
                    rep_args = _build_args(
                        args=args,
                        num_drones=num_drones,
                        speed_case=speed_case,
                        response_policy=response_policy,
                    )
                    env = viz.BeliefCoverageEnv(**viz._build_env_kwargs(scenario_cfg, rep_args))
                    try:
                        rep_history, rep_final = viz._run_rollout(
                            env,
                            steps=args.steps,
                            seed=int(representative["seed"]),
                        )
                        viz._plot_rollout_diagnostics(
                            env,
                            rep_history,
                            rep_final,
                            output_path=(
                                output_dir
                                / f"ekf_{response_policy}_{speed_case}_{num_drones}d_seed{representative['seed']}.png"
                            ),
                            show=False,
                        )
                    finally:
                        env.close()

    grouped: dict[tuple[int, str, str], dict[str, float]] = {}
    summary_rows: list[dict[str, object]] = []
    for num_drones in drone_cases:
        drone_grouped: dict[tuple[str, str], dict[str, float]] = {}
        for speed_case in speed_cases:
            for response_policy in policies:
                subset = [
                    row
                    for row in rows
                    if row["num_drones"] == num_drones
                    and row["speed_case"] == speed_case
                    and row["response_policy"] == response_policy
                ]
                summary = _aggregate(subset) if subset else {}
                grouped[(num_drones, response_policy, speed_case)] = summary
                drone_grouped[(response_policy, speed_case)] = summary
                if summary:
                    summary_rows.append(
                        {
                            "num_drones": num_drones,
                            "speed_case": speed_case,
                            "response_policy": response_policy,
                            **summary,
                        }
                    )

        if not args.skip_diagnostic_plots:
            comparison_path = output_dir / f"ekf_response_policy_comparison_{num_drones}d.png"
            _plot_comparison(drone_grouped, speed_cases, policies, comparison_path)

    summary_path, run_path, json_path = _write_metrics_outputs(
        output_dir=output_dir,
        args=args,
        drone_cases=drone_cases,
        speed_cases=speed_cases,
        seeds=seeds,
        rows=rows,
        summaries=summary_rows,
    )

    print("=" * 110)
    print("EKF response policy comparison")
    print("=" * 110)
    print(f"Drone cases   : {drone_cases}")
    print(f"Speed cases   : {speed_cases}")
    print(f"Policies      : {policies}")
    print(f"Dynamics      : {args.validation_dynamics}")
    print(f"Threat belief : {args.threat_belief_mode}")
    print(f"Seeds         : {seeds}")
    print(f"Output dir    : {output_dir}")
    print(f"Summary CSV   : {summary_path}")
    print(f"Run CSV       : {run_path}")
    print(f"Metrics JSON  : {json_path}")
    print("-" * 110)
    for num_drones in drone_cases:
        print(f"{num_drones} drones")
        for speed_case in speed_cases:
            for response_policy in policies:
                summary = grouped.get((num_drones, response_policy, speed_case), {})
                if not summary:
                    continue
                print(
                    f"{speed_case:>6} | {response_policy:>8} | base_fail={summary['base_compromise_rate']:.3f} | "
                    f"success={summary['intercept_success_rate']:.3f} | confirm={summary['mean_confirmation_step']:.1f} | "
                    f"base_confirm={summary['mean_base_confirmation_step']:.1f} | "
                    f"launch={summary['mean_launch_step']:.1f} | intercept={summary['mean_intercept_step']:.1f} | "
                    f"urgency={summary['mean_urgency']:.3f} | track_err={summary['mean_track_error']:.3f} | "
                    f"home_frac={summary['mean_home_fraction']:.3f} | neglect={summary['mean_neglect_pressure']:.3f} | "
                    f"base_det={summary['mean_final_base_sensor_detections']:.1f} | "
                    f"base_upd={summary['mean_final_base_sensor_updates']:.1f} | "
                    f"comm_lag={summary['mean_base_confirmation_lag_steps']:.1f} | "
                    f"runtime={summary['mean_runtime_seconds']:.2f}s"
                )
    print("=" * 110)


if __name__ == "__main__":
    main()
