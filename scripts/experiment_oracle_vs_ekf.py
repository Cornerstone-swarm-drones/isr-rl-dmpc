#!/usr/bin/env python3
"""
Oracle-vs-EKF comparison for moving persistent-threat interception.

Runs deterministic rollouts for both guidance modes and reports measured
coordination/interception outcomes without training.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    parser = argparse.ArgumentParser(description="Compare oracle vs EKF moving-threat guidance.")
    parser.add_argument("--scenario", default="area_surveillance")
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
    )
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--max-threat-cycles", type=int, default=1)
    parser.add_argument("--speed-cases", default="slow,medium,fast")
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "visualizations" / "oracle_vs_ekf"),
    )
    parser.add_argument("--ignore-suspicious-zones", action="store_true")
    parser.add_argument("--threat-comm-delay-per-hop-steps", type=int, default=2)
    parser.add_argument("--threat-measurement-noise-std", type=float, default=2.0)
    parser.add_argument("--threat-measurement-staleness-gain", type=float, default=0.4)
    parser.add_argument("--interceptor-launch-confidence", type=float, default=0.55)
    parser.add_argument("--response-policy", choices=["baseline", "improved"], default="baseline")
    return parser.parse_args()


def _build_visualizer_args(
    *,
    num_drones: int,
    max_threat_cycles: int,
    ignore_suspicious_zones: bool,
    speed_case: str,
    mode: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_drones=num_drones,
        disable_neighbor_sharing=False,
        disable_goal_projection=False,
        disable_persistent_threats=False,
        max_threat_cycles=max_threat_cycles,
        ignore_suspicious_zones=ignore_suspicious_zones,
        threat_speed_case=speed_case,
        threat_speed=None,
        interceptor_guidance_mode=mode,
        response_policy=args.response_policy,
        interceptor_launch_confidence=args.interceptor_launch_confidence,
        threat_comm_delay_per_hop_steps=args.threat_comm_delay_per_hop_steps,
        threat_measurement_noise_std=args.threat_measurement_noise_std,
        threat_measurement_staleness_gain=args.threat_measurement_staleness_gain,
        assist_spike_drone=None,
        assist_spike_step=4,
        assist_spike_level=0.95,
        force_threat_home_drone=None,
    )


def _event_step(history: dict, key: str) -> int | None:
    return viz._first_event_step(history, key)


def _single_run(
    scenario_cfg: dict,
    *,
    speed_case: str,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    viz_args = _build_visualizer_args(
        num_drones=args.num_drones,
        max_threat_cycles=args.max_threat_cycles,
        ignore_suspicious_zones=args.ignore_suspicious_zones,
        speed_case=speed_case,
        mode=mode,
        args=args,
    )
    env_kwargs = viz._build_env_kwargs(scenario_cfg, viz_args)
    env = viz.BeliefCoverageEnv(**env_kwargs)
    try:
        history, final_info = viz._run_rollout(env, steps=args.steps, seed=seed)
        finite_track_errors = np.asarray(history["track_error"], dtype=np.float64)
        finite_track_errors = finite_track_errors[np.isfinite(finite_track_errors)]
        return {
            "mode": mode,
            "speed_case": speed_case,
            "seed": int(seed),
            "history": history,
            "final_info": final_info,
            "confirmation_step": _event_step(history, "threat_confirmed"),
            "launch_step": _event_step(history, "interceptor_dispatched"),
            "intercept_step": _event_step(history, "threat_removed"),
            "mission_fail_step": _event_step(history, "mission_failed"),
            "intercept_success": bool(_event_step(history, "threat_removed") is not None),
            "mission_failed": bool(final_info["mission_failed"]),
            "max_tracker_count": int(np.max(np.sum(np.asarray(history["tracking_bias_drones"]), axis=1))),
            "mean_home_fraction": float(np.mean(history["home_fraction"])),
            "mean_assist_fraction": float(np.mean(history["assist_fraction"])),
            "mean_tracking_fraction": float(np.mean(history["tracking_fraction"])),
            "mean_track_confidence": float(np.mean(history["track_confidence"])),
            "mean_track_error": float(np.mean(finite_track_errors)) if finite_track_errors.size > 0 else float("nan"),
            "final_never_observed_fraction": float(final_info["never_observed_fraction"]),
            "final_low_risk_fraction": float(final_info["low_risk_fraction"]),
            "final_neglect_pressure": float(final_info["neglect_pressure"]),
            "env_dt": float(env.dt),
        }
    finally:
        env.close()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    intercept_success = np.array([float(row["intercept_success"]) for row in rows], dtype=np.float64)
    mission_fail = np.array([float(row["mission_failed"]) for row in rows], dtype=np.float64)
    def _mean_steps(key: str) -> float:
        vals = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "n_runs": float(len(rows)),
        "intercept_success_rate": float(np.mean(intercept_success)),
        "base_compromise_rate": float(np.mean(mission_fail)),
        "mean_confirmation_step": _mean_steps("confirmation_step"),
        "mean_launch_step": _mean_steps("launch_step"),
        "mean_intercept_step": _mean_steps("intercept_step"),
        "mean_tracker_count": float(np.mean([row["max_tracker_count"] for row in rows])),
        "mean_track_confidence": float(np.mean([row["mean_track_confidence"] for row in rows])),
        "mean_track_error": float(np.nanmean([row["mean_track_error"] for row in rows])),
        "mean_home_fraction": float(np.mean([row["mean_home_fraction"] for row in rows])),
        "mean_assist_fraction": float(np.mean([row["mean_assist_fraction"] for row in rows])),
        "mean_tracking_fraction": float(np.mean([row["mean_tracking_fraction"] for row in rows])),
        "mean_final_never_observed_fraction": float(np.mean([row["final_never_observed_fraction"] for row in rows])),
        "mean_final_low_risk_fraction": float(np.mean([row["final_low_risk_fraction"] for row in rows])),
        "mean_final_neglect_pressure": float(np.mean([row["final_neglect_pressure"] for row in rows])),
    }


def _plot_mode_comparison(
    grouped: dict[tuple[str, str], dict[str, float]],
    speed_cases: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    modes = ("oracle", "ekf")
    width = 0.35
    x = np.arange(len(speed_cases), dtype=np.float64)

    def _vals(metric: str, mode: str) -> np.ndarray:
        values = []
        for speed_case in speed_cases:
            summary = grouped.get((mode, speed_case))
            values.append(float(summary[metric]) if summary is not None else np.nan)
        return np.asarray(values, dtype=np.float64)

    axes[0, 0].bar(x - width / 2, _vals("base_compromise_rate", modes[0]), width=width, label="oracle")
    axes[0, 0].bar(x + width / 2, _vals("base_compromise_rate", modes[1]), width=width, label="ekf")
    axes[0, 0].set_title("Base Compromise Rate")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(speed_cases)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, _vals("intercept_success_rate", modes[0]), width=width, label="oracle")
    axes[0, 1].bar(x + width / 2, _vals("intercept_success_rate", modes[1]), width=width, label="ekf")
    axes[0, 1].set_title("Intercept Success Rate")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(speed_cases)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(x - width / 2, _vals("mean_intercept_step", modes[0]), width=width, label="oracle")
    axes[1, 0].bar(x + width / 2, _vals("mean_intercept_step", modes[1]), width=width, label="ekf")
    axes[1, 0].set_title("Mean Intercept Step")
    axes[1, 0].set_ylabel("Step")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(speed_cases)
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x - width / 2, _vals("mean_final_neglect_pressure", modes[0]), width=width, label="oracle")
    axes[1, 1].bar(x + width / 2, _vals("mean_final_neglect_pressure", modes[1]), width=width, label="ekf")
    axes[1, 1].set_title("Mean Final Neglect Pressure")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(speed_cases)
    axes[1, 1].grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    speed_cases = [token.strip() for token in args.speed_cases.split(",") if token.strip()]
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    scenario_cfg = viz._load_scenario(args.scenario_config, args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for speed_case in speed_cases:
        for mode in ("oracle", "ekf"):
            for seed in seeds:
                row = _single_run(
                    scenario_cfg,
                    speed_case=speed_case,
                    mode=mode,
                    seed=seed,
                    args=args,
                )
                rows.append(row)
            # Save one representative diagnostic panel per mode/speed.
            representative = next(item for item in rows if item["mode"] == mode and item["speed_case"] == speed_case)
            rep_path = output_dir / f"{mode}_{speed_case}_seed{representative['seed']}.png"
            env_args = _build_visualizer_args(
                num_drones=args.num_drones,
                max_threat_cycles=args.max_threat_cycles,
                ignore_suspicious_zones=args.ignore_suspicious_zones,
                speed_case=speed_case,
                mode=mode,
                args=args,
            )
            env_kwargs = viz._build_env_kwargs(scenario_cfg, env_args)
            env = viz.BeliefCoverageEnv(**env_kwargs)
            try:
                rep_history, rep_info = viz._run_rollout(
                    env,
                    steps=args.steps,
                    seed=int(representative["seed"]),
                )
                viz._plot_rollout_diagnostics(
                    env,
                    rep_history,
                    rep_info,
                    output_path=rep_path,
                    show=False,
                )
            finally:
                env.close()

    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for speed_case in speed_cases:
        for mode in ("oracle", "ekf"):
            subset = [row for row in rows if row["speed_case"] == speed_case and row["mode"] == mode]
            grouped[(mode, speed_case)] = _aggregate(subset) if subset else {}

    comparison_path = output_dir / "oracle_vs_ekf_comparison.png"
    _plot_mode_comparison(grouped, speed_cases, comparison_path)

    print("=" * 96)
    print("Oracle vs EKF moving-threat comparison")
    print("=" * 96)
    print(f"Speed cases   : {speed_cases}")
    print(f"Seeds         : {seeds}")
    print(f"Output dir    : {output_dir}")
    print(f"Comparison fig: {comparison_path}")
    print("-" * 96)
    for speed_case in speed_cases:
        for mode in ("oracle", "ekf"):
            summary = grouped.get((mode, speed_case), {})
            if not summary:
                continue
            print(
                f"{speed_case:>6} | {mode:>6} | base_fail={summary['base_compromise_rate']:.3f} | "
                f"success={summary['intercept_success_rate']:.3f} | "
                f"confirm={summary['mean_confirmation_step']:.1f} | launch={summary['mean_launch_step']:.1f} | "
                f"intercept={summary['mean_intercept_step']:.1f} | track_err={summary['mean_track_error']:.3f} | "
                f"home_frac={summary['mean_home_fraction']:.3f} | neglect={summary['mean_final_neglect_pressure']:.3f}"
            )
    print("=" * 96)


if __name__ == "__main__":
    main()
