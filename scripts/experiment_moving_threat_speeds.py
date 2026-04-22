#!/usr/bin/env python3
"""
Controlled speed-case experiment for the moving persistent-threat patrol loop.

This script runs the same deterministic patrol policy under three moving-threat
speed cases, saves one diagnostic figure per case, and writes a compact
comparison plot from the actual rollout histories.
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
    parser = argparse.ArgumentParser(
        description="Run slow/medium/fast moving-threat experiments for BeliefCoverageEnv.",
    )
    parser.add_argument("--scenario", default="area_surveillance")
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
    )
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--max-threat-cycles", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "visualizations" / "moving_threat_speed_cases"),
    )
    parser.add_argument(
        "--ignore-suspicious-zones",
        action="store_true",
        help="Run the speed cases without the extra suspicious-zone spawn preference.",
    )
    return parser.parse_args()


def _build_visualizer_args(base_args: argparse.Namespace, speed_case: str) -> SimpleNamespace:
    return SimpleNamespace(
        num_drones=base_args.num_drones,
        disable_neighbor_sharing=False,
        disable_goal_projection=False,
        disable_persistent_threats=False,
        max_threat_cycles=base_args.max_threat_cycles,
        ignore_suspicious_zones=base_args.ignore_suspicious_zones,
        threat_speed_case=speed_case,
        threat_speed=None,
        assist_spike_drone=None,
        assist_spike_step=4,
        assist_spike_level=0.95,
        force_threat_home_drone=None,
    )


def _case_summary(env, history: dict, final_info: dict) -> dict[str, object]:
    summary = viz._summarize_rollout(env, history, final_info)
    confirmation_step = viz._first_event_step(history, "threat_confirmed")
    launch_step = viz._first_event_step(history, "interceptor_dispatched")
    intercept_step = viz._first_event_step(history, "threat_removed")
    mission_fail_step = viz._first_event_step(history, "mission_failed")
    steps_run = len(history["mean_patrol_risk_belief"]) - 1
    dt = float(env.dt)
    return {
        "speed_case": final_info["threat_speed_case"],
        "speed_mps": float(final_info["threat_speed"]),
        "steps_run": int(steps_run),
        "confirmation_step": confirmation_step,
        "confirmation_time_s": None if confirmation_step is None else float(confirmation_step * dt),
        "launch_step": launch_step,
        "launch_time_s": None if launch_step is None else float(launch_step * dt),
        "intercept_step": intercept_step,
        "intercept_time_s": None if intercept_step is None else float(intercept_step * dt),
        "mission_fail_step": mission_fail_step,
        "mission_fail_time_s": None if mission_fail_step is None else float(mission_fail_step * dt),
        "intercept_success": bool(intercept_step is not None),
        "mission_failed": bool(final_info["mission_failed"]),
        "max_tracker_count": int(np.max(np.sum(np.asarray(history["tracking_bias_drones"], dtype=np.int32), axis=1))),
        "max_tracking_fraction": float(np.max(history["tracking_fraction"])),
        "mean_home_selection_fraction": float(np.mean(history["home_fraction"])),
        "final_home_selection_fraction": float(np.mean(final_info["selected_in_home"])),
        "final_low_risk_fraction": float(final_info["low_risk_fraction"]),
        "final_never_observed_fraction": float(final_info["never_observed_fraction"]),
        "final_mean_patrol_belief": float(final_info["mean_patrol_risk_belief"]),
        "final_mean_threat_belief": float(final_info["mean_threat_belief"]),
        "summary": summary,
    }


def _plot_comparison(results: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    for result in results:
        summary = result["summary"]
        label = f"{summary['speed_case']} ({summary['speed_mps']:.1f} m/s)"
        steps = np.arange(len(result["history"]["threat_distance_to_base"]))
        axes[0, 0].plot(
            steps,
            np.asarray(result["history"]["threat_distance_to_base"], dtype=np.float64),
            label=label,
            linewidth=1.8,
        )
        axes[0, 1].plot(
            steps,
            np.asarray(result["history"]["tracking_fraction"], dtype=np.float64),
            label=label,
            linewidth=1.8,
        )
        axes[1, 0].plot(
            steps,
            np.asarray(result["history"]["home_fraction"], dtype=np.float64),
            label=label,
            linewidth=1.8,
        )

    axes[0, 0].set_title("Threat Distance To Base")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Distance [m]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Tracking Fraction")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].set_title("Home-Selection Fraction")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(alpha=0.25)

    labels = [str(result["speed_case"]) for result in results]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.18
    confirmation = [
        float(result["summary"]["confirmation_step"]) if result["summary"]["confirmation_step"] is not None else np.nan
        for result in results
    ]
    launch = [
        float(result["summary"]["launch_step"]) if result["summary"]["launch_step"] is not None else np.nan
        for result in results
    ]
    intercept = [
        float(result["summary"]["intercept_step"]) if result["summary"]["intercept_step"] is not None else np.nan
        for result in results
    ]
    fail = [
        float(result["summary"]["mission_fail_step"]) if result["summary"]["mission_fail_step"] is not None else np.nan
        for result in results
    ]
    axes[1, 1].bar(x - 1.5 * width, confirmation, width=width, label="Confirm", color="#457B9D")
    axes[1, 1].bar(x - 0.5 * width, launch, width=width, label="Launch", color="#2A9D8F")
    axes[1, 1].bar(x + 0.5 * width, intercept, width=width, label="Intercept", color="#E9C46A")
    axes[1, 1].bar(x + 1.5 * width, fail, width=width, label="Fail", color="#D62828")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Event Steps")
    axes[1, 1].set_xlabel("Speed case")
    axes[1, 1].set_ylabel("Step")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    scenario_cfg = viz._load_scenario(args.scenario_config, args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for speed_case in ("slow", "medium", "fast"):
        viz_args = _build_visualizer_args(args, speed_case)
        env_kwargs = viz._build_env_kwargs(scenario_cfg, viz_args)
        env = viz.BeliefCoverageEnv(**env_kwargs)
        try:
            history, final_info = viz._run_rollout(
                env,
                steps=args.steps,
                seed=args.seed,
                force_threat_home_drone=None,
            )
            output_path = output_dir / f"moving_threat_{speed_case}.png"
            viz._plot_rollout_diagnostics(
                env,
                history,
                final_info,
                output_path=output_path,
                show=False,
            )
            results.append(
                {
                    "speed_case": speed_case,
                    "history": history,
                    "final_info": final_info,
                    "figure": output_path,
                    "summary": _case_summary(env, history, final_info),
                }
            )
        finally:
            env.close()

    comparison_path = output_dir / "moving_threat_speed_comparison.png"
    _plot_comparison(results, comparison_path)

    print("=" * 84)
    print("Moving persistent-threat speed experiment")
    print("=" * 84)
    print(f"Output directory : {output_dir}")
    print(f"Comparison plot  : {comparison_path}")
    print("-" * 84)
    for result in results:
        summary = result["summary"]
        print(
            f"{summary['speed_case']:>6} | speed={summary['speed_mps']:.1f} m/s | "
            f"confirm={summary['confirmation_step']} | launch={summary['launch_step']} | "
            f"intercept={summary['intercept_step']} | fail={summary['mission_fail_step']} | "
            f"success={summary['intercept_success']} | mission_failed={summary['mission_failed']} | "
            f"max_trackers={summary['max_tracker_count']} | "
            f"home_frac={summary['mean_home_selection_fraction']:.3f}"
        )
        print(f"         figure={result['figure']}")
    print("=" * 84)


if __name__ == "__main__":
    main()
