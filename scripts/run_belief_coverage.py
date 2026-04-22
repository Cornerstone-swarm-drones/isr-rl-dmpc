#!/usr/bin/env python3
"""
Smoke runner for the parallel Phase 1 belief-coverage environment.

This keeps the existing continuous-control scripts untouched while providing
an easy entrypoint to exercise the cell-selection action path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.gym_env import BeliefCoverageEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 belief-coverage smoke scenario."
    )
    parser.add_argument(
        "--scenario",
        default="area_surveillance",
        help="Scenario name from config/mission_scenarios.yaml",
    )
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
        help="Path to mission scenario YAML",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of env steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--summary-every",
        type=int,
        default=5,
        help="Print metrics every N steps",
    )
    parser.add_argument(
        "--num-drones",
        type=int,
        default=None,
        help="Override scenario num_drones",
    )
    parser.add_argument(
        "--disable-neighbor-sharing",
        action="store_true",
        help="Turn off Phase B neighbor delta sharing",
    )
    parser.add_argument(
        "--disable-goal-projection",
        action="store_true",
        help="Turn off connectivity-aware goal projection",
    )
    parser.add_argument(
        "--policy",
        choices=("patrol", "greedy"),
        default="patrol",
        help="High-level baseline policy to run",
    )
    return parser.parse_args()


def _load_scenario(path: str, scenario_name: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        scenarios = yaml.safe_load(handle) or {}
    if scenario_name not in scenarios:
        available = ", ".join(sorted(scenarios)) or "<none>"
        raise KeyError(f"Scenario '{scenario_name}' not found. Available: {available}")
    return scenarios[scenario_name]


def _build_env_kwargs(scenario_cfg: dict, args: argparse.Namespace) -> dict:
    belief_cfg = dict(scenario_cfg.get("belief_coverage", {}))
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
        "base_station": tuple(belief_cfg.get("base_station", [20.0, 20.0])),
        "suspicious_zones": belief_cfg.get("suspicious_zones", []),
        "enable_neighbor_sharing": not args.disable_neighbor_sharing,
        "enable_goal_projection": not args.disable_goal_projection,
    }


def _format_threshold_counts(counts: dict) -> str:
    return (
        f">{0.1:.1f}:{counts.get('gt_0_1', 0)} "
        f">{0.4:.1f}:{counts.get('gt_0_4', 0)} "
        f">{0.7:.1f}:{counts.get('gt_0_7', 0)}"
    )


def main() -> None:
    args = _parse_args()
    scenario_cfg = _load_scenario(args.scenario_config, args.scenario)
    env_kwargs = _build_env_kwargs(scenario_cfg, args)

    print("=" * 72)
    print("BeliefCoverageEnv smoke run")
    print("=" * 72)
    print(f"Scenario              : {args.scenario}")
    print(f"Num drones            : {env_kwargs['num_drones']}")
    print(f"Area size             : {env_kwargs['area_size']}")
    print(f"Policy                : {args.policy}")
    print(f"Neighbor sharing      : {env_kwargs['enable_neighbor_sharing']}")
    print(f"Goal projection       : {env_kwargs['enable_goal_projection']}")
    print("=" * 72)

    env = BeliefCoverageEnv(**env_kwargs)
    observation, info = env.reset(seed=args.seed)
    print(
        "Reset:"
        f" obs_dim={observation.shape[0]}"
        f" mean_uncertainty={info['mean_uncertainty']:.3f}"
        f" mean_risk={info['mean_risk_score']:.3f}"
        f" connectivity={info['connectivity_state']['component_count']} component(s)"
        f" risk={_format_threshold_counts(info['risk_counts'])}"
    )

    last_reward = 0.0
    last_info = info
    for step in range(1, args.steps + 1):
        if args.policy == "patrol":
            action = env.select_patrol_action()
        else:
            action = env.select_greedy_action(unique=True)
        _, reward, terminated, truncated, last_info = env.step(action)
        last_reward = float(reward)

        if step % max(args.summary_every, 1) == 0 or terminated or truncated:
            connectivity = last_info["connectivity_state"]
            print(
                f"step={step:04d} "
                f"reward={reward:+.3f} "
                f"mean_u={last_info['mean_uncertainty']:.3f} "
                f"mean_risk={last_info['mean_risk_score']:.3f} "
                f"total_u={last_info['total_uncertainty']:.3f} "
                f"neglect={last_info['neglect_pressure']:.3f} "
                f"components={connectivity['component_count']} "
                f"risk={_format_threshold_counts(last_info['risk_counts'])} "
                f"home={int(np.sum(last_info['selected_in_home']))}/{env.num_drones} "
                f"assist={int(np.sum(last_info['selected_in_assist']))}/{env.num_drones} "
                f"detours={int(np.sum(last_info['patrol_detouring']))}"
            )

        if terminated or truncated:
            break

    print("=" * 72)
    print(
        "Done:"
        f" reward={last_reward:+.3f}"
        f" mean_uncertainty={last_info['mean_uncertainty']:.3f}"
        f" mean_risk={last_info['mean_risk_score']:.3f}"
        f" connectivity={last_info['connectivity_state']['component_count']}"
        f" risk={_format_threshold_counts(last_info['risk_counts'])}"
    )
    env.close()


if __name__ == "__main__":
    main()
