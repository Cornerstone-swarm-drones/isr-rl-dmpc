#!/usr/bin/env python3
"""
run_dmpc.py — Pure DMPC scenario runner.

Executes a swarm mission using the Distributed Model Predictive Controller
(DMPC) with *fixed* cost weights (no RL policy).  The MARLDMPCEnv is driven
with all-ones Q/R scales so that each drone's DMPC optimises the baseline
cost matrices without any learned adaptation.

Scenario-specific reference trajectories (lawnmower, wedge-patrol, expanding
square) are generated inside MARLDMPCEnv and updated at every step, giving the
DMPC a meaningful goal rather than a static hover point.

Task scenarios (defined in config/mission_scenarios.yaml):
    area_surveillance  — Wide-area lawnmower coverage (400 × 200 m strips, 4 drones)
    threat_response    — Wedge patrol toward cycled waypoints (2 targets)
    search_and_track   — Expanding-square search with line formation (3 targets)

Usage
-----
    python scripts/run_dmpc.py --scenario area_surveillance
    python scripts/run_dmpc.py --scenario threat_response --episodes 5
    python scripts/run_dmpc.py --scenario search_and_track --num-drones 4 --seed 0

Options
-------
    --scenario      Task scenario name (default: area_surveillance)
    --config        DMPC config YAML (default: config/dmpc_config.yaml)
    --num-drones    Override number of drones from scenario
    --episodes      Episodes to run (default: 5 for statistical validity)
    --max-steps     Max steps per episode — overrides scenario duration
    --seed          Random seed (default: 42)
    --output        Output directory (default: data/results/dmpc)
    --render        Render the episode (flag)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Ensure the package is importable when running from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.gym_env.marl_env import MARLDMPCEnv, ACT_DIM

logger = logging.getLogger(__name__)

# ── Scenario → MARLDMPCEnv parameter mapping ──────────────────────────────

SCENARIO_STEPS_PER_SEC = 50  # MARLDMPCEnv runs at 50 Hz


def _load_scenarios(config_dir: Path) -> dict:
    path = config_dir / "mission_scenarios.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _scenario_to_env_kwargs(
    scenario_name: str,
    scenario_cfg: dict,
    num_drones_override: int | None,
    max_steps_override: int | None,
    dmpc_cfg: dict,
) -> dict:
    """Map mission_scenarios.yaml entries to MARLDMPCEnv constructor kwargs."""
    num_drones = num_drones_override or scenario_cfg.get("num_drones", 4)
    max_targets = scenario_cfg.get("num_targets", 3)

    # Convert max_duration (seconds) → episode steps at 50 Hz, cap at 2000
    max_dur_s = float(scenario_cfg.get("max_duration", 400.0))
    mission_duration = min(int(max_dur_s * SCENARIO_STEPS_PER_SEC), 2000)
    if max_steps_override is not None:
        mission_duration = max_steps_override

    # DMPC override from scenario (e.g. threat_response specifies accel_max)
    dmpc_override = scenario_cfg.get("dmpc_override", {})
    accel_max = float(dmpc_override.get("accel_max", dmpc_cfg.get("accel_max", 8.0)))
    collision_radius = float(
        dmpc_override.get("collision_radius", dmpc_cfg.get("collision_radius", 3.0))
    )

    return dict(
        num_drones=num_drones,
        max_targets=max(max_targets, 0),
        mission_duration=mission_duration,
        horizon=int(dmpc_cfg.get("prediction_horizon", 20)),
        dt=float(dmpc_cfg.get("dt", 0.02)),
        accel_max=accel_max,
        collision_radius=collision_radius,
        solver_timeout=float(dmpc_cfg.get("solver_timeout", 0.02)),
        osqp_max_iter=int(dmpc_cfg.get("osqp_max_iter", 4000)),
        scenario=scenario_name,
    )


# ── Episode runner ─────────────────────────────────────────────────────────

def _run_episode(env: MARLDMPCEnv, max_steps: int, render: bool) -> dict:
    """Run one episode with all-ones Q/R scales (pure DMPC, no RL)."""
    obs, _ = env.reset()
    terminated = False
    truncated = False
    done = False
    step = 0
    total_reward = 0.0
    solve_times: list[float] = []
    battery_levels: list[float] = []

    # All Q/R scales set to 1.0 → baseline DMPC cost matrices, no RL adaptation
    neutral_action = np.ones(env.num_drones * ACT_DIM, dtype=np.float32)

    while not done and step < max_steps:
        obs, reward, terminated, truncated, info = env.step(neutral_action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        st = info.get("solve_times", [])
        if st:
            solve_times.append(float(np.mean(st)))

        bat = info.get("battery", [])
        if bat:
            battery_levels.append(float(np.mean(bat)))

        if render:
            env.render()

    metrics = {
        "method": "pure_dmpc",
        "total_reward": total_reward,
        "episode_steps": step,
        "mean_solve_time_ms": float(np.mean(solve_times) * 1e3) if solve_times else 0.0,
        "final_battery_mean": float(battery_levels[-1]) if battery_levels else 1.0,
        "min_battery": float(np.min(battery_levels)) if battery_levels else 1.0,
        "terminated_early": bool(terminated),
    }
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Pure DMPC scenario runner")
    parser.add_argument(
        "--scenario", default="area_surveillance",
        choices=["area_surveillance", "threat_response", "search_and_track"],
        help="Task scenario from config/mission_scenarios.yaml",
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "dmpc_config.yaml"),
                        help="DMPC config YAML")
    parser.add_argument("--num-drones", type=int, default=None,
                        help="Override num_drones from scenario")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes to run (default: 5 for statistical validity)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (overrides scenario duration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "dmpc"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Load configs
    dmpc_cfg: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            dmpc_cfg = yaml.safe_load(f).get("dmpc", {})

    scenarios = _load_scenarios(ROOT / "config")
    if args.scenario not in scenarios:
        logger.error("Unknown scenario '%s'. Available: %s", args.scenario,
                     list(scenarios.keys()))
        sys.exit(1)
    scenario_cfg = scenarios[args.scenario]

    env_kwargs = _scenario_to_env_kwargs(
        args.scenario, scenario_cfg, args.num_drones, args.max_steps, dmpc_cfg
    )

    print("=" * 60)
    print("  ISR-RL-DMPC — Pure DMPC Runner")
    print("=" * 60)
    print(f"  Scenario      : {args.scenario}")
    print(f"  Drones        : {env_kwargs['num_drones']}")
    print(f"  Max targets   : {env_kwargs['max_targets']}")
    print(f"  Duration      : {env_kwargs['mission_duration']} steps")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Solver timeout: {env_kwargs['solver_timeout'] * 1000:.0f} ms")
    print(f"  OSQP max iter : {env_kwargs['osqp_max_iter']}")
    print(f"  Output        : {args.output}")
    print("=" * 60)

    env = MARLDMPCEnv(**env_kwargs)
    env.reset(seed=args.seed)

    all_metrics: list[dict] = []
    for ep in range(1, args.episodes + 1):
        t0 = time.perf_counter()
        metrics = _run_episode(env, env_kwargs["mission_duration"], args.render)
        elapsed = time.perf_counter() - t0
        metrics["scenario"] = args.scenario
        metrics["episode"] = ep
        metrics["wall_time_s"] = round(elapsed, 3)

        logger.info(
            "Episode %d/%d | reward=%.2f  steps=%d  solve=%.2f ms  bat=%.3f  (%.1f s)",
            ep, args.episodes,
            metrics["total_reward"],
            metrics["episode_steps"],
            metrics["mean_solve_time_ms"],
            metrics["final_battery_mean"],
            elapsed,
        )
        all_metrics.append(metrics)

    env.close()

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{args.scenario}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Results saved → %s", out_file)

    rewards = [m["total_reward"] for m in all_metrics]
    print(f"\nSummary: mean_reward={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")
    print(f"         min={np.min(rewards):.2f}  max={np.max(rewards):.2f}")


if __name__ == "__main__":
    main()
