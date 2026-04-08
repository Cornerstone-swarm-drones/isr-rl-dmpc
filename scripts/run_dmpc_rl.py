#!/usr/bin/env python3
"""
run_dmpc_rl.py — DMPC-RL (MAPPO-adaptive DMPC) scenario runner.

Executes a swarm mission using the MAPPO policy trained by
``scripts/train_mappo.py``.  The policy outputs per-drone Q/R scale vectors
at every step, dynamically adapting the DMPC cost matrices in response to the
current observation.

Task scenarios (defined in config/mission_scenarios.yaml):
    area_surveillance  — Wide-area persistent coverage (400 × 400 m, 4 drones)
    threat_response    — Rapid detection & tracking of hostile targets
    search_and_track   — Locate and track mobile targets in large area

Usage
-----
    python scripts/run_dmpc_rl.py --scenario area_surveillance
    python scripts/run_dmpc_rl.py --scenario threat_response --model models/mappo_dmpc/final
    python scripts/run_dmpc_rl.py --scenario search_and_track --episodes 3 --seed 0

Options
-------
    --scenario      Task scenario name (default: area_surveillance)
    --model         Path to trained MAPPO model zip (without .zip extension)
                    Default: models/mappo_dmpc/final
    --config        DMPC config YAML (default: config/dmpc_config.yaml)
    --num-drones    Override number of drones from scenario
    --episodes      Episodes to run (default: 1)
    --max-steps     Max steps per episode — overrides scenario duration
    --seed          Random seed (default: 42)
    --output        Output directory (default: data/results/dmpc_rl)
    --deterministic Use deterministic MAPPO policy (flag, recommended for eval)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.gym_env.marl_env import MARLDMPCEnv, ACT_DIM
from isr_rl_dmpc.agents.mappo_agent import MAPPOAgent

logger = logging.getLogger(__name__)

SCENARIO_STEPS_PER_SEC = 50


def _load_scenarios(config_dir: Path) -> dict:
    path = config_dir / "mission_scenarios.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _scenario_to_env_kwargs(scenario_cfg: dict, num_drones_override: int | None,
                             max_steps_override: int | None, dmpc_cfg: dict) -> dict:
    """Map mission_scenarios.yaml entries to MARLDMPCEnv constructor kwargs."""
    num_drones = num_drones_override or scenario_cfg.get("num_drones", 4)
    max_targets = scenario_cfg.get("num_targets", 3)

    max_dur_s = float(scenario_cfg.get("max_duration", 400.0))
    mission_duration = min(int(max_dur_s * SCENARIO_STEPS_PER_SEC), 2000)
    if max_steps_override is not None:
        mission_duration = max_steps_override

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
    )


def _run_episode(
    env: MARLDMPCEnv,
    agent: MAPPOAgent,
    max_steps: int,
    deterministic: bool,
    render: bool,
) -> dict:
    """Run one episode with the MAPPO policy providing adaptive Q/R scales."""
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    solve_times: list[float] = []
    battery_levels: list[float] = []
    q_scale_history: list[float] = []  # mean q_scale across drones/dims

    while not done and step < max_steps:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        st = info.get("solve_times", [])
        if st:
            solve_times.append(float(np.mean(st)))

        bat = info.get("battery", [])
        if bat:
            battery_levels.append(float(np.mean(bat)))

        # Track mean q_scale to measure RL adaptation activity
        action_flat = np.asarray(action, dtype=np.float32).ravel()
        expected_size = env.num_drones * ACT_DIM
        assert action_flat.size == expected_size, (
            f"Expected action size {expected_size}, got {action_flat.size}"
        )
        acts = action_flat.reshape(env.num_drones, ACT_DIM)
        q_scale_history.append(float(np.mean(acts[:, :11])))

        if render:
            env.render()

    metrics = {
        "method": "dmpc_rl",
        "total_reward": total_reward,
        "episode_steps": step,
        "mean_solve_time_ms": float(np.mean(solve_times) * 1e3) if solve_times else 0.0,
        "final_battery_mean": float(battery_levels[-1]) if battery_levels else 1.0,
        "min_battery": float(np.min(battery_levels)) if battery_levels else 1.0,
        "mean_q_scale": float(np.mean(q_scale_history)) if q_scale_history else 1.0,
        "std_q_scale": float(np.std(q_scale_history)) if q_scale_history else 0.0,
    }
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="DMPC-RL (MAPPO) scenario runner")
    parser.add_argument(
        "--scenario", default="area_surveillance",
        choices=["area_surveillance", "threat_response", "search_and_track"],
        help="Task scenario from config/mission_scenarios.yaml",
    )
    parser.add_argument(
        "--model", default=str(ROOT / "models" / "mappo_dmpc" / "final"),
        help="Path to trained MAPPO model (without .zip extension)",
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "dmpc_config.yaml"),
                        help="DMPC config YAML")
    parser.add_argument("--num-drones", type=int, default=None,
                        help="Override num_drones from scenario")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (overrides scenario duration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "dmpc_rl"))
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic MAPPO policy (recommended for eval)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false",
                        help="Use stochastic MAPPO policy")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Load DMPC config
    dmpc_cfg: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            dmpc_cfg = yaml.safe_load(f).get("dmpc", {})

    # Load scenario
    scenarios = _load_scenarios(ROOT / "config")
    if args.scenario not in scenarios:
        logger.error("Unknown scenario '%s'. Available: %s", args.scenario,
                     list(scenarios.keys()))
        sys.exit(1)
    scenario_cfg = scenarios[args.scenario]

    env_kwargs = _scenario_to_env_kwargs(
        scenario_cfg, args.num_drones, args.max_steps, dmpc_cfg
    )

    print("=" * 60)
    print("  ISR-RL-DMPC — DMPC-RL (MAPPO) Runner")
    print("=" * 60)
    print(f"  Scenario      : {args.scenario}")
    print(f"  Model         : {args.model}")
    print(f"  Drones        : {env_kwargs['num_drones']}")
    print(f"  Max targets   : {env_kwargs['max_targets']}")
    print(f"  Duration      : {env_kwargs['mission_duration']} steps")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Deterministic : {args.deterministic}")
    print(f"  Output        : {args.output}")
    print("=" * 60)

    # Create environment
    env = MARLDMPCEnv(**env_kwargs)
    env.reset(seed=args.seed)

    # Load MAPPO agent
    model_path = Path(args.model)
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        logger.error(
            "Model not found at '%s'. Train first with:\n"
            "  python scripts/train_mappo.py",
            args.model,
        )
        sys.exit(1)

    logger.info("Loading MAPPO policy from %s …", args.model)
    agent = MAPPOAgent.load(args.model, env=env)
    logger.info("Policy loaded.")

    all_metrics: list[dict] = []
    for ep in range(1, args.episodes + 1):
        t0 = time.perf_counter()
        metrics = _run_episode(
            env, agent, env_kwargs["mission_duration"], args.deterministic, args.render
        )
        elapsed = time.perf_counter() - t0
        metrics["scenario"] = args.scenario
        metrics["episode"] = ep
        metrics["wall_time_s"] = round(elapsed, 3)
        metrics["model"] = str(args.model)

        logger.info(
            "Episode %d/%d | reward=%.2f  steps=%d  solve=%.2f ms  "
            "bat=%.3f  q̄=%.2f±%.2f  (%.1f s)",
            ep, args.episodes,
            metrics["total_reward"],
            metrics["episode_steps"],
            metrics["mean_solve_time_ms"],
            metrics["final_battery_mean"],
            metrics["mean_q_scale"],
            metrics["std_q_scale"],
            elapsed,
        )
        all_metrics.append(metrics)

    env.close()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{args.scenario}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Results saved → %s", out_file)

    rewards = [m["total_reward"] for m in all_metrics]
    print(f"\nSummary: mean_reward={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
