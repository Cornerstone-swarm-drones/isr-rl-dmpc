"""
ISR-DMPC Mission Runner

Executes a DMPC-controlled ISR swarm mission and reports performance
metrics.  This script replaces the former RL training script
(train_agent.py) — no neural networks or experience replay are involved.

Usage
-----
    python scripts/run_mission.py [OPTIONS]

Options
-------
    --config     Path to YAML config file (default: config/dmpc_config.yaml)
    --num-drones Number of drones (default: 4)
    --task       Mission preset: recon | intel | target_pursuit (default: recon)
    --episodes   Number of episodes to run (default: 1)
    --max-steps  Maximum steps per episode (default: 1000)
    --seed       Random seed (default: 42)
    --output     Output directory for metrics (default: results/)
    --render     Render mission (flag)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

from isr_rl_dmpc.gym_env import ISRGridEnv
from isr_rl_dmpc.agents import DMPCAgent
from isr_rl_dmpc.gym_env.reward_shaper import TASK_REWARD_PRESETS, RewardWeights
from isr_rl_dmpc.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def build_reference_from_obs(obs: dict, horizon: int = 20, state_dim: int = 11) -> np.ndarray:
    """Build a simple reference trajectory by repeating the current observation."""
    flat = np.zeros(state_dim, dtype=np.float32)
    if "swarm_states" in obs:
        drone0 = np.asarray(obs["swarm_states"][0], dtype=np.float32)
        flat[: min(len(drone0), state_dim)] = drone0[: state_dim]
    x_ref = np.tile(flat, (horizon + 1, 1))
    return x_ref


def run_episode(
    env: ISRGridEnv,
    agent: DMPCAgent,
    max_steps: int,
    render: bool,
) -> dict:
    """Run one episode and return performance metrics."""
    obs, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # Flatten observation to get first drone's state
        flat_obs = agent.flatten_obs(obs)
        state = flat_obs[:11] if len(flat_obs) >= 11 else np.pad(flat_obs, (0, 11 - len(flat_obs)))
        x_ref = build_reference_from_obs(obs, horizon=agent.dmpc.config.horizon)

        motor_cmds, info = agent.act(state, x_ref)

        # Step environment with first-drone action (simplified)
        action = motor_cmds if env.action_space.contains(motor_cmds) else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        steps += 1

        if render:
            env.render()

    metrics = agent.get_metrics()
    metrics["total_reward"] = total_reward
    metrics["episode_steps"] = steps
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ISR-DMPC Mission Runner")
    parser.add_argument("--config", default="config/dmpc_config.yaml")
    parser.add_argument("--num-drones", type=int, default=4)
    parser.add_argument("--task", default="recon",
                        choices=["recon", "intel", "target_pursuit"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    setup_logger(level=logging.INFO)
    logger.info("ISR-DMPC Mission Runner")
    logger.info("Task: %s | Drones: %d | Episodes: %d",
                args.task, args.num_drones, args.episodes)

    # Load DMPC config
    dmpc_cfg: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            dmpc_cfg = yaml.safe_load(f).get("dmpc", {})

    # Task reward weights
    weights: RewardWeights = TASK_REWARD_PRESETS.get(args.task, RewardWeights())

    # Create environment
    env = ISRGridEnv(
        num_drones=args.num_drones,
        reward_weights=weights,
    )
    env.reset(seed=args.seed)

    # Create pure DMPC agent
    agent = DMPCAgent(
        horizon=dmpc_cfg.get("prediction_horizon", 20),
        dt=dmpc_cfg.get("dt", 0.02),
        accel_max=dmpc_cfg.get("accel_max", 10.0),
        collision_radius=dmpc_cfg.get("collision_radius", 5.0),
    )

    # Run episodes
    all_metrics = []
    for ep in range(1, args.episodes + 1):
        t0 = time.perf_counter()
        metrics = run_episode(env, agent, args.max_steps, args.render)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Episode %d/%d: reward=%.2f steps=%d rmse=%.4f success_rate=%.3f "
            "(%.2f s)",
            ep, args.episodes,
            metrics["total_reward"],
            metrics["episode_steps"],
            metrics["rmse_tracking"],
            metrics["solve_success_rate"],
            elapsed,
        )
        all_metrics.append(metrics)

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"dmpc_{args.task}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Results saved to %s", out_file)

    # Summary
    rewards = [m["total_reward"] for m in all_metrics]
    logger.info("Summary: mean_reward=%.2f  std_reward=%.2f",
                float(np.mean(rewards)), float(np.std(rewards)))

    env.close()


if __name__ == "__main__":
    main()
