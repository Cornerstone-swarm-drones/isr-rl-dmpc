"""
ISR-RL-DMPC Training Script
Trains DMPC agent parameters via reinforcement learning.

Compatible with existing isr_rl_dmpc package:
  - ISRGridEnv  (gym_env)
  - DMPCAgent   (agents)
  - MetricsLogger / setup_logger (utils.logging_utils)
"""

import argparse
import asyncio
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, Optional

from isr_rl_dmpc.gym_env import ISRGridEnv
from isr_rl_dmpc.agents import DMPCAgent
from isr_rl_dmpc.config import load_config
from isr_rl_dmpc.utils.logging_utils import setup_logger, MetricsLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ISR-RL-DMPC Agent')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num-episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--save-freq', type=int, default=50,
                        help='Checkpoint save frequency (episodes)')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--output-dir', type=str, default='data/training_logs',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--foxglove', action='store_true',
                        help='Enable live Foxglove Studio visualization during training')
    parser.add_argument('--foxglove-port', type=int, default=8765,
                        help='Foxglove WebSocket server port (default: 8765)')
    return parser.parse_args()


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _obs_dim(env: ISRGridEnv) -> int:
    """Compute the flat observation dimension from the environment."""
    obs, _ = env.reset()
    flat = DMPCAgent.flatten_obs(obs)
    return flat.shape[0]


def train(config_path: str,
          num_episodes: int = 500,
          num_steps_per_episode: int = 1000,
          save_freq: int = 50,
          eval_freq: int = 100,
          output_dir: str = 'data/training_logs',
          device: str = 'cpu',
          seed: int = 42,
          foxglove_bridge: Optional[object] = None) -> Dict:
    """
    Main training loop for ISR-RL-DMPC agent.

    Args:
        config_path: Path to configuration YAML file
        num_episodes: Number of training episodes
        num_steps_per_episode: Maximum steps per episode
        save_freq: Checkpoint save frequency (episodes)
        eval_freq: Evaluation frequency (episodes)
        output_dir: Output directory for logs and checkpoints
        device: Device to use ('cuda' or 'cpu')
        seed: Random seed
        foxglove_bridge: Optional FoxgloveBridge instance for live visualization

    Returns:
        Dictionary containing training statistics
    """
    # Setup
    set_seeds(seed)
    output_path = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('train_agent', str(output_path / 'train.log'))
    metrics_logger = MetricsLogger(name='train_metrics', log_dir=str(output_path))

    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Device: {device}, Seed: {seed}")
    logger.info(f"Episodes: {num_episodes}, Steps per episode: {num_steps_per_episode}")

    # Load configuration
    cfg = load_config(config_path)

    # Load environment (using keyword arguments as ISRGridEnv expects)
    env = ISRGridEnv(
        num_drones=10,
        max_targets=5,
        mission_duration=num_steps_per_episode,
    )

    # Determine observation dimension from the environment
    state_dim = _obs_dim(env)
    action_dim = int(np.prod(env.action_space.shape))

    # Create agent
    agent = DMPCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate_critic=cfg.learning.learning_rate_critic,
        learning_rate_actor=cfg.learning.learning_rate_actor,
        gamma=cfg.learning.discount_factor,
        batch_size=cfg.learning.batch_size,
        buffer_size=int(cfg.learning.buffer_size),
        device=device,
    )

    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'coverage_efficiency': [],
        'threat_detection_rate': [],
        'collision_count': [],
        'energy_efficiency': [],
        'critic_losses': [],
        'actor_losses': [],
    }

    # Training loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_metrics = {
            'coverage': [],
            'energy': [],
            'threats_detected': 0,
            'threats_engaged': 0,
            'collisions': 0,
        }

        # Episode rollout
        for step in range(num_steps_per_episode):
            # Policy: DMPC with learned parameters
            action = agent.act(obs, training=True)

            # Reshape action to match env action space shape
            action_env = np.clip(
                action.reshape(env.action_space.shape), 0.0, 1.0
            )

            # Environment step (5-tuple: obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # Publish to Foxglove Studio if bridge is active
            if foxglove_bridge is not None and foxglove_bridge.is_running:
                from isr_rl_dmpc.utils.foxglove_bridge import extract_targets_from_obs
                ts_ns = int(time.time() * 1e9)
                swarm = next_obs["swarm"]
                tgt_pos, tgt_cls = extract_targets_from_obs(next_obs["targets"])
                foxglove_bridge.publish_scene(
                    drone_positions=swarm[:, :3],
                    drone_quaternions=swarm[:, 9:13],
                    drone_batteries=swarm[:, 16],
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts_ns,
                )
                foxglove_bridge.publish_metrics(info, reward=reward, timestamp_ns=ts_ns)

            # Store experience
            agent.remember(obs, action, reward, next_obs, done)

            # Update networks (if enough experiences)
            if agent.ready_to_train():
                loss_critic, loss_actor = agent.train_on_batch()
                stats['critic_losses'].append(loss_critic)
                stats['actor_losses'].append(loss_actor)

            # Track metrics
            episode_reward += reward
            episode_steps += 1
            episode_metrics['coverage'].append(info.get('coverage', 0))
            episode_metrics['energy'].append(info.get('avg_battery', 0))
            if info.get('collisions', 0) > 0:
                episode_metrics['collisions'] += info['collisions']

            obs = next_obs

            if done:
                break

        # Episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)
        final_coverage = (
            episode_metrics['coverage'][-1]
            if episode_metrics['coverage'] else 0.0
        )
        stats['coverage_efficiency'].append(final_coverage)
        stats['collision_count'].append(episode_metrics['collisions'])

        # Energy efficiency: coverage per 100 Wh
        total_energy = (
            sum(episode_metrics['energy']) if episode_metrics['energy'] else 1.0
        )
        energy_eff = (final_coverage * 100) / (total_energy / 100 + 1e-6)
        stats['energy_efficiency'].append(energy_eff)

        # Logging
        log_metrics = {
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'coverage': final_coverage,
            'collisions': episode_metrics['collisions'],
            'energy_eff': energy_eff,
            'critic_loss': (
                float(np.mean(stats['critic_losses'][-100:]))
                if stats['critic_losses'] else 0
            ),
            'actor_loss': (
                float(np.mean(stats['actor_losses'][-100:]))
                if stats['actor_losses'] else 0
            ),
        }

        metrics_logger.log_dict(log_metrics, step=episode)

        if (episode + 1) % 10 == 0:
            rolling_reward = np.mean(stats['episode_rewards'][-100:])
            rolling_coverage = np.mean(stats['coverage_efficiency'][-100:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} (avg: {rolling_reward:.2f}) | "
                f"Coverage: {final_coverage:.1%} "
                f"(avg: {rolling_coverage:.1%}) | "
                f"Steps: {episode_steps} | "
                f"Collisions: {episode_metrics['collisions']}"
            )

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = output_path / f"checkpoint_ep{episode + 1}.pt"
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_metrics = evaluate_policy(agent, env, num_episodes=10)
            logger.info(f"Evaluation at episode {episode + 1}:")
            logger.info(f"  Mean Coverage: {eval_metrics['mean_coverage']:.1%}")
            logger.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
            logger.info(
                f"  Collision Rate: {eval_metrics['collision_rate']:.2%}"
            )

    # Final checkpoint
    final_path = output_path / "final_model.pt"
    agent.save(final_path)
    logger.info(f"Training complete. Final model saved: {final_path}")

    # Save training statistics
    stats_path = output_path / "training_stats.json"
    with open(stats_path, 'w') as f:
        json_stats = {}
        for k, vals in stats.items():
            if isinstance(vals, list):
                json_stats[k] = [float(v) for v in vals]
            else:
                json_stats[k] = float(vals)
        json.dump(json_stats, f, indent=2)

    return stats


def evaluate_policy(agent, env, num_episodes: int = 10) -> Dict:
    """
    Evaluate learned policy on test missions.

    Args:
        agent: Trained DMPCAgent
        env: ISRGridEnv instance
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'rewards': [],
        'coverage': [],
        'collisions': 0,
        'episode_lengths': [],
    }

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Deterministic policy (no exploration)
            action = agent.act(obs, training=False, deterministic=True)
            action_env = np.clip(
                action.reshape(env.action_space.shape), 0.0, 1.0
            )
            obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1

            results['collisions'] += info.get('collisions', 0)

        results['rewards'].append(episode_reward)
        results['coverage'].append(info.get('coverage', 0))
        results['episode_lengths'].append(episode_steps)

    return {
        'mean_reward': float(np.mean(results['rewards'])),
        'std_reward': float(np.std(results['rewards'])),
        'mean_coverage': float(np.mean(results['coverage'])),
        'std_coverage': float(np.std(results['coverage'])),
        'collision_rate': results['collisions'] / max(num_episodes, 1),
        'mean_episode_length': float(np.mean(results['episode_lengths'])),
    }


if __name__ == '__main__':
    args = parse_args()

    bridge = None
    if args.foxglove:
        from isr_rl_dmpc.utils.foxglove_bridge import FoxgloveBridge

        async def _start_bridge(port: int) -> FoxgloveBridge:
            b = FoxgloveBridge(host="0.0.0.0", port=port)
            await b.start()
            return b

        bridge = asyncio.get_event_loop().run_until_complete(
            _start_bridge(args.foxglove_port)
        )
        print(
            f"Foxglove bridge started on ws://0.0.0.0:{args.foxglove_port} — "
            "connect Foxglove Studio to visualize training"
        )

    try:
        stats = train(
            config_path=args.config,
            num_episodes=args.num_episodes,
            num_steps_per_episode=args.num_steps,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            foxglove_bridge=bridge,
        )
    finally:
        if bridge is not None:
            asyncio.get_event_loop().run_until_complete(bridge.stop())

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {len(stats['episode_rewards'])}")
    print(
        f"Final avg reward (last 100): "
        f"{np.mean(stats['episode_rewards'][-100:]):.2f}"
    )
    print(
        f"Final avg coverage (last 100): "
        f"{np.mean(stats['coverage_efficiency'][-100:]):.1%}"
    )
    print(f"Total collisions: {sum(stats['collision_count'])}")
    print("=" * 60)
