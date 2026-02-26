"""
ISR-RL-DMPC Hyperparameter Search
Grid/random search over key hyperparameters for optimal configuration.

Compatible with existing isr_rl_dmpc package:
  - ISRGridEnv  (gym_env)
  - DMPCAgent   (agents)
  - setup_logger (utils.logging_utils)
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple

from isr_rl_dmpc.gym_env import ISRGridEnv
from isr_rl_dmpc.agents import DMPCAgent
from isr_rl_dmpc.utils.logging_utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search for ISR-RL-DMPC'
    )
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Base configuration file')
    parser.add_argument('--num-trials', type=int, default=100,
                        help='Episodes per configuration')
    parser.add_argument('--output-dir', type=str,
                        default='data/hyperparameter_search',
                        help='Output directory')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--parallel', action='store_true',
                        help='Run configurations in parallel')
    return parser.parse_args()


def get_search_space() -> Dict[str, List]:
    """
    Define hyperparameter search space.

    Returns:
        Dictionary mapping parameter names to lists of values to try
    """
    return {
        # Learning rates
        'learning_rate_critic': [0.0001, 0.0005, 0.001, 0.005],
        'learning_rate_actor': [0.0001, 0.0005, 0.001],

        # Discount factor
        'gamma': [0.95, 0.99, 0.995],

        # Batch size
        'batch_size': [16, 32, 64],

        # Buffer size
        'buffer_size': [10000, 50000, 100000],

        # Exploration noise
        'exploration_noise': [0.1, 0.2, 0.3],
    }


def evaluate_config(config: Dict,
                    num_episodes: int = 100,
                    device: str = 'cpu') -> Dict:
    """
    Evaluate a single hyperparameter configuration.

    Args:
        config: Hyperparameter configuration to evaluate
        num_episodes: Number of episodes for evaluation
        device: Device to use

    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    max_steps = 500
    env = ISRGridEnv(
        num_drones=10,
        max_targets=5,
        mission_duration=max_steps,
    )

    # Derive state/action dims
    obs, _ = env.reset()
    state_dim = DMPCAgent.flatten_obs(obs).shape[0]
    action_dim = int(np.prod(env.action_space.shape))

    # Create agent with this configuration
    agent = DMPCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate_critic=float(config['learning_rate_critic']),
        learning_rate_actor=float(config['learning_rate_actor']),
        gamma=float(config['gamma']),
        batch_size=int(config['batch_size']),
        buffer_size=int(config['buffer_size']),
        exploration_noise=float(config['exploration_noise']),
        device=device,
    )

    train_episodes = min(num_episodes, 200)

    results = {
        'episode_rewards': [],
        'coverage_efficiency': [],
        'collision_count': 0,
        'training_time': 0.0,
    }

    start_time = time.time()

    for episode in range(train_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            action = agent.act(obs, training=True)
            action_env = np.clip(
                action.reshape(env.action_space.shape), 0.0, 1.0
            )

            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            agent.remember(obs, action, reward, next_obs, done)

            if agent.ready_to_train():
                agent.train_on_batch()

            episode_reward += reward
            step_count += 1
            obs = next_obs

            results['collision_count'] += info.get('collisions', 0)

        results['episode_rewards'].append(episode_reward)
        results['coverage_efficiency'].append(info.get('coverage', 0))

    results['training_time'] = time.time() - start_time

    # Summary statistics (last 50 episodes)
    tail = min(50, len(results['episode_rewards']))
    return {
        'mean_reward': float(np.mean(results['episode_rewards'][-tail:])),
        'std_reward': float(np.std(results['episode_rewards'][-tail:])),
        'mean_coverage': float(np.mean(results['coverage_efficiency'][-tail:])),
        'std_coverage': float(np.std(results['coverage_efficiency'][-tail:])),
        'collision_rate': results['collision_count'] / max(train_episodes, 1),
        'training_time': results['training_time'],
        'convergence_speed': compute_convergence_speed(
            results['episode_rewards']
        ),
    }


def compute_convergence_speed(rewards: List[float]) -> float:
    """
    Compute convergence speed metric.

    Args:
        rewards: List of episode rewards

    Returns:
        Episode number at which reward reaches 90% of final value
    """
    if len(rewards) < 50:
        return float(len(rewards))

    final_reward = np.mean(rewards[-20:])
    target_reward = 0.9 * final_reward

    window = 20
    for i in range(window, len(rewards)):
        rolling_mean = np.mean(rewards[i - window:i])
        if rolling_mean >= target_reward:
            return float(i)

    return float(len(rewards))


def _make_json_safe(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def grid_search(base_config_path: str,
                num_trials_per_config: int = 100,
                output_dir: str = 'data/hyperparameter_search',
                device: str = 'cpu') -> Tuple[Dict, List[Dict]]:
    """
    Perform random search over hyperparameters.

    Args:
        base_config_path: Path to base configuration
        num_trials_per_config: Episodes per configuration
        output_dir: Output directory
        device: Device to use

    Returns:
        Tuple of (best_config, all_results)
    """
    output_path = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        'hyperparam_search', str(output_path / 'search.log')
    )
    logger.info("Starting hyperparameter search")

    search_space = get_search_space()

    # Random search with a reasonable number of samples
    num_random_samples = 50
    logger.info(f"Search space has {len(search_space)} parameters")
    logger.info(
        f"Performing random search with {num_random_samples} configurations"
    )

    # Random sampling
    all_configs = []
    for _ in range(num_random_samples):
        config = {
            param: np.random.choice(values)
            for param, values in search_space.items()
        }
        all_configs.append(config)

    results: List[Dict] = []
    best_score = -np.inf
    best_config = None

    for idx, config in enumerate(all_configs):
        logger.info(
            f"\nEvaluating configuration {idx + 1}/{len(all_configs)}"
        )
        logger.info(f"Config: {config}")

        try:
            metrics = evaluate_config(
                config,
                num_episodes=num_trials_per_config,
                device=device,
            )

            # Composite score
            time_bonus = min(10.0, 1.0 / (metrics['training_time'] / 60 + 1e-6))
            score = (
                metrics['mean_reward'] * 0.4
                + metrics['mean_coverage'] * 100 * 0.4
                + (1 - metrics['collision_rate']) * 100 * 0.1
                + time_bonus * 0.1
            )

            result = {
                'config': _make_json_safe(config),
                'metrics': _make_json_safe(metrics),
                'score': float(score),
            }
            results.append(result)

            logger.info(
                f"Results: Score={score:.2f}, "
                f"Reward={metrics['mean_reward']:.2f}, "
                f"Coverage={metrics['mean_coverage']:.1%}, "
                f"Collision Rate={metrics['collision_rate']:.2%}"
            )

            if score > best_score:
                best_score = score
                best_config = _make_json_safe(config)
                logger.info(
                    f"*** NEW BEST CONFIG (score={best_score:.2f}) ***"
                )

        except Exception as e:
            logger.error(f"Error evaluating config {idx + 1}: {e}")
            continue

        # Save intermediate results
        if (idx + 1) % 10 == 0:
            intermediate_path = output_path / f'results_iter{idx + 1}.json'
            with open(intermediate_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Save final results
    final_results_path = output_path / 'final_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if best_config is not None:
        best_config_path = output_path / 'best_config.yaml'
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f)

    logger.info(f"\n{'=' * 60}")
    logger.info("HYPERPARAMETER SEARCH COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Best score: {best_score:.2f}")
    logger.info(f"All results saved to: {final_results_path}")

    return best_config, results


if __name__ == '__main__':
    args = parse_args()

    best_config, all_results = grid_search(
        base_config_path=args.config,
        num_trials_per_config=args.num_trials,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    if best_config:
        for key, value in best_config.items():
            print(f"{key}: {value}")
    print("=" * 60)
