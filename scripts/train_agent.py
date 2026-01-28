"""
train_agent.py - Main Training Script for ISR-RL-DMPC Swarm

Complete training pipeline for multi-agent RL with mission simulation.
Integrates all 9 modules with PyTorch optimization and checkpointing.
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isr_rl_dmpc.modules import (
    LearningBasedDMPC, LearningAgent, MultiAgentLearner,
    ThreatAssessor, TaskAllocator, TaskType
)
from src.isr_rl_dmpc.gym_env import ISRGridEnvironment, RewardShaper
from src.isr_rl_dmpc.utils import setup_logging, log_metrics


class TrainingConfig:
    """Training configuration."""
    
    def __init__(self, config_path: str = None):
        """Load configuration from file or defaults."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Environment
        self.num_drones = 4
        self.num_targets = 5
        self.mission_duration = 1000  # steps
        
        # Training
        self.num_episodes = 100
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.epsilon_decay = 0.995
        
        # Checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_freq = 10  # episodes
        self.save_best = True
        
        # Logging
        self.log_dir = Path('logs')
        self.log_freq = 1
        self.verbose = True
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, path: str):
        """Save configuration to JSON."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not isinstance(v, (Path, type))}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class SwarmTrainer:
    """Main trainer for ISR swarm."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        self.device = config.device
        
        # Create directories
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        self.config.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            name='SwarmTrainer',
            log_file=self.config.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # Initialize environment
        self.logger.info("Initializing environment...")
        self.env = ISRGridEnvironment(
            num_drones=config.num_drones,
            num_targets=config.num_targets
        )
        
        # Initialize modules
        self.logger.info("Initializing modules...")
        self._init_modules()
        
        # Initialize learner
        self.logger.info("Initializing multi-agent learner...")
        self.swarm_learner = MultiAgentLearner(
            num_agents=config.num_drones,
            device=self.device
        )
        
        for i, agent in enumerate(self.agents):
            self.swarm_learner.register_agent(i, agent)
        
        # Metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.best_reward = float('-inf')
        
        self.logger.info(f"Training configured on {self.device}")
    
    def _init_modules(self):
        """Initialize all control modules."""
        self.threat_assessor = ThreatAssessor()
        self.task_allocator = TaskAllocator(num_drones=self.config.num_drones)
        self.reward_shaper = RewardShaper()
        
        # Create DMPC controllers
        self.dmpc_controllers = [
            LearningBasedDMPC(i, device=self.device)
            for i in range(self.config.num_drones)
        ]
        
        # Create learning agents
        self.agents = [
            LearningAgent(i, device=self.device)
            for i in range(self.config.num_drones)
        ]
    
    def train_episode(self, episode: int) -> Dict:
        """Train one episode."""
        # Reset environment
        states = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(self.config.mission_duration):
            # Get state vectors
            state_vectors = [s.to_vector() if hasattr(s, 'to_vector') else s 
                            for s in states]
            
            # Select actions
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(state_vectors[i], deterministic=False)
                actions.append(action)
            
            # Execute actions
            next_states, rewards, dones, infos = self.env.step(actions)
            episode_reward += np.mean(rewards)
            
            # Store experiences
            next_state_vectors = [s.to_vector() if hasattr(s, 'to_vector') else s 
                                 for s in next_states]
            
            for i, agent in enumerate(self.agents):
                agent.store_experience(
                    state_vectors[i], actions[i], rewards[i],
                    next_state_vectors[i], dones[i], infos[i]
                )
            
            # Train periodically
            if step % 10 == 0 and step > 0:
                results = self.swarm_learner.train_all_agents(
                    batch_size=self.config.batch_size
                )
                
                # Aggregate losses
                for r in results.values():
                    if 'critic_loss' in r:
                        episode_loss += r['critic_loss']
                        loss_count += 1
            
            # Synchronize policies
            if step % 100 == 0 and step > 0:
                self.swarm_learner.synchronize_policies()
            
            states = next_states
            
            if all(dones):
                break
        
        # Normalize metrics
        avg_reward = episode_reward / max(1, self.config.mission_duration)
        avg_loss = episode_loss / max(1, loss_count)
        
        # Reset episodes for agents
        for agent in self.agents:
            agent.reset_episode()
        
        return {
            'episode': episode,
            'reward': avg_reward,
            'loss': avg_loss,
            'steps': step
        }
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Episodes: {self.config.num_episodes}")
        self.logger.info(f"Device: {self.device}")
        
        try:
            for episode in range(self.config.num_episodes):
                metrics = self.train_episode(episode)
                
                self.episode_rewards.append(metrics['reward'])
                self.episode_losses.append(metrics['loss'])
                
                # Logging
                if (episode + 1) % self.config.log_freq == 0:
                    self.logger.info(
                        f"Episode {episode + 1}/{self.config.num_episodes} | "
                        f"Reward: {metrics['reward']:.4f} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Steps: {metrics['steps']}"
                    )
                
                # Checkpointing
                if (episode + 1) % self.config.checkpoint_freq == 0:
                    self._save_checkpoint(episode + 1)
                    
                    if self.config.save_best and metrics['reward'] > self.best_reward:
                        self.best_reward = metrics['reward']
                        self._save_checkpoint(episode + 1, best=True)
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
        finally:
            self._save_final_results()
    
    def _save_checkpoint(self, episode: int, best: bool = False):
        """Save training checkpoint."""
        suffix = '_best' if best else ''
        checkpoint_path = self.config.checkpoint_dir / f'checkpoint_ep{episode}{suffix}.pt'
        
        checkpoint = {
            'episode': episode,
            'agents': [a.actor.state_dict() for a in self.agents],
            'critics': [a.critic.state_dict() for a in self.agents],
            'reward_history': self.episode_rewards,
            'loss_history': self.episode_losses,
            'best_reward': self.best_reward
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self):
        """Save final training results."""
        results = {
            'config': {
                'num_episodes': self.config.num_episodes,
                'num_drones': self.config.num_drones,
                'device': self.device
            },
            'metrics': {
                'total_episodes': len(self.episode_rewards),
                'best_reward': float(self.best_reward),
                'avg_reward': float(np.mean(self.episode_rewards[-10:])),
                'final_loss': float(self.episode_losses[-1]) if self.episode_losses else 0
            },
            'history': {
                'rewards': [float(r) for r in self.episode_rewards],
                'losses': [float(l) for l in self.episode_losses]
            }
        }
        
        results_path = self.config.log_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training complete! Results saved to {results_path}")
        self.logger.info(f"Best reward: {self.best_reward:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ISR-RL-DMPC swarm')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration JSON file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--drones', type=int, default=4,
                       help='Number of drones')
    parser.add_argument('--targets', type=int, default=5,
                       help='Number of targets')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU even if GPU available')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Logging directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(args.config)
    config.num_episodes = args.episodes
    config.num_drones = args.drones
    config.num_targets = args.targets
    config.log_dir = Path(args.log_dir)
    config.checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.cpu:
        config.device = 'cpu'
    
    # Create and run trainer
    trainer = SwarmTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
