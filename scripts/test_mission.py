"""
test_mission.py - Single Mission Evaluation Script

Evaluates trained swarm on a single ISR mission with full logging and metrics.
Compatible with trained checkpoints from train_agent.py
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isr_rl_dmpc.modules import (
    LearningModule, ThreatAssessor, TaskAllocator,
    DMPC, DMPCConfig, AttitudeController, DroneParameters
)
from src.isr_rl_dmpc.gym_env import ISRGridEnv, RewardShaper
from src.isr_rl_dmpc.utils import setup_logger


class MissionEvaluator:
    """Evaluate trained swarm on a mission."""
    
    def __init__(self, num_drones: int = 4, num_targets: int = 5,
                 device: str = 'cpu', checkpoint_dir: str = 'checkpoints'):
        """Initialize evaluator."""
        self.device = device
        self.num_drones = num_drones
        self.num_targets = num_targets
        
        # Setup logging
        self.logger = setup_logger(
            'MissionEvaluator',
            log_file=str(Path('logs') / 'mission_test.log')
        )
        
        # Initialize environment
        self.env = ISRGridEnv(
            num_drones=num_drones,
            max_targets=num_targets,
            mission_duration=1000,
        )
        self.reward_shaper = RewardShaper(
            num_drones=num_drones,
            max_targets=num_targets,
            grid_cells=400,
        )
        
        # Initialize modules
        self._init_modules()
        
        # Load checkpoint if available
        self.checkpoint_dir = Path(checkpoint_dir)
        self._load_checkpoint()
        
        # Metrics
        self.mission_data = {
            'drone_positions': [],
            'target_detections': [],
            'threat_assessments': [],
            'task_assignments': [],
            'rewards': [],
            'control_metrics': []
        }
    
    def _init_modules(self):
        """Initialize modules."""
        self.threat_assessor = ThreatAssessor()
        self.task_allocator = TaskAllocator(num_drones=self.num_drones)
        
        self.agents = [
            LearningModule(state_dim=18, action_dim=4, device=self.device)
            for i in range(self.num_drones)
        ]
        
        self.dmpc_controllers = [
            DMPC(DMPCConfig(device=self.device))
            for i in range(self.num_drones)
        ]
        
        self.flight_controllers = [
            AttitudeController(DroneParameters(mass=1.0, device=self.device))
            for i in range(self.num_drones)
        ]
    
    def _load_checkpoint(self):
        """Load trained checkpoint if available."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_ep*_best.pt'))
        
        if not checkpoints:
            self.logger.warning("No trained checkpoint found, using random policies")
            return
        
        latest_checkpoint = checkpoints[-1]
        self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        for i, agent in enumerate(self.agents):
            if 'agents' in checkpoint and i < len(checkpoint['agents']):
                agent.value_network.load_state_dict(checkpoint['agents'][i])
    
    def run_mission(self, mission_steps: int = 1000, deterministic: bool = True) -> Dict:
        """Run single mission evaluation."""
        self.logger.info(f"Starting mission (deterministic={deterministic})")
        
        obs, info = self.env.reset()
        mission_reward = 0
        detections = 0
        classifications = 0
        
        for step in range(mission_steps):
            # 1. Select action using first agent (centralized policy)
            state_vector = np.asarray(obs['swarm'], dtype=np.float32).ravel()
            if self.agents[0].policy_network is not None:
                state_t = torch.FloatTensor(state_vector).to(self.device)
                self.agents[0].policy_network.eval()
                with torch.no_grad():
                    mean, _ = self.agents[0].policy_network(state_t)
                action = mean.squeeze(0).cpu().numpy()
            else:
                action = np.random.randn(int(np.prod(self.env.action_space.shape)))
            
            # Reshape action to match env action space shape
            expected_size = int(np.prod(self.env.action_space.shape))
            if action.size == expected_size:
                action_env = action.reshape(self.env.action_space.shape)
            else:
                action_env = np.random.uniform(0, 1, self.env.action_space.shape)
            action_env = np.clip(action_env, 0.0, 1.0)
            
            # 2. Execute in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_env)
            done = terminated or truncated
            mission_reward += reward
            
            # 3. Log metrics
            self._log_step_metrics(step, next_obs, reward, info)
            
            obs = next_obs
            
            if done:
                self.logger.info(f"Mission ended at step {step}")
                break
        
        # Compute final metrics
        avg_reward = mission_reward / max(step + 1, 1)
        metrics = {
            'total_reward': float(mission_reward),
            'avg_reward': float(avg_reward),
            'detections': int(detections),
            'classifications': int(classifications),
            'mission_steps': int(step + 1),
            'success': avg_reward > 0.5
        }
        
        return metrics
    
    def _log_step_metrics(self, step: int, obs, reward: float, info: Dict):
        """Log metrics for current step."""
        self.mission_data['drone_positions'].append({
            'step': step,
            'positions': obs['swarm'][:, :3].tolist() if isinstance(obs, dict) and 'swarm' in obs else []
        })
        
        self.mission_data['rewards'].append({
            'step': step,
            'rewards': [float(reward)],
            'avg': float(reward)
        })
    
    def get_mission_summary(self) -> Dict:
        """Get mission summary statistics."""
        if not self.mission_data['rewards']:
            return {}
        
        rewards = np.array([r['avg'] for r in self.mission_data['rewards']])
        
        return {
            'total_reward': float(np.sum(rewards)),
            'avg_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'std_reward': float(np.std(rewards)),
            'total_detections': len(self.mission_data['target_detections']),
            'steps_completed': len(self.mission_data['rewards'])
        }
    
    def save_results(self, output_path: str = 'mission_results.json'):
        """Save mission results."""
        results = {
            'configuration': {
                'num_drones': self.num_drones,
                'num_targets': self.num_targets,
                'device': self.device
            },
            'summary': self.get_mission_summary(),
            'detailed_data': self.mission_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        self.logger.info(f"Results saved to {output_path}")
        
        return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Test trained ISR swarm on mission')
    parser.add_argument('--drones', type=int, default=4,
                       help='Number of drones')
    parser.add_argument('--targets', type=int, default=5,
                       help='Number of targets')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Mission duration in steps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--output', type=str, default='mission_results.json',
                       help='Output results file')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU')
    parser.add_argument('--random', action='store_true',
                       help='Use random policy (no checkpoint loading)')
    parser.add_argument('--num-missions', type=int, default=1,
                       help='Run multiple missions')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create evaluator
    evaluator = MissionEvaluator(
        num_drones=args.drones,
        num_targets=args.targets,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Run missions
    all_results = []
    for mission in range(args.num_missions):
        print(f"\n{'='*50}")
        print(f"Running Mission {mission + 1}/{args.num_missions}")
        print(f"{'='*50}\n")
        
        metrics = evaluator.run_mission(args.steps, deterministic=True)
        summary = evaluator.get_mission_summary()
        
        print(f"Mission Summary:")
        print(f"  Total Reward: {summary['total_reward']:.2f}")
        print(f"  Avg Reward: {summary['avg_reward']:.4f}")
        print(f"  Detections: {summary['total_detections']}")
        print(f"  Steps: {summary['steps_completed']}")
        
        all_results.append(summary)
    
    # Save aggregated results
    aggregated = {
        'num_missions': args.num_missions,
        'missions': all_results,
        'average': {
            'avg_reward': float(np.mean([r['avg_reward'] for r in all_results])),
            'total_detections': int(np.mean([r['total_detections'] for r in all_results]))
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Aggregated Results ({args.num_missions} missions):")
    print(f"  Average Reward: {aggregated['average']['avg_reward']:.4f}")
    print(f"  Avg Detections: {aggregated['average']['total_detections']:.1f}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
