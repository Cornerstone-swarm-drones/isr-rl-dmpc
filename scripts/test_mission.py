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
    LearningAgent, ThreatAssessor, TaskAllocator,
    LearningBasedDMPC, FlightController, DMPCState
)
from src.isr_rl_dmpc.gym_env import ISRGridEnvironment, RewardShaper
from src.isr_rl_dmpc.utils import setup_logging


class MissionEvaluator:
    """Evaluate trained swarm on a mission."""
    
    def __init__(self, num_drones: int = 4, num_targets: int = 5,
                 device: str = 'cpu', checkpoint_dir: str = 'checkpoints'):
        """Initialize evaluator."""
        self.device = device
        self.num_drones = num_drones
        self.num_targets = num_targets
        
        # Setup logging
        self.logger = setup_logging(
            name='MissionEvaluator',
            log_file=Path('logs') / 'mission_test.log'
        )
        
        # Initialize environment
        self.env = ISRGridEnvironment(num_drones, num_targets)
        self.reward_shaper = RewardShaper()
        
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
            LearningAgent(i, device=self.device)
            for i in range(self.num_drones)
        ]
        
        self.dmpc_controllers = [
            LearningBasedDMPC(i, device=self.device)
            for i in range(self.num_drones)
        ]
        
        self.flight_controllers = [
            FlightController(mass=1.0)
            for i in range(self.num_drones)
        ]
    
    def _load_checkpoint(self):
        """Load trained checkpoint if available."""
        best_checkpoint = self.checkpoint_dir / 'checkpoint_ep*_best.pt'
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_ep*_best.pt'))
        
        if not checkpoints:
            self.logger.warning("No trained checkpoint found, using random policies")
            return
        
        latest_checkpoint = checkpoints[-1]
        self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        for i, agent in enumerate(self.agents):
            if i < len(checkpoint['agents']):
                agent.actor.load_state_dict(checkpoint['agents'][i])
                agent.critic.load_state_dict(checkpoint['critics'][i])
    
    def run_mission(self, mission_steps: int = 1000, deterministic: bool = True) -> Dict:
        """Run single mission evaluation."""
        self.logger.info(f"Starting mission (deterministic={deterministic})")
        
        states = self.env.reset()
        mission_reward = 0
        detections = 0
        classifications = 0
        
        for step in range(mission_steps):
            state_vectors = [s.to_vector() if hasattr(s, 'to_vector') else s 
                           for s in states]
            
            # 1. Select actions (deterministic for evaluation)
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(state_vectors[i], deterministic=True)
                actions.append(action)
            
            # 2. Execute in environment
            next_states, rewards, dones, infos = self.env.step(actions)
            mission_reward += np.mean(rewards)
            
            # 3. Log metrics
            self._log_step_metrics(step, next_states, rewards, infos)
            
            states = next_states
            
            # Update counters
            for info in infos:
                if info.get('target_detected'):
                    detections += 1
                if info.get('target_classified'):
                    classifications += 1
            
            if all(dones):
                self.logger.info(f"Mission ended at step {step}")
                break
        
        # Compute final metrics
        avg_reward = mission_reward / mission_steps
        metrics = {
            'total_reward': float(mission_reward),
            'avg_reward': float(avg_reward),
            'detections': int(detections),
            'classifications': int(classifications),
            'mission_steps': int(step),
            'success': avg_reward > 0.5
        }
        
        return metrics
    
    def _log_step_metrics(self, step: int, states, rewards: np.ndarray, infos: List[Dict]):
        """Log metrics for current step."""
        self.mission_data['drone_positions'].append({
            'step': step,
            'positions': [s.position.tolist() if hasattr(s, 'position') else [0, 0, 0]
                         for s in states]
        })
        
        self.mission_data['rewards'].append({
            'step': step,
            'rewards': [float(r) for r in rewards],
            'avg': float(np.mean(rewards))
        })
        
        # Log detections
        for info in infos:
            if info.get('target_detected'):
                self.mission_data['target_detections'].append({
                    'step': step,
                    'target_id': info.get('target_id'),
                    'drone_id': info.get('drone_id')
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
