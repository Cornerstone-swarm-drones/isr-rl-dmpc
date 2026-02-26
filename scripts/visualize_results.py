"""
visualize_results.py - Post-Analysis Visualization

Generates plots and visualizations from training and mission results.
Creates publication-quality figures for performance analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


class ResultsVisualizer:
    """Visualize training and mission results."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    def plot_training_curves(self, log_file: str = 'logs/final_results.json',
                           output_name: str = 'training_curves.png'):
        """Plot training reward and loss curves."""
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {log_file}")
            return
        
        rewards = data.get('history', {}).get('rewards', [])
        losses = data.get('history', {}).get('losses', [])
        
        if not rewards:
            print("No training history found")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Reward curve
        episodes = range(len(rewards))
        ax1.plot(episodes, rewards, linewidth=2, color=self.colors[0], label='Episode Reward')
        
        # Moving average
        window = 10
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), ma, linewidth=2, 
                    color=self.colors[1], linestyle='--', label=f'{window}-Episode MA')
        
        ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Training Reward Progression', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Loss curve
        if losses:
            ax2.plot(episodes[:len(losses)], losses, linewidth=2, color=self.colors[2],
                    label='Critic Loss')
            ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax2.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_mission_results(self, mission_file: str = 'mission_results.json',
                            output_name: str = 'mission_analysis.png'):
        """Plot mission evaluation results."""
        try:
            with open(mission_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {mission_file}")
            return
        
        if 'missions' in data:
            # Multiple missions (aggregated)
            missions = data['missions']
            num_missions = len(missions)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Reward distribution
            rewards = [m['avg_reward'] for m in missions]
            axes[0, 0].bar(range(num_missions), rewards, color=self.colors[0], alpha=0.7)
            axes[0, 0].axhline(y=np.mean(rewards), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(rewards):.3f}')
            axes[0, 0].set_xlabel('Mission', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Average Reward', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Mission Rewards', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # 2. Detections
            detections = [m['total_detections'] for m in missions]
            axes[0, 1].bar(range(num_missions), detections, color=self.colors[1], alpha=0.7)
            axes[0, 1].set_xlabel('Mission', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Number of Detections', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Target Detections', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Success rate
            success = [1 if m['success'] else 0 for m in missions]
            success_rate = np.mean(success) * 100
            axes[1, 0].pie([success_rate, 100-success_rate],
                          labels=['Success', 'Failure'],
                          autopct='%1.1f%%',
                          colors=[self.colors[2], self.colors[3]],
                          startangle=90)
            axes[1, 0].set_title(f'Success Rate: {success_rate:.1f}%', 
                                fontsize=12, fontweight='bold')
            
            # 4. Summary statistics
            axes[1, 1].axis('off')
            summary_text = f"""
            MISSION SUMMARY
            ━━━━━━━━━━━━━━━━━━━━━━━━
            
            Missions Completed: {num_missions}
            
            Reward Statistics:
              • Average: {np.mean(rewards):.4f}
              • Max: {np.max(rewards):.4f}
              • Min: {np.min(rewards):.4f}
              • Std: {np.std(rewards):.4f}
            
            Detection Statistics:
              • Total: {sum(detections)}
              • Avg per mission: {np.mean(detections):.1f}
              • Max: {np.max(detections)}
            
            Success Rate: {success_rate:.1f}%
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                           fontfamily='monospace', verticalalignment='center')
            
        else:
            # Single mission
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Get reward history
            detailed = data.get('detailed_data', {})
            rewards = detailed.get('rewards', [])
            
            if rewards:
                reward_values = [r['avg'] for r in rewards]
                axes[0, 0].plot(reward_values, linewidth=2, color=self.colors[0])
                axes[0, 0].fill_between(range(len(reward_values)), reward_values,
                                       alpha=0.3, color=self.colors[0])
                axes[0, 0].set_xlabel('Step', fontsize=11, fontweight='bold')
                axes[0, 0].set_ylabel('Reward', fontsize=11, fontweight='bold')
                axes[0, 0].set_title('Reward Over Time', fontsize=12, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Detections over time
            detections = detailed.get('target_detections', [])
            if detections:
                detection_steps = [d['step'] for d in detections]
                axes[0, 1].scatter(detection_steps, [1]*len(detection_steps), 
                                  s=100, color=self.colors[1], alpha=0.6)
                axes[0, 1].set_xlabel('Step', fontsize=11, fontweight='bold')
                axes[0, 1].set_ylabel('Detection', fontsize=11, fontweight='bold')
                axes[0, 1].set_title(f'Target Detections ({len(detections)} total)',
                                    fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Summary
            summary = data.get('summary', {})
            axes[1, 0].axis('off')
            summary_text = f"""
            MISSION SUMMARY
            ━━━━━━━━━━━━━━━━━━━━━━━━
            
            Total Reward: {summary.get('total_reward', 0):.2f}
            Avg Reward: {summary.get('avg_reward', 0):.4f}
            Max Reward: {summary.get('max_reward', 0):.4f}
            Min Reward: {summary.get('min_reward', 0):.4f}
            
            Detections: {summary.get('total_detections', 0)}
            Steps: {summary.get('steps_completed', 0)}
            """
            axes[1, 0].text(0.1, 0.5, summary_text, fontsize=10,
                           fontfamily='monospace', verticalalignment='center')
            
            # Drone positions heatmap
            positions = detailed.get('drone_positions', [])
            if positions:
                axes[1, 1].axis('off')
                axes[1, 1].text(0.5, 0.5, f'Position Log\n{len(positions)} steps recorded',
                               ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_performance_comparison(self, results_list: List[str],
                                   labels: List[str] = None,
                                   output_name: str = 'comparison.png'):
        """Compare multiple results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        all_rewards = []
        all_labels = []
        all_detections = []
        
        for i, result_file in enumerate(results_list):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                label = labels[i] if labels and i < len(labels) else f"Run {i+1}"
                
                if 'missions' in data:
                    rewards = [m['avg_reward'] for m in data['missions']]
                    detections = [m['total_detections'] for m in data['missions']]
                else:
                    summary = data.get('summary', {})
                    rewards = [summary.get('avg_reward', 0)]
                    detections = [summary.get('total_detections', 0)]
                
                all_rewards.append(rewards)
                all_detections.append(detections)
                all_labels.append(label)
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
        
        if not all_rewards:
            print("No data to compare")
            return
        
        # Reward comparison
        positions = np.arange(len(all_labels))
        means = [np.mean(r) for r in all_rewards]
        stds = [np.std(r) for r in all_rewards]
        
        axes[0].bar(positions, means, yerr=stds, capsize=5, color=self.colors[0], alpha=0.7)
        axes[0].set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        axes[0].set_title('Reward Comparison', fontsize=13, fontweight='bold')
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(all_labels)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Detection comparison
        det_means = [np.mean(d) for d in all_detections]
        det_stds = [np.std(d) for d in all_detections]
        
        axes[1].bar(positions, det_means, yerr=det_stds, capsize=5, 
                   color=self.colors[1], alpha=0.7)
        axes[1].set_ylabel('Detections', fontsize=12, fontweight='bold')
        axes[1].set_title('Detection Comparison', fontsize=13, fontweight='bold')
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(all_labels)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_report(self, training_log: str = None, mission_results: str = None,
                       output_name: str = 'report.txt'):
        """Generate text report."""
        report = []
        report.append("="*60)
        report.append("ISR-RL-DMPC SWARM EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        report.append("")
        
        # Training results
        if training_log and Path(training_log).exists():
            report.append("TRAINING RESULTS")
            report.append("-"*60)
            with open(training_log, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {})
            report.append(f"Total Episodes: {metrics.get('total_episodes', 'N/A')}")
            report.append(f"Best Reward: {metrics.get('best_reward', 'N/A'):.4f}")
            report.append(f"Avg Reward (last 10): {metrics.get('avg_reward', 'N/A'):.4f}")
            report.append(f"Final Loss: {metrics.get('final_loss', 'N/A'):.4f}")
            report.append("")
        
        # Mission results
        if mission_results and Path(mission_results).exists():
            report.append("MISSION EVALUATION")
            report.append("-"*60)
            with open(mission_results, 'r') as f:
                data = json.load(f)
            
            if 'missions' in data:
                report.append(f"Missions: {data['num_missions']}")
                avg_data = data.get('average', {})
                report.append(f"Avg Reward: {avg_data.get('avg_reward', 0):.4f}")
                report.append(f"Avg Detections: {avg_data.get('total_detections', 0):.1f}")
            else:
                summary = data.get('summary', {})
                report.append(f"Total Reward: {summary.get('total_reward', 0):.2f}")
                report.append(f"Avg Reward: {summary.get('avg_reward', 0):.4f}")
                report.append(f"Detections: {summary.get('total_detections', 0)}")
            report.append("")
        
        report.append("="*60)
        report_text = "\n".join(report)
        
        output_path = self.output_dir / output_name
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to: {output_path}")


def main():
    """Main visualization entry point."""
    parser = argparse.ArgumentParser(description='Visualize training and mission results')
    parser.add_argument('--training-log', type=str, default='logs/final_results.json',
                       help='Training results JSON file')
    parser.add_argument('--mission-results', type=str, default='mission_results.json',
                       help='Mission results JSON file')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                       help='Compare multiple result files')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate text report')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.output_dir)
    
    # Plot training curves
    if Path(args.training_log).exists():
        print("Generating training curves...")
        visualizer.plot_training_curves(args.training_log)
    
    # Plot mission results
    if Path(args.mission_results).exists():
        print("Generating mission analysis...")
        visualizer.plot_mission_results(args.mission_results)
    
    # Compare multiple results
    if args.compare:
        print("Generating comparison plots...")
        visualizer.plot_performance_comparison(args.compare)
    
    # Generate report
    if args.report:
        print("Generating report...")
        visualizer.generate_report(args.training_log, args.mission_results)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
