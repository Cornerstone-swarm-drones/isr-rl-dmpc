"""
benchmark.py - Performance Benchmarking Suite

Comprehensive benchmarking of all ISR-RL-DMPC modules with timing analysis.
GPU vs CPU performance comparison and memory profiling.
"""

import sys
import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isr_rl_dmpc.modules import (
    LearningBasedDMPC, LearningAgent, ThreatAssessor,
    TaskAllocator, FlightController, DMPCState
)


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""
    
    def __init__(self, num_runs: int = 10, device: str = 'cpu'):
        """Initialize benchmarks."""
        self.num_runs = num_runs
        self.device = device
        self.results = {
            'device': device,
            'num_runs': num_runs,
            'modules': {},
            'system': {}
        }
    
    def benchmark_threat_assessor(self) -> Dict:
        """Benchmark threat assessment."""
        print("  Benchmarking Threat Assessor...", end='', flush=True)
        
        assessor = ThreatAssessor()
        times = []
        
        for _ in range(self.num_runs):
            target_data = {
                'position': np.random.rand(3) * 1000,
                'rf_strength': np.random.uniform(-100, -30),
                'velocity': np.random.rand(3) * 100,
                'classification': np.random.choice(['civilian', 'commercial', 'military'])
            }
            own_pos = np.random.rand(3) * 1000
            
            start = time.perf_counter()
            assessment = assessor.assess_target(target_data, own_pos)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'throughput_per_sec': float(1000 / np.mean(times))
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_task_allocator(self, num_drones: int = 4, num_tasks: int = 10) -> Dict:
        """Benchmark task allocation."""
        print(f"  Benchmarking Task Allocator ({num_drones} drones, {num_tasks} tasks)...", 
              end='', flush=True)
        
        allocator = TaskAllocator(num_drones=num_drones)
        times = []
        
        for _ in range(self.num_runs):
            # Generate random tasks and drones
            tasks = [{'id': i, 'priority': np.random.rand()} for i in range(num_tasks)]
            drones = [{'id': i, 'load': np.random.rand()} for i in range(num_drones)]
            
            start = time.perf_counter()
            assignments = allocator.allocate_tasks(tasks, drones)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times))
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_dmpc(self, horizon: int = 10) -> Dict:
        """Benchmark DMPC optimization."""
        print(f"  Benchmarking DMPC (horizon={horizon})...", end='', flush=True)
        
        dmpc = LearningBasedDMPC(0, device=self.device)
        times = []
        
        for _ in range(self.num_runs):
            # Create random state
            position = np.random.rand(3) * 1000
            velocity = np.random.rand(3) * 50
            attitude = np.random.rand(3)
            
            state = DMPCState(position, velocity, attitude, 100, 0)
            waypoints = np.random.rand(3, 3) * 1000
            
            start = time.perf_counter()
            trajectory = dmpc.plan_trajectory(state, waypoints)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times))
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_learning_agent(self, batch_size: int = 32) -> Dict:
        """Benchmark learning agent."""
        print(f"  Benchmarking Learning Agent (batch_size={batch_size})...", 
              end='', flush=True)
        
        agent = LearningAgent(0, device=self.device)
        times = []
        
        # Pre-populate buffer
        for _ in range(batch_size * 2):
            state = np.random.randn(18)
            action = np.random.randn(5)
            reward = np.random.rand()
            next_state = np.random.randn(18)
            agent.store_experience(state, action, reward, next_state, False)
        
        # Benchmark learning
        for _ in range(self.num_runs):
            start = time.perf_counter()
            metrics = agent.learn_from_batch(batch_size)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times))
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_attitude_controller(self) -> Dict:
        """Benchmark attitude control."""
        print("  Benchmarking Attitude Controller...", end='', flush=True)
        
        controller = FlightController(mass=1.0)
        times = []
        
        for _ in range(self.num_runs):
            accel = np.random.randn(3)
            # Mock state
            class MockState:
                def __init__(self):
                    self.roll = np.random.rand()
                    self.pitch = np.random.rand()
                    self.yaw = np.random.rand()
                    self.roll_rate = np.random.randn() * 0.1
                    self.pitch_rate = np.random.randn() * 0.1
                    self.yaw_rate = np.random.randn() * 0.1
            
            state = MockState()
            
            start = time.perf_counter()
            motors = controller.update(accel, state)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'control_freq_hz': float(1000 / np.mean(times))
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_full_control_loop(self, num_drones: int = 4) -> Dict:
        """Benchmark integrated control loop."""
        print(f"  Benchmarking Full Control Loop ({num_drones} drones)...", 
              end='', flush=True)
        
        # Initialize all modules
        threat_assessors = [ThreatAssessor() for _ in range(num_drones)]
        dmpc_controllers = [LearningBasedDMPC(i, device=self.device) 
                           for i in range(num_drones)]
        flight_controllers = [FlightController(mass=1.0) for _ in range(num_drones)]
        
        times = []
        
        for _ in range(self.num_runs):
            start = time.perf_counter()
            
            # Simulate complete control cycle
            for i in range(num_drones):
                # Threat assessment
                target_data = {
                    'position': np.random.rand(3) * 1000,
                    'rf_strength': np.random.uniform(-100, -30)
                }
                own_pos = np.random.rand(3) * 1000
                assessment = threat_assessors[i].assess_target(target_data, own_pos)
                
                # DMPC planning
                state = DMPCState(own_pos, np.random.rand(3) * 50, 
                                 np.random.rand(3), 100, 0)
                waypoints = np.random.rand(3, 3) * 1000
                trajectory = dmpc_controllers[i].plan_trajectory(state, waypoints)
                
                # Attitude control
                class MockState:
                    roll = pitch = yaw = 0
                    roll_rate = pitch_rate = yaw_rate = 0
                
                motors = flight_controllers[i].update(
                    trajectory.control_sequence[0],
                    MockState()
                )
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'control_freq_hz': float(1000 / np.mean(times) / num_drones)
        }
        
        print(f" {results['mean_time_ms']:.3f} ms")
        return results
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage."""
        print("  Measuring memory usage...", end='', flush=True)
        
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create modules
        dmpc = LearningBasedDMPC(0, device=self.device)
        agent = LearningAgent(0, device=self.device)
        
        if torch.cuda.is_available() and 'cuda' in self.device:
            allocated = torch.cuda.memory_allocated() / 1e6  # MB
            reserved = torch.cuda.memory_reserved() / 1e6    # MB
            results = {
                'gpu_allocated_mb': float(allocated),
                'gpu_reserved_mb': float(reserved)
            }
        else:
            results = {
                'note': 'GPU memory tracking not available on CPU'
            }
        
        print(f" Done")
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite."""
        print(f"\nRunning benchmarks on {self.device} ({self.num_runs} runs each)...\n")
        
        print("Module Benchmarks:")
        self.results['modules']['threat_assessor'] = self.benchmark_threat_assessor()
        self.results['modules']['task_allocator_4_10'] = self.benchmark_task_allocator(4, 10)
        self.results['modules']['dmpc'] = self.benchmark_dmpc(horizon=10)
        self.results['modules']['learning_agent'] = self.benchmark_learning_agent(batch_size=32)
        self.results['modules']['attitude_controller'] = self.benchmark_attitude_controller()
        
        print("\nIntegrated System Benchmarks:")
        self.results['system']['full_control_loop_4drones'] = self.benchmark_full_control_loop(4)
        self.results['system']['memory_usage'] = self.benchmark_memory_usage()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Runs per benchmark: {self.num_runs}\n")
        
        print("Individual Module Performance:")
        print("-"*70)
        for name, metrics in self.results['modules'].items():
            print(f"{name:30s}: {metrics['mean_time_ms']:8.3f} ms "
                  f"(±{metrics['std_time_ms']:.3f} ms)")
        
        print("\nIntegrated System Performance:")
        print("-"*70)
        for name, metrics in self.results['system'].items():
            if 'control_freq' in metrics:
                print(f"{name:30s}: {metrics['mean_time_ms']:8.3f} ms "
                      f"({metrics['control_freq_hz']:.1f} Hz)")
            else:
                print(f"{name:30s}: {metrics}")
        
        # Check real-time compliance
        total_time = self.results['system']['full_control_loop_4drones']['mean_time_ms']
        print("\nReal-Time Compliance:")
        print("-"*70)
        print(f"Total cycle time: {total_time:.2f} ms")
        print(f"Target: <30 ms")
        print(f"Status: {'✓ PASS' if total_time < 30 else '✗ FAIL'}")
        print("="*70 + "\n")
    
    def save_results(self, output_file: str = 'benchmark_results.json'):
        """Save benchmark results."""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main benchmarking entry point."""
    parser = argparse.ArgumentParser(description='Benchmark ISR-RL-DMPC modules')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per benchmark')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU benchmarking')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU benchmarking')
    parser.add_argument('--compare', action='store_true',
                       help='Compare CPU vs GPU')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output results file')
    
    args = parser.parse_args()
    
    devices = []
    
    if args.compare:
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
    elif args.gpu:
        devices = ['cuda'] if torch.cuda.is_available() else ['cpu']
    else:
        devices = ['cpu']
    
    all_results = {}
    
    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmark")
            continue
        
        print(f"\n{'='*70}")
        print(f"BENCHMARKING ON {device.upper()}")
        print(f"{'='*70}")
        
        suite = BenchmarkSuite(num_runs=args.runs, device=device)
        suite.run_all_benchmarks()
        suite.print_summary()
        
        all_results[device] = suite.results
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All results saved to: {output_path}")
    
    # Performance comparison if comparing
    if args.compare and 'cuda' in all_results and 'cpu' in all_results:
        print("\n" + "="*70)
        print("CPU vs GPU SPEEDUP ANALYSIS")
        print("="*70)
        
        cpu_time = all_results['cpu']['system']['full_control_loop_4drones']['mean_time_ms']
        gpu_time = all_results['cuda']['system']['full_control_loop_4drones']['mean_time_ms']
        speedup = cpu_time / gpu_time
        
        print(f"CPU time: {cpu_time:.3f} ms")
        print(f"GPU time: {gpu_time:.3f} ms")
        print(f"Speedup: {speedup:.1f}x")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
