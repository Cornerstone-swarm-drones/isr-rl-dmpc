"""
Integration tests for ISR-RL-DMPC gym environment.

Tests module interactions, state consistency, reward computation,
and end-to-end mission workflows.

Test suites:
1. Mission workflow tests (5 tests)
2. Learning integration tests (3 tests)  
3. Reward computation tests (6 tests)
4. State consistency tests (3 tests)
5. Performance tests (2 tests)

Author: Autonomous Systems Research Group
Date: January 2026
"""

import pytest
import numpy as np
from typing import Dict, Tuple

# Import environment and dependencies
try:
    from isr_rl_dmpc.gym_env import ISRGridEnv, VectorEnv, make_env
    from isr_rl_dmpc.gym_env.simulator import EnvironmentSimulator, TargetType
    from isr_rl_dmpc.gym_env.reward_shaper import RewardShaper, RewardWeights
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestMissionWorkflows:
    """Test complete mission workflows."""

    def test_basic_mission_execution(self):
        """Test basic mission runs without errors."""
        env = ISRGridEnv(num_drones=5, max_targets=2, mission_duration=100)
        obs, info = env.reset()
        
        assert obs is not None
        assert 'swarm' in obs
        assert obs['swarm'].shape == (5, 18)
        
        # Run mission
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            assert isinstance(reward, (float, np.floating))
            assert isinstance(done, (bool, np.bool_))
            assert obs['swarm'].shape == (5, 18)
        
        env.close()

    def test_coverage_tracking(self):
        """Test coverage map updates correctly."""
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=200)
        obs, _ = env.reset()
        
        initial_coverage = np.mean(env.coverage_map)
        
        # Move drones around
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
        
        # Coverage should increase (or stay same)
        final_coverage = np.mean(env.coverage_map)
        assert final_coverage >= initial_coverage
        assert 0.0 <= final_coverage <= 1.0

    def test_threat_detection(self):
        """Test threat detection mechanism."""
        env = ISRGridEnv(num_drones=5, max_targets=3, mission_duration=200)
        obs, _ = env.reset()
        
        # Initialize targets
        if hasattr(env, 'simulator') and env.simulator is not None:
            env.simulator.add_target(
                np.array([100.0, 100.0, 50.0]),
                TargetType.HOSTILE
            )
            env.simulator.add_target(
                np.array([200.0, 0.0, 50.0]),
                TargetType.FRIENDLY
            )
        
        # Run mission
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, _, _, info = env.step(action)
        
        assert obs['targets'].shape == (3, 12)

    def test_task_allocation(self):
        """Test task allocation to drones."""
        env = ISRGridEnv(num_drones=6, max_targets=2, mission_duration=100)
        obs, _ = env.reset()
        
        # Check adjacency matrix (formation topology)
        adjacency = obs['adjacency']
        assert adjacency.shape == (6, 6)
        assert np.all(adjacency >= 0)
        
        # Adjacency should be symmetric
        assert np.allclose(adjacency, adjacency.T)

    def test_formation_maintenance(self):
        """Test formation control maintains coherence."""
        env = ISRGridEnv(num_drones=4, max_targets=1, mission_duration=100)
        obs, _ = env.reset()
        
        # Get initial formation (distance between drones)
        drone_positions = obs['swarm'][:, :3]
        initial_distances = []
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                dist = np.linalg.norm(drone_positions[i] - drone_positions[j])
                initial_distances.append(dist)
        
        initial_avg_dist = np.mean(initial_distances) if initial_distances else 0
        
        # Run mission with coordinated control
        for _ in range(50):
            action = np.ones((4, 4)) * 0.5  # Hover command
            obs, _, _, _, _ = env.step(action)
        
        # Check if formation is maintained (distances similar)
        drone_positions = obs['swarm'][:, :3]
        final_distances = []
        for i in range(len(drone_positions)):
            for j in range(i + 1, len(drone_positions)):
                dist = np.linalg.norm(drone_positions[i] - drone_positions[j])
                final_distances.append(dist)
        
        final_avg_dist = np.mean(final_distances) if final_distances else 0
        
        # Should not drift too far (allowing reasonable movement)
        assert final_avg_dist < 5000.0  # Large buffer for movement


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestLearningIntegration:
    """Test learning system integration."""

    def test_value_network_training(self):
        """Test value network can be trained with environment."""
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=50)
        obs, _ = env.reset()
        
        # Dummy value network (mock)
        class DummyValueNet:
            def __call__(self, obs):
                return np.random.randn()
            
            def update(self, td_error):
                pass
        
        value_net = DummyValueNet()
        
        # Collect experience
        rewards = []
        for _ in range(40):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            
            # Compute TD error (mock)
            v_pred = value_net(obs)
            v_target = reward + 0.99 * value_net(obs) if not done else reward
            td_error = v_target - v_pred
            value_net.update(td_error)
            
            if done:
                break
        
        assert len(rewards) > 0
        assert all(isinstance(r, (float, np.floating)) for r in rewards)

    def test_dmpc_parameter_update(self):
        """Test DMPC parameter updates from learning."""
        env = ISRGridEnv(num_drones=4, max_targets=1, mission_duration=100)
        obs, _ = env.reset()
        
        # Simulate DMPC with learnable parameters
        dmpc_weights = np.array([0.3, 0.3, 0.2, 0.2])
        learning_rate = 0.01
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            
            # Update DMPC weights based on reward
            if reward > 0:
                dmpc_weights *= (1.0 + learning_rate)
            else:
                dmpc_weights *= (1.0 - learning_rate)
            
            # Normalize
            dmpc_weights = dmpc_weights / np.sum(dmpc_weights)
            
            if done:
                break
        
        # Weights should remain valid
        assert np.all(dmpc_weights >= 0)
        assert np.isclose(np.sum(dmpc_weights), 1.0)

    def test_stochastic_robustness(self):
        """Test learning robustness to stochasticity."""
        rewards_run1 = []
        rewards_run2 = []
        
        # Run 1
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=50, )
        obs, _ = env.reset(seed=42)
        for _ in range(40):
            action = np.ones((3, 4)) * 0.5
            _, reward, done, _, _ = env.step(action)
            rewards_run1.append(reward)
            if done:
                break
        
        # Run 2 (same seed)
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=50)
        obs, _ = env.reset(seed=42)
        for _ in range(40):
            action = np.ones((3, 4)) * 0.5
            _, reward, done, _, _ = env.step(action)
            rewards_run2.append(reward)
            if done:
                break
        
        # Rewards should be identical with same seed
        assert np.allclose(rewards_run1, rewards_run2)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestRewardComputation:
    """Test reward function components."""

    def test_coverage_reward(self):
        """Test coverage reward increases with coverage."""
        reward_shaper = RewardShaper(
            num_drones=5, max_targets=2, grid_cells=400
        )
        
        # No coverage
        coverage_map = np.zeros(400)
        drone_states = np.zeros((5, 18))
        target_states = np.zeros((2, 12))
        motor_commands = np.ones((5, 4)) * 0.5
        
        r1 = reward_shaper.compute_reward(
            drone_states, target_states, np.zeros(404),
            motor_commands, coverage_map, step=0
        )
        
        # Some coverage
        coverage_map = np.zeros(400)
        coverage_map[:50] = 1.0  # 50/400 = 12.5% coverage
        
        r2 = reward_shaper.compute_reward(
            drone_states, target_states, np.zeros(404),
            motor_commands, coverage_map, step=1
        )
        
        # More coverage should give more reward
        assert r2 > r1

    def test_energy_penalty(self):
        """Test energy penalty for consumption."""
        reward_shaper = RewardShaper(
            num_drones=3, max_targets=1, grid_cells=100
        )
        
        # Low battery usage
        drone_states_low = np.ones((3, 18))
        drone_states_low[:, 13] = 1.0  # Full battery (column 13)
        
        coverage_map = np.zeros(100)
        motor_commands = np.zeros((3, 4))  # No thrust = no power
        
        r_low = reward_shaper.compute_reward(
            drone_states_low, np.zeros((1, 12)), np.zeros(104),
            motor_commands, coverage_map, step=0
        )
        
        # High battery usage
        drone_states_high = np.ones((3, 18))
        drone_states_high[:, 13] = 0.5  # Half battery
        
        r_high = reward_shaper.compute_reward(
            drone_states_high, np.zeros((1, 12)), np.zeros(104),
            motor_commands, coverage_map, step=1
        )
        
        # High energy usage should penalize
        assert r_high < r_low

    def test_safety_reward(self):
        """Test safety reward for collision avoidance."""
        reward_shaper = RewardShaper(
            num_drones=4, max_targets=1, grid_cells=100
        )
        
        # Safe configuration (drones far apart)
        drone_states_safe = np.zeros((4, 18))
        for i in range(4):
            drone_states_safe[i, :3] = np.array([i*100, i*100, 50])  # Well separated
            drone_states_safe[i, 15] = 1.0  # All active
        
        r_safe = reward_shaper.compute_reward(
            drone_states_safe, np.zeros((1, 12)), np.zeros(104),
            np.zeros((4, 4)), np.zeros(100), step=0
        )
        
        # Collision configuration
        drone_states_collision = np.zeros((4, 18))
        drone_states_collision[:, :3] = np.array([0, 0, 50])  # All at same position
        drone_states_collision[0, 15] = 0.0  # One inactive (collision)
        
        r_collision = reward_shaper.compute_reward(
            drone_states_collision, np.zeros((1, 12)), np.zeros(104),
            np.zeros((4, 4)), np.zeros(100), step=1
        )
        
        # Collision should severely penalize
        assert r_collision < r_safe
        assert r_collision < -5.0

    def test_threat_reward(self):
        """Test threat engagement reward."""
        reward_shaper = RewardShaper(
            num_drones=3, max_targets=2, grid_cells=100
        )
        
        # No targets detected
        target_states_none = np.zeros((2, 12))
        target_states_none[:, 10] = 0.0  # Not detected
        
        r_none = reward_shaper.compute_reward(
            np.zeros((3, 18)), target_states_none, np.zeros(104),
            np.zeros((3, 4)), np.zeros(100), step=0
        )
        
        # Hostile target detected
        target_states_hostile = np.zeros((2, 12))
        target_states_hostile[0, 8] = 2.0  # Type = Hostile
        target_states_hostile[0, 10] = 1.0  # Detected
        
        r_hostile = reward_shaper.compute_reward(
            np.zeros((3, 18)), target_states_hostile, np.zeros(104),
            np.zeros((3, 4)), np.zeros(100), step=1
        )
        
        # Detecting hostile should reward
        assert r_hostile > r_none

    def test_learning_reward(self):
        """Test learning signal from TD error."""
        reward_shaper = RewardShaper(
            num_drones=2, max_targets=1, grid_cells=50
        )
        
        # Small TD error
        r_small = reward_shaper.compute_reward(
            np.zeros((2, 18)), np.zeros((1, 12)), np.zeros(54),
            np.zeros((2, 4)), np.zeros(50), step=0, td_error=0.01
        )
        
        # Large TD error
        r_large = reward_shaper.compute_reward(
            np.zeros((2, 18)), np.zeros((1, 12)), np.zeros(54),
            np.zeros((2, 4)), np.zeros(50), step=1, td_error=0.5
        )
        
        # Larger TD error should penalize more
        assert r_large < r_small

    def test_component_stats(self):
        """Test component statistics tracking."""
        reward_shaper = RewardShaper(
            num_drones=2, max_targets=1, grid_cells=50,
            enable_component_stats=True
        )
        
        for i in range(10):
            reward_shaper.compute_reward(
                np.zeros((2, 18)), np.zeros((1, 12)), np.zeros(54),
                np.zeros((2, 4)), np.zeros(50), step=i
            )
        
        stats = reward_shaper.get_component_stats()
        
        assert 'r_cov' in stats
        assert 'r_eng' in stats
        assert 'r_safe' in stats
        assert 'r_threat' in stats
        assert 'r_learn' in stats
        assert 'r_total' in stats
        
        for key, stat in stats.items():
            assert stat['count'] == 10
            assert 'mean' in stat
            assert 'std' in stat
            assert 'min' in stat
            assert 'max' in stat


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestStateConsistency:
    """Test state consistency and observation validity."""

    def test_observation_completeness(self):
        """Test all observation components are present and valid."""
        env = ISRGridEnv(num_drones=4, max_targets=2, mission_duration=50)
        obs, _ = env.reset()
        
        # Check all keys present
        assert 'swarm' in obs
        assert 'targets' in obs
        assert 'environment' in obs
        assert 'adjacency' in obs
        
        # Check shapes
        assert obs['swarm'].shape == (4, 18)
        assert obs['targets'].shape == (2, 12)
        assert obs['environment'].shape == (400 + 4,)  # 400 grid cells + 4
        assert obs['adjacency'].shape == (4, 4)
        
        # Check dtypes
        assert obs['swarm'].dtype == np.float32
        assert obs['targets'].dtype == np.float32
        assert obs['environment'].dtype == np.float32
        assert obs['adjacency'].dtype == np.float32
        
        # Check values are finite
        assert np.all(np.isfinite(obs['swarm']))
        assert np.all(np.isfinite(obs['targets']))
        assert np.all(np.isfinite(obs['environment']))
        assert np.all(np.isfinite(obs['adjacency']))

    def test_normalization_correctness(self):
        """Test observation normalization is correct."""
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=50)
        obs, _ = env.reset()
        
        # Drone states should have reasonable ranges
        # Positions: typically within geofence
        positions = obs['swarm'][:, :3]
        assert np.all(np.abs(positions) < 3000)  # Within geofence
        
        # Battery: 0-1 scale
        battery = obs['swarm'][:, 13]
        assert np.all((battery >= 0) & (battery <= 1))
        
        # Active flag: 0 or 1
        active = obs['swarm'][:, 15]
        assert np.all((active == 0) | (active == 1))
        
        # Coverage: 0-1 scale
        coverage = obs['environment'][:400]
        assert np.all((coverage >= 0) & (coverage <= 1))

    def test_action_clipping(self):
        """Test motor commands are properly clipped."""
        env = ISRGridEnv(num_drones=2, max_targets=0, mission_duration=20)
        obs, _ = env.reset()
        
        # Test clipping
        action_invalid = np.array([[-0.5, 1.5, 0.5, 0.5],
                                    [0.5, 0.5, -1.0, 2.0]])
        
        # Step with invalid action (should be clipped)
        obs, reward, done, _, _ = env.step(action_invalid)
        
        assert np.all(np.isfinite(obs['swarm']))
        assert np.all(np.isfinite(reward))


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestPerformance:
    """Test performance metrics and benchmarks."""

    def test_episode_termination(self):
        """Test episode terminates correctly."""
        env = ISRGridEnv(num_drones=3, max_targets=1, mission_duration=50)
        obs, _ = env.reset()
        
        done = False
        steps = 0
        for i in range(60):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            steps += 1
            
            if done:
                break
        
        # Should terminate at or before mission_duration
        assert done
        assert steps <= 50

    def test_reward_statistics(self):
        """Test reward statistics are reasonable."""
        env = ISRGridEnv(num_drones=4, max_targets=2, mission_duration=100)
        obs, _ = env.reset()
        
        rewards = []
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        rewards = np.array(rewards)
        
        # Rewards should be finite
        assert np.all(np.isfinite(rewards))
        
        # Average reward should be reasonable
        mean_reward = np.mean(rewards)
        assert -10000 < mean_reward < 10000  # Broad range


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ISR modules not installed")
class TestVectorizedEnvironment:
    """Test vectorized environment for parallel training."""

    def test_vectorized_env_reset(self):
        """Test vectorized environment reset."""
        envs = [ISRGridEnv(num_drones=3, max_targets=1) for _ in range(4)]
        vec_env = VectorEnv(envs)
        
        obs = vec_env.reset()
        
        # Check batch shapes
        assert obs['swarm'].shape == (4, 3, 18)
        assert obs['targets'].shape == (4, 1, 12)

    def test_vectorized_env_step(self):
        """Test vectorized environment step."""
        envs = [ISRGridEnv(num_drones=2, max_targets=1) for _ in range(3)]
        vec_env = VectorEnv(envs)
        
        obs = vec_env.reset()
        
        # Step all envs together
        actions = np.ones((3, 2, 4)) * 0.5
        obs, rewards, dones, infos = vec_env.step(actions)
        
        # Check output shapes
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3
        assert obs['swarm'].shape == (3, 2, 18)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
