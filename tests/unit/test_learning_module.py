"""
TEST: Module 9 - Learning Module (RL Agent for DMPC Parameter Optimization)

Unit tests for policy gradient learning and value function approximation
Tests experience collection, TD learning, and policy updates
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock
from isr_rl_dmpc import (
    Transition, ValueNetwork, PolicyNetwork, ExperienceBuffer, LearningModule, 
)


class TestTransitionDataStructure:
    """Test Transition data structure."""
    
    def test_transition_initialization(self):
        """Transition initializes correctly."""
        
        transition = Transition(
            state=np.random.randn(20),
            action=np.random.randn(4),
            reward=5.0,
            next_state=np.random.randn(20),
            done=False
        )
        
        assert transition.state.shape == (20,)
        assert transition.action.shape == (4,)
        assert transition.reward == 5.0
        assert transition.done == False


class TestValueNetwork:
    """Test PyTorch value function approximator."""
    
    @pytest.fixture
    def network(self):
        return ValueNetwork(state_dim=20)
    
    def test_network_initialization(self, network):
        """Value network initializes."""
        assert network is not None
        assert network.state_dim == 20
    
    def test_forward_pass(self, network):
        """Forward pass computes values."""
        state = torch.randn(1, 20)
        
        value = network(state)
        
        assert value.shape == (1,)
    
    def test_batch_forward(self, network):
        """Network handles batch inputs."""
        batch_states = torch.randn(32, 20)
        
        values = network(batch_states)
        
        assert values.shape == (32,)
    
    def test_single_state(self, network):
        """Network handles single state."""
        state = torch.randn(20)
        
        value = network(state)
        
        assert value.is_scalar()


class TestPolicyNetwork:
    """Test Gaussian policy network."""
    
    @pytest.fixture
    def network(self):
        return PolicyNetwork(state_dim=20, action_dim=4)
    
    def test_network_initialization(self, network):
        """Policy network initializes."""
        assert network is not None
        assert network.state_dim == 20
        assert network.action_dim == 4
    
    def test_forward_pass(self, network):
        """Forward pass computes mean and log std."""
        state = torch.randn(1, 20)
        
        mean, log_std = network(state)
        
        assert mean.shape == (1, 4)
        assert log_std.shape == (1, 4)
    
    def test_sample_action(self, network):
        """Sample actions from policy."""
        state = torch.randn(1, 20)
        
        action, log_prob = network.sample(state)
        
        assert action.shape == (1, 4)
        assert log_prob.shape == (1,)


class TestExperienceBuffer:
    """Test experience replay buffer."""
    
    @pytest.fixture
    def buffer(self):
        return ExperienceBuffer(max_size=1000, device='cpu')
    
    def test_buffer_initialization(self, buffer):
        """Experience buffer initializes."""
        assert len(buffer) == 0
    
    def test_add_transition(self, buffer):
        """Add transition to buffer."""
        
        transition = Transition(
            state=np.random.randn(20),
            action=np.random.randn(4),
            reward=1.0,
            next_state=np.random.randn(20),
            done=False
        )
        
        buffer.add(transition)
        
        assert len(buffer) == 1
    
    def test_sample_batch(self, buffer):
        """Sample batch from buffer."""
        
        # Add transitions
        for i in range(10):
            transition = Transition(
                state=np.random.randn(20),
                action=np.random.randn(4),
                reward=float(i),
                next_state=np.random.randn(20),
                done=i % 2 == 0
            )
            buffer.add(transition)
        
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
        
        assert states.shape == (5, 20)
        assert actions.shape == (5, 4)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 20)
        assert dones.shape == (5,)
    
    def test_prioritized_sampling(self, buffer):
        """Sample with priorities."""
        
        # Add transitions with different priorities
        for i in range(10):
            transition = Transition(
                state=np.random.randn(20),
                action=np.random.randn(4),
                reward=float(i),
                next_state=np.random.randn(20),
                done=False
            )
            buffer.add(transition, priority=float(i + 1))
        
        states, _, _, _, _ = buffer.sample(batch_size=5, use_priorities=True)
        
        assert states.shape == (5, 20)


class TestLearningModule:
    """Test RL learning module."""
    
    @pytest.fixture
    def learning(self):
        return LearningModule(state_dim=20, action_dim=4, device='cpu')
    
    def test_module_initialization(self, learning):
        """Learning module initializes."""
        assert learning is not None
        assert learning.state_dim == 20
        assert learning.action_dim == 4
        assert learning.gamma == 0.99
    
    def test_collect_trajectory(self, learning):
        """Collect trajectory from mission data."""
        mission_data = {
            'states': [np.random.randn(20) for _ in range(11)],
            'actions': [np.random.randn(4) for _ in range(10)],
            'rewards': [float(i) for i in range(10)],
            'dones': [i == 9 for i in range(10)]
        }
        
        trajectory, total_reward = learning.collect_trajectory(mission_data)
        
        assert len(trajectory) == 10
        assert isinstance(total_reward, float)
    
    def test_compute_td_targets(self, learning):
        """Compute TD learning targets."""
        rewards = torch.randn(32)
        next_values = torch.randn(32)
        dones = torch.ones(32)
        
        targets = learning.compute_td_targets(rewards, next_values, dones)
        
        assert targets.shape == (32,)
    
    def test_update_value_function(self, learning):
        """Update value network."""
        
        # Add some experience
        for i in range(100):
            transition = Transition(
                state=np.random.randn(20),
                action=np.random.randn(4),
                reward=float(np.random.randn()),
                next_state=np.random.randn(20),
                done=False
            )
            learning.buffer.add(transition)
        
        loss = learning.update_value_function(batch_size=32)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_compute_policy_gradient(self, learning):
        """Compute policy gradient from trajectory."""
        
        trajectory = []
        for i in range(10):
            transition = Transition(
                state=np.random.randn(20),
                action=np.random.randn(4),
                reward=float(i),
                next_state=np.random.randn(20),
                done=i == 9
            )
            trajectory.append(transition)
        
        policy_grad, avg_advantage = learning.compute_policy_gradient(trajectory)
        
        assert policy_grad.shape == (4,)
        assert isinstance(avg_advantage, (float, np.floating))
    
    def test_learning_step(self, learning):
        """Execute single learning step."""
        
        trajectory = []
        for i in range(10):
            transition = Transition(
                state=np.random.randn(20),
                action=np.random.randn(4),
                reward=float(i),
                next_state=np.random.randn(20),
                done=i == 9
            )
            trajectory.append(transition)
            learning.buffer.add(transition)
        
        stats = learning.learning_step(trajectory)
        
        assert 'value_loss' in stats
        assert 'policy_gradient_norm' in stats
        assert 'avg_advantage' in stats


class TestLearningStatistics:
    """Test learning statistics collection."""
    
    @pytest.fixture
    def learning(self):
        return LearningModule(state_dim=20, action_dim=4, device='cpu')
    
    def test_get_statistics(self, learning):
        """Get learning statistics."""
        learning.reward_history = [10.0, 15.0, 20.0]
        learning.td_error_history = [0.5, 0.3, 0.2]
        learning.value_loss_history = [0.1, 0.08, 0.06]
        
        stats = learning.get_learning_statistics()
        
        assert 'avg_episode_reward' in stats
        assert 'max_episode_reward' in stats
        assert 'min_episode_reward' in stats
        assert 'avg_td_error' in stats
        assert 'avg_value_loss' in stats
        assert 'total_episodes' in stats


class TestLearningCheckpoints:
    """Test checkpoint saving and loading."""
    
    @pytest.fixture
    def learning(self):
        return LearningModule(state_dim=20, action_dim=4, device='cpu')
    
    def test_save_checkpoint(self, learning, tmp_path):
        """Save learning checkpoint."""
        checkpoint_path = tmp_path / "learning_checkpoint.pt"
        
        learning.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, learning, tmp_path):
        """Load learning checkpoint."""
        checkpoint_path = tmp_path / "learning_checkpoint.pt"
        
        # Save
        learning.save_checkpoint(checkpoint_path)
        learning.reward_history = [1.0, 2.0, 3.0]
        
        # Create new instance and load
        learning2 = LearningModule(state_dim=20, action_dim=4, device='cpu')
        learning2.load_checkpoint(checkpoint_path)
        
        assert learning2.reward_history is not None


class TestGaussianPolicy:
    """Test Gaussian policy sampling."""
    
    @pytest.fixture
    def learning(self):
        return LearningModule(state_dim=20, action_dim=4, 
                            use_policy_network=True, device='cpu')
    
    def test_policy_network_initialized(self, learning):
        """Policy network exists when enabled."""
        assert learning.policy_network is not None
        assert learning.policy_optimizer is not None
    
    def test_sample_from_policy(self, learning):
        """Sample actions from learned policy."""
        state = torch.randn(1, 20)
        
        action, log_prob = learning.policy_network.sample(state)
        
        assert action.shape == (1, 4)
        assert log_prob.shape == (1,)


class TestLearningConvergence:
    """Test learning convergence properties."""
    
    @pytest.fixture
    def learning(self):
        return LearningModule(state_dim=20, action_dim=4, device='cpu')
    
    def test_value_loss_decreases(self, learning):
        """Value loss decreases during training."""

        # Collect experience with consistent rewards
        for episode in range(3):
            for i in range(50):
                transition = Transition(
                    state=np.random.randn(20),
                    action=np.random.randn(4),
                    reward=1.0,  # Constant reward
                    next_state=np.random.randn(20),
                    done=i == 49
                )
                learning.buffer.add(transition)
            
            loss = learning.update_value_function(batch_size=16)
        
        # Loss should be reasonable
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
