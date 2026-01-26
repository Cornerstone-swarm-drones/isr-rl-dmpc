"""
Module 9: Learning Module - RL Agent for DMPC Parameter Optimization (PyTorch)

Implements policy gradient learning for DMPC cost function parameters.
Uses TD(λ) value function learning + policy gradient for parameter adaptation.
All neural networks implemented using PyTorch for GPU acceleration and flexibility.

Mathematical Framework:
  State: Augmented system state [drone, targets, coverage, battery, ...]
  Action: DMPC parameters θ = [Q, R, P, slack]
  Reward: Multi-objective (coverage, energy, threat neutralization, safety)
  
  Value function: V(s) ≈ E[Σ_k γ^k r_k | s]
  Policy gradient: ∇J(θ) = E[∇_θ log π(a|s) * Q(s,a)]
  
  Learning updates:
    TD error: δ_k = r_k + γ V(s_{k+1}) - V(s_k)
    Value update: V(s) ← V(s) + α δ_k
    Policy update: θ ← θ + β ∇_θ log π(a|s) * δ_k
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from collections import deque
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single transition tuple (s, a, r, s', done)."""
    
    state: np.ndarray  # Observation vector
    action: np.ndarray  # DMPC parameters θ
    reward: float  # Scalar reward
    next_state: np.ndarray  # Next observation
    done: bool  # Episode termination flag
    info: Dict = field(default_factory=dict)  # Additional info


class ValueNetwork(nn.Module):
    """
    PyTorch-based neural network value function approximator V(s) ≈ E[cumulative_reward | s]
    
    Architecture:
      Input (state_dim) → Linear(256) → ReLU → BatchNorm → 
      Linear(128) → ReLU → BatchNorm → Linear(1)
    
    Features:
    - Batch normalization for stable training
    - Dropout for regularization
    - LayerNorm option for alternative normalization
    - GPU acceleration via PyTorch
    """
    
    def __init__(self, state_dim: int, hidden_dim_1: int = 256, hidden_dim_2: int = 128,
                 use_batch_norm: bool = True, dropout_rate: float = 0.1):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim_1: Size of first hidden layer
            hidden_dim_2: Size of second hidden layer
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
        """
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1) if use_batch_norm else None
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2) if use_batch_norm else None
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim_2, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
        logger.info(f"ValueNetwork initialized (state_dim={state_dim}, "
                   f"hidden=[{hidden_dim_1}, {hidden_dim_2}], "
                   f"batch_norm={use_batch_norm})")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: (batch_size, state_dim) tensor or (state_dim,) tensor
            
        Returns:
            Value tensor (batch_size,) or scalar
        """
        # Ensure 2D input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Layer 1
        x = self.fc1(state)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Output
        value = self.fc3(x)
        
        if squeeze_output:
            value = value.squeeze()
        else:
            value = value.squeeze(1)
        
        return value


class PolicyNetwork(nn.Module):
    """
    PyTorch policy network for generating DMPC parameters.
    
    Outputs mean μ and log standard deviation σ for Gaussian policy.
    Action = μ + σ * ε, where ε ~ N(0, I)
    
    Architecture:
      Input (state_dim) → Linear(256) → ReLU → BatchNorm → 
      Linear(128) → ReLU → BatchNorm → [μ branch, σ branch]
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                hidden_dim_1: int = 256, hidden_dim_2: int = 128,
                use_batch_norm: bool = True, dropout_rate: float = 0.1):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (DMPC params)
            hidden_dim_1: First hidden layer size
            hidden_dim_2: Second hidden layer size
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout probability
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_batch_norm = use_batch_norm
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1) if use_batch_norm else None
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2) if use_batch_norm else None
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Mean branch (μ)
        self.mu_head = nn.Linear(hidden_dim_2, action_dim)
        
        # Log std branch (log σ)
        self.log_std_head = nn.Linear(hidden_dim_2, action_dim)
        
        self.relu = nn.ReLU()
        
        logger.info(f"PolicyNetwork initialized (state_dim={state_dim}, "
                   f"action_dim={action_dim})")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute policy mean and std.
        
        Args:
            state: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Tuple of (mean, log_std) tensors
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared layers
        x = self.fc1(state)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Compute mean and log std
        mean = self.mu_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy: a = μ + σ * ε
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_probability)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from Gaussian
        normal = torch.randn_like(mean)
        action = mean + std * normal
        
        # Compute log probability
        log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob


class ExperienceBuffer:
    """
    Replay buffer for storing trajectory data.
    
    Used to aggregate experience across multiple episodes before learning update.
    Implements prioritized experience sampling.
    GPU-optimized with PyTorch tensors.
    """
    
    def __init__(self, max_size: int = 10000, device: str = 'cpu'):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum buffer size
            device: 'cpu' or 'cuda' for PyTorch tensors
        """
        self.max_size = max_size
        self.device = device
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def add(self, transition: Transition, priority: float = 1.0) -> None:
        """Add transition to buffer with priority."""
        self.buffer.append(transition)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, 
              use_priorities: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample batch of transitions and convert to PyTorch tensors.
        
        Args:
            batch_size: Number of samples
            use_priorities: Use priority-weighted sampling
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
        """
        if use_priorities:
            priorities = np.array(self.priorities)
            priorities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=priorities)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
        
        batch = [self.buffer[i] for i in indices]
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([1.0 - t.done for t in batch])).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def get_trajectory(self) -> List[Transition]:
        """Get all transitions."""
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.priorities.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class LearningModule:
    """
    Main RL learning module with PyTorch networks for DMPC parameter optimization.
    
    Workflow:
    1. Collect trajectory: execute mission, store (s, a, r, s', done)
    2. Compute TD targets: r + γ V(s')
    3. Update value function: ∇ V(s) ← ∇(V(s) - target)²
    4. Compute policy gradient: ∇θ J = E[∇ log π(θ|s) * TD_error]
    5. Update DMPC parameters: θ ← θ + α ∇θ J
    6. Log performance metrics
    
    Features:
    - GPU acceleration via PyTorch
    - Batch normalization for stable training
    - Optional policy network for Gaussian policy
    - Checkpoint saving/loading with PyTorch
    """
    
    def __init__(self, state_dim: int, 
                action_dim: int = 4,
                learning_rate_value: float = 1e-3,
                learning_rate_policy: float = 1e-4,
                gamma: float = 0.99,
                batch_size: int = 32,
                device: str = 'cpu',
                use_policy_network: bool = False):
        """
        Initialize learning module with PyTorch.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action (DMPC params)
            learning_rate_value: Learning rate for value network
            learning_rate_policy: Learning rate for policy network
            gamma: Discount factor
            batch_size: Batch size for updates
            device: 'cpu' or 'cuda'
            use_policy_network: Use separate policy network (vs direct parameter update)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.use_policy_network = use_policy_network
        
        logger.info(f"LearningModule initializing on device: {device}")
        
        # Value network
        self.value_network = ValueNetwork(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), 
                                          lr=learning_rate_value)
        self.value_loss_fn = nn.MSELoss()
        
        # Optional policy network
        if use_policy_network:
            self.policy_network = PolicyNetwork(state_dim, action_dim).to(device)
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(),
                                              lr=learning_rate_policy)
            logger.info("Policy network enabled for Gaussian policy")
        else:
            self.policy_network = None
            self.policy_optimizer = None
        
        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=10000, device=device)
        
        # Statistics
        self.theta_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.td_error_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.policy_loss_history: List[float] = []
        
        # Hyperparameters
        self.policy_lr = learning_rate_policy
        self.entropy_coeff = 0.01
        
        self.logger = logging.getLogger("LearningModule")
        self.logger.info(f"LearningModule initialized (state_dim={state_dim}, "
                        f"action_dim={action_dim}, device={device})")
    
    def collect_trajectory(self, 
                          mission_data: Dict) -> Tuple[List[Transition], float]:
        """
        Convert mission data to trajectory transitions.
        
        Args:
            mission_data: Dictionary with mission results
                - 'states': List of state vectors
                - 'actions': List of action vectors
                - 'rewards': List of scalar rewards
                - 'dones': List of done flags
                
        Returns:
            Tuple of (trajectory, total_reward)
        """
        trajectory = []
        total_reward = 0.0
        
        states = mission_data.get('states', [])
        actions = mission_data.get('actions', [])
        rewards = mission_data.get('rewards', [])
        dones = mission_data.get('dones', [])
        
        for i in range(len(actions)):
            transition = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=states[i+1],
                done=dones[i]
            )
            trajectory.append(transition)
            self.buffer.add(transition, priority=abs(rewards[i]))
            total_reward += rewards[i]
        
        self.reward_history.append(total_reward)
        return trajectory, total_reward
    
    def compute_td_targets(self, rewards: torch.Tensor, next_values: torch.Tensor, 
                          dones: torch.Tensor) -> torch.Tensor:
        """
        Compute TD learning targets: r_k + γ V(s_{k+1})
        
        Args:
            rewards: (batch_size,) reward tensor
            next_values: (batch_size,) next state values
            dones: (batch_size,) continuation flags (1 if not done, 0 if done)
            
        Returns:
            (batch_size,) TD target tensor
        """
        targets = rewards + self.gamma * next_values * dones
        return targets
    
    def update_value_function(self, batch_size: Optional[int] = None) -> float:
        """
        Update value function using collected experience.
        
        Args:
            batch_size: Batch size (uses self.batch_size if None)
            
        Returns:
            Average loss over batch
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, _, rewards, next_states, dones = self.buffer.sample(batch_size, 
                                                                     use_priorities=False)
        
        # Forward pass: compute values
        with torch.no_grad():
            next_values = self.value_network(next_states)
        
        # Compute TD targets
        targets = self.compute_td_targets(rewards, next_values, dones)
        
        # Forward pass for current values
        current_values = self.value_network(states)
        
        # Compute loss
        loss = self.value_loss_fn(current_values, targets)
        
        # Backward pass
        self.value_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        
        self.value_optimizer.step()
        
        self.value_loss_history.append(loss.item())
        
        return loss.item()
    
    def compute_policy_gradient(self, trajectory: List[Transition]) -> Tuple[np.ndarray, float]:
        """
        Compute policy gradient for DMPC parameters.
        
        Uses advantage estimation: A(s,a) = r + γ V(s') - V(s)
        
        Args:
            trajectory: List of transitions
            
        Returns:
            Tuple of (parameter_gradient, average_advantage)
        """
        self.value_network.eval()
        
        advantages = []
        policy_grad = np.zeros(self.action_dim)
        
        with torch.no_grad():
            for transition in trajectory:
                # Convert to torch tensors
                state_tensor = torch.FloatTensor(transition.state).to(self.device)
                next_state_tensor = torch.FloatTensor(transition.next_state).to(self.device)
                
                # Compute advantage
                state_value = self.value_network(state_tensor).item()
                
                if transition.done:
                    next_value = 0.0
                else:
                    next_value = self.value_network(next_state_tensor).item()
                
                advantage = transition.reward + self.gamma * next_value - state_value
                advantages.append(advantage)
                
                # Policy gradient: ∇ log π(a|s) = (a - μ(s)) / σ²
                action_diff = transition.action
                policy_grad += action_diff * advantage
        
        self.value_network.train()
        
        if len(advantages) > 0:
            policy_grad /= len(advantages)
            avg_advantage = np.mean(advantages)
        else:
            avg_advantage = 0.0
        
        return policy_grad, avg_advantage
    
    def update_parameters(self, policy_gradient: np.ndarray, 
                         current_theta: np.ndarray) -> np.ndarray:
        """
        Update DMPC parameters using policy gradient.
        
        θ ← θ + α ∇θ J
        
        Args:
            policy_gradient: Computed policy gradient
            current_theta: Current parameter vector
            
        Returns:
            Updated parameter vector
        """
        updated_theta = current_theta + self.policy_lr * policy_gradient
        
        # Clip to valid ranges
        updated_theta = np.clip(updated_theta, 0.1, 100.0)
        
        self.theta_history.append(updated_theta)
        return updated_theta
    
    def learning_step(self, trajectory: List[Transition]) -> Dict:
        """
        Single learning step: value update + policy gradient.
        
        Args:
            trajectory: Collected trajectory from mission
            
        Returns:
            Dictionary with learning statistics
        """
        # Update value function
        value_loss = self.update_value_function()
        
        # Compute policy gradient
        policy_grad, avg_advantage = self.compute_policy_gradient(trajectory)
        
        # Compute TD errors for statistics
        self.value_network.eval()
        with torch.no_grad():
            for transition in trajectory:
                state_tensor = torch.FloatTensor(transition.state).to(self.device)
                next_state_tensor = torch.FloatTensor(transition.next_state).to(self.device)
                
                state_value = self.value_network(state_tensor).item()
                if transition.done:
                    next_value = 0.0
                else:
                    next_value = self.value_network(next_state_tensor).item()
                
                td_error = transition.reward + self.gamma * next_value - state_value
                self.td_error_history.append(abs(td_error))
        self.value_network.train()
        
        return {
            'value_loss': value_loss,
            'policy_gradient_norm': np.linalg.norm(policy_grad),
            'avg_advantage': avg_advantage,
            'buffer_size': len(self.buffer)
        }
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics for monitoring."""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-10:]
        recent_td_errors = self.td_error_history[-100:] if self.td_error_history else []
        recent_losses = self.value_loss_history[-10:] if self.value_loss_history else []
        
        return {
            'avg_episode_reward': float(np.mean(recent_rewards)),
            'max_episode_reward': float(np.max(recent_rewards)),
            'min_episode_reward': float(np.min(recent_rewards)),
            'avg_td_error': float(np.mean(recent_td_errors)) if recent_td_errors else 0.0,
            'avg_value_loss': float(np.mean(recent_losses)) if recent_losses else 0.0,
            'total_episodes': len(self.reward_history),
            'buffer_size': len(self.buffer),
            'device': self.device
        }
    
    def save_checkpoint(self, filepath: Path) -> None:
        """Save learning checkpoint with PyTorch models."""
        checkpoint = {
            'value_network_state': self.value_network.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'reward_history': self.reward_history,
            'td_error_history': self.td_error_history,
            'value_loss_history': self.value_loss_history,
            'theta_history': self.theta_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'device': self.device
            }
        }
        
        if self.policy_network is not None:
            checkpoint['policy_network_state'] = self.policy_network.state_dict()
            checkpoint['policy_optimizer_state'] = self.policy_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load learning checkpoint with PyTorch models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.value_network.load_state_dict(checkpoint['value_network_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        
        if self.policy_network is not None and 'policy_network_state' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_network_state'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        
        self.reward_history = checkpoint.get('reward_history', [])
        self.td_error_history = checkpoint.get('td_error_history', [])
        self.value_loss_history = checkpoint.get('value_loss_history', [])
        self.theta_history = checkpoint.get('theta_history', [])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def to_device(self, device: str) -> None:
        """Move networks to device."""
        self.device = device
        self.value_network.to(device)
        if self.policy_network is not None:
            self.policy_network.to(device)
        self.logger.info(f"Moved networks to {device}")
