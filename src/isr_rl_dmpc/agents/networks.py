"""Networks and Replay Buffer for Learning Module"""

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

