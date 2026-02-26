"""
Critic Network - General-Purpose Value Function Approximator

Implements a flexible critic architecture supporting both state-value V(s)
and action-value Q(s,a) modes for reinforcement learning in DMPC optimization.

Complements the ValueNetwork in modules/learning_module.py by providing a more
general architecture that can be used as a standalone critic in actor-critic
methods or as a Q-function approximator.

Mathematical Framework:
  State-value mode:  V(s) ≈ E[Σ_k γ^k r_k | s]
  Action-value mode: Q(s,a) ≈ E[Σ_k γ^k r_k | s, a]

Architecture:
  Input (state_dim [+ action_dim]) → [Linear → BatchNorm → ReLU → Dropout] × N → Linear(1)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    """
    General critic network supporting both V(s) and Q(s,a) estimation.

    When action_dim=0, operates as a state-value function V(s).
    When action_dim>0, operates as an action-value function Q(s,a)
    by concatenating state and action inputs.

    Architecture:
      Input → [Linear → BatchNorm → ReLU → Dropout] × len(hidden_dims) → Linear(1)

    Features:
    - Configurable hidden layer sizes via tuple
    - Batch normalization for stable training
    - Dropout for regularization
    - Automatic V(s)/Q(s,a) mode selection based on action_dim
    """

    def __init__(self, state_dim: int, action_dim: int = 0,
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 use_batch_norm: bool = True, dropout_rate: float = 0.1):
        """
        Initialize critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (0 for V(s) mode)
            hidden_dims: Sizes of hidden layers
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
        """
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self._input_dim = state_dim + action_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm

        # Build hidden layers
        layers: List[nn.Module] = []
        prev_dim = self._input_dim
        self._fc_layers = nn.ModuleList()
        self._bn_layers = nn.ModuleList()
        self._dropout_layers = nn.ModuleList()

        for h_dim in hidden_dims:
            self._fc_layers.append(nn.Linear(prev_dim, h_dim))
            self._bn_layers.append(
                nn.BatchNorm1d(h_dim) if use_batch_norm else nn.Identity()
            )
            self._dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

        # Activation
        self.relu = nn.ReLU()

        mode = "Q(s,a)" if action_dim > 0 else "V(s)"
        logger.info(f"CriticNetwork initialized (mode={mode}, state_dim={state_dim}, "
                    f"action_dim={action_dim}, hidden={list(hidden_dims)}, "
                    f"batch_norm={use_batch_norm})")

    def forward(self, state: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
            action: Optional action tensor of shape (batch_size, action_dim) or
                    (action_dim,). Required when action_dim > 0.

        Returns:
            Value tensor of shape (batch_size,) or scalar
        """
        # Ensure 2D input
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Concatenate action if in Q(s,a) mode
        if self.action_dim > 0:
            if action is None:
                raise ValueError(
                    "action must be provided when action_dim > 0 (Q-mode)"
                )
            if action.dim() == 1:
                action = action.unsqueeze(0)
            x = torch.cat([state, action], dim=1)
        else:
            x = state

        # Hidden layers
        for fc, bn, dropout in zip(
            self._fc_layers, self._bn_layers, self._dropout_layers
        ):
            x = fc(x)
            x = bn(x)
            x = self.relu(x)
            x = dropout(x)

        # Output
        value = self.output_layer(x)

        if squeeze_output:
            value = value.squeeze()
        else:
            value = value.squeeze(1)

        return value

    @property
    def input_dim(self) -> int:
        """Total input dimension (state_dim + action_dim)."""
        return self._input_dim

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
