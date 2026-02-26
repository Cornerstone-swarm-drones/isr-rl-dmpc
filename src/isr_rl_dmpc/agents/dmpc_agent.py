"""
DMPCAgent - RL Agent wrapping DMPC controller and Learning Module.

Provides a unified agent interface for training scripts with:
- act(): select actions from current policy
- remember(): store experience transitions
- ready_to_train(): check if enough experience is available
- train_on_batch(): perform one gradient update step
- save()/load(): checkpoint management
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

from isr_rl_dmpc.modules.learning_module import (
    LearningModule, Transition, ExperienceBuffer,
    ValueNetwork, PolicyNetwork,
)


class DMPCAgent:
    """
    Reinforcement learning agent for DMPC parameter optimization.

    Wraps the existing LearningModule (value + policy networks)
    and provides a simple act/remember/train interface used by
    the training and hyperparameter-search scripts.
    """

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 4,
        learning_rate_critic: float = 1e-3,
        learning_rate_actor: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 100000,
        exploration_noise: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize DMPCAgent.

        Args:
            state_dim: Dimension of the (flattened) state vector.
            action_dim: Dimension of the action vector.
            learning_rate_critic: Learning rate for value network.
            learning_rate_actor: Learning rate for policy network.
            gamma: Discount factor.
            batch_size: Mini-batch size for training.
            buffer_size: Maximum replay-buffer size.
            exploration_noise: Std-dev of Gaussian exploration noise.
            device: ``'cpu'`` or ``'cuda'``.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.device = device

        # Core learning module (value + optional policy networks)
        self.learner = LearningModule(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate_value=learning_rate_critic,
            learning_rate_policy=learning_rate_actor,
            gamma=gamma,
            batch_size=batch_size,
            device=device,
            use_policy_network=True,
        )

        # Override buffer size if requested
        self.learner.buffer = ExperienceBuffer(
            max_size=int(buffer_size), device=device
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def flatten_obs(obs) -> np.ndarray:
        """Flatten a dict observation into a 1-D numpy array."""
        if isinstance(obs, dict):
            parts = []
            for key in sorted(obs.keys()):
                arr = np.asarray(obs[key], dtype=np.float32).ravel()
                parts.append(arr)
            return np.concatenate(parts)
        return np.asarray(obs, dtype=np.float32).ravel()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(
        self,
        observation,
        training: bool = True,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action given the current observation.

        Args:
            observation: Environment observation (dict or array).
            training: Whether the agent is being trained (enables noise).
            deterministic: If ``True``, return the policy mean.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        state = self.flatten_obs(observation)
        state_t = torch.FloatTensor(state).to(self.device)

        policy = self.learner.policy_network
        if policy is not None:
            policy.eval()
            with torch.no_grad():
                mean, log_std = policy(state_t)
                mean = mean.squeeze(0)
                log_std = log_std.squeeze(0)

            if deterministic or not training:
                action = mean.cpu().numpy()
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean) * std
                action = (mean + noise).cpu().numpy()
            policy.train()
        else:
            # Fallback: random action with exploration noise
            action = np.random.randn(self.action_dim) * self.exploration_noise

        return action

    def remember(
        self,
        observation,
        action: np.ndarray,
        reward: float,
        next_observation,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        state = self.flatten_obs(observation)
        next_state = self.flatten_obs(next_observation)
        transition = Transition(
            state=state,
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )
        self.learner.buffer.add(transition, priority=abs(reward) + 1e-6)

    def ready_to_train(self) -> bool:
        """Return ``True`` when enough experience is available."""
        return len(self.learner.buffer) >= self.batch_size

    def train_on_batch(self) -> Tuple[float, float]:
        """
        Perform one gradient-update step.

        Returns:
            ``(critic_loss, actor_loss)`` floats.
        """
        critic_loss = self.learner.update_value_function(self.batch_size)

        actor_loss = 0.0
        policy = self.learner.policy_network
        if policy is not None and len(self.learner.buffer) >= self.batch_size:
            actor_loss = self._update_policy()

        return critic_loss, actor_loss

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save(self, filepath) -> None:
        """Save agent checkpoint."""
        filepath = Path(filepath)
        self.learner.save_checkpoint(filepath)

    def load(self, filepath) -> None:
        """Load agent checkpoint."""
        filepath = Path(filepath)
        self.learner.load_checkpoint(filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_policy(self) -> float:
        """Update policy network via advantage-weighted log-prob."""
        states, actions, rewards, next_states, dones = self.learner.buffer.sample(
            self.batch_size
        )

        # Compute advantages
        # Note: dones from ExperienceBuffer is already a continuation mask
        # (1.0 - done_flag), so multiplying zeroes out next_values at terminal states
        with torch.no_grad():
            values = self.learner.value_network(states)
            next_values = self.learner.value_network(next_states)
            targets = rewards + self.learner.gamma * next_values * dones
            advantages = targets - values

        # Policy log-prob
        mean, log_std = self.learner.policy_network(states)
        std = torch.exp(log_std)
        log_prob = (
            -0.5 * ((actions - mean) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1)

        # Policy gradient loss (negative because we maximise)
        policy_loss = -(log_prob * advantages.detach()).mean()

        self.learner.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.learner.policy_network.parameters(), max_norm=1.0
        )
        self.learner.policy_optimizer.step()

        self.learner.policy_loss_history.append(policy_loss.item())
        return policy_loss.item()
