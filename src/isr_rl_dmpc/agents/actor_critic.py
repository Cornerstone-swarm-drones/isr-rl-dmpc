"""
Actor-Critic training loop for DMPC parameter optimisation.

Complements the existing ``DMPCAgent`` by providing a standalone
Actor-Critic trainer that re-uses the ``ValueNetwork`` (critic) and
``PolicyNetwork`` (actor) already defined in ``learning_module``.

Features:
    - GAE (Generalised Advantage Estimation) for low-variance advantage
    - Entropy bonus for exploration
    - Gradient clipping for stable training
    - Checkpoint save / load

Mathematical framework:
    Advantage:  Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
                δ_t  = r_t + γ V(s_{t+1}) - V(s_t)
    Actor loss: L_π  = -E[log π(a|s) Â] - c_ent H[π]
    Critic loss: L_V = E[(V(s) - R̂)²]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path

from isr_rl_dmpc.modules.learning_module import (
    ValueNetwork,
    PolicyNetwork,
    Transition,
    ExperienceBuffer,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for Actor-Critic training."""

    gamma: float = 0.99  # Discount factor
    lr_actor: float = 1e-4  # Actor learning rate
    lr_critic: float = 1e-3  # Critic learning rate
    batch_size: int = 32  # Mini-batch size
    max_grad_norm: float = 0.5  # Gradient clipping threshold
    entropy_coeff: float = 0.01  # Entropy bonus weight
    value_loss_coeff: float = 0.5  # Value-loss weight in total loss
    gae_lambda: float = 0.95  # GAE λ parameter


class ActorCriticTrainer:
    """
    Standard Actor-Critic trainer for DMPC parameter optimisation.

    Uses the project's ``PolicyNetwork`` as actor and ``ValueNetwork``
    as critic.  Training follows the advantage actor-critic (A2C)
    algorithm with GAE.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize actor-critic trainer.

        Args:
            state_dim: Dimension of the state vector.
            action_dim: Dimension of the action vector.
            config: Training hyperparameters (uses defaults if ``None``).
            device: ``'cpu'`` or ``'cuda'``.
        """
        self._config = config or TrainingConfig()
        self._device = device
        self._training_steps: int = 0

        # Networks
        self._actor = PolicyNetwork(state_dim, action_dim).to(device)
        self._critic = ValueNetwork(state_dim).to(device)

        # Optimisers
        self._actor_optim = torch.optim.Adam(
            self._actor.parameters(), lr=self._config.lr_actor
        )
        self._critic_optim = torch.optim.Adam(
            self._critic.parameters(), lr=self._config.lr_critic
        )

        logger.info(
            "ActorCriticTrainer initialised (state_dim=%d, action_dim=%d, "
            "device=%s, gamma=%.3f, gae_lambda=%.3f)",
            state_dim,
            action_dim,
            device,
            self._config.gamma,
            self._config.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def training_steps(self) -> int:
        """Total number of ``update`` calls performed."""
        return self._training_steps

    @property
    def actor(self) -> PolicyNetwork:
        """The policy (actor) network."""
        return self._actor

    @property
    def critic(self) -> ValueNetwork:
        """The value (critic) network."""
        return self._critic

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Sample an action from the current policy.

        Args:
            state: State vector of shape ``(state_dim,)``.

        Returns:
            Tuple of (action, info) where *action* is a NumPy array
            of shape ``(action_dim,)`` and *info* contains ``log_prob``
            and ``value``.
        """
        state_t = torch.FloatTensor(state).to(self._device)

        self._actor.eval()
        self._critic.eval()
        with torch.no_grad():
            action_t, log_prob_t = self._actor.sample(state_t)
            value_t = self._critic(state_t)
        self._actor.train()
        self._critic.train()

        action = action_t.squeeze(0).cpu().numpy()
        info = {
            "log_prob": float(log_prob_t.squeeze(0).cpu()),
            "value": float(value_t.cpu()),
        }
        return action, info

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalised Advantage Estimation.

        Args:
            rewards: Tensor of shape ``(T,)`` with per-step rewards.
            values: Tensor of shape ``(T,)`` with value estimates V(s_t).
            dones: Tensor of shape ``(T,)`` with episode-done flags (1.0 = done).
            next_value: Scalar tensor V(s_{T+1}).

        Returns:
            Tuple of (advantages, returns) each of shape ``(T,)``.
        """
        gamma = self._config.gamma
        lam = self._config.gae_lambda
        T = len(rewards)

        advantages = torch.zeros(T, device=self._device)
        gae = torch.tensor(0.0, device=self._device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single actor-critic update.

        Args:
            batch: Dictionary with keys ``states``, ``actions``,
                ``log_probs``, ``returns``, ``advantages``
                (all tensors on the correct device).

        Returns:
            Dictionary of training metrics::

                {
                    "actor_loss": ...,
                    "critic_loss": ...,
                    "entropy": ...,
                    "total_loss": ...,
                }
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Normalise advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Critic loss ---
        values = self._critic(states)
        critic_loss = nn.functional.mse_loss(values, returns)

        # --- Actor loss ---
        mean, log_std = self._actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        actor_loss = -(new_log_probs * advantages.detach()).mean()

        # --- Total loss ---
        total_loss = (
            actor_loss
            + self._config.value_loss_coeff * critic_loss
            - self._config.entropy_coeff * entropy
        )

        # --- Optimise critic first (graph not yet freed) ---
        self._critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self._critic.parameters(), self._config.max_grad_norm
        )
        self._critic_optim.step()

        # --- Optimise actor (recompute to avoid retain_graph) ---
        mean2, log_std2 = self._actor(states)
        std2 = torch.exp(log_std2)
        dist2 = torch.distributions.Normal(mean2, std2)
        new_log_probs2 = dist2.log_prob(actions).sum(dim=-1)
        entropy2 = dist2.entropy().sum(dim=-1).mean()
        actor_loss_bp = -(new_log_probs2 * advantages.detach()).mean() - self._config.entropy_coeff * entropy2

        self._actor_optim.zero_grad()
        actor_loss_bp.backward()
        nn.utils.clip_grad_norm_(
            self._actor.parameters(), self._config.max_grad_norm
        )
        self._actor_optim.step()

        self._training_steps += 1

        metrics = {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy.detach()),
            "total_loss": float(total_loss.detach()),
        }

        if self._training_steps % 100 == 0:
            logger.info(
                "Step %d | actor=%.4f critic=%.4f entropy=%.4f",
                self._training_steps,
                metrics["actor_loss"],
                metrics["critic_loss"],
                metrics["entropy"],
            )

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Save actor, critic, and optimiser states.

        Args:
            path: File path for the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self._actor.state_dict(),
                "critic_state_dict": self._critic.state_dict(),
                "actor_optim_state_dict": self._actor_optim.state_dict(),
                "critic_optim_state_dict": self._critic_optim.state_dict(),
                "training_steps": self._training_steps,
            },
            path,
        )
        logger.info("Checkpoint saved to %s (step %d)", path, self._training_steps)

    def load(self, path) -> None:
        """
        Load actor, critic, and optimiser states.

        Args:
            path: File path of a previously saved checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self._actor.load_state_dict(checkpoint["actor_state_dict"])
        self._critic.load_state_dict(checkpoint["critic_state_dict"])
        self._actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
        self._training_steps = checkpoint.get("training_steps", 0)
        logger.info("Checkpoint loaded from %s (step %d)", path, self._training_steps)
