"""
MAPPOAgent — Multi-Agent PPO using Stable-Baselines3.

Wraps the SB3 ``PPO`` algorithm to provide a MAPPO-style
Centralised Training / Decentralised Execution (CTDE) interface for
the :class:`~isr_rl_dmpc.gym_env.marl_env.MARLDMPCEnv` environment.

The environment already flattens per-drone observations and actions into
single vectors, so a standard SB3 ``PPO`` instance with a default
``MlpPolicy`` is sufficient.  The ``MAPPOAgent`` class adds a thin
convenience wrapper with MARL-specific documentation, helper factories,
and default hyper-parameters sourced from ``config/mappo_config.yaml``.

Usage
-----
>>> from isr_rl_dmpc.gym_env.marl_env import MARLDMPCEnv
>>> from isr_rl_dmpc.agents.mappo_agent import MAPPOAgent
>>> env = MARLDMPCEnv(num_drones=4)
>>> agent = MAPPOAgent(env=env)
>>> agent.learn(total_timesteps=500_000)
>>> agent.save("models/mappo_dmpc_v1")
>>> loaded = MAPPOAgent.load("models/mappo_dmpc_v1", env=env)
>>> obs, _ = env.reset()
>>> actions, _ = loaded.predict(obs, deterministic=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gymnasium as gym


class MAPPOAgent:
    """
    MAPPO agent built on top of Stable-Baselines3 PPO.

    Parameters
    ----------
    env : gym.Env
        A :class:`~isr_rl_dmpc.gym_env.marl_env.MARLDMPCEnv` instance (or
        any Gymnasium-compatible environment).
    policy : str
        SB3 policy identifier.  Defaults to ``'MlpPolicy'``.
    learning_rate : float
        Adam learning rate.
    n_steps : int
        Steps collected per rollout before a PPO update.
    batch_size : int
        Mini-batch size for PPO gradient updates.
    n_epochs : int
        Number of epochs per PPO update.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_range : float
        PPO clipping parameter ε.
    ent_coef : float
        Entropy regularisation coefficient.
    vf_coef : float
        Value-function loss coefficient.
    max_grad_norm : float
        Global gradient norm clip.
    tensorboard_log : str or None
        Directory for TensorBoard logs.
    device : str
        PyTorch device (``'auto'``, ``'cpu'``, ``'cuda'``).
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    **kwargs
        Additional keyword arguments forwarded to ``stable_baselines3.PPO``.
    """

    # Default hyper-parameters aligned with mappo_config.yaml
    _DEFAULTS: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    def __init__(
        self,
        env: gym.Env,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = "logs/mappo_dmpc",
        device: str = "auto",
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        self._env = env
        self._ppo = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=verbose,
            **kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    # SB3-compatible delegation
    # ──────────────────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int = 1_000_000,
        callback=None,
        log_interval: int = 10,
        tb_log_name: str = "MAPPOAgent",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "MAPPOAgent":
        """
        Train the MAPPO policy.

        Args:
            total_timesteps:      Total number of environment steps.
            callback:             SB3 callback or list of callbacks.
            log_interval:         Log every N updates.
            tb_log_name:          TensorBoard run name.
            reset_num_timesteps:  Whether to reset the step counter.
            progress_bar:         Show training progress bar.

        Returns:
            self (for chaining).
        """
        self._ppo.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return self

    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """
        Compute action from observation using the trained policy.

        Args:
            observation:   Flat observation array.
            state:         Hidden state (for recurrent policies; ignored here).
            episode_start: Episode-start mask (ignored).
            deterministic: Use the deterministic policy mode.

        Returns:
            ``(actions, state)`` tuple as returned by SB3.
        """
        return self._ppo.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the policy to *path* (SB3 zip format)."""
        self._ppo.save(str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> "MAPPOAgent":
        """
        Load a previously saved policy.

        Args:
            path: Path to the saved model (SB3 zip file, without extension).
            env:  Environment to attach to the loaded model.
            **kwargs: Forwarded to ``PPO.load``.

        Returns:
            ``MAPPOAgent`` wrapping the loaded PPO instance.
        """
        ppo = PPO.load(str(path), env=env, **kwargs)
        agent = cls.__new__(cls)
        agent._env = env
        agent._ppo = ppo
        return agent

    # ──────────────────────────────────────────────────────────────────
    # Convenience training helpers
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def make_callbacks(
        cls,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: str = "models/checkpoints",
        eval_freq: int = 10_000,
        checkpoint_freq: int = 50_000,
    ):
        """
        Build standard SB3 callbacks for evaluation and checkpointing.

        Args:
            eval_env:        Environment for periodic evaluation (optional).
            checkpoint_dir:  Directory to save model checkpoints.
            eval_freq:       Evaluate every *n* timesteps.
            checkpoint_freq: Save a checkpoint every *n* timesteps.

        Returns:
            List of SB3 callbacks.
        """
        callbacks = []

        if eval_env is not None:
            eval_cb = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=0,
            )
            callbacks.append(eval_cb)

        ckpt_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="mappo_dmpc",
            verbose=0,
        )
        callbacks.append(ckpt_cb)

        return callbacks

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def policy(self):
        """The underlying SB3 policy network."""
        return self._ppo.policy

    @property
    def num_timesteps(self) -> int:
        """Total number of environment steps taken so far."""
        return self._ppo.num_timesteps

    def __repr__(self) -> str:
        return (
            f"MAPPOAgent("
            f"lr={self._ppo.learning_rate}, "
            f"n_steps={self._ppo.n_steps}, "
            f"timesteps={self.num_timesteps})"
        )
