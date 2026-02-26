"""
Standalone replay and trajectory buffer utilities.

Re-exports the base ``ExperienceBuffer`` from ``learning_module`` and
provides two extended buffer types:

- ``PrioritizedReplayBuffer`` — proportional-priority sampling with
  importance-sampling weight correction.
- ``TrajectoryBuffer`` — episode-aware buffer that stores complete
  trajectories and supports trajectory-level sampling.

Classes:
    PrioritizedReplayBuffer: Proportional prioritised experience replay.
    TrajectoryBuffer: Episode-segmented transition storage.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import deque

from isr_rl_dmpc.modules.learning_module import (
    Transition,
    ExperienceBuffer as _BaseExperienceBuffer,
)

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """
    Proportional prioritised experience replay buffer.

    Priority sampling follows Schaul et al. (2016):
        P(i) = p_i^α / Σ_j p_j^α
        w_i  = (N · P(i))^{-β}   (importance-sampling weights)

    ``beta`` is annealed toward 1.0 over training to correct the
    bias introduced by non-uniform sampling.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """
        Initialize prioritised replay buffer.

        Args:
            capacity: Maximum number of stored transitions.
            alpha: Priority exponent (0 = uniform, 1 = full prioritisation).
            beta: Initial importance-sampling exponent.
            beta_increment: Per-sample increment applied to ``beta``
                toward 1.0.
        """
        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = beta_increment

        self._buffer: List[Optional[Transition]] = []
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos: int = 0  # write cursor
        self._size: int = 0

        logger.info(
            "PrioritizedReplayBuffer created (capacity=%d, α=%.2f, β=%.2f)",
            capacity,
            alpha,
            beta,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def add(self, transition: Transition, td_error: Optional[float] = None) -> None:
        """
        Store a transition with optional TD-error priority.

        Args:
            transition: Experience tuple to store.
            td_error: Absolute TD error used as priority.
                Falls back to max existing priority (or 1.0).
        """
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self._alpha
        elif self._size > 0:
            priority = float(self._priorities[: self._size].max())
        else:
            priority = 1.0

        if self._pos < len(self._buffer):
            self._buffer[self._pos] = transition
        else:
            self._buffer.append(transition)

        self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a prioritised mini-batch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of ``(transitions, is_weights, indices)`` where
            ``is_weights`` are importance-sampling corrections and
            ``indices`` can later be passed to ``update_priorities``.
        """
        prios = self._priorities[: self._size]
        probs = prios / prios.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probs)
        transitions = [self._buffer[i] for i in indices]

        # Importance-sampling weights
        self._beta = min(self._beta + self._beta_increment, 1.0)
        min_prob = probs.min()
        max_weight = (self._size * min_prob) ** (-self._beta)
        weights = (self._size * probs[indices]) ** (-self._beta) / max(max_weight, 1e-8)

        return transitions, weights.astype(np.float32), indices

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices returned by ``sample``.
            td_errors: New absolute TD errors for those transitions.
        """
        for idx, td in zip(indices, td_errors):
            self._priorities[idx] = (abs(td) + 1e-6) ** self._alpha


class TrajectoryBuffer:
    """
    Episode-segmented transition buffer.

    Maintains complete trajectories so that algorithms requiring
    sequential data (e.g., GAE, n-step returns) can sample full
    episodes.
    """

    def __init__(self, max_trajectories: int = 100):
        """
        Initialize trajectory buffer.

        Args:
            max_trajectories: Maximum number of stored trajectories.
        """
        self._max_trajectories = max_trajectories
        self._trajectories: deque = deque(maxlen=max_trajectories)
        self._current: Optional[List[Transition]] = None
        self._total_steps: int = 0

        logger.info(
            "TrajectoryBuffer created (max_trajectories=%d)", max_trajectories
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_trajectories(self) -> int:
        """Number of complete trajectories stored."""
        return len(self._trajectories)

    @property
    def total_steps(self) -> int:
        """Total transitions across all stored trajectories."""
        return self._total_steps

    # ------------------------------------------------------------------
    # Trajectory lifecycle
    # ------------------------------------------------------------------

    def start_trajectory(self) -> None:
        """Begin recording a new trajectory.

        If a trajectory is already in progress it is discarded.
        """
        if self._current is not None:
            logger.warning(
                "Discarding incomplete trajectory (%d steps)", len(self._current)
            )
        self._current = []

    def add_step(self, transition: Transition) -> None:
        """
        Append a transition to the current trajectory.

        Args:
            transition: Experience tuple for this time step.

        Raises:
            RuntimeError: If ``start_trajectory`` was not called first.
        """
        if self._current is None:
            raise RuntimeError(
                "No active trajectory. Call start_trajectory() first."
            )
        self._current.append(transition)

    def end_trajectory(self) -> List[Transition]:
        """
        Finalise the current trajectory and store it.

        Returns:
            The completed trajectory as a list of ``Transition`` objects.

        Raises:
            RuntimeError: If no trajectory is in progress.
        """
        if self._current is None:
            raise RuntimeError("No active trajectory to end.")

        trajectory = list(self._current)

        # If buffer is full the oldest trajectory is evicted
        if len(self._trajectories) == self._max_trajectories:
            evicted = self._trajectories[0]
            self._total_steps -= len(evicted)

        self._trajectories.append(trajectory)
        self._total_steps += len(trajectory)
        self._current = None

        logger.debug(
            "Trajectory stored (%d steps, %d total trajectories)",
            len(trajectory),
            len(self._trajectories),
        )
        return trajectory

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_trajectories(self, n: int) -> List[List[Transition]]:
        """
        Sample *n* trajectories uniformly at random.

        Args:
            n: Number of trajectories to sample.

        Returns:
            List of trajectories (each a list of ``Transition``).
        """
        n = min(n, len(self._trajectories))
        indices = np.random.choice(len(self._trajectories), size=n, replace=False)
        return [list(self._trajectories[i]) for i in indices]

    def get_all_transitions(self) -> List[Transition]:
        """
        Flatten all stored trajectories into a single list.

        Returns:
            List of every ``Transition`` across all trajectories.
        """
        transitions: List[Transition] = []
        for traj in self._trajectories:
            transitions.extend(traj)
        return transitions
