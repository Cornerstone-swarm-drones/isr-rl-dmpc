"""
actor_critic.py — removed.

The Actor-Critic neural-network trainer has been replaced by the pure
DMPC controller.  This stub is kept for backward-compatibility of any
import that previously referenced this module; it raises a clear error.
"""

__all__: list = []


def __getattr__(name: str):
    raise ImportError(
        f"'{name}' is not available: the Actor-Critic neural-network trainer "
        "has been removed.  Use the pure DMPC controller via "
        "'isr_rl_dmpc.modules.dmpc_controller.DMPC' instead."
    )
