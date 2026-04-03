"""
experience_buffer.py — removed.

The RL replay buffer has been removed along with the neural-network
learning pipeline.  This stub is kept for backward-compatibility.
"""

__all__: list = []


def __getattr__(name: str):
    raise ImportError(
        f"'{name}' is not available: the RL experience buffer has been "
        "removed.  The system now uses a pure DMPC controller."
    )
