"""
value_network.py — removed.

The CriticNetwork neural-network value function has been removed along
with the RL learning pipeline.  This stub is kept for import
backward-compatibility.
"""

__all__: list = []


def __getattr__(name: str):
    raise ImportError(
        f"'{name}' is not available: neural-network value/critic functions "
        "have been removed.  The system now uses a pure DMPC controller."
    )
