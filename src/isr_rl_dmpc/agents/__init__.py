"""
ISR-RL-DMPC Agents module.

Exports the pure DMPC agent (DMPCAgent) and the MARL MAPPO agent
(MAPPOAgent) that wraps Stable-Baselines3 PPO for adaptive DMPC
cost-weight tuning.
"""

from .dmpc_agent import DMPCAgent

try:  # SB3 may not be installed in minimal setups
    from .mappo_agent import MAPPOAgent
    __all__ = ["DMPCAgent", "MAPPOAgent"]
except ImportError:  # pragma: no cover
    __all__ = ["DMPCAgent"]
