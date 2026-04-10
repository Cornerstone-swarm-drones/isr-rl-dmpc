"""
pybullet_sim — PyBullet-based ISR-DMPC swarm simulation package.

Usage
-----
    python pybullet_sim/swarm_pybullet_sim.py [options]
    python pybullet_sim/swarm_pybullet_sim.py --help
"""

from .swarm_pybullet_sim import SwarmPyBulletSim, main

__all__ = ["SwarmPyBulletSim", "main"]
