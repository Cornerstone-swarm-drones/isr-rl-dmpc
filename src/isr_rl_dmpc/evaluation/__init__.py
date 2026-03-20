"""Evaluation utilities for ISR-RL-DMPC.

This package provides a runnable, application-grouped task suite for
cornerstone-style scoring (coverage, threat, safety, battery, formation).
"""

from .swarm_task_suite import run_swarm_task_suite

__all__ = ["run_swarm_task_suite"]

