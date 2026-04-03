"""
train_agent.py — deprecated.

This script previously trained a neural-network RL agent.  The RL
components have been removed.  To run ISR missions use:

    python scripts/run_mission.py [OPTIONS]

See ``scripts/run_mission.py --help`` for available options.
"""

import sys

print(
    "WARNING: train_agent.py is deprecated.\n"
    "The neural-network RL agent has been removed.\n"
    "Use 'python scripts/run_mission.py' to run DMPC-controlled ISR missions.",
    file=sys.stderr,
)

import subprocess
cmd = [sys.executable, "scripts/run_mission.py"] + sys.argv[1:]
sys.exit(subprocess.call(cmd))
