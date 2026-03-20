#!/usr/bin/env python3
"""
Run grouped “swarm system” evaluation tasks and compute cornerstone scores.

Outputs:
- data/evaluation_runs/<run_id>/cornerstone_score.json
- data/evaluation_runs/<run_id>/<task_id>.json
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    from isr_rl_dmpc.evaluation.swarm_task_suite import main as suite_main

    suite_main()


if __name__ == "__main__":
    main()

