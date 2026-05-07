#!/usr/bin/env python3
"""Generate canonical Phase 1-3 presentation artifacts.

The script is intentionally a thin wrapper around the same public demo and
validation scripts teammates can run directly. It does not train policies or
modify simulator state; it only writes deterministic PNG/CSV/JSON outputs under
``visualizations/presentation``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "visualizations" / "presentation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-ready Phase 1-3 outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Generate only PNG demo figures; skip CSV/JSON validation sweeps.",
    )
    return parser.parse_args()


def _run(name: str, command: list[str]) -> dict[str, object]:
    print("=" * 72)
    print(name)
    print(" ".join(command))
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=_env(), text=True)
    runtime = time.perf_counter() - start
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    return {
        "name": name,
        "command": command,
        "runtime_seconds": runtime,
    }


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    return env


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    runs: list[dict[str, object]] = []

    phase1_png = output_dir / "phase1_patrol_risk.png"
    runs.append(
        _run(
            "Phase 1 patrol-risk demo",
            [
                python,
                "scripts/visualize_belief_coverage.py",
                "--steps",
                "120",
                "--seed",
                "7",
                "--num-drones",
                "6",
                "--disable-persistent-threats",
                "--validation-dynamics",
                "fast_planar",
                "--output",
                str(phase1_png),
            ],
        )
    )

    phase2_png = output_dir / "phase2_moving_threat_ekf.png"
    runs.append(
        _run(
            "Phase 2 moving-threat EKF/interceptor demo",
            [
                python,
                "scripts/visualize_belief_coverage.py",
                "--steps",
                "220",
                "--seed",
                "7",
                "--num-drones",
                "6",
                "--threat-speed-case",
                "fast",
                "--interceptor-guidance-mode",
                "ekf",
                "--response-policy",
                "phase2",
                "--threat-belief-mode",
                "limited_strict",
                "--validation-dynamics",
                "fast_planar",
                "--max-threat-cycles",
                "1",
                "--output",
                str(phase2_png),
            ],
        )
    )

    phase3_png = output_dir / "phase3_sequential_threats.png"
    runs.append(
        _run(
            "Phase 3 bounded sequential-threat demo",
            [
                python,
                "scripts/visualize_belief_coverage.py",
                "--steps",
                "260",
                "--seed",
                "11",
                "--num-drones",
                "6",
                "--threat-speed-case",
                "fast",
                "--interceptor-guidance-mode",
                "ekf",
                "--response-policy",
                "phase2",
                "--threat-belief-mode",
                "limited_strict",
                "--validation-dynamics",
                "fast_planar",
                "--max-threat-cycles",
                "2",
                "--enable-sequential-pending-threats",
                "--pending-threat-delay-steps",
                "12",
                "--output",
                str(phase3_png),
            ],
        )
    )

    if not args.skip_validation:
        phase2_metrics = output_dir / "phase2_policy_metrics"
        runs.append(
            _run(
                "Phase 2 improved-vs-phase2 policy metrics",
                [
                    python,
                    "scripts/experiment_ekf_response_policy.py",
                    "--steps",
                    "140",
                    "--num-drones-cases",
                    "4,5,6",
                    "--speed-cases",
                    "medium,fast",
                    "--seeds",
                    "7,11",
                    "--policies",
                    "improved,phase2",
                    "--skip-diagnostic-plots",
                    "--validation-dynamics",
                    "fast_planar",
                    "--threat-belief-mode",
                    "limited_strict",
                    "--output-dir",
                    str(phase2_metrics),
                ],
            )
        )

        phase3_metrics = output_dir / "phase3_sequential_metrics"
        runs.append(
            _run(
                "Phase 3 sequential-threat metrics",
                [
                    python,
                    "scripts/experiment_sequential_threats.py",
                    "--steps",
                    "260",
                    "--num-drones-cases",
                    "4,5,6",
                    "--speed-cases",
                    "medium,fast",
                    "--seeds",
                    "7,11",
                    "--max-threat-cycles",
                    "2",
                    "--pending-threat-delay-steps",
                    "12",
                    "--validation-dynamics",
                    "fast_planar",
                    "--threat-belief-mode",
                    "limited_strict",
                    "--output-dir",
                    str(phase3_metrics),
                ],
            )
        )

    manifest = {
        "generated_at_unix": time.time(),
        "output_dir": str(output_dir),
        "runs": runs,
        "primary_outputs": {
            "phase1_patrol_risk_png": str(phase1_png),
            "phase2_moving_threat_ekf_png": str(phase2_png),
            "phase3_sequential_threats_png": str(phase3_png),
            "phase2_metrics_dir": str(output_dir / "phase2_policy_metrics"),
            "phase3_metrics_dir": str(output_dir / "phase3_sequential_metrics"),
        },
    }
    manifest_path = output_dir / "presentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Presentation artifacts written under: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
