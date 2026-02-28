#!/usr/bin/env python3
"""Launch the ISR-RL-DMPC Dashboard.

Starts the FastAPI backend (and optionally the Vite dev server for
frontend hot-reload during development).

Usage:
    # Production (frontend pre-built into dashboard/frontend/dist):
    python scripts/launch_dashboard.py

    # Development (auto-opens Vite dev server + FastAPI):
    python scripts/launch_dashboard.py --dev

Open your browser at http://localhost:5173 (dev) or http://localhost:8000 (prod).
"""

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "dashboard" / "frontend"


def main():
    parser = argparse.ArgumentParser(description="Launch ISR-RL-DMPC Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument(
        "--dev", action="store_true",
        help="Run in development mode with Vite hot-reload",
    )
    args = parser.parse_args()

    # Ensure PYTHONPATH includes src/
    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    vite_proc = None
    if args.dev:
        print(f"[Dashboard] Starting Vite dev server at {FRONTEND_DIR} ...")
        vite_proc = subprocess.Popen(
            ["npx", "vite", "--host"],
            cwd=str(FRONTEND_DIR),
            env=env,
        )

    print(f"[Dashboard] Starting FastAPI backend on http://{args.host}:{args.port} ...")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "dashboard.backend.app:app",
                "--host", args.host,
                "--port", str(args.port),
                "--reload" if args.dev else "--workers=1",
            ],
            cwd=str(ROOT),
            env=env,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if vite_proc is not None:
            vite_proc.terminate()
            vite_proc.wait()


if __name__ == "__main__":
    main()
