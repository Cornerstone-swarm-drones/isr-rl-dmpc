#!/usr/bin/env python3
"""
Train MAPPO policy for adaptive DMPC cost-weight tuning.

Usage
-----
    python scripts/train_mappo.py                              # default config
    python scripts/train_mappo.py --config config/mappo_config.yaml
    python scripts/train_mappo.py --timesteps 500000 --num-drones 4
    python scripts/train_mappo.py --eval                      # with eval env

The trained policy is saved to ``models/mappo_dmpc/final`` (SB3 zip format).
TensorBoard logs are written to ``logs/mappo_dmpc``.

References
----------
- docs/GYM_DESIGN.md
- math_docs/09_MAPPO_AGENT.md
- config/mappo_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure the package is importable when running from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isr_rl_dmpc.gym_env import BeliefCoverageEnv, MARLDMPCEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MAPPO policy on MARLDMPCEnv or BeliefCoverageEnv"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "mappo_config.yaml"),
        help="Path to MAPPO config YAML (default: config/mappo_config.yaml)",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--num-drones", type=int, default=None, help="Override num_drones")
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Create a separate evaluation env and run EvalCallback",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Override device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--env-kind",
        default="marl",
        choices=["marl", "belief_coverage"],
        help="Select the training environment. Defaults to the legacy MARL env.",
    )
    parser.add_argument(
        "--scenario-config",
        default=str(ROOT / "config" / "mission_scenarios.yaml"),
        help="Scenario YAML used to seed belief-coverage defaults.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_scenarios(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_belief_env_kwargs(env_cfg: dict, scenario_cfg: dict) -> dict:
    belief_cfg = dict(scenario_cfg.get("belief_coverage", {}))
    scenario_name = str(env_cfg.get("scenario", "area_surveillance"))
    return {
        "scenario": scenario_name,
        "num_drones": int(env_cfg.get("num_drones", scenario_cfg.get("num_drones", 4))),
        "mission_duration": int(
            env_cfg.get("mission_duration", scenario_cfg.get("max_duration", 1000))
        ),
        "horizon": int(env_cfg.get("horizon", 20)),
        "dt": float(env_cfg.get("dt", 0.02)),
        "accel_max": float(env_cfg.get("accel_max", 8.0)),
        "collision_radius": float(env_cfg.get("collision_radius", 3.0)),
        "area_size": tuple(env_cfg.get("area_size", scenario_cfg.get("area_size", [400.0, 400.0]))),
        "communication_range": float(
            env_cfg.get(
                "communication_range",
                scenario_cfg.get("communication_range", 250.0),
            )
        ),
        "fixed_altitude": float(
            env_cfg.get("fixed_altitude", scenario_cfg.get("min_altitude", 30.0))
        ),
        "sensor_range": float(belief_cfg.get("sensor_range", 120.0)),
        "growth_rate": float(belief_cfg.get("growth_rate", 0.03)),
        "global_sync_steps": int(belief_cfg.get("global_sync_steps", 25)),
        "base_station": tuple(belief_cfg.get("base_station", [20.0, 20.0])),
        "suspicious_zones": belief_cfg.get("suspicious_zones", []),
    }


def _make_env(args: argparse.Namespace, env_cfg: dict):
    if args.env_kind == "marl":
        return MARLDMPCEnv(**env_cfg)

    scenario_name = str(env_cfg.get("scenario", "area_surveillance"))
    scenarios = _load_scenarios(args.scenario_config)
    if scenario_name not in scenarios:
        available = ", ".join(sorted(scenarios)) or "<none>"
        raise KeyError(
            f"Scenario '{scenario_name}' not found in {args.scenario_config}. "
            f"Available: {available}"
        )
    belief_env_cfg = _build_belief_env_kwargs(env_cfg, scenarios[scenario_name])
    return BeliefCoverageEnv(**belief_env_cfg)


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    from isr_rl_dmpc.agents.mappo_agent import MAPPOAgent

    env_cfg: dict = cfg.get("environment", {})
    train_cfg: dict = cfg.get("training", {})
    ppo_cfg: dict = cfg.get("ppo", {})

    # CLI overrides
    if args.num_drones is not None:
        env_cfg["num_drones"] = args.num_drones
    if args.timesteps is not None:
        train_cfg["total_timesteps"] = args.timesteps
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    total_timesteps: int = int(train_cfg.get("total_timesteps", 1_000_000))
    log_dir: str = str(train_cfg.get("log_dir", "logs/mappo_dmpc"))
    model_dir: str = str(train_cfg.get("model_dir", "models/mappo_dmpc"))
    device: str = str(train_cfg.get("device", "auto"))
    eval_freq: int = int(train_cfg.get("eval_freq", 10_000))
    checkpoint_freq: int = int(train_cfg.get("checkpoint_freq", 50_000))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ISR-RL-DMPC  — MAPPO Training")
    print("=" * 60)
    print(f"  Config        : {args.config}")
    print(f"  Env kind      : {args.env_kind}")
    print(f"  Num drones    : {env_cfg.get('num_drones', 4)}")
    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Device        : {device}")
    print(f"  Log dir       : {log_dir}")
    print(f"  Model dir     : {model_dir}")
    print("=" * 60)

    # ── Create training environment ──────────────────────────────────────
    train_env = _make_env(args, env_cfg)

    # ── Optional evaluation environment ─────────────────────────────────
    eval_env = _make_env(args, env_cfg) if args.eval else None

    # ── Build agent ──────────────────────────────────────────────────────
    agent = MAPPOAgent(
        env=train_env,
        policy=str(ppo_cfg.get("policy", "MlpPolicy")),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = MAPPOAgent.make_callbacks(
        eval_env=eval_env,
        checkpoint_dir=str(Path(model_dir) / "checkpoints"),
        eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nStarting training for {total_timesteps:,} timesteps …\n")
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────────────────────────
    save_path = Path(model_dir) / "final"
    agent.save(save_path)
    print(f"\n✓ Training complete.  Model saved to: {save_path}.zip")

    train_env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
