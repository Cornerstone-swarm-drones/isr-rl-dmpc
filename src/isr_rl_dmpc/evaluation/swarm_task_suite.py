from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from isr_rl_dmpc.gym_env import ISRGridEnv
from isr_rl_dmpc.gym_env.simulator import TargetType
from isr_rl_dmpc.modules.task_allocator import (
    DroneCapability,
    ISRTask,
    TaskAllocator,
    TaskType,
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _pairwise_min_distance(positions: np.ndarray) -> float:
    # positions: (N,3)
    n = positions.shape[0]
    if n <= 1:
        return float("inf")
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    # Exclude diagonal (avoid inf-math warnings by masking directly)
    diag_mask = np.eye(n, dtype=bool)
    d = d.astype(float, copy=False)
    d[diag_mask] = np.inf
    return float(np.min(d))


def _compute_lambda2_from_radius(
    positions: np.ndarray, communication_radius: float
) -> Dict[str, float]:
    """
    Compute connectivity quality (lambda2 of the unweighted Laplacian).
    """
    n = positions.shape[0]
    if n <= 1:
        return {"lambda2": 0.0, "lambda2_norm": 0.0}

    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = (d <= communication_radius).astype(float)
    np.fill_diagonal(adjacency, 0.0)

    degrees = np.sum(adjacency, axis=1)
    lap = np.diag(degrees) - adjacency

    eigvals = np.linalg.eigvalsh(lap)
    eigvals = np.sort(eigvals)
    lambda2 = float(eigvals[1]) if eigvals.shape[0] >= 2 else 0.0

    # For an unweighted complete graph K_n, lambda2 == n.
    lambda2_norm = _clamp01(lambda2 / float(n))
    return {"lambda2": lambda2, "lambda2_norm": lambda2_norm}


def _safe_hover_action(env: ISRGridEnv) -> np.ndarray:
    """
    Deterministic hover-like motor commands:
    - all four motors per drone equal -> zero roll/pitch/yaw torques
    - thrust ~= weight -> minimal horizontal motion
    """
    if env.simulator is None:
        return env.action_space.sample().astype(np.float32)

    cfg = env.simulator.drone_config
    hover_pwm = float(cfg.hover_thrust / cfg.max_thrust)
    hover_pwm = float(max(0.0, min(1.0, hover_pwm)))
    return np.full((env.num_drones, 4), hover_pwm, dtype=np.float32)


def _extract_targets(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Return target matrix with only "real" targets (non-zero padded rows).
    """
    targets = obs["targets"]
    if targets.size == 0:
        return targets
    mask = np.any(targets[:, :3] != 0.0, axis=1)
    return targets[mask]


def _threat_metrics_from_obs(obs: Dict[str, np.ndarray]) -> Dict[str, float]:
    targets = _extract_targets(obs)
    if targets.size == 0:
        return {
            "hostile_total": 0.0,
            "hostile_detected": 0.0,
            "hostile_detection_rate": 0.0,
            "non_hostile_total": 0.0,
            "non_hostile_detected": 0.0,
            "false_positive_rate": 0.0,
        }

    type_ids = targets[:, 8].astype(int)
    is_detected = targets[:, 10] > 0.5

    hostile_type = TargetType.HOSTILE.value
    hostile_mask = type_ids == hostile_type

    hostile_total = int(np.sum(hostile_mask))
    hostile_detected = int(np.sum(is_detected & hostile_mask))

    non_hostile_total = int(np.sum(~hostile_mask))
    non_hostile_detected = int(np.sum(is_detected & (~hostile_mask)))

    hostile_detection_rate = (
        hostile_detected / hostile_total if hostile_total > 0 else 0.0
    )
    false_positive_rate = (
        non_hostile_detected / non_hostile_total if non_hostile_total > 0 else 0.0
    )

    return {
        "hostile_total": float(hostile_total),
        "hostile_detected": float(hostile_detected),
        "hostile_detection_rate": float(hostile_detection_rate),
        "non_hostile_total": float(non_hostile_total),
        "non_hostile_detected": float(non_hostile_detected),
        "false_positive_rate": float(false_positive_rate),
    }


def _threat_score(threat: Dict[str, float]) -> float:
    hostile_det = _clamp01(threat.get("hostile_detection_rate", 0.0))
    fp_rate = _clamp01(threat.get("false_positive_rate", 0.0))
    return _clamp01(0.7 * hostile_det + 0.3 * (1.0 - fp_rate))


def _battery_norm(initial_avg: float, final_avg: float) -> float:
    if initial_avg <= 0:
        return 0.0
    return _clamp01(final_avg / initial_avg)


def _safety_norm(collisions: int, geofence_violations: int) -> float:
    penalty = float(max(0, collisions)) + float(max(0, geofence_violations))
    return float(np.exp(-0.2 * penalty))


def _runtime_norm(steps: int, mission_duration: int) -> float:
    if mission_duration <= 0:
        return 0.0
    return _clamp01(float(steps) / float(mission_duration))


def _formation_metrics_from_obs(obs: Dict[str, np.ndarray], env: ISRGridEnv) -> Dict[str, float]:
    positions = obs["swarm"][:, :3]
    # The current gym env MissionConfig may not expose these fields in all versions.
    # Formation adjacency in ISRGridEnv is built using a hardcoded ~500m threshold,
    # and the simulator collision check uses a default minimum distance.
    min_sep = float(getattr(env.mission_config, "min_swarm_separation", 1.0))
    min_distance = _pairwise_min_distance(positions)
    separation_norm = _clamp01(min_distance / max(min_sep, 1e-6))

    comm_radius = float(getattr(env.mission_config, "communication_radius", 500.0))
    lam = _compute_lambda2_from_radius(positions, comm_radius)

    return {
        "min_inter_drone_distance": float(min_distance),
        "separation_norm": float(separation_norm),
        "lambda2": float(lam["lambda2"]),
        "lambda2_norm": float(lam["lambda2_norm"]),
        "communication_radius": float(comm_radius),
    }


def _task_allocation_metrics(env: ISRGridEnv, obs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Task allocation quality computed via existing Hungarian assignment logic.

    This does not change simulator physics; it is purely an evaluation metric.
    """
    if env.simulator is None:
        return {}

    targets = _extract_targets(obs)
    if targets.size == 0:
        return {}

    swarm = obs["swarm"]
    drone_positions = swarm[:, :3]
    battery_norm = swarm[:, 13]  # simulator-normalized in obs

    sim_drone_cfg = env.simulator.drone_config
    max_speed = float(sim_drone_cfg.max_linear_velocity)
    communication_range = float(getattr(env.mission_config, "communication_radius", 500.0))

    drones: Dict[int, DroneCapability] = {}
    for i in range(env.num_drones):
        drones[i] = DroneCapability(
            drone_id=i,
            position=drone_positions[i].astype(np.float32),
            fuel_remaining=float(battery_norm[i] * 100.0),
            sensors=["radar", "optical", "rf", "acoustic"],
            current_load=0.0,
            max_speed=max_speed,
            endurance=1e9,
            communication_range=communication_range,
        )

    tasks: List[ISRTask] = []
    for i, t in enumerate(targets):
        pos = t[:3].astype(np.float32)
        type_id = int(t[8])
        is_hostile = type_id == TargetType.HOSTILE.value

        # Application-realistic mapping: hostile targets -> track/pursue; others -> classify.
        task_type = TaskType.TRACK if is_hostile else TaskType.CLASSIFY
        priority = 1.0 if is_hostile else 0.4

        tasks.append(
            ISRTask(
                task_id=i,
                task_type=task_type,
                target_position=pos,
                priority=float(priority),
                required_sensors=["radar", "optical"],
                estimated_duration=0.0,
            )
        )

    allocator = TaskAllocator(num_drones=env.num_drones)
    allocations = allocator.allocate_tasks(tasks=tasks, drones=drones)
    metrics = allocator.get_allocation_metrics()

    allocation_quality_norm = 0.0
    if metrics and "average_cost" in metrics:
        avg_cost = float(metrics["average_cost"])
        allocation_quality_norm = float(1.0 / (1.0 + avg_cost))

    return {
        "total_assignments": float(metrics.get("total_assignments", 0.0)) if metrics else 0.0,
        "average_cost": float(metrics.get("average_cost", 0.0)) if metrics else 0.0,
        "best_cost": float(metrics.get("best_cost", 0.0)) if metrics else 0.0,
        "worst_cost": float(metrics.get("worst_cost", 0.0)) if metrics else 0.0,
        "allocation_quality_norm": allocation_quality_norm,
        "assignment_coverage_norm": float(len(allocations) / max(len(tasks), 1)),
    }


def _score_for_task(
    task_id: str,
    *,
    coverage_norm: float,
    safety_norm: float,
    threat_score: float,
    battery_norm: float,
    runtime_norm: float,
    formation_norm: float,
    allocation_quality_norm: float,
) -> float:
    """
    Deterministic rubric to keep cornerstone comparable across runs.
    All inputs are already normalized to [0,1].
    """
    if task_id == "recon_coverage":
        score = (
            0.55 * coverage_norm
            + 0.20 * safety_norm
            + 0.15 * battery_norm
            + 0.10 * runtime_norm
        )
    elif task_id in ("intel_search_classify", "target_pursuit_threat"):
        score = (
            0.40 * threat_score
            + 0.25 * safety_norm
            + 0.15 * allocation_quality_norm
            + 0.10 * formation_norm
            + 0.10 * runtime_norm
        )
    elif task_id == "safety_cooperative_flight":
        score = (
            0.65 * safety_norm
            + 0.20 * formation_norm
            + 0.10 * battery_norm
            + 0.05 * runtime_norm
        )
    elif task_id == "battery_endurance":
        score = (
            0.60 * battery_norm
            + 0.15 * safety_norm
            + 0.15 * formation_norm
            + 0.10 * runtime_norm
        )
    elif task_id == "formation_connectivity":
        score = (
            0.50 * formation_norm
            + 0.20 * safety_norm
            + 0.20 * coverage_norm
            + 0.10 * runtime_norm
        )
    else:
        score = 0.25 * coverage_norm + 0.25 * safety_norm + 0.25 * threat_score + 0.25 * runtime_norm

    return float(max(0.0, min(100.0, 100.0 * _clamp01(score))))


@dataclass(frozen=True)
class SwarmTaskDefinition:
    task_id: str
    application: str
    num_drones: int
    max_targets: int
    mission_duration: int
    target_counts: Dict[TargetType, int]


TASKS: List[SwarmTaskDefinition] = [
    SwarmTaskDefinition(
        task_id="recon_coverage",
        application="Recon / Coverage",
        num_drones=10,
        max_targets=0,
        mission_duration=300,
        target_counts={},
    ),
    SwarmTaskDefinition(
        task_id="intel_search_classify",
        application="Intel / Search & Classify",
        num_drones=8,
        max_targets=6,
        mission_duration=300,
        target_counts={
            TargetType.HOSTILE: 2,
            TargetType.FRIENDLY: 2,
            TargetType.NEUTRAL: 2,
        },
    ),
    SwarmTaskDefinition(
        task_id="target_pursuit_threat",
        application="Threat Pursuit",
        num_drones=10,
        max_targets=6,
        mission_duration=250,
        target_counts={
            TargetType.HOSTILE: 4,
            TargetType.FRIENDLY: 1,
            TargetType.NEUTRAL: 1,
        },
    ),
    SwarmTaskDefinition(
        task_id="safety_cooperative_flight",
        application="Safety-Critical Cooperative Flight",
        num_drones=12,
        max_targets=0,
        mission_duration=250,
        target_counts={},
    ),
    SwarmTaskDefinition(
        task_id="battery_endurance",
        application="Battery-Constrained Endurance",
        num_drones=8,
        max_targets=0,
        mission_duration=300,
        target_counts={},
    ),
    SwarmTaskDefinition(
        task_id="formation_connectivity",
        application="Formation & Connectivity",
        num_drones=10,
        max_targets=0,
        mission_duration=250,
        target_counts={},
    ),
]


def _build_target_types(task: SwarmTaskDefinition) -> List[TargetType]:
    types: List[TargetType] = []
    for ttype, count in task.target_counts.items():
        types.extend([ttype] * int(count))
    return types


def _place_targets_for_task(
    env: ISRGridEnv,
    rng: np.random.RandomState,
    task: SwarmTaskDefinition,
) -> List[Tuple[np.ndarray, TargetType]]:
    if env.simulator is None or task.max_targets <= 0:
        return []

    min_det = float(env.simulator.target_config.min_detection_distance)
    max_det = float(env.simulator.target_config.max_detection_distance)
    drone_positions = np.array([d.position for d in env.simulator.drones], dtype=float)

    desired_types = _build_target_types(task)
    placed: List[Tuple[np.ndarray, TargetType]] = []

    for ttype in desired_types:
        ok = False
        for _attempt in range(2000):
            anchor = int(rng.randint(0, env.num_drones))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius = float(rng.uniform(min_det + 20.0, max_det - 20.0))
            dx = radius * float(np.cos(angle))
            dy = radius * float(np.sin(angle))

            z = float(drone_positions[anchor, 2])
            pos = drone_positions[anchor] + np.array([dx, dy, 0.0], dtype=float)
            pos[2] = z

            dists = np.linalg.norm(drone_positions - pos[None, :], axis=1)
            min_distance = float(np.min(dists))

            if not (min_det <= min_distance <= max_det):
                continue
            if float(np.linalg.norm(pos[:2])) < 1e-3:
                continue

            placed.append((pos.astype(np.float32), ttype))
            ok = True
            break

        if not ok:
            pos = drone_positions[0].copy()
            pos[0] += min_det + 30.0
            placed.append((pos.astype(np.float32), ttype))

    return placed


def _make_policy(
    *,
    policy: str,
    checkpoint_path: Optional[str],
) -> Callable[[ISRGridEnv, Dict[str, np.ndarray]], np.ndarray]:
    if policy not in ("hover", "random", "agent"):
        raise ValueError("policy must be one of: hover, random, agent")

    agent_cache: Dict[Tuple[int, int], Any] = {}

    def _policy(env: ISRGridEnv, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if policy == "hover":
            return _safe_hover_action(env)
        if policy == "random":
            return env.action_space.sample().astype(np.float32)

        # policy == "agent"
        if checkpoint_path is None:
            return _safe_hover_action(env)

        # Lazy import so evaluation can run without `torch`.
        from isr_rl_dmpc.agents.dmpc_agent import DMPCAgent

        state_vec = DMPCAgent.flatten_obs(obs)
        state_dim = int(state_vec.shape[0])
        action_dim = int(np.prod(env.action_space.shape))
        cache_key = (state_dim, action_dim)

        if cache_key not in agent_cache:
            agent = DMPCAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cpu",
            )
            agent.load(checkpoint_path)
            agent_cache[cache_key] = agent

        action_vec = agent_cache[cache_key].act(obs, training=False, deterministic=True)
        action_env = np.clip(action_vec.reshape(env.action_space.shape), 0.0, 1.0)
        return action_env.astype(np.float32)

    return _policy


def _run_single_episode(
    env: ISRGridEnv,
    task: SwarmTaskDefinition,
    *,
    seed: int,
    policy_fn: Callable[[ISRGridEnv, Dict[str, np.ndarray]], np.ndarray],
    mission_duration_override: Optional[int] = None,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)

    initial_avg_battery = float(info.get("avg_battery", 0.0))

    # Override initial random targets with deterministic task targets.
    if env.simulator is not None and task.max_targets > 0:
        env.simulator.num_targets = 0
        placed = _place_targets_for_task(
            env=env, rng=np.random.RandomState(seed), task=task
        )
        for pos, ttype in placed:
            env.simulator.add_target(pos, ttype)

    total_reward = 0.0
    done = False
    last_obs = obs
    last_info = info
    steps_completed = 0

    mission_duration = int(mission_duration_override) if mission_duration_override is not None else int(task.mission_duration)

    for _step_idx in range(mission_duration):
        action_env = policy_fn(env, last_obs)
        last_obs, reward, terminated, _truncated, last_info = env.step(action_env)
        total_reward += float(reward)
        steps_completed = int(last_info.get("step", _step_idx + 1))
        done = bool(terminated)
        if done:
            break

    coverage = float(last_info.get("coverage", 0.0))
    collisions = int(last_info.get("collisions", 0))
    geofence_violations = int(last_info.get("geofence_violations", 0))
    avg_battery = float(last_info.get("avg_battery", 0.0))
    active_drones = int(last_info.get("active_drones", 0))

    coverage_goal = float(
        getattr(env.mission_config, "coverage_goal", getattr(env.mission_config, "coverage_target", 1.0))
    )
    coverage_norm = _clamp01(coverage / max(coverage_goal, 1e-6))
    safety_norm = _safety_norm(collisions, geofence_violations)
    runtime_norm = _runtime_norm(steps_completed, mission_duration)

    battery_norm = _battery_norm(initial_avg_battery, avg_battery)
    if active_drones <= 0:
        battery_norm = 0.0

    threat = _threat_metrics_from_obs(last_obs)
    threat_score = _threat_score(threat)

    formation_m = _formation_metrics_from_obs(last_obs, env)
    formation_norm = _clamp01(0.6 * formation_m["lambda2_norm"] + 0.4 * formation_m["separation_norm"])

    allocation_m = _task_allocation_metrics(env=env, obs=last_obs)
    allocation_quality_norm = float(allocation_m.get("allocation_quality_norm", 0.0)) if allocation_m else 0.0

    score = _score_for_task(
        task.task_id,
        coverage_norm=coverage_norm,
        safety_norm=safety_norm,
        threat_score=threat_score,
        battery_norm=battery_norm,
        runtime_norm=runtime_norm,
        formation_norm=formation_norm,
        allocation_quality_norm=allocation_quality_norm,
    )

    return {
        "seed": int(seed),
        "task_id": task.task_id,
        "application": task.application,
        "mission_duration": mission_duration,
        "steps_completed": int(steps_completed),
        "total_reward": float(total_reward),
        "metrics": {
            "coverage": coverage,
            "coverage_norm": coverage_norm,
            "collisions": collisions,
            "geofence_violations": geofence_violations,
            "safety_norm": safety_norm,
            "avg_battery": avg_battery,
            "battery_norm": battery_norm,
            "active_drones": active_drones,
            "runtime_norm": runtime_norm,
            "threat": threat,
            "threat_score": threat_score,
            "formation": formation_m,
            "formation_norm": formation_norm,
            "allocation": allocation_m,
            "score": score,
        },
        "episode_score": float(score),
    }


def run_swarm_task_suite(
    *,
    task_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
    policy: str = "hover",
    checkpoint_path: Optional[str] = None,
    output_dir: str = "data/evaluation_runs",
    duration_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Execute the grouped swarm task suite and compute cornerstone scores.
    """
    task_list = [t for t in TASKS if task_ids is None or t.task_id in task_ids]
    if not task_list:
        raise ValueError("No tasks selected")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed + i) for i in range(int(episodes))]
    policy_fn = _make_policy(policy=policy, checkpoint_path=checkpoint_path)

    per_task: List[Dict[str, Any]] = []
    all_scores: List[float] = []

    for task in task_list:
        episodes_out: List[Dict[str, Any]] = []

        for ep_seed in seeds:
            env = ISRGridEnv(
                num_drones=task.num_drones,
                max_targets=task.max_targets,
                mission_duration=int(max(1, task.mission_duration * float(duration_multiplier))),
            )
            ep_out = _run_single_episode(
                env,
                task,
                seed=ep_seed,
                policy_fn=policy_fn,
                mission_duration_override=int(max(1, task.mission_duration * float(duration_multiplier))),
            )
            episodes_out.append(ep_out)
            env.close()

        scores = [float(e["episode_score"]) for e in episodes_out]
        task_score = float(np.mean(scores)) if scores else 0.0
        all_scores.append(task_score)

        per_task_entry = {
            "task_id": task.task_id,
            "application": task.application,
            "episodes": episodes_out,
            "score": task_score,
            "score_std": float(np.std(scores)) if scores else 0.0,
        }
        per_task.append(per_task_entry)

        (run_dir / f"{task.task_id}.json").write_text(
            json.dumps(per_task_entry, indent=2, default=str),
            encoding="utf-8",
        )

    overall = float(np.mean(all_scores)) if all_scores else 0.0
    cornerstone = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "policy": policy,
        "checkpoint_path": checkpoint_path,
        "seed_start": seed,
        "episodes": episodes,
        "overall_score": overall,
        "tasks": [{"task_id": t["task_id"], "application": t["application"], "score": t["score"]} for t in per_task],
    }

    (run_dir / "cornerstone_score.json").write_text(
        json.dumps(cornerstone, indent=2, default=str),
        encoding="utf-8",
    )

    return cornerstone


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ISR-RL-DMPC with grouped swarm tasks.")
    p.add_argument("--tasks", type=str, default="", help="Comma-separated task ids to run.")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes (seeds).")
    p.add_argument("--seed", type=int, default=42, help="Seed start.")
    p.add_argument("--policy", type=str, default="hover", choices=["hover", "random", "agent"])
    p.add_argument("--checkpoint", type=str, default=None, help="Agent checkpoint path (for policy=agent).")
    p.add_argument("--output-dir", type=str, default="data/evaluation_runs", help="Where to write reports.")
    p.add_argument("--duration-multiplier", type=float, default=1.0, help="Multiply mission duration per task.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    out = run_swarm_task_suite(
        task_ids=task_ids,
        episodes=args.episodes,
        seed=args.seed,
        policy=args.policy,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        duration_multiplier=float(args.duration_multiplier),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

