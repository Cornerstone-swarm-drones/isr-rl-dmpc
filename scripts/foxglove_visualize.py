"""
foxglove_visualize.py - Run ISR-RL-DMPC simulation with Foxglove Studio visualization.

Starts a Foxglove WebSocket server and runs the ISRGridEnv simulation,
streaming real-time data for 3D visualization, metrics plotting, and
mission analysis in Foxglove Studio.

Usage:
    # Live visualization (connect Foxglove Studio to ws://localhost:8765):
    python scripts/foxglove_visualize.py --mode live

    # Record to MCAP file for offline playback:
    python scripts/foxglove_visualize.py --mode record --output mission.mcap

    # Both live + recording simultaneously:
    python scripts/foxglove_visualize.py --mode both --output mission.mcap

Foxglove Studio Setup:
    1. Download Foxglove Studio from https://foxglove.dev/download
    2. Open Foxglove Studio
    3. Click "Open connection" -> "Foxglove WebSocket"
    4. Enter ws://localhost:8765
    5. Import layout from config/foxglove_layout.json (optional)
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _create_synthetic_drone_positions(n_drones: int, step: int, grid_extent: float):
    """Generate synthetic drone positions for demonstration."""
    positions = np.zeros((n_drones, 3))
    grid_side = int(np.ceil(np.sqrt(n_drones)))
    spacing = grid_extent / (grid_side + 1)

    for i in range(n_drones):
        row = i // grid_side
        col = i % grid_side
        base_x = (col + 1) * spacing
        base_y = (row + 1) * spacing
        # Add circular patrol motion
        angle = step * 0.02 + i * (2 * np.pi / n_drones)
        radius = 30.0 + 10.0 * np.sin(step * 0.01 + i)
        positions[i] = [
            base_x + radius * np.cos(angle),
            base_y + radius * np.sin(angle),
            50.0 + 10.0 * np.sin(step * 0.05 + i),
        ]
    return positions


def _create_synthetic_targets(n_targets: int, step: int, grid_extent: float):
    """Generate synthetic target positions for demonstration."""
    positions = {}
    classifications = {}
    cls_types = ["hostile", "friendly", "unknown"]
    for i in range(n_targets):
        angle = step * 0.005 + i * 1.5
        positions[f"T{i}"] = np.array([
            grid_extent * 0.3 + 100 * np.cos(angle + i),
            grid_extent * 0.3 + 100 * np.sin(angle + i),
            80.0 + 20 * np.sin(step * 0.03 + i),
        ])
        classifications[f"T{i}"] = cls_types[i % len(cls_types)]
    return positions, classifications


async def run_live(args):
    """Run simulation with live Foxglove WebSocket server."""
    from isr_rl_dmpc.utils.foxglove_bridge import FoxgloveBridge, extract_targets_from_obs

    bridge = FoxgloveBridge(
        host=args.host,
        port=args.port,
        server_name="ISR-RL-DMPC Simulation",
    )
    bridge.start()

    logger.info(
        "Foxglove bridge ready — connect Foxglove Studio to ws://%s:%d",
        args.host, args.port,
    )
    logger.info("Press Ctrl+C to stop")

    grid_extent = args.grid_x * 100.0

    try:
        env = None
        try:
            from isr_rl_dmpc.gym_env.isr_env import ISRGridEnv
            env = ISRGridEnv(
                num_drones=args.num_drones,
                max_targets=args.max_targets,
                grid_size=(args.grid_x, args.grid_y),
                mission_duration=args.num_steps,
            )
            obs, info = env.reset()
            logger.info("ISRGridEnv initialized with %d drones", args.num_drones)
        except Exception as e:
            logger.warning("Could not initialize ISRGridEnv: %s", e)
            logger.info("Running with synthetic data for demonstration")

        for step in range(args.num_steps):
            ts_ns = int(time.time() * 1e9)

            if env is not None:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # Extract drone state from observation
                swarm = obs["swarm"]  # (N, 18)
                drone_positions = swarm[:, :3]
                drone_quats = swarm[:, 9:13]
                drone_batteries = swarm[:, 16]

                # Extract target positions and classifications from obs
                tgt_pos, tgt_cls = extract_targets_from_obs(obs["targets"])

                bridge.publish_scene(
                    drone_positions=drone_positions,
                    drone_quaternions=drone_quats,
                    drone_batteries=drone_batteries,
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts_ns,
                    grid_extent=grid_extent,
                )
                bridge.publish_metrics(info, reward=reward, timestamp_ns=ts_ns)
                bridge.publish_coverage(
                    env.coverage_map, env.grid_size, timestamp_ns=ts_ns,
                )

                if terminated or truncated:
                    obs, info = env.reset()
                    logger.info("Episode reset at step %d", step)
            else:
                # Synthetic data
                positions = _create_synthetic_drone_positions(
                    args.num_drones, step, grid_extent
                )
                batteries = np.full(args.num_drones, 5000.0 - step * 2)
                batteries = np.clip(batteries, 0, 5000)
                tgt_pos, tgt_cls = _create_synthetic_targets(
                    args.max_targets, step, grid_extent
                )

                bridge.publish_scene(
                    drone_positions=positions,
                    drone_batteries=batteries,
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts_ns,
                    grid_extent=grid_extent,
                )
                coverage_map = np.zeros(args.grid_x * args.grid_y)
                coverage_map[:int(step * 0.5)] = 1.0
                bridge.publish_metrics(
                    info={
                        "step": step,
                        "coverage": float(np.mean(coverage_map)),
                        "avg_battery": float(np.mean(batteries)),
                        "active_drones": args.num_drones,
                        "total_drones": args.num_drones,
                        "collisions": 0,
                    },
                    reward=0.01 * (1.0 - step / args.num_steps),
                    timestamp_ns=ts_ns,
                )
                bridge.publish_coverage(
                    coverage_map, (args.grid_x, args.grid_y), timestamp_ns=ts_ns,
                )

            # Control loop rate
            await asyncio.sleep(1.0 / args.fps)

            if step % 100 == 0:
                logger.info("Step %d/%d", step, args.num_steps)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if env is not None:
            env.close()
        bridge.stop()
def run_record(args):
    """Run simulation and record to MCAP file."""
    from isr_rl_dmpc.utils.mcap_logger import MCAPRecorder
    from isr_rl_dmpc.utils.foxglove_bridge import extract_targets_from_obs

    output_path = args.output or f"data/recordings/mission_{datetime.now():%Y%m%d_%H%M%S}.mcap"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    grid_extent = args.grid_x * 100.0

    with MCAPRecorder(output_path) as recorder:
        logger.info("Recording to %s", output_path)

        env = None
        try:
            from isr_rl_dmpc.gym_env.isr_env import ISRGridEnv
            env = ISRGridEnv(
                num_drones=args.num_drones,
                max_targets=args.max_targets,
                grid_size=(args.grid_x, args.grid_y),
                mission_duration=args.num_steps,
            )
            obs, info = env.reset()
            logger.info("ISRGridEnv initialized with %d drones", args.num_drones)
        except Exception as e:
            logger.warning("Could not initialize ISRGridEnv: %s", e)
            logger.info("Recording with synthetic data for demonstration")

        for step in range(args.num_steps):
            ts_ns = int(time.time() * 1e9)

            if env is not None:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                swarm = obs["swarm"]
                drone_positions = swarm[:, :3]
                drone_quats = swarm[:, 9:13]
                drone_batteries = swarm[:, 16]

                # Extract target positions and classifications from obs
                tgt_pos, tgt_cls = extract_targets_from_obs(obs["targets"])

                recorder.record_scene(
                    drone_positions=drone_positions,
                    drone_quaternions=drone_quats,
                    drone_batteries=drone_batteries,
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts_ns,
                )
                recorder.record_metrics(info, reward=reward, timestamp_ns=ts_ns)
                recorder.record_coverage(
                    env.coverage_map, env.grid_size, timestamp_ns=ts_ns,
                )

                if terminated or truncated:
                    obs, info = env.reset()
            else:
                positions = _create_synthetic_drone_positions(
                    args.num_drones, step, grid_extent
                )
                batteries = np.clip(
                    np.full(args.num_drones, 5000.0 - step * 2), 0, 5000,
                )
                tgt_pos, tgt_cls = _create_synthetic_targets(
                    args.max_targets, step, grid_extent
                )

                recorder.record_scene(
                    drone_positions=positions,
                    drone_batteries=batteries,
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts_ns,
                )
                coverage_map = np.zeros(args.grid_x * args.grid_y)
                coverage_map[:int(step * 0.5)] = 1.0
                recorder.record_metrics(
                    info={
                        "step": step,
                        "coverage": float(np.mean(coverage_map)),
                        "avg_battery": float(np.mean(batteries)),
                        "active_drones": args.num_drones,
                        "total_drones": args.num_drones,
                        "collisions": 0,
                    },
                    reward=0.01 * (1.0 - step / args.num_steps),
                    timestamp_ns=ts_ns,
                )
                recorder.record_coverage(
                    coverage_map, (args.grid_x, args.grid_y), timestamp_ns=ts_ns,
                )

            if step % 100 == 0:
                logger.info("Recorded step %d/%d", step, args.num_steps)

        if env is not None:
            env.close()

    logger.info("Recording complete: %s", output_path)
    logger.info("Open this file in Foxglove Studio for playback")


async def run_both(args):
    """Run live visualization and MCAP recording simultaneously."""
    from isr_rl_dmpc.utils.foxglove_bridge import FoxgloveBridge, extract_targets_from_obs
    from isr_rl_dmpc.utils.mcap_logger import MCAPRecorder

    output_path = args.output or f"data/recordings/mission_{datetime.now():%Y%m%d_%H%M%S}.mcap"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bridge = FoxgloveBridge(
        host=args.host,
        port=args.port,
        server_name="ISR-RL-DMPC Simulation",
    )
    bridge.start()

    logger.info(
        "Foxglove bridge ready — connect Foxglove Studio to ws://%s:%d",
        args.host, args.port,
    )
    logger.info("Recording to %s", output_path)
    logger.info("Press Ctrl+C to stop")

    grid_extent = args.grid_x * 100.0

    with MCAPRecorder(output_path) as recorder:
        try:
            env = None
            try:
                from isr_rl_dmpc.gym_env.isr_env import ISRGridEnv
                env = ISRGridEnv(
                    num_drones=args.num_drones,
                    max_targets=args.max_targets,
                    grid_size=(args.grid_x, args.grid_y),
                    mission_duration=args.num_steps,
                )
                obs, info = env.reset()
                logger.info("ISRGridEnv initialized with %d drones", args.num_drones)
            except Exception as e:
                logger.warning("Could not initialize ISRGridEnv: %s", e)
                logger.info("Running with synthetic data for demonstration")

            for step in range(args.num_steps):
                ts_ns = int(time.time() * 1e9)

                if env is not None:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                    swarm = obs["swarm"]
                    drone_positions = swarm[:, :3]
                    drone_quats = swarm[:, 9:13]
                    drone_batteries = swarm[:, 16]
                    tgt_pos, tgt_cls = extract_targets_from_obs(obs["targets"])

                    bridge.publish_scene(
                        drone_positions=drone_positions,
                        drone_quaternions=drone_quats,
                        drone_batteries=drone_batteries,
                        target_positions=tgt_pos,
                        target_classifications=tgt_cls,
                        timestamp_ns=ts_ns,
                        grid_extent=grid_extent,
                    )
                    bridge.publish_metrics(info, reward=reward, timestamp_ns=ts_ns)
                    bridge.publish_coverage(
                        env.coverage_map, env.grid_size, timestamp_ns=ts_ns,
                    )

                    recorder.record_scene(
                        drone_positions=drone_positions,
                        drone_quaternions=drone_quats,
                        drone_batteries=drone_batteries,
                        target_positions=tgt_pos,
                        target_classifications=tgt_cls,
                        timestamp_ns=ts_ns,
                    )
                    recorder.record_metrics(info, reward=reward, timestamp_ns=ts_ns)
                    recorder.record_coverage(
                        env.coverage_map, env.grid_size, timestamp_ns=ts_ns,
                    )

                    if terminated or truncated:
                        obs, info = env.reset()
                        logger.info("Episode reset at step %d", step)
                else:
                    positions = _create_synthetic_drone_positions(
                        args.num_drones, step, grid_extent
                    )
                    batteries = np.clip(
                        np.full(args.num_drones, 5000.0 - step * 2), 0, 5000,
                    )
                    tgt_pos, tgt_cls = _create_synthetic_targets(
                        args.max_targets, step, grid_extent
                    )

                    bridge.publish_scene(
                        drone_positions=positions,
                        drone_batteries=batteries,
                        target_positions=tgt_pos,
                        target_classifications=tgt_cls,
                        timestamp_ns=ts_ns,
                        grid_extent=grid_extent,
                    )
                    recorder.record_scene(
                        drone_positions=positions,
                        drone_batteries=batteries,
                        target_positions=tgt_pos,
                        target_classifications=tgt_cls,
                        timestamp_ns=ts_ns,
                    )
                    coverage_map = np.zeros(args.grid_x * args.grid_y)
                    coverage_map[:int(step * 0.5)] = 1.0
                    metrics_info = {
                        "step": step,
                        "coverage": float(np.mean(coverage_map)),
                        "avg_battery": float(np.mean(batteries)),
                        "active_drones": args.num_drones,
                        "total_drones": args.num_drones,
                        "collisions": 0,
                    }
                    reward_val = 0.01 * (1.0 - step / args.num_steps)
                    bridge.publish_metrics(
                        info=metrics_info, reward=reward_val, timestamp_ns=ts_ns,
                    )
                    bridge.publish_coverage(
                        coverage_map, (args.grid_x, args.grid_y), timestamp_ns=ts_ns,
                    )
                    recorder.record_metrics(
                        info=metrics_info, reward=reward_val, timestamp_ns=ts_ns,
                    )
                    recorder.record_coverage(
                        coverage_map, (args.grid_x, args.grid_y), timestamp_ns=ts_ns,
                    )

                await asyncio.sleep(1.0 / args.fps)

                if step % 100 == 0:
                    logger.info("Step %d/%d", step, args.num_steps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if env is not None:
                env.close()
            bridge.stop()

    logger.info("Recording complete: %s", output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ISR-RL-DMPC Foxglove Studio Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live visualization:
  python scripts/foxglove_visualize.py --mode live

  # Record to MCAP:
  python scripts/foxglove_visualize.py --mode record --output mission.mcap

  # Live + record simultaneously:
  python scripts/foxglove_visualize.py --mode both --output mission.mcap
        """,
    )

    parser.add_argument(
        "--mode", choices=["live", "record", "both"], default="live",
        help="Visualization mode (default: live)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--output", type=str, default=None, help="MCAP output file path")
    parser.add_argument("--num-drones", type=int, default=4, help="Number of drones")
    parser.add_argument("--max-targets", type=int, default=3, help="Maximum targets")
    parser.add_argument("--grid-x", type=int, default=20, help="Grid X size")
    parser.add_argument("--grid-y", type=int, default=20, help="Grid Y size")
    parser.add_argument("--num-steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--fps", type=float, default=10.0, help="Publishing rate (Hz)")

    args = parser.parse_args()

    if args.mode == "live":
        asyncio.run(run_live(args))
    elif args.mode == "record":
        run_record(args)
    elif args.mode == "both":
        logger.info("Running live visualization + MCAP recording simultaneously")
        asyncio.run(run_both(args))


if __name__ == "__main__":
    main()
