# Foxglove Studio Integration

This guide covers the Foxglove Studio integration for real-time and offline visualization of ISR-RL-DMPC simulation data.

## Overview

[Foxglove Studio](https://foxglove.dev/) is an open-source robotics visualization tool that provides interactive 3D views, time-series plots, and data inspection panels. The ISR-RL-DMPC integration supports:

- **Live visualization** via WebSocket — stream simulation data in real-time
- **MCAP recording** — record simulation runs for offline playback and analysis
- **StateManager integration** — publish directly from the existing state management system

## Quick Start

### 1. Install Dependencies

```bash
pip install foxglove-websocket>=0.1.4 mcap>=1.1.0 protobuf>=5.29.6
```

Or install all project dependencies:

```bash
pip install -r requirements/base.txt
```

### 2. Live Visualization

Start the simulation with Foxglove WebSocket server:

```bash
python scripts/foxglove_visualize.py --mode live
```

Then connect Foxglove Studio:
1. Open [Foxglove Studio](https://foxglove.dev/download)
2. Click **Open connection** → **Foxglove WebSocket**
3. Enter `ws://localhost:8765`
4. Optionally import the layout from `config/foxglove_layout.json`

### 3. Record to MCAP

Record a simulation run for later analysis:

```bash
python scripts/foxglove_visualize.py --mode record --output data/recordings/mission.mcap
```

Open the `.mcap` file in Foxglove Studio for full playback with timeline scrubbing.

## Architecture

```
┌─────────────────────┐     WebSocket (ws://localhost:8765)
│  ISRGridEnv / Agent  │ ──────────────────────────────────── ► Foxglove Studio
│                      │     FoxgloveBridge                     (3D, Plots, Data)
│  StateManager        │ ──┐
└─────────────────────┘   │
                           │  MCAPRecorder
                           └──────────────────────────────── ► recording.mcap
                                                                (Offline Playback)
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `FoxgloveBridge` | `src/isr_rl_dmpc/utils/foxglove_bridge.py` | WebSocket server for live streaming |
| `MCAPRecorder` | `src/isr_rl_dmpc/utils/mcap_logger.py` | MCAP file recording |
| `foxglove_visualize.py` | `scripts/foxglove_visualize.py` | CLI script for visualization |
| `foxglove_config.yaml` | `config/foxglove_config.yaml` | Configuration settings |
| `foxglove_layout.json` | `config/foxglove_layout.json` | Default Foxglove Studio layout |

## Channels

The integration publishes data on these topics:

| Topic | Schema | Content |
|-------|--------|---------|
| `/swarm/scene` | `foxglove.SceneUpdate` | 3D scene with drone cubes and target spheres |
| `/swarm/metrics` | `isr.SwarmMetrics` | Coverage, battery, reward, collisions |
| `/mission/coverage` | `isr.CoverageGrid` | Grid coverage map and percentage |
| `/mission/info` | `isr.MissionInfo` | Mission progress, duration, efficiency |

## API Reference

### FoxgloveBridge

```python
from isr_rl_dmpc.utils import FoxgloveBridge

# Initialize and start
bridge = FoxgloveBridge(host="0.0.0.0", port=8765)
await bridge.start()

# Publish during simulation loop
bridge.publish_scene(drone_positions, drone_quaternions, drone_batteries,
                     target_positions, target_classifications, timestamp_ns)
bridge.publish_metrics(info_dict, reward, timestamp_ns)
bridge.publish_coverage(coverage_map, grid_size, timestamp_ns)
bridge.publish_mission_info(elapsed_time, mission_duration, coverage_efficiency,
                            num_waypoints, timestamp_ns)

# Or publish directly from StateManager
bridge.publish_from_state_manager(state_manager, timestamp_ns)

# Stop
await bridge.stop()
```

### MCAPRecorder

```python
from isr_rl_dmpc.utils import MCAPRecorder

# Context manager usage (recommended)
with MCAPRecorder("recording.mcap") as recorder:
    for step in range(num_steps):
        recorder.record_scene(drone_positions, timestamp_ns=ts)
        recorder.record_metrics(info, reward=reward, timestamp_ns=ts)
        recorder.record_coverage(coverage_map, grid_size, timestamp_ns=ts)
        recorder.record_mission_info(elapsed_time, mission_duration,
                                     timestamp_ns=ts)
```

### Integration with Training Loop

```python
import asyncio
from isr_rl_dmpc.gym_env.isr_env import ISRGridEnv
from isr_rl_dmpc.utils import FoxgloveBridge, MCAPRecorder

async def train_with_visualization():
    env = ISRGridEnv(num_drones=4)
    bridge = FoxgloveBridge(port=8765)
    await bridge.start()

    obs, info = env.reset()
    for step in range(1000):
        action = agent.act(obs)  # Your RL agent
        obs, reward, done, truncated, info = env.step(action)

        # Publish to Foxglove
        bridge.publish_scene(
            drone_positions=obs['swarm'][:, :3],
            drone_batteries=obs['swarm'][:, 16],
        )
        bridge.publish_metrics(info, reward=reward)

        if done:
            obs, info = env.reset()

    await bridge.stop()
```

## CLI Options

```
python scripts/foxglove_visualize.py [OPTIONS]

Options:
  --mode {live,record,both}  Visualization mode (default: live)
  --host HOST                WebSocket host (default: 0.0.0.0)
  --port PORT                WebSocket port (default: 8765)
  --output PATH              MCAP output file path
  --num-drones N             Number of drones (default: 4)
  --max-targets N            Maximum targets (default: 3)
  --grid-x N                 Grid X size (default: 20)
  --grid-y N                 Grid Y size (default: 20)
  --num-steps N              Simulation steps (default: 500)
  --fps RATE                 Publishing rate in Hz (default: 10)
```

## Configuration

Edit `config/foxglove_config.yaml` to customize:

```yaml
foxglove:
  websocket:
    host: "0.0.0.0"
    port: 8765
  recording:
    enabled: true
    output_dir: "data/recordings"
  visualization:
    drone_marker_size: [2.0, 2.0, 0.5]
    target_marker_size: [3.0, 3.0, 3.0]
```

## Foxglove Studio Layout

Import the default layout from `config/foxglove_layout.json` for a pre-configured dashboard with:

- **3D Panel** — Drone positions (cubes) and target markers (spheres)
- **Plot Panel** — Real-time coverage, reward, and battery metrics
- **Raw Messages** — Coverage grid and mission info inspection
