# Foxglove Studio Integration

This guide covers the Foxglove Studio integration for real-time and offline visualization of ISR-RL-DMPC simulation data.

## Overview

[Foxglove Studio](https://foxglove.dev/) is an open-source robotics visualization tool that provides interactive 3D views, time-series plots, and data inspection panels. The ISR-RL-DMPC integration supports:

- **Live visualization** via WebSocket — stream simulation data in real-time
- **MCAP recording** — record simulation runs for offline playback and analysis
- **Simultaneous live + recording** — stream and record at the same time
- **Training visualization** — visualize agent training in real-time with the `--foxglove` flag
- **StateManager integration** — publish directly from the existing state management system

## Quick Start

### 1. Install Dependencies

```bash
pip install foxglove-sdk>=0.18.0 mcap>=1.1.0 protobuf>=5.29.6
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

### 4. Live + Record Simultaneously

Stream to Foxglove Studio **and** record to an MCAP file at the same time:

```bash
python scripts/foxglove_visualize.py --mode both --output data/recordings/mission.mcap
```

### 5. Visualize During Training

Enable live Foxglove visualization while training the RL agent:

```bash
python scripts/train_agent.py --foxglove --foxglove-port 8765
```

Then connect Foxglove Studio to `ws://localhost:8765` to watch drones, targets, and metrics update in real-time as the agent trains.

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
| `extract_targets_from_obs` | `src/isr_rl_dmpc/utils/foxglove_bridge.py` | Extract target data from env observations |
| `foxglove_visualize.py` | `scripts/foxglove_visualize.py` | CLI script for visualization |
| `foxglove_config.yaml` | `config/foxglove_config.yaml` | Configuration settings |
| `foxglove_layout.json` | `config/foxglove_layout.json` | Default Foxglove Studio layout |

## Channels

The integration publishes data on these topics:

| Topic | Schema | Content |
|-------|--------|---------|
| `/swarm/scene` | `foxglove.SceneUpdate` | 3D scene with drone models, target models, and ground plane |
| `/swarm/metrics` | `isr.SwarmMetrics` | Coverage, battery, reward, collisions |
| `/mission/coverage` | `isr.CoverageGrid` | Grid coverage map and percentage |
| `/mission/info` | `isr.MissionInfo` | Mission progress, duration, efficiency |

### Scene Objects

The 3D scene includes:

- **Drone 3D models** — hector_quadrotor .glb model (converted from the [hector_quadrotor](https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor) URDF, bundled locally under `src/isr_rl_dmpc/models/meshes/`) color-coded by battery level (green = full, red = empty) with labels showing drone ID and battery
- **Target 3D models** — distinct models per threat class (CesiumMilkTruck = hostile, CesiumAir = friendly, Box = unknown) with color tinting, loaded from local .glb files
- **Target labels** — showing target ID and classification
- **Ground plane** — semi-transparent reference plane showing the mission area extent

All 3D models are embedded directly in SceneUpdate messages from local files — **no network access is required at runtime**.

## API Reference

### FoxgloveBridge

```python
from isr_rl_dmpc.utils import FoxgloveBridge

# Initialize and start
bridge = FoxgloveBridge(host="0.0.0.0", port=8765)
bridge.start()

# Publish during simulation loop
bridge.publish_scene(drone_positions, drone_quaternions, drone_batteries,
                     target_positions, target_classifications, timestamp_ns,
                     grid_extent=2000.0)  # explicit extent; auto-computed when omitted
bridge.publish_metrics(info_dict, reward, timestamp_ns)
bridge.publish_coverage(coverage_map, grid_size, timestamp_ns)
bridge.publish_mission_info(elapsed_time, mission_duration, coverage_efficiency,
                            num_waypoints, timestamp_ns)

# Or publish directly from StateManager
bridge.publish_from_state_manager(state_manager, timestamp_ns)

# Stop
bridge.stop()
```

### extract_targets_from_obs

Helper to extract target positions and classifications from the gym environment's observation array:

```python
from isr_rl_dmpc.utils import extract_targets_from_obs

# obs["targets"] is (max_targets, 12) array from ISRGridEnv
target_positions, target_classifications = extract_targets_from_obs(obs["targets"])
# target_positions: {"T0": np.array([x, y, z]), ...}
# target_classifications: {"T0": "hostile", "T1": "friendly", ...}
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
from isr_rl_dmpc.gym_env.isr_env import ISRGridEnv
from isr_rl_dmpc.utils import FoxgloveBridge, extract_targets_from_obs

def train_with_visualization():
    env = ISRGridEnv(num_drones=4, max_targets=3)
    bridge = FoxgloveBridge(port=8765)
    bridge.start()

    obs, info = env.reset()
    for step in range(1000):
        action = agent.act(obs)  # Your RL agent
        obs, reward, done, truncated, info = env.step(action)

        # Extract targets from observation
        tgt_pos, tgt_cls = extract_targets_from_obs(obs["targets"])

        # Publish to Foxglove with all objects visible
        bridge.publish_scene(
            drone_positions=obs['swarm'][:, :3],
            drone_batteries=obs['swarm'][:, 16],
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
            grid_extent=2000.0,
        )
        bridge.publish_metrics(info, reward=reward)

        if done:
            obs, info = env.reset()

    bridge.stop()
```

Or use the built-in training flag:

```bash
python scripts/train_agent.py --foxglove --num-episodes 100
```

## CLI Options

### foxglove_visualize.py

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

### train_agent.py (Foxglove options)

```
python scripts/train_agent.py [OPTIONS]

Foxglove Options:
  --foxglove                 Enable live Foxglove Studio visualization
  --foxglove-port PORT       WebSocket server port (default: 8765)
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

- **3D Panel** — Follows `swarm_center` frame so all drones stay visible. Camera distance 300 m, angled overhead view. Drone 3D models (local hector_quadrotor .glb), target models (local per-threat-class .glb), and ground plane
- **Plot Panel** — Real-time coverage, reward, and battery metrics
- **Raw Messages** — Coverage grid and mission info inspection

## Troubleshooting

### Known Issues

- The project uses `foxglove-sdk` (v0.18.0+) which replaced the deprecated `foxglove-websocket` library for a clean, warning-free experience.

### Common Problems

| Problem | Solution |
|---------|----------|
| Cannot connect to WebSocket | Ensure the server is running and check firewall settings for port 8765 |
| No 3D objects visible | Add a **3D Panel** in Foxglove Studio and subscribe to `/swarm/scene` |
| Targets not showing | Ensure the simulation has active targets (non-zero positions) |
| Metrics not plotting | Add a **Plot Panel** and select fields from `/swarm/metrics` |
