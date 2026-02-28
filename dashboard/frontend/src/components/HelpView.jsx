import React, { useState } from "react";

const SECTIONS = [
  {
    title: "Getting Started",
    icon: "🚀",
    content: `The ISR-RL-DMPC Dashboard lets you control and monitor autonomous drone swarm missions.
There are two main modes:

**Operate Mode** – Run simulations, view real-time drone/target positions, and analyze mission metrics.
**Train Mode** – Launch RL training runs, monitor learning curves, and compare experiments.

Use the sidebar on the left to switch modes and view/edit configuration parameters.`,
  },
  {
    title: "Operate Mode – Mission Tab",
    icon: "🎮",
    content: `1. **Configure** the number of drones, targets, and mission duration in the Simulation Control panel.
2. **Select a trained model** from the "Model" dropdown to use a saved policy, or keep "Random" for untrained actions.
3. Click **Reset** to initialize the environment.
4. Click **Step** to advance one time step, or **▶ Auto** to run continuously.
5. Click **🌐 Foxglove 3D** to open Foxglove Studio for rich 3D visualization of the swarm.
6. The **2D map** shows combined drone and target positions in real time.
7. **Metric tiles** show step reward, cumulative reward, coverage, battery, collisions, and targets tracked.
8. **Charts** display reward, coverage, collisions, and battery over time.`,
  },
  {
    title: "Operate Mode – Swarm Tab",
    icon: "🛸",
    content: `The Swarm tab shows the drone formation and inter-drone relationships:

- **Connectivity metrics**: Number of nodes, edges, and the algebraic connectivity λ₂ (Fiedler value).
- **2D Formation plot**: Scatter chart of drone positions colored by drone ID.
- **Formation error**: How far the swarm deviates from the desired formation over time.
- **Distance heatmap**: Matrix of pairwise drone distances. Red cells indicate violations of the safe distance.`,
  },
  {
    title: "Operate Mode – Targets Tab",
    icon: "🎯",
    content: `The Targets tab shows detected targets and their classification:

- **Summary tiles**: Total targets, hostile, friendly, and unknown counts.
- **Target map**: 2D scatter plot color-coded by classification (red = hostile, green = friendly, yellow = unknown).
- **Target details table**: ID, classification, confidence level, threat level, and position for each target.`,
  },
  {
    title: "Train Mode – Training Tab",
    icon: "🧪",
    content: `1. **Launch Training**: Set episodes, steps per episode, seed, device (CPU/CUDA), and task type, then click **Train**.
2. **Hyperparameter Sweep**: Configure the number of trials and launch a grid search.
3. **Training Runs**: View a list of completed runs, select one to see learning curves (reward, coverage, losses).
4. **Compare**: Check multiple runs and click **Compare** to overlay their learning curves.
5. **Reward Components**: Breakdown of reward into coverage, energy efficiency, and collision penalties.
6. **Statistics**: Averaged metrics from the last 100 episodes.`,
  },
  {
    title: "Sidebar – Configuration",
    icon: "⚙️",
    content: `The sidebar displays all loaded configuration parameters:

- **Drone Specs**: Max velocity, acceleration, yaw rate, battery capacity, mass.
- **Sensors**: Control frequency, radar/optical/RF ranges and update rates.
- **Mission**: Grid cell size, coverage radius/goal, communication radius, separation limits.
- **DMPC Controller**: Prediction/control horizons, constraint tightening, solver settings.
- **Learning**: Algorithm, learning rates, discount factor, batch/buffer sizes.
- **Reward Weights**: Coverage, energy, collision, formation, communication weights.

Select a different YAML config file from the dropdown to load alternative parameter sets.`,
  },
  {
    title: "3D Visualization with Foxglove",
    icon: "🌐",
    content: `Foxglove Studio provides advanced 3D visualization of the drone swarm:

1. Click the **🌐 Foxglove 3D** button in the Mission tab to open Foxglove Studio.
2. Connect to the WebSocket server at **ws://localhost:8765** (configured in foxglove_config.yaml).
3. Available topics:
   - **/swarm/scene** – 3D markers for drones and targets
   - **/swarm/metrics** – Real-time performance data
   - **/mission/coverage** – Coverage grid overlay
   - **/mission/info** – Mission status updates
4. Recorded sessions are saved as MCAP files in **data/recordings/** for offline playback.`,
  },
  {
    title: "Tips & Troubleshooting",
    icon: "💡",
    content: `- **Values persist** across tab and mode switches – your settings are saved automatically.
- If the simulation seems stuck, check the **Status Bar** at the top for error indicators.
- Use the **Math tab** to understand the physics and RL formulations behind the system.
- All configuration files are in the **config/** directory and can be edited via the API or YAML.
- Training logs are stored in **data/training_logs/** and models in **data/trained_models/**.`,
  },
];

export default function HelpView() {
  const [expanded, setExpanded] = useState(SECTIONS.map(() => true));

  const toggle = (idx) => {
    setExpanded((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <h3>❓ Dashboard Help Guide</h3>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
          Welcome to the ISR-RL-DMPC Dashboard. This guide explains each section and how to use the system effectively.
          Click on any section below to expand or collapse it.
        </p>
      </div>

      {SECTIONS.map((sec, idx) => (
        <div className="card help-section" key={idx}>
          <h3
            onClick={() => toggle(idx)}
            style={{ cursor: "pointer", userSelect: "none" }}
          >
            {sec.icon} {sec.title}
            <span style={{ marginLeft: "auto", fontSize: 14 }}>{expanded[idx] ? "▾" : "▸"}</span>
          </h3>
          {expanded[idx] && (
            <div className="help-body">
              {sec.content.split("\n").map((line, i) => {
                const trimmed = line.trim();
                if (!trimmed) return <br key={i} />;
                /* Render bold text wrapped in ** */
                const parts = trimmed.split(/\*\*(.*?)\*\*/g);
                return (
                  <p key={i} style={{ margin: "4px 0", fontSize: 12, lineHeight: 1.7, color: "var(--text-primary)" }}>
                    {parts.map((part, j) =>
                      j % 2 === 1 ? <strong key={j} style={{ color: "var(--accent-cyan)" }}>{part}</strong> : part
                    )}
                  </p>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
