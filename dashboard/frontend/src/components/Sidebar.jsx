import React, { useState, useEffect } from "react";

const FORMATION_TYPES = ["linear", "circular", "wedge", "grid"];

export default function Sidebar({ config, setConfig, mode, setMode }) {
  const [configs, setConfigs] = useState([]);
  const [selectedFile, setSelectedFile] = useState("default_config.yaml");
  const [learningCfg, setLearningCfg] = useState(null);

  useEffect(() => {
    fetch("/api/config/list")
      .then((r) => r.json())
      .then((d) => setConfigs(d.files || []))
      .catch(() => {});
    fetch("/api/config/learning_config.yaml")
      .then((r) => r.json())
      .then((d) => setLearningCfg(d))
      .catch(() => {});
  }, []);

  const loadConfig = async (fname) => {
    setSelectedFile(fname);
    try {
      const r = await fetch(`/api/config/${fname}`);
      if (r.ok) {
        const data = await r.json();
        setConfig(data);
      }
    } catch { /* ignore */ }
  };

  const drone = config?.drone || {};
  const sensor = config?.sensor || {};
  const mission = config?.mission || {};
  const learning = config?.learning || {};
  const dmpc = config?.dmpc || {};
  const rewardWeights = learningCfg?.reward_weights || {};
  const training = learningCfg?.training || {};

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>⬡ ISR-RL-DMPC</h1>
        <div className="subtitle">Mission + Training Control Center</div>
      </div>

      {/* Mode switch */}
      <div className="sidebar-section">
        <h3>Mode</h3>
        <div className="btn-group">
          <button className={`btn${mode === "operate" ? " primary" : ""}`} onClick={() => setMode("operate")}>
            Operate
          </button>
          <button className={`btn${mode === "train" ? " primary" : ""}`} onClick={() => setMode("train")}>
            Train
          </button>
        </div>
      </div>

      {/* Config file selector */}
      <div className="sidebar-section">
        <h3>Configuration</h3>
        <select
          style={{ width: "100%" }}
          value={selectedFile}
          onChange={(e) => loadConfig(e.target.value)}
        >
          {configs.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </div>

      {/* Drone params */}
      <div className="sidebar-section">
        <h3>Drone Specs</h3>
        <Field label="Max Velocity" value={drone.max_velocity} unit="m/s" />
        <Field label="Max Accel" value={drone.max_acceleration} unit="m/s²" />
        <Field label="Max Yaw Rate" value={drone.max_yaw_rate} unit="rad/s" />
        <Field label="Battery" value={drone.battery_capacity} unit="Wh" />
        <Field label="Mass" value={drone.mass} unit="kg" />
        <Field label="Max Angular Vel" value={drone.max_angular_velocity} unit="rad/s" />
      </div>

      {/* Sensor params */}
      <div className="sidebar-section">
        <h3>Sensors</h3>
        <Field label="Control Freq" value={sensor.control_frequency} unit="Hz" />
        <Field label="Radar Range" value={sensor.radar_range} unit="m" />
        <Field label="Radar Rate" value={sensor.radar_update_rate} unit="Hz" />
        <Field label="Optical FOV" value={sensor.optical_fov} unit="°" />
        <Field label="Optical Rate" value={sensor.optical_update_rate} unit="Hz" />
        <Field label="RF Range" value={sensor.rf_range} unit="m" />
        <Field label="RF Rate" value={sensor.rf_update_rate} unit="Hz" />
        <Field label="Max Radar Tgt" value={sensor.max_radar_targets} />
        <Field label="Max Optical Tgt" value={sensor.max_optical_targets} />
      </div>

      {/* Mission params */}
      <div className="sidebar-section">
        <h3>Mission</h3>
        <Field label="Grid Cell" value={mission.grid_cell_size} unit="m" />
        <Field label="Coverage Radius" value={mission.coverage_radius} unit="m" />
        <Field label="Comm Radius" value={mission.communication_radius} unit="m" />
        <Field label="Coverage Goal" value={mission.coverage_goal ? `${(mission.coverage_goal * 100).toFixed(0)}%` : "–"} />
        <Field label="Min Separation" value={mission.min_swarm_separation} unit="m" />
        <Field label="Max Spread" value={mission.max_swarm_spread} unit="m" />
      </div>

      {/* DMPC params */}
      <div className="sidebar-section">
        <h3>DMPC Controller</h3>
        <Field label="Prediction T" value={dmpc.prediction_horizon} />
        <Field label="Control T" value={dmpc.control_horizon} />
        <Field label="Receding Step" value={dmpc.receding_horizon_step} />
        <Field label="Tightening" value={dmpc.constraint_tightening} />
        <Field label="Max Iters" value={dmpc.solver_max_iterations} />
        <Field label="Tolerance" value={dmpc.solver_tolerance} />
      </div>

      {/* Learning params */}
      <div className="sidebar-section">
        <h3>Learning</h3>
        <Field label="Algorithm" value={training.algorithm || "SAC"} />
        <Field label="LR Critic" value={learning.learning_rate_critic} />
        <Field label="LR Actor" value={learning.learning_rate_actor} />
        <Field label="Gamma" value={learning.discount_factor} />
        <Field label="Batch Size" value={learning.batch_size} />
        <Field label="Buffer Size" value={learning.buffer_size} />
        <Field label="Target Update" value={learning.target_update_frequency} />
      </div>

      {/* Reward weights */}
      <div className="sidebar-section">
        <h3>Reward Weights</h3>
        <Field label="Coverage" value={rewardWeights.coverage ?? learning.weight_coverage} />
        <Field label="Energy" value={rewardWeights.energy ?? learning.weight_energy} />
        <Field label="Collision" value={rewardWeights.collision ?? learning.weight_collision} />
        <Field label="Target Engage" value={rewardWeights.target_engagement ?? learning.weight_target_engagement} />
        <Field label="Formation" value={rewardWeights.formation ?? learning.weight_formation} />
        <Field label="Communication" value={rewardWeights.communication} />
        <Field label="Revisit" value={rewardWeights.revisit} />
      </div>

      {/* Formation type */}
      <div className="sidebar-section">
        <h3>Formation</h3>
        <div className="field-row">
          <span className="field-label">Type</span>
          <select defaultValue="linear">
            {FORMATION_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
      </div>
    </aside>
  );
}

function Field({ label, value, unit }) {
  const display = value !== undefined && value !== null ? `${value}${unit ? ` ${unit}` : ""}` : "–";
  return (
    <div className="field-row">
      <span className="field-label">{label}</span>
      <span className="field-value">{display}</span>
    </div>
  );
}
