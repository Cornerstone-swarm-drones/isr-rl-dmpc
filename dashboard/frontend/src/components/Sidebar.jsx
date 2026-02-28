import React, { useState, useEffect } from "react";

const FORMATION_TYPES = ["linear", "circular", "wedge", "grid"];

export default function Sidebar({ config, setConfig, mode, setMode }) {
  const [configs, setConfigs] = useState([]);
  const [selectedFile, setSelectedFile] = useState("default_config.yaml");

  useEffect(() => {
    fetch("/api/config/list")
      .then((r) => r.json())
      .then((d) => setConfigs(d.files || []))
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
      </div>

      {/* Sensor params */}
      <div className="sidebar-section">
        <h3>Sensors</h3>
        <Field label="Control Freq" value={sensor.control_frequency} unit="Hz" />
        <Field label="Radar Range" value={sensor.radar_range} unit="m" />
        <Field label="Optical FOV" value={sensor.optical_fov} unit="°" />
        <Field label="RF Range" value={sensor.rf_range} unit="m" />
      </div>

      {/* Mission params */}
      <div className="sidebar-section">
        <h3>Mission</h3>
        <Field label="Comm Radius" value={mission.communication_radius} unit="m" />
        <Field label="Grid Cell" value={mission.grid_cell_size} unit="m" />
        <Field label="Coverage Goal" value={mission.coverage_goal ? `${(mission.coverage_goal * 100).toFixed(0)}%` : "–"} />
        <Field label="Min Separation" value={mission.min_swarm_separation} unit="m" />
      </div>

      {/* DMPC params */}
      <div className="sidebar-section">
        <h3>DMPC Controller</h3>
        <Field label="Horizon T" value={dmpc.prediction_horizon} />
        <Field label="Control Horizon" value={dmpc.control_horizon} />
        <Field label="Tightening" value={dmpc.constraint_tightening} />
        <Field label="Max Iterations" value={dmpc.solver_max_iterations} />
      </div>

      {/* Learning params */}
      <div className="sidebar-section">
        <h3>Learning</h3>
        <Field label="LR Critic" value={learning.learning_rate_critic} />
        <Field label="LR Actor" value={learning.learning_rate_actor} />
        <Field label="Gamma" value={learning.discount_factor} />
        <Field label="Batch Size" value={learning.batch_size} />
        <Field label="Buffer Size" value={learning.buffer_size} />
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
