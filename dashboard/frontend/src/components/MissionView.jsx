import React, { useState, useCallback, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area,
} from "recharts";

export default function MissionView({ status, config, mode }) {
  const cfgMission = config?.mission || {};
  const [simConfig, setSimConfig] = useState({
    num_drones: 4,
    max_targets: 10,
    mission_duration: 200,
  });
  const [stepHistory, setStepHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [autoStep, setAutoStep] = useState(false);

  /* Sync simConfig from loaded config */
  useEffect(() => {
    if (!config) return;
    setSimConfig((prev) => ({
      ...prev,
      num_drones: config?.drone?.num_drones ?? prev.num_drones,
      max_targets: config?.mission?.max_targets ?? prev.max_targets,
      mission_duration: config?.mission?.max_duration
        ? Math.round(config.mission.max_duration / (1.0 / (config?.sensor?.control_frequency || 50)))
        : prev.mission_duration,
    }));
  }, [config]);

  const postJSON = useCallback(async (url, body) => {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.ok ? r.json() : null;
  }, []);

  const handleReset = async () => {
    setStepHistory([]);
    setAutoStep(false);
    const data = await postJSON("/api/mission/reset", simConfig);
    if (data) setRunning(true);
  };

  const handleStep = useCallback(async () => {
    const data = await postJSON("/api/mission/step", {});
    if (data) {
      setStepHistory((prev) => [
        ...prev,
        {
          step: data.step_count,
          reward: data.reward,
          total: data.total_reward,
          coverage: data.info?.coverage ?? 0,
          battery: data.info?.avg_battery ?? 0,
          collisions: data.info?.collisions ?? 0,
        },
      ]);
      if (data.done) {
        setRunning(false);
        setAutoStep(false);
      }
    }
  }, [postJSON]);

  /* Auto-step loop */
  React.useEffect(() => {
    if (!autoStep) return;
    const id = setInterval(handleStep, 150);
    return () => clearInterval(id);
  }, [autoStep, handleStep]);

  const latest = stepHistory[stepHistory.length - 1] || {};

  return (
    <div>
      {/* Controls */}
      <div className="card">
        <h3>🎮 Simulation Control</h3>
        <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
          <label style={{ fontSize: 11 }}>
            Drones
            <input
              type="number" min={1} max={20}
              value={simConfig.num_drones}
              onChange={(e) => setSimConfig((s) => ({ ...s, num_drones: +e.target.value }))}
              style={{ display: "block", marginTop: 4 }}
            />
          </label>
          <label style={{ fontSize: 11 }}>
            Targets
            <input
              type="number" min={1} max={30}
              value={simConfig.max_targets}
              onChange={(e) => setSimConfig((s) => ({ ...s, max_targets: +e.target.value }))}
              style={{ display: "block", marginTop: 4 }}
            />
          </label>
          <label style={{ fontSize: 11 }}>
            Duration
            <input
              type="number" min={50} max={5000}
              value={simConfig.mission_duration}
              onChange={(e) => setSimConfig((s) => ({ ...s, mission_duration: +e.target.value }))}
              style={{ display: "block", marginTop: 4 }}
            />
          </label>
          <div className="btn-group">
            <button className="btn primary" onClick={handleReset}>Reset</button>
            <button className="btn success" onClick={handleStep} disabled={!running}>Step</button>
            <button
              className={`btn${autoStep ? " danger" : " success"}`}
              onClick={() => setAutoStep(!autoStep)}
              disabled={!running}
            >
              {autoStep ? "⏸ Pause" : "▶ Auto"}
            </button>
          </div>
        </div>
      </div>

      {/* Metric tiles */}
      <div className="metric-tiles">
        <Tile label="Step" value={latest.step ?? 0} />
        <Tile label="Reward" value={(latest.reward ?? 0).toFixed(2)} />
        <Tile label="Total Reward" value={(latest.total ?? 0).toFixed(2)} />
        <Tile label="Coverage" value={`${((latest.coverage ?? 0) * 100).toFixed(1)}%`} />
        <Tile label="Avg Battery" value={(latest.battery ?? 0).toFixed(0)} unit="Wh" />
        <Tile label="Collisions" value={latest.collisions ?? 0} />
      </div>

      {/* Charts */}
      <div className="card-grid">
        <div className="card">
          <h3>📈 Episode Reward</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stepHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Line type="monotone" dataKey="reward" stroke="#00e5ff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="total" stroke="#2979ff" strokeWidth={1} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>📊 Coverage Progress</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={stepHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis domain={[0, 1]} />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Area type="monotone" dataKey="coverage" stroke="#00e676" fill="rgba(0,230,118,0.15)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function Tile({ label, value, unit }) {
  return (
    <div className="metric-tile">
      <div className="metric-value">{value}{unit ? <span style={{ fontSize: 12, marginLeft: 2 }}>{unit}</span> : null}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
