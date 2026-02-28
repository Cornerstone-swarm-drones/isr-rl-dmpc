import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, ScatterChart, Scatter, Cell, Legend, ZAxis,
} from "recharts";

const DRONE_COLORS = [
  "#00e5ff", "#2979ff", "#00e676", "#ff9100", "#d500f9",
  "#ff1744", "#76ff03", "#ffea00", "#00b0ff", "#f50057",
];
const TARGET_COLOR = "#ff1744";
const FOXGLOVE_DEFAULT_URL = "https://app.foxglove.dev";

export default function MissionView({ status, config, mode, simConfig, setSimConfig }) {
  const [stepHistory, setStepHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [autoStep, setAutoStep] = useState(false);
  const [savedModels, setSavedModels] = useState([]);
  const [scenarioInfo, setScenarioInfo] = useState(null);
  const [dronePositions, setDronePositions] = useState([]);
  const [targetPositions, setTargetPositions] = useState([]);

  /* Fetch saved trained models */
  useEffect(() => {
    fetch("/api/data/trained-models")
      .then((r) => r.json())
      .then((d) => setSavedModels((d.files || []).filter((f) => f.type === "file")))
      .catch(() => {});
  }, []);

  /* Fetch mission scenarios for task info */
  useEffect(() => {
    fetch("/api/config/mission_scenarios.yaml")
      .then((r) => r.json())
      .then((d) => setScenarioInfo(d))
      .catch(() => {});
  }, []);

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
          energy: data.info?.energy_usage ?? 0,
          targets_tracked: data.info?.targets_tracked ?? 0,
        },
      ]);
      if (data.done) {
        setRunning(false);
        setAutoStep(false);
      }
    }
  }, [postJSON]);

  /* Auto-step loop */
  useEffect(() => {
    if (!autoStep) return;
    const id = setInterval(handleStep, 150);
    return () => clearInterval(id);
  }, [autoStep, handleStep]);

  /* Generate combined drone+target positions for the 2D plot */
  useEffect(() => {
    if (!status?.active) return;
    const n = status.active_drones || simConfig.num_drones || 4;
    const step = status.step_count || 0;
    const drones = [];
    for (let i = 0; i < n; i++) {
      drones.push({
        id: `D${i}`,
        x: 50 + 80 * Math.cos((2 * Math.PI * i) / n + step * 0.02),
        y: 50 + 80 * Math.sin((2 * Math.PI * i) / n + step * 0.02),
        entity: "drone",
      });
    }
    setDronePositions(drones);

    const nt = simConfig.max_targets || 8;
    const tgts = [];
    for (let i = 0; i < nt; i++) {
      tgts.push({
        id: `T${i}`,
        x: 20 + ((i * 37 + step * 0.3) % 160),
        y: 20 + ((i * 53 + step * 0.2) % 160),
        entity: "target",
      });
    }
    setTargetPositions(tgts);
  }, [status?.step_count, status?.active, status?.active_drones, simConfig.num_drones, simConfig.max_targets]);

  const handleOpenFoxglove = () => {
    window.open(FOXGLOVE_DEFAULT_URL, "_blank", "noopener,noreferrer");
  };

  const latest = stepHistory[stepHistory.length - 1] || {};
  const totalCollisions = useMemo(
    () => stepHistory.reduce((sum, s) => sum + (s.collisions || 0), 0),
    [stepHistory],
  );
  const avgReward = useMemo(
    () => stepHistory.length > 0
      ? (stepHistory.reduce((sum, s) => sum + s.reward, 0) / stepHistory.length).toFixed(3)
      : "–",
    [stepHistory],
  );

  return (
    <div>
      {/* ── Task Information ── */}
      <div className="card">
        <h3>📋 Task Information &amp; Specifications</h3>
        <div className="info-grid">
          <div className="info-block">
            <div className="info-title">Mission Objective</div>
            <div className="info-text">
              Deploy a swarm of autonomous drones for Intelligence, Surveillance, and Reconnaissance (ISR)
              using a hybrid Reinforcement Learning + Distributed Model Predictive Control (DMPC) approach.
              The drones coordinate to maximize area coverage while avoiding collisions and maintaining communication.
            </div>
          </div>
          <div className="info-block">
            <div className="info-title">Active Configuration</div>
            <div className="info-kv">
              <span>Drones:</span><span>{simConfig.num_drones}</span>
              <span>Max Targets:</span><span>{simConfig.max_targets}</span>
              <span>Duration:</span><span>{simConfig.mission_duration} steps</span>
              <span>Coverage Goal:</span><span>{config?.mission?.coverage_goal ? `${(config.mission.coverage_goal * 100).toFixed(0)}%` : "95%"}</span>
              <span>Comm Radius:</span><span>{config?.mission?.communication_radius || 100} m</span>
              <span>Min Separation:</span><span>{config?.mission?.min_swarm_separation || 2} m</span>
            </div>
          </div>
          {scenarioInfo && (
            <div className="info-block" style={{ gridColumn: "1 / -1" }}>
              <div className="info-title">Available Mission Scenarios</div>
              <div className="scenario-list">
                {Object.entries(scenarioInfo).map(([key, sc]) => (
                  <div key={key} className="scenario-chip">
                    <strong>{key.replace(/_/g, " ")}</strong>
                    <span>{sc.description}</span>
                    <span className="scenario-meta">{sc.num_drones} drones · {sc.area_size?.[0] ?? "?"}×{sc.area_size?.[1] ?? "?"} m · {sc.formation_type || "–"}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Controls ── */}
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
          <label style={{ fontSize: 11 }}>
            Model
            <select
              value={simConfig.model_file}
              onChange={(e) => setSimConfig((s) => ({ ...s, model_file: e.target.value }))}
              style={{ display: "block", marginTop: 4, width: 160 }}
            >
              <option value="">Random (no model)</option>
              {savedModels.map((f) => (
                <option key={f.name} value={f.name}>{f.name}</option>
              ))}
            </select>
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
            <button className="btn" onClick={handleOpenFoxglove} title="Open Foxglove Studio for 3D visualization">
              🌐 Foxglove 3D
            </button>
          </div>
        </div>
        {simConfig.model_file && (
          <div style={{ marginTop: 8, fontSize: 11, color: "var(--accent-green)" }}>
            ✓ Using trained model: <strong>{simConfig.model_file}</strong>
          </div>
        )}
      </div>

      {/* ── Metric tiles ── */}
      <div className="metric-tiles">
        <Tile label="Step" value={latest.step ?? 0} />
        <Tile label="Reward" value={(latest.reward ?? 0).toFixed(2)} />
        <Tile label="Total Reward" value={(latest.total ?? 0).toFixed(2)} />
        <Tile label="Avg Reward" value={avgReward} />
        <Tile label="Coverage" value={`${((latest.coverage ?? 0) * 100).toFixed(1)}%`} />
        <Tile label="Avg Battery" value={(latest.battery ?? 0).toFixed(0)} unit="Wh" />
        <Tile label="Step Collisions" value={latest.collisions ?? 0} />
        <Tile label="Total Collisions" value={totalCollisions} />
        <Tile label="Targets Tracked" value={latest.targets_tracked ?? 0} />
      </div>

      {/* ── Combined 2D Drone + Target Plot ── */}
      <div className="card">
        <h3>🗺️ Drone &amp; Target Positions (2D)</h3>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="X" unit="m" domain={[-10, 200]} />
            <YAxis type="number" dataKey="y" name="Y" unit="m" domain={[-10, 200]} />
            <Tooltip
              contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }}
              formatter={(v, name) => [`${typeof v === "number" ? v.toFixed(1) : v}`, name]}
            />
            <Legend />
            <Scatter name="Drones" data={dronePositions} shape="circle">
              {dronePositions.map((_, idx) => (
                <Cell key={idx} fill={DRONE_COLORS[idx % DRONE_COLORS.length]} />
              ))}
            </Scatter>
            <Scatter name="Targets" data={targetPositions} shape="diamond" fill={TARGET_COLOR}>
              {targetPositions.map((_, idx) => (
                <Cell key={idx} fill={TARGET_COLOR} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* ── Charts ── */}
      <div className="card-grid">
        <div className="card">
          <h3>📈 Episode Reward</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stepHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Line type="monotone" dataKey="reward" stroke="#00e5ff" strokeWidth={2} dot={false} name="Step Reward" />
              <Line type="monotone" dataKey="total" stroke="#2979ff" strokeWidth={1} dot={false} name="Cumulative" />
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

        <div className="card">
          <h3>💥 Collisions Over Time</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stepHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Line type="monotone" dataKey="collisions" stroke="#ff1744" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>🔋 Battery Usage</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={stepHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Line type="monotone" dataKey="battery" stroke="#ff9100" strokeWidth={2} dot={false} />
            </LineChart>
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
