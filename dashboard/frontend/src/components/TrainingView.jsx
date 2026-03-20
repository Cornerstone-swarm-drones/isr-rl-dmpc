import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts";

const CURVE_COLORS = ["#00e5ff", "#2979ff", "#00e676", "#ff9100", "#d500f9", "#ff1744", "#76ff03"];

export default function TrainingView({ status, config, trainCfg, setTrainCfg, sweepCfg, setSweepCfg }) {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [stats, setStats] = useState(null);
  const [compareIds, setCompareIds] = useState([]);
  const [compareData, setCompareData] = useState({});

  const fetchJSON = useCallback(async (url) => {
    try {
      const r = await fetch(url);
      return r.ok ? r.json() : null;
    } catch { return null; }
  }, []);

  const postJSON = useCallback(async (url, body) => {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return r.ok ? r.json() : null;
    } catch { return null; }
  }, []);

  /* Load runs */
  const loadRuns = useCallback(async () => {
    const d = await fetchJSON("/api/training/runs");
    if (d) setRuns(d.runs || []);
  }, [fetchJSON]);

  useEffect(() => {
    loadRuns();
    const id = setInterval(loadRuns, 5000);
    return () => clearInterval(id);
  }, [loadRuns]);

  /* Load selected run metrics */
  const selectRun = async (runId) => {
    setSelectedRun(runId);
    const [m, s] = await Promise.all([
      fetchJSON(`/api/training/runs/${runId}/metrics`),
      fetchJSON(`/api/training/runs/${runId}/stats`),
    ]);
    if (m) setMetrics(m.metrics || []);
    if (s) setStats(s);
  };

  /* Compare runs */
  const loadComparison = async () => {
    if (compareIds.length < 2) return;
    const d = await fetchJSON(`/api/training/compare?run_ids=${compareIds.join(",")}`);
    if (d) setCompareData(d.runs || {});
  };

  /* Training control */
  const startTraining = async () => {
    await postJSON("/api/training/start", trainCfg);
    setTimeout(loadRuns, 2000);
  };

  const stopTraining = async () => {
    await postJSON("/api/training/stop", {});
  };

  const startSweep = async () => {
    await postJSON("/api/training/sweep", sweepCfg);
    setTimeout(loadRuns, 2000);
  };

  const toggleCompare = (runId) => {
    setCompareIds((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]
    );
  };

  return (
    <div>
      {/* Training controls */}
      <div className="card-grid">
        <div className="card">
          <h3>🚀 Launch Training</h3>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
            <label style={{ fontSize: 11 }}>
              Episodes
              <input type="number" min={1} value={trainCfg.num_episodes}
                onChange={(e) => setTrainCfg((c) => ({ ...c, num_episodes: +e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              />
            </label>
            <label style={{ fontSize: 11 }}>
              Steps/Ep
              <input type="number" min={1} value={trainCfg.num_steps}
                onChange={(e) => setTrainCfg((c) => ({ ...c, num_steps: +e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              />
            </label>
            <label style={{ fontSize: 11 }}>
              Seed
              <input type="number" value={trainCfg.seed}
                onChange={(e) => setTrainCfg((c) => ({ ...c, seed: +e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              />
            </label>
            <label style={{ fontSize: 11 }}>
              Device
              <select value={trainCfg.device}
                onChange={(e) => setTrainCfg((c) => ({ ...c, device: e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              >
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
            <label style={{ fontSize: 11 }}>
              Task
              <select value={trainCfg.task}
                onChange={(e) => setTrainCfg((c) => ({ ...c, task: e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              >
                <option value="recon">Recon</option>
                <option value="intel">Intel</option>
                <option value="target_pursuit">Target Pursuit</option>
              </select>
            </label>
            <div className="btn-group">
              <button className="btn primary" onClick={startTraining}
                disabled={status?.status === "running"}>
                Train
              </button>
              <button className="btn danger" onClick={stopTraining}
                disabled={status?.status !== "running"}>
                Stop
              </button>
            </div>
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: "#8899aa" }}>
            Status: <span className={`badge ${status?.status || "idle"}`}>{(status?.status || "idle").toUpperCase()}</span>
            {status?.elapsed_seconds != null && <span style={{ marginLeft: 8 }}>({status.elapsed_seconds.toFixed(0)}s)</span>}
          </div>
        </div>

        <div className="card">
          <h3>🔬 Hyperparameter Sweep</h3>
          <div style={{ fontSize: 11, color: "#8899aa", marginBottom: 8 }}>
            Grid search over learning rates, γ, batch size, buffer size, reward weights.
          </div>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
            <label style={{ fontSize: 11 }}>
              Trials
              <input type="number" min={1} value={sweepCfg.num_trials}
                onChange={(e) => setSweepCfg((c) => ({ ...c, num_trials: +e.target.value }))}
                style={{ display: "block", marginTop: 4 }}
              />
            </label>
            <button className="btn primary" onClick={startSweep}
              disabled={status?.status === "running"}>
              Launch Sweep
            </button>
          </div>
        </div>
      </div>

      {/* Run list */}
      <div className="card">
        <h3>📁 Training Runs</h3>
        {runs.length === 0 ? (
          <div className="empty-state" style={{ padding: 16 }}>No runs yet. Start a training run above.</div>
        ) : (
          <div style={{ overflowY: "auto", maxHeight: 200 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Compare</th>
                  <th>Run ID</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id}>
                    <td>
                      <input type="checkbox" checked={compareIds.includes(r.run_id)}
                        onChange={() => toggleCompare(r.run_id)} />
                    </td>
                    <td style={{ fontWeight: selectedRun === r.run_id ? 700 : 400,
                      color: selectedRun === r.run_id ? "#00e5ff" : "inherit" }}>
                      {r.run_id}
                    </td>
                    <td>
                      <button className="btn" style={{ padding: "2px 10px", fontSize: 10 }}
                        onClick={() => selectRun(r.run_id)}>
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {compareIds.length >= 2 && (
          <div style={{ marginTop: 8 }}>
            <button className="btn success" onClick={loadComparison}>Compare {compareIds.length} Runs</button>
          </div>
        )}
      </div>

      {/* Selected run metrics */}
      {selectedRun && metrics.length > 0 && (
        <>
          <div className="card">
            <h3>📈 Learning Curves – {selectedRun}</h3>
            <div className="card-grid">
              <ChartPanel title="Episode Reward" data={metrics} dataKey="reward" color="#00e5ff" />
              <ChartPanel title="Coverage" data={metrics} dataKey="coverage" color="#00e676" />
              <ChartPanel title="Critic Loss" data={metrics} dataKey="critic_loss" color="#ff9100" />
              <ChartPanel title="Actor Loss" data={metrics} dataKey="actor_loss" color="#d500f9" />
            </div>
          </div>

          {/* Reward components */}
          <div className="card">
            <h3>🧩 Reward Components</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
                <Legend />
                <Line type="monotone" dataKey="reward" stroke="#00e5ff" strokeWidth={2} dot={false} name="Total" />
                <Line type="monotone" dataKey="coverage" stroke="#00e676" strokeWidth={1} dot={false} name="Coverage" />
                <Line type="monotone" dataKey="energy_eff" stroke="#ff9100" strokeWidth={1} dot={false} name="Energy" />
                <Line type="monotone" dataKey="collisions" stroke="#ff1744" strokeWidth={1} dot={false} name="Collisions" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Stats summary */}
      {stats && (
        <div className="card">
          <h3>📊 Run Statistics</h3>
          <div className="metric-tiles">
            <StatTile label="Avg Reward (last 100)"
              value={stats.episode_rewards ? (stats.episode_rewards.slice(-100).reduce((a, b) => a + b, 0) / Math.min(100, stats.episode_rewards.length)).toFixed(2) : "–"} />
            <StatTile label="Avg Coverage (last 100)"
              value={stats.coverage_efficiency ? `${(stats.coverage_efficiency.slice(-100).reduce((a, b) => a + b, 0) / Math.min(100, stats.coverage_efficiency.length) * 100).toFixed(1)}%` : "–"} />
            <StatTile label="Total Collisions"
              value={stats.collision_count ? stats.collision_count.reduce((a, b) => a + b, 0) : 0} />
            <StatTile label="Episodes" value={stats.episode_rewards ? stats.episode_rewards.length : 0} />
          </div>
        </div>
      )}

      {/* Compare runs view */}
      {Object.keys(compareData).length >= 2 && (
        <div className="card">
          <h3>📊 Compare Runs (Synchronized)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="episode" type="number" allowDuplicatedCategory={false} />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Legend />
              {Object.entries(compareData).map(([runId, data], idx) => {
                const m = data.metrics || [];
                return (
                  <Line
                    key={runId}
                    data={m}
                    type="monotone"
                    dataKey="reward"
                    stroke={CURVE_COLORS[idx % CURVE_COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    name={runId}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ChartPanel({ title, data, dataKey, color }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#8899aa", marginBottom: 4 }}>{title}</div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" />
          <YAxis />
          <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function StatTile({ label, value }) {
  return (
    <div className="metric-tile">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
