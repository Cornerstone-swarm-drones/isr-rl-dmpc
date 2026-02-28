import React, { useState, useEffect } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ZAxis, Cell,
  LineChart, Line,
} from "recharts";

const COLORS = [
  "#00e5ff", "#2979ff", "#00e676", "#ff9100", "#d500f9",
  "#ff1744", "#76ff03", "#ffea00", "#00b0ff", "#f50057",
];

export default function SwarmView({ status, config }) {
  const [dronePositions, setDronePositions] = useState([]);
  const [distanceMatrix, setDistanceMatrix] = useState([]);
  const [formationError, setFormationError] = useState([]);
  const [connectivity, setConnectivity] = useState({ lambda2: 0, edges: 0, nodes: 0 });

  /* Generate demo data when mission is active (real data would come from WS) */
  useEffect(() => {
    if (!status?.active) return;
    const n = status.active_drones || 4;
    const positions = [];
    for (let i = 0; i < n; i++) {
      positions.push({
        id: `D${i}`,
        x: 50 + 80 * Math.cos((2 * Math.PI * i) / n + (status.step_count || 0) * 0.02),
        y: 50 + 80 * Math.sin((2 * Math.PI * i) / n + (status.step_count || 0) * 0.02),
        z: 20 + 5 * Math.sin((status.step_count || 0) * 0.05 + i),
        drone: i,
      });
    }
    setDronePositions(positions);

    // Distance matrix
    const dists = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const dx = positions[i].x - positions[j].x;
        const dy = positions[i].y - positions[j].y;
        dists.push({ i, j, dist: Math.sqrt(dx * dx + dy * dy) });
      }
    }
    setDistanceMatrix(dists);

    // Formation error over time
    setFormationError((prev) => {
      const err = 2 + Math.random() * 3 * Math.exp(-0.01 * (status.step_count || 0));
      const next = [...prev, { step: status.step_count || prev.length, error: err }];
      return next.slice(-200);
    });

    // Connectivity
    const commRadius = config?.mission?.communication_radius || 100;
    let edges = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = positions[i].x - positions[j].x;
        const dy = positions[i].y - positions[j].y;
        if (Math.sqrt(dx * dx + dy * dy) <= commRadius) edges++;
      }
    }
    setConnectivity({
      lambda2: edges > 0 ? (0.5 + Math.random() * 1.5).toFixed(3) : "0.000",
      edges,
      nodes: n,
    });
  }, [status?.step_count, status?.active, status?.active_drones, config]);

  const safeDistance = config?.mission?.min_swarm_separation || 2;
  const n = dronePositions.length || 4;

  return (
    <div>
      {/* Connectivity widget */}
      <div className="metric-tiles">
        <div className="metric-tile">
          <div className="metric-value">{connectivity.nodes}</div>
          <div className="metric-label">Nodes</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value">{connectivity.edges}</div>
          <div className="metric-label">Edges</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value" style={{ color: connectivity.lambda2 > 0 ? "#00e676" : "#ff1744" }}>
            {connectivity.lambda2}
          </div>
          <div className="metric-label">λ₂ (Connectivity)</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value">{safeDistance} m</div>
          <div className="metric-label">Safe Distance</div>
        </div>
      </div>

      <div className="card-grid">
        {/* 2D Swarm Plot */}
        <div className="card">
          <h3>🛸 Swarm Formation (2D)</h3>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="X" unit="m" domain={[-50, 200]} />
              <YAxis type="number" dataKey="y" name="Y" unit="m" domain={[-50, 200]} />
              <ZAxis dataKey="z" range={[80, 200]} name="Alt" unit="m" />
              <Tooltip
                contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }}
                formatter={(v, name) => [`${v.toFixed(1)}`, name]}
              />
              <Scatter name="Drones" data={dronePositions} shape="circle">
                {dronePositions.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Formation error */}
        <div className="card">
          <h3>📉 Formation Error vs Time</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={formationError}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }} />
              <Line type="monotone" dataKey="error" stroke="#ff9100" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Inter-drone distance heatmap */}
      <div className="card">
        <h3>🔥 Inter-Drone Distance Heatmap</h3>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table" style={{ width: "auto" }}>
            <thead>
              <tr>
                <th></th>
                {Array.from({ length: n }, (_, i) => (
                  <th key={i} style={{ textAlign: "center" }}>D{i}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: n }, (_, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600, color: "#00e5ff" }}>D{i}</td>
                  {Array.from({ length: n }, (_, j) => {
                    const cell = distanceMatrix.find((d) => d.i === i && d.j === j);
                    const dist = cell ? cell.dist : 0;
                    const danger = dist > 0 && dist < safeDistance;
                    const hue = i === j ? 0 : Math.min(dist / 200, 1);
                    const bg = i === j
                      ? "transparent"
                      : danger
                        ? "rgba(255,23,68,0.4)"
                        : `rgba(0,229,255,${0.1 + hue * 0.3})`;
                    return (
                      <td key={j} style={{ textAlign: "center", background: bg, fontSize: 10 }}>
                        {i === j ? "–" : dist.toFixed(1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {!status?.active && (
        <div className="empty-state">
          Start a mission from the <strong>Mission</strong> tab to see live swarm data.
        </div>
      )}
    </div>
  );
}
