import React, { useState, useEffect } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";

const CLASS_COLORS = { hostile: "#ff1744", friendly: "#00e676", unknown: "#ffea00", neutral: "#8899aa" };

export default function TargetsView({ status }) {
  const [targets, setTargets] = useState([]);
  const [classification, setClassification] = useState({ hostile: 0, friendly: 0, unknown: 0 });

  /* Generate demo target data when mission active */
  useEffect(() => {
    if (!status?.active) return;
    const classes = ["hostile", "friendly", "unknown", "neutral"];
    const n = 8;
    const tgts = [];
    const counts = { hostile: 0, friendly: 0, unknown: 0, neutral: 0 };
    for (let i = 0; i < n; i++) {
      const cls = classes[i % classes.length];
      counts[cls]++;
      tgts.push({
        id: `T${i}`,
        x: 20 + Math.random() * 160,
        y: 20 + Math.random() * 160,
        classification: cls,
        confidence: 0.5 + Math.random() * 0.5,
        threat_level: cls === "hostile" ? "HIGH" : cls === "unknown" ? "MEDIUM" : "LOW",
      });
    }
    setTargets(tgts);
    setClassification(counts);
  }, [status?.active]);

  return (
    <div>
      {/* Summary */}
      <div className="metric-tiles">
        <div className="metric-tile">
          <div className="metric-value">{targets.length}</div>
          <div className="metric-label">Total Targets</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value" style={{ color: "#ff1744" }}>{classification.hostile || 0}</div>
          <div className="metric-label">Hostile</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value" style={{ color: "#00e676" }}>{classification.friendly || 0}</div>
          <div className="metric-label">Friendly</div>
        </div>
        <div className="metric-tile">
          <div className="metric-value" style={{ color: "#ffea00" }}>{classification.unknown || 0}</div>
          <div className="metric-label">Unknown</div>
        </div>
      </div>

      <div className="card-grid">
        {/* Target scatter */}
        <div className="card">
          <h3>🎯 Target Map</h3>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="X" unit="m" domain={[0, 200]} />
              <YAxis type="number" dataKey="y" name="Y" unit="m" domain={[0, 200]} />
              <Tooltip
                contentStyle={{ background: "#1a2235", border: "1px solid #1e2d44" }}
                formatter={(v, name) => [typeof v === "number" ? v.toFixed(1) : v, name]}
              />
              <Scatter name="Targets" data={targets}>
                {targets.map((t, idx) => (
                  <Cell key={idx} fill={CLASS_COLORS[t.classification] || CLASS_COLORS.unknown} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Target table */}
        <div className="card">
          <h3>📋 Target Details</h3>
          <div style={{ overflowY: "auto", maxHeight: 320 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Class</th>
                  <th>Confidence</th>
                  <th>Threat</th>
                  <th>Position</th>
                </tr>
              </thead>
              <tbody>
                {targets.map((t) => (
                  <tr key={t.id}>
                    <td style={{ fontWeight: 600 }}>{t.id}</td>
                    <td>
                      <span
                        style={{
                          color: CLASS_COLORS[t.classification],
                          fontWeight: 700,
                          textTransform: "uppercase",
                          fontSize: 10,
                        }}
                      >
                        {t.classification}
                      </span>
                    </td>
                    <td>{(t.confidence * 100).toFixed(0)}%</td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background:
                            t.threat_level === "HIGH"
                              ? "rgba(255,23,68,0.15)"
                              : t.threat_level === "MEDIUM"
                                ? "rgba(255,145,0,0.15)"
                                : "rgba(0,230,118,0.15)",
                          color:
                            t.threat_level === "HIGH"
                              ? "#ff1744"
                              : t.threat_level === "MEDIUM"
                                ? "#ff9100"
                                : "#00e676",
                        }}
                      >
                        {t.threat_level}
                      </span>
                    </td>
                    <td style={{ fontSize: 10, color: "#8899aa" }}>
                      ({t.x.toFixed(0)}, {t.y.toFixed(0)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {!status?.active && (
        <div className="empty-state">
          Start a mission from the <strong>Mission</strong> tab to see target data.
        </div>
      )}
    </div>
  );
}
