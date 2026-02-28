import React from "react";

export default function StatusBar({ mode, missionStatus, trainingStatus }) {
  const simTime = missionStatus?.step_count ?? 0;
  const controlHz = 50;
  const isActive = missionStatus?.active ?? false;
  const trStatus = trainingStatus?.status ?? "idle";

  return (
    <header className="topbar">
      <span className={`mode-label ${mode}`}>{mode === "operate" ? "OPERATE" : "TRAIN"}</span>

      <div className="status-item">
        <span className={`status-dot ${isActive ? "active" : "inactive"}`} />
        <span>SIM {isActive ? "LIVE" : "OFF"}</span>
      </div>

      <div className="status-item">
        <span>Loop: {controlHz} Hz</span>
      </div>

      <div className="status-item">
        <span>Step: {simTime}</span>
      </div>

      {missionStatus?.total_reward !== undefined && (
        <div className="status-item">
          <span>Reward: {missionStatus.total_reward.toFixed(2)}</span>
        </div>
      )}

      <div className="status-item">
        <span className={`status-dot ${trStatus === "running" ? "active" : trStatus === "completed" ? "warning" : "inactive"}`} />
        <span>Training: <span className={`badge ${trStatus}`}>{trStatus.toUpperCase()}</span></span>
      </div>
    </header>
  );
}
