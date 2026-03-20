import React from "react";

export default function StatusBar({ mode, missionStatus, trainingStatus }) {
  const simTime = missionStatus?.step_count ?? 0;
  const isActive = missionStatus?.active ?? false;
  const trStatus = trainingStatus?.status ?? "idle";

  // Live training progress: prefer training API progress when training
  const displayStep = trStatus === "running" || trStatus === "completed"
    ? (trainingStatus?.current_step ?? simTime)
    : simTime;
  const displayLoop = trStatus === "running" || trStatus === "completed"
    ? `Ep ${trainingStatus?.current_episode ?? 0}`
    : "50 Hz";

  return (
    <header className="topbar">
      <span className={`mode-label ${mode}`}>{mode === "operate" ? "OPERATE" : "TRAIN"}</span>

      <div className="status-item">
        <span className={`status-dot ${isActive ? "active" : "inactive"}`} />
        <span>SIM {isActive ? "LIVE" : "OFF"}</span>
      </div>

      <div className="status-item">
        <span>Loop: {displayLoop}</span>
      </div>

      <div className="status-item">
        <span>Step: {displayStep}</span>
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
