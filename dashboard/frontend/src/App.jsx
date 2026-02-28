import React, { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar.jsx";
import StatusBar from "./components/StatusBar.jsx";
import MissionView from "./components/MissionView.jsx";
import SwarmView from "./components/SwarmView.jsx";
import TargetsView from "./components/TargetsView.jsx";
import TrainingView from "./components/TrainingView.jsx";

const MODE_TABS = {
  operate: ["Mission", "Swarm", "Targets"],
  train: ["Training", "Mission"],
};

export default function App() {
  const [mode, setMode] = useState("operate"); // "operate" | "train"
  const [activeTab, setActiveTab] = useState("Mission");
  const [config, setConfig] = useState(null);
  const [missionStatus, setMissionStatus] = useState({ active: false });
  const [trainingStatus, setTrainingStatus] = useState({ status: "idle" });

  /* When mode changes, switch to first tab of that mode */
  useEffect(() => {
    const tabs = MODE_TABS[mode] || MODE_TABS.operate;
    if (!tabs.includes(activeTab)) {
      setActiveTab(tabs[0]);
    }
  }, [mode, activeTab]);

  /* ── Fetch helpers ── */
  const fetchJSON = useCallback(async (url) => {
    try {
      const r = await fetch(url);
      return r.ok ? await r.json() : null;
    } catch { return null; }
  }, []);

  /* ── Load initial data ── */
  useEffect(() => {
    fetchJSON("/api/config/default_config.yaml").then((d) => d && setConfig(d));
  }, [fetchJSON]);

  /* ── Poll status every 2 s ── */
  useEffect(() => {
    const id = setInterval(() => {
      fetchJSON("/api/mission/status").then((d) => d && setMissionStatus(d));
      fetchJSON("/api/training/status").then((d) => d && setTrainingStatus(d));
    }, 2000);
    return () => clearInterval(id);
  }, [fetchJSON]);

  const tabs = MODE_TABS[mode] || MODE_TABS.operate;

  const renderTab = () => {
    switch (activeTab) {
      case "Mission": return <MissionView status={missionStatus} config={config} mode={mode} />;
      case "Swarm":   return <SwarmView status={missionStatus} config={config} />;
      case "Targets": return <TargetsView status={missionStatus} />;
      case "Training": return <TrainingView status={trainingStatus} config={config} />;
      default: return null;
    }
  };

  return (
    <div className="app-layout" data-mode={mode}>
      <Sidebar config={config} setConfig={setConfig} mode={mode} setMode={setMode} />
      <StatusBar
        mode={mode}
        missionStatus={missionStatus}
        trainingStatus={trainingStatus}
      />
      <div className="main-canvas">
        <div className="tab-bar">
          {tabs.map((t) => (
            <button
              key={t}
              className={`tab-btn${activeTab === t ? " active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="tab-content">{renderTab()}</div>
      </div>
    </div>
  );
}
