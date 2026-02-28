import React, { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar.jsx";
import StatusBar from "./components/StatusBar.jsx";
import MissionView from "./components/MissionView.jsx";
import SwarmView from "./components/SwarmView.jsx";
import TargetsView from "./components/TargetsView.jsx";
import TrainingView from "./components/TrainingView.jsx";

const TABS = ["Mission", "Swarm", "Targets", "Training"];

export default function App() {
  const [activeTab, setActiveTab] = useState("Mission");
  const [mode, setMode] = useState("operate"); // "operate" | "train"
  const [config, setConfig] = useState(null);
  const [missionStatus, setMissionStatus] = useState({ active: false });
  const [trainingStatus, setTrainingStatus] = useState({ status: "idle" });
  const [moduleChain, setModuleChain] = useState([]);

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
    fetchJSON("/api/mission/module-chain").then((d) => d && setModuleChain(d.modules || []));
  }, [fetchJSON]);

  /* ── Poll status every 2 s ── */
  useEffect(() => {
    const id = setInterval(() => {
      fetchJSON("/api/mission/status").then((d) => d && setMissionStatus(d));
      fetchJSON("/api/training/status").then((d) => d && setTrainingStatus(d));
    }, 2000);
    return () => clearInterval(id);
  }, [fetchJSON]);

  const renderTab = () => {
    switch (activeTab) {
      case "Mission": return <MissionView status={missionStatus} config={config} />;
      case "Swarm":   return <SwarmView status={missionStatus} config={config} />;
      case "Targets": return <TargetsView status={missionStatus} />;
      case "Training": return <TrainingView status={trainingStatus} />;
      default: return null;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar config={config} setConfig={setConfig} mode={mode} setMode={setMode} />
      <StatusBar
        mode={mode}
        missionStatus={missionStatus}
        trainingStatus={trainingStatus}
        moduleChain={moduleChain}
      />
      <div className="main-canvas">
        <div className="tab-bar">
          {TABS.map((t) => (
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
