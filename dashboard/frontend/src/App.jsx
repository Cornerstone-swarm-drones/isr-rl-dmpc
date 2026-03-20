import React, { useState, useEffect, useCallback, useRef } from "react";
import Sidebar from "./components/Sidebar.jsx";
import StatusBar from "./components/StatusBar.jsx";
import MissionView from "./components/MissionView.jsx";
import SwarmView from "./components/SwarmView.jsx";
import TargetsView from "./components/TargetsView.jsx";
import TrainingView from "./components/TrainingView.jsx";
import HelpView from "./components/HelpView.jsx";
import MathView from "./components/MathView.jsx";

const MODE_TABS = {
  operate: ["Mission", "Swarm", "Targets", "Help", "Math"],
  train: ["Training", "Mission", "Help", "Math"],
};

/* ── localStorage helpers for state persistence ── */
function loadPersisted(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}
function savePersisted(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}

export default function App() {
  const [mode, setMode] = useState(() => loadPersisted("isr_mode", "operate"));
  const [activeTab, setActiveTab] = useState(() => loadPersisted("isr_activeTab", "Mission"));
  const [config, setConfig] = useState(null);
  const [missionStatus, setMissionStatus] = useState({ active: false });
  const [trainingStatus, setTrainingStatus] = useState({ status: "idle" });

  /* Persisted simulation config – survives tab/mode switches */
  const [simConfig, setSimConfig] = useState(() =>
    loadPersisted("isr_simConfig", { num_drones: 4, max_targets: 10, mission_duration: 200, model_file: "" })
  );

  /* Persisted training config */
  const [trainCfg, setTrainCfg] = useState(() =>
    loadPersisted("isr_trainCfg", { num_episodes: 100, num_steps: 500, seed: 42, device: "cpu", task: "recon" })
  );

  /* Persisted sweep config */
  const [sweepCfg, setSweepCfg] = useState(() =>
    loadPersisted("isr_sweepCfg", { num_trials: 10, device: "cpu" })
  );

  /* Persist state changes */
  useEffect(() => { savePersisted("isr_mode", mode); }, [mode]);
  useEffect(() => { savePersisted("isr_activeTab", activeTab); }, [activeTab]);
  useEffect(() => { savePersisted("isr_simConfig", simConfig); }, [simConfig]);
  useEffect(() => { savePersisted("isr_trainCfg", trainCfg); }, [trainCfg]);
  useEffect(() => { savePersisted("isr_sweepCfg", sweepCfg); }, [sweepCfg]);

  /* When mode changes, switch to first tab of that mode if current tab not valid */
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
      case "Mission":
        return (
          <MissionView
            status={missionStatus}
            config={config}
            mode={mode}
            simConfig={simConfig}
            setSimConfig={setSimConfig}
          />
        );
      case "Swarm":   return <SwarmView status={missionStatus} config={config} />;
      case "Targets": return <TargetsView status={missionStatus} />;
      case "Training":
        return (
          <TrainingView
            status={trainingStatus}
            config={config}
            trainCfg={trainCfg}
            setTrainCfg={setTrainCfg}
            sweepCfg={sweepCfg}
            setSweepCfg={setSweepCfg}
          />
        );
      case "Help": return <HelpView />;
      case "Math": return <MathView />;
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
