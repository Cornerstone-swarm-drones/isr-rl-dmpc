# ISR-RL-DMPC Dashboard

A web-based mission + training control center for the ISR-RL-DMPC autonomous swarm system.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn

# Build the frontend (requires Node.js)
cd dashboard/frontend
npm install
npm run build
cd ../..

# Launch the dashboard
python scripts/launch_dashboard.py
```

Open **http://127.0.0.1:8000** in your browser.

### Development Mode

```bash
python scripts/launch_dashboard.py --dev
```

This starts both the FastAPI backend (port 8000) and the Vite dev server (port 5173)
with hot-reload. Open **http://localhost:5173** for development.

## Architecture

```
dashboard/
├── backend/
│   ├── app.py            # FastAPI application
│   ├── config_api.py     # YAML config CRUD (/api/config/*)
│   ├── mission_api.py    # Gym env control  (/api/mission/*)
│   ├── training_api.py   # Training launch  (/api/training/*)
│   └── data_api.py       # Data browsing    (/api/data/*)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── Sidebar.jsx       # Config panel
    │       ├── StatusBar.jsx     # Top status bar
    │       ├── MissionView.jsx   # Operate/simulate
    │       ├── SwarmView.jsx     # Formation & connectivity
    │       ├── TargetsView.jsx   # Target classification
    │       └── TrainingView.jsx  # Train & analyze
    └── package.json
```

## Features

### Operate / Simulate Mode
- **Mission View**: Reset/step/auto-step the ISRGridEnv, live reward & coverage charts,
  per-step module chain visualization (planner → formation → DMPC → attitude → sensors/EKF
  → classification → threat/alloc → reward)
- **Swarm View**: 2D formation scatter plot, algebraic connectivity λ₂ widget,
  inter-drone distance heatmap vs safe distance, formation error vs time
- **Targets View**: Target map with classification colors, threat level badges,
  confidence scores

### Train / Analyze Mode
- **Training View**: Launch training runs, hyperparameter sweep launcher,
  run-centric metrics viewer with learning curves (reward, coverage, critic/actor loss),
  reward component breakdown, compare multiple runs with synchronized x-axes

### Sidebar
Configuration loaded from YAML files: drone specs (max velocity, accel, yaw rate,
battery), sensor params (radar range, optical FOV, RF range), mission params
(comm radius, grid cell size, coverage goal), DMPC controller settings
(horizon, tightening), learning hyperparameters (LR, gamma, batch size),
and formation type selector.

### Data Storage
Persists artifacts in the existing project structure:
- `data/training_logs/` – metrics CSV + training stats JSON
- `data/trained_models/` – model checkpoints
- `data/mission_results/` – recorded missions

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/config/list` | GET | List YAML config files |
| `/api/config/{filename}` | GET | Read config as JSON |
| `/api/config/{filename}` | PUT | Update config from JSON |
| `/api/mission/reset` | POST | Create & reset ISRGridEnv |
| `/api/mission/step` | POST | Advance simulation one step |
| `/api/mission/status` | GET | Current simulation status |
| `/api/mission/module-chain` | GET | Module processing chain |
| `/api/training/start` | POST | Launch training subprocess |
| `/api/training/stop` | POST | Stop active training |
| `/api/training/status` | GET | Training process status |
| `/api/training/sweep` | POST | Launch hyperparameter sweep |
| `/api/training/runs` | GET | List training run directories |
| `/api/training/runs/{id}/metrics` | GET | Run metrics (CSV → JSON) |
| `/api/training/runs/{id}/stats` | GET | Run statistics JSON |
| `/api/training/compare` | GET | Compare multiple runs |
| `/api/data/training-logs` | GET | Browse training logs |
| `/api/data/trained-models` | GET | Browse trained models |
| `/api/data/mission-results` | GET | Browse mission results |
| `/api/data/file/{path}` | GET | Read data file contents |
