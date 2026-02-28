"""Tests for the ISR-RL-DMPC Dashboard backend API."""

from fastapi.testclient import TestClient

from dashboard.backend.app import app

client = TestClient(app)


# ── Config API ──────────────────────────────────────────────────────

class TestConfigAPI:
    """Tests for /api/config endpoints."""

    def test_list_configs(self):
        r = client.get("/api/config/list")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert isinstance(data["files"], list)
        assert "default_config.yaml" in data["files"]

    def test_read_default_config(self):
        r = client.get("/api/config/default_config.yaml")
        assert r.status_code == 200
        data = r.json()
        assert "drone" in data
        assert "sensor" in data
        assert "mission" in data
        assert "learning" in data
        assert "dmpc" in data

    def test_read_learning_config(self):
        r = client.get("/api/config/learning_config.yaml")
        assert r.status_code == 200
        data = r.json()
        assert "training" in data or "value_network" in data

    def test_read_nonexistent_config(self):
        r = client.get("/api/config/nonexistent.yaml")
        assert r.status_code == 404


# ── Mission API ─────────────────────────────────────────────────────

class TestMissionAPI:
    """Tests for /api/mission endpoints."""

    def test_mission_status_no_env(self):
        r = client.get("/api/mission/status")
        assert r.status_code == 200
        data = r.json()
        # May be active from previous test or not
        assert "active" in data

    def test_module_chain(self):
        r = client.get("/api/mission/module-chain")
        assert r.status_code == 200
        data = r.json()
        assert "modules" in data
        modules = data["modules"]
        assert "planner" in modules
        assert "formation" in modules
        assert "dmpc" in modules
        assert "attitude" in modules
        assert "reward" in modules

    def test_step_without_reset(self):
        """Step should fail when no environment is loaded."""
        # Reset the sim state first to ensure clean state
        from dashboard.backend.mission_api import _sim_state
        _sim_state["env"] = None
        _sim_state["done"] = False

        r = client.post("/api/mission/step")
        assert r.status_code == 400


# ── Training API ────────────────────────────────────────────────────

class TestTrainingAPI:
    """Tests for /api/training endpoints."""

    def test_training_status_idle(self):
        r = client.get("/api/training/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data

    def test_list_runs(self):
        r = client.get("/api/training/runs")
        assert r.status_code == 200
        data = r.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_get_nonexistent_run_metrics(self):
        r = client.get("/api/training/runs/nonexistent_run/metrics")
        assert r.status_code == 404

    def test_get_nonexistent_run_stats(self):
        r = client.get("/api/training/runs/nonexistent_run/stats")
        assert r.status_code == 404

    def test_stop_without_active_process(self):
        r = client.post("/api/training/stop")
        assert r.status_code == 400


# ── Data API ────────────────────────────────────────────────────────

class TestDataAPI:
    """Tests for /api/data endpoints."""

    def test_list_training_logs(self):
        r = client.get("/api/data/training-logs")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data

    def test_list_trained_models(self):
        r = client.get("/api/data/trained-models")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data

    def test_list_mission_results(self):
        r = client.get("/api/data/mission-results")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data

    def test_list_recordings(self):
        r = client.get("/api/data/recordings")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data

    def test_path_traversal_blocked(self):
        # HTTP clients normalise ".." so we use an encoded path
        r = client.get("/api/data/file/%2e%2e/%2e%2e/etc/passwd")
        assert r.status_code in (400, 404)

    def test_read_nonexistent_file(self):
        r = client.get("/api/data/file/nonexistent.txt")
        assert r.status_code == 404
