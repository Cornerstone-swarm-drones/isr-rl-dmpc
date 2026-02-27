"""
Tests for Foxglove Studio integration: FoxgloveBridge and MCAPRecorder.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from isr_rl_dmpc.utils.foxglove_bridge import (
    FoxgloveBridge,
    _timestamp_obj,
    _color,
    _vec3,
    _quat,
    _now_ns,
    _SCENE_UPDATE_SCHEMA,
    extract_targets_from_obs,
    get_drone_model_data,
    get_target_model_data,
    DRONE_MODEL_PATH,
    TARGET_MODEL_PATHS,
)
from isr_rl_dmpc.utils.mcap_logger import MCAPRecorder
from isr_rl_dmpc.core.data_structures import DroneState, TargetState, MissionState
from isr_rl_dmpc.core.state_manager import StateManager


# ---------------------------------------------------------------------------
# Helper utilities tests
# ---------------------------------------------------------------------------

class TestHelperUtilities:
    """Tests for bridge helper functions."""

    def test_timestamp_obj_conversion(self):
        """Timestamp nanoseconds correctly split into sec/nsec."""
        ts = _timestamp_obj(1_500_000_000)
        assert ts["sec"] == 1
        assert ts["nsec"] == 500_000_000

    def test_timestamp_obj_zero(self):
        """Zero timestamp produces zero sec/nsec."""
        ts = _timestamp_obj(0)
        assert ts["sec"] == 0
        assert ts["nsec"] == 0

    def test_color_dict(self):
        """_color creates RGBA dict."""
        c = _color(0.1, 0.2, 0.3, 0.4)
        assert c == {"r": 0.1, "g": 0.2, "b": 0.3, "a": 0.4}

    def test_vec3_dict(self):
        """_vec3 creates xyz dict."""
        v = _vec3(1.0, 2.0, 3.0)
        assert v == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_quat_dict(self):
        """_quat creates xyzw dict."""
        q = _quat(0.1, 0.2, 0.3, 0.4)
        assert q == {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4}

    def test_now_ns_returns_positive(self):
        """_now_ns returns a positive nanosecond timestamp."""
        ns = _now_ns()
        assert ns > 0
        assert isinstance(ns, int)


# ---------------------------------------------------------------------------
# FoxgloveBridge tests (without actual server start)
# ---------------------------------------------------------------------------

class TestFoxgloveBridgeInit:
    """Test FoxgloveBridge initialization and properties."""

    def test_default_init(self):
        """Bridge initializes with default parameters."""
        bridge = FoxgloveBridge()
        assert bridge.host == "0.0.0.0"
        assert bridge.port == 8765
        assert bridge.server_name == "ISR-RL-DMPC Simulation"
        assert bridge.is_running is False

    def test_custom_init(self):
        """Bridge accepts custom parameters."""
        bridge = FoxgloveBridge(host="127.0.0.1", port=9999, server_name="Test")
        assert bridge.host == "127.0.0.1"
        assert bridge.port == 9999
        assert bridge.server_name == "Test"

    def test_publish_scene_before_start_no_error(self):
        """Publishing before start does not raise an error."""
        bridge = FoxgloveBridge()
        positions = np.array([[0, 0, 50], [100, 100, 50]])
        # Should silently return without error
        bridge.publish_scene(drone_positions=positions)

    def test_publish_metrics_before_start_no_error(self):
        """Publishing metrics before start does not raise."""
        bridge = FoxgloveBridge()
        bridge.publish_metrics({"step": 0, "coverage": 0.5})

    def test_publish_coverage_before_start_no_error(self):
        """Publishing coverage before start does not raise."""
        bridge = FoxgloveBridge()
        bridge.publish_coverage(np.zeros(100), (10, 10))

    def test_publish_mission_info_before_start_no_error(self):
        """Publishing mission info before start does not raise."""
        bridge = FoxgloveBridge()
        bridge.publish_mission_info(elapsed_time=10.0, mission_duration=3600.0)


# ---------------------------------------------------------------------------
# MCAPRecorder tests
# ---------------------------------------------------------------------------

class TestMCAPRecorderInit:
    """Test MCAPRecorder initialization."""

    def test_default_init(self):
        """Recorder initializes with default path."""
        recorder = MCAPRecorder()
        assert recorder.filepath == Path("recording.mcap")
        assert recorder.is_recording is False

    def test_custom_path(self):
        """Recorder accepts custom path."""
        recorder = MCAPRecorder("/tmp/test.mcap")
        assert recorder.filepath == Path("/tmp/test.mcap")


class TestMCAPRecorderRecording:
    """Test MCAPRecorder recording functionality."""

    def test_start_stop(self, tmp_path):
        """Recorder can start and stop cleanly."""
        filepath = str(tmp_path / "test.mcap")
        recorder = MCAPRecorder(filepath)
        recorder.start()
        assert recorder.is_recording is True
        recorder.stop()
        assert recorder.is_recording is False
        assert Path(filepath).exists()

    def test_context_manager(self, tmp_path):
        """Recorder works as context manager."""
        filepath = str(tmp_path / "test_ctx.mcap")
        with MCAPRecorder(filepath) as recorder:
            assert recorder.is_recording is True
        assert recorder.is_recording is False
        assert Path(filepath).exists()

    def test_record_scene(self, tmp_path):
        """Recording scene data produces a valid MCAP file."""
        filepath = str(tmp_path / "scene.mcap")
        with MCAPRecorder(filepath) as recorder:
            positions = np.array([[10.0, 20.0, 50.0], [100.0, 200.0, 60.0]])
            batteries = np.array([4500.0, 3000.0])
            recorder.record_scene(
                drone_positions=positions,
                drone_batteries=batteries,
                timestamp_ns=int(time.time() * 1e9),
            )
        assert Path(filepath).stat().st_size > 0

    def test_record_scene_with_targets(self, tmp_path):
        """Recording scene with targets produces valid output."""
        filepath = str(tmp_path / "scene_targets.mcap")
        with MCAPRecorder(filepath) as recorder:
            positions = np.array([[10.0, 20.0, 50.0]])
            target_positions = {
                "T0": np.array([50.0, 50.0, 80.0]),
                "T1": np.array([150.0, 150.0, 90.0]),
            }
            target_classifications = {"T0": "hostile", "T1": "friendly"}
            recorder.record_scene(
                drone_positions=positions,
                target_positions=target_positions,
                target_classifications=target_classifications,
                timestamp_ns=int(time.time() * 1e9),
            )
        assert Path(filepath).stat().st_size > 0

    def test_record_metrics(self, tmp_path):
        """Recording metrics works correctly."""
        filepath = str(tmp_path / "metrics.mcap")
        with MCAPRecorder(filepath) as recorder:
            info = {
                "step": 42,
                "coverage": 0.75,
                "avg_battery": 3500.0,
                "active_drones": 4,
                "total_drones": 4,
                "collisions": 0,
            }
            recorder.record_metrics(info, reward=0.5, timestamp_ns=int(time.time() * 1e9))
        assert Path(filepath).stat().st_size > 0

    def test_record_coverage(self, tmp_path):
        """Recording coverage grid works correctly."""
        filepath = str(tmp_path / "coverage.mcap")
        with MCAPRecorder(filepath) as recorder:
            coverage = np.zeros(400)
            coverage[:200] = 1.0
            recorder.record_coverage(coverage, (20, 20), timestamp_ns=int(time.time() * 1e9))
        assert Path(filepath).stat().st_size > 0

    def test_record_mission_info(self, tmp_path):
        """Recording mission info works correctly."""
        filepath = str(tmp_path / "mission.mcap")
        with MCAPRecorder(filepath) as recorder:
            recorder.record_mission_info(
                elapsed_time=120.0,
                mission_duration=3600.0,
                coverage_efficiency=0.85,
                num_waypoints=10,
                timestamp_ns=int(time.time() * 1e9),
            )
        assert Path(filepath).stat().st_size > 0

    def test_record_before_start_no_error(self, tmp_path):
        """Recording before start silently does nothing."""
        recorder = MCAPRecorder(str(tmp_path / "nostart.mcap"))
        # Should not raise
        recorder.record_metrics({"step": 0}, reward=0.0)

    def test_mcap_file_readable(self, tmp_path):
        """Recorded MCAP file can be read back."""
        from mcap.reader import make_reader

        filepath = str(tmp_path / "readable.mcap")
        with MCAPRecorder(filepath) as recorder:
            for i in range(5):
                positions = np.random.randn(3, 3) * 100
                recorder.record_scene(
                    drone_positions=positions,
                    timestamp_ns=int(time.time() * 1e9) + i * 1_000_000,
                )
                recorder.record_metrics(
                    {"step": i, "coverage": i * 0.1},
                    reward=0.01,
                    timestamp_ns=int(time.time() * 1e9) + i * 1_000_000,
                )

        # Read back
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            assert summary is not None
            # Should have channels registered
            assert len(summary.channels) > 0

    def test_multi_step_recording(self, tmp_path):
        """Recording multiple steps creates a growing file."""
        filepath = str(tmp_path / "multi.mcap")
        with MCAPRecorder(filepath) as recorder:
            for step in range(20):
                ts = int(time.time() * 1e9) + step * 20_000_000  # 20ms intervals
                positions = np.random.randn(4, 3) * 100 + 500
                batteries = np.random.uniform(1000, 5000, size=4)
                recorder.record_scene(
                    drone_positions=positions,
                    drone_batteries=batteries,
                    timestamp_ns=ts,
                )
                recorder.record_metrics(
                    info={
                        "step": step,
                        "coverage": step / 20.0,
                        "avg_battery": float(np.mean(batteries)),
                        "active_drones": 4,
                        "total_drones": 4,
                        "collisions": 0,
                    },
                    reward=0.01 * (1.0 - step / 20.0),
                    timestamp_ns=ts,
                )
        assert Path(filepath).stat().st_size > 100


# ---------------------------------------------------------------------------
# Integration with StateManager
# ---------------------------------------------------------------------------

class TestStateManagerIntegration:
    """Test FoxgloveBridge integration with StateManager."""

    def test_publish_from_state_manager_no_error(self):
        """Publishing from StateManager does not raise (bridge not started)."""
        bridge = FoxgloveBridge()
        sm = StateManager(n_drones=3)

        # Populate states
        for i in range(3):
            sm.update_drone_state(i, DroneState(
                position=np.array([i * 100.0, 0.0, 50.0]),
                battery_energy=4000.0 - i * 500,
            ))

        sm.update_target_state("hostile_1", TargetState(
            position=np.array([200.0, 200.0, 80.0]),
            classification_confidence=-0.8,
            target_id="hostile_1",
        ))

        sm.update_mission_state(MissionState(
            elapsed_time=60.0,
            mission_duration=3600.0,
            coverage_efficiency=0.5,
        ))

        # Should not raise even though bridge is not started
        bridge.publish_from_state_manager(sm)


# ---------------------------------------------------------------------------
# extract_targets_from_obs tests
# ---------------------------------------------------------------------------

class TestExtractTargetsFromObs:
    """Tests for the extract_targets_from_obs helper."""

    def test_empty_targets(self):
        """All-zero observation returns empty dicts."""
        obs = np.zeros((5, 12))
        pos, cls = extract_targets_from_obs(obs)
        assert len(pos) == 0
        assert len(cls) == 0

    def test_single_hostile_target(self):
        """A single hostile target is extracted correctly."""
        obs = np.zeros((3, 12))
        obs[0, :3] = [100.0, 200.0, 50.0]
        obs[0, 8] = 2.0  # HOSTILE
        pos, cls = extract_targets_from_obs(obs)
        assert len(pos) == 1
        assert "T0" in pos
        np.testing.assert_array_almost_equal(pos["T0"], [100.0, 200.0, 50.0])
        assert cls["T0"] == "hostile"

    def test_multiple_target_types(self):
        """Multiple targets with different types."""
        obs = np.zeros((4, 12))
        obs[0, :3] = [10.0, 20.0, 30.0]
        obs[0, 8] = 1.0  # FRIENDLY
        obs[1, :3] = [40.0, 50.0, 60.0]
        obs[1, 8] = 2.0  # HOSTILE
        obs[2, :3] = [70.0, 80.0, 90.0]
        obs[2, 8] = 0.0  # UNKNOWN
        # obs[3] is all-zero -> should be skipped
        pos, cls = extract_targets_from_obs(obs)
        assert len(pos) == 3
        assert cls["T0"] == "friendly"
        assert cls["T1"] == "hostile"
        assert cls["T2"] == "unknown"

    def test_neutral_target(self):
        """Neutral target type (3) is recognized."""
        obs = np.zeros((1, 12))
        obs[0, :3] = [5.0, 5.0, 5.0]
        obs[0, 8] = 3.0  # NEUTRAL
        pos, cls = extract_targets_from_obs(obs)
        assert cls["T0"] == "neutral"

    def test_skips_zero_position(self):
        """Targets at origin with zero position are skipped (padding)."""
        obs = np.zeros((3, 12))
        obs[0, :3] = [0.0, 0.0, 0.0]  # zero -> skipped
        obs[0, 8] = 2.0
        obs[1, :3] = [100.0, 200.0, 50.0]
        obs[1, 8] = 1.0
        pos, cls = extract_targets_from_obs(obs)
        assert len(pos) == 1
        assert "T1" in pos


# ---------------------------------------------------------------------------
# Ground plane in scene tests
# ---------------------------------------------------------------------------

class TestGroundPlane:
    """Tests for the ground plane in publish_scene."""

    def _build_scene(self, bridge, **kwargs):
        """Capture the scene dict produced by publish_scene."""
        captured = {}

        def _capture(channel_key, data, timestamp_ns):
            captured[channel_key] = data

        bridge._send = _capture
        bridge._started = True
        bridge.publish_scene(**kwargs)
        bridge._started = False
        return captured.get("scene")

    def test_publish_scene_with_ground_plane_no_error(self):
        """Publishing scene with grid_extent does not raise."""
        bridge = FoxgloveBridge()
        positions = np.array([[0, 0, 50], [100, 100, 50]])
        bridge.publish_scene(
            drone_positions=positions,
            grid_extent=2000.0,
        )

    def test_publish_scene_without_ground_plane_no_error(self):
        """Publishing scene without grid_extent still works."""
        bridge = FoxgloveBridge()
        positions = np.array([[0, 0, 50]])
        bridge.publish_scene(drone_positions=positions)

    def test_ground_plane_centered_at_origin(self):
        """Ground plane cube must be positioned at (0, 0) when grid_extent is given."""
        bridge = FoxgloveBridge()
        positions = np.array([[1000.0, 1000.0, 50.0]])
        scene = self._build_scene(bridge, drone_positions=positions, grid_extent=2000.0)
        ground = next(e for e in scene["entities"] if e["id"] == "ground_plane")
        cube_pos = ground["cubes"][0]["pose"]["position"]
        assert cube_pos["x"] == 0.0
        assert cube_pos["y"] == 0.0

    def test_drone_positions_shifted_by_half(self):
        """Drone model positions must be shifted by -half when grid_extent is set."""
        bridge = FoxgloveBridge()
        grid_extent = 2000.0
        half = grid_extent / 2.0
        raw_pos = np.array([[1200.0, 800.0, 50.0]])
        scene = self._build_scene(
            bridge, drone_positions=raw_pos, grid_extent=grid_extent
        )
        drones = next(e for e in scene["entities"] if e["id"] == "drones")
        model_pos = drones["models"][0]["pose"]["position"]
        assert model_pos["x"] == pytest.approx(1200.0 - half)
        assert model_pos["y"] == pytest.approx(800.0 - half)
        assert model_pos["z"] == pytest.approx(50.0)

    def test_target_positions_shifted_by_half(self):
        """Target model positions must be shifted by -half when grid_extent is set."""
        bridge = FoxgloveBridge()
        grid_extent = 2000.0
        half = grid_extent / 2.0
        drone_pos = np.array([[1000.0, 1000.0, 50.0]])
        tgt_pos = {"T0": np.array([600.0, 400.0, 80.0])}
        tgt_cls = {"T0": "hostile"}
        scene = self._build_scene(
            bridge,
            drone_positions=drone_pos,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
            grid_extent=grid_extent,
        )
        targets = next(e for e in scene["entities"] if e["id"] == "targets")
        model_pos = targets["models"][0]["pose"]["position"]
        assert model_pos["x"] == pytest.approx(600.0 - half)
        assert model_pos["y"] == pytest.approx(400.0 - half)
        assert model_pos["z"] == pytest.approx(80.0)

    def test_no_shift_without_grid_extent(self):
        """Without grid_extent, drone positions are published as-is."""
        bridge = FoxgloveBridge()
        raw_pos = np.array([[300.0, 400.0, 50.0]])
        scene = self._build_scene(bridge, drone_positions=raw_pos)
        drones = next(e for e in scene["entities"] if e["id"] == "drones")
        model_pos = drones["models"][0]["pose"]["position"]
        assert model_pos["x"] == pytest.approx(300.0)
        assert model_pos["y"] == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Scene structure verification tests
# ---------------------------------------------------------------------------

class TestSceneStructure:
    """Verify the scene entity structure matches what Foxglove Studio expects."""

    def _build_scene(self, bridge, **kwargs):
        """Capture the scene dict by calling publish_scene on an unstarted bridge."""
        # Directly call the internal scene-building logic via the public method.
        # Since the bridge is not started, _send is a no-op, but we can
        # inspect the arguments it would send by monkey-patching _send.
        captured = {}

        def _capture(channel_key, data, timestamp_ns):
            captured[channel_key] = data

        bridge._send = _capture
        bridge._started = True  # allow _send to be called
        bridge.publish_scene(**kwargs)
        bridge._started = False
        return captured.get("scene")

    def test_drone_entity_has_models_and_texts(self):
        """Drones entity should contain models (3D) and text labels."""
        bridge = FoxgloveBridge()
        positions = np.array([[100.0, 200.0, 50.0], [300.0, 400.0, 60.0]])
        batteries = np.array([4000.0, 2000.0])
        scene = self._build_scene(
            bridge,
            drone_positions=positions,
            drone_batteries=batteries,
        )
        assert scene is not None
        drones_entity = next(e for e in scene["entities"] if e["id"] == "drones")

        # 3D models: 1 per drone
        assert len(drones_entity["models"]) == 2
        # No cubes, cylinders, or arrows (replaced by models)
        assert len(drones_entity["cubes"]) == 0
        assert len(drones_entity["cylinders"]) == 0
        assert len(drones_entity["arrows"]) == 0
        # Labels: 1 per drone
        assert len(drones_entity["texts"]) == 2

    def test_drone_model_has_required_fields(self):
        """Each drone model must have pose, scale, data, media_type, and color."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        scene = self._build_scene(bridge, drone_positions=positions)
        model = scene["entities"][0]["models"][0]
        assert "pose" in model
        assert "position" in model["pose"]
        assert "orientation" in model["pose"]
        assert "scale" in model
        assert "data" in model
        assert model["data"] != ""
        assert "media_type" in model
        assert "color" in model
        assert model["media_type"] == "model/gltf-binary"

    def test_target_entity_has_models_and_labels(self):
        """Targets should have 3D models and text labels."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        tgt_pos = {"T0": np.array([100.0, 100.0, 80.0])}
        tgt_cls = {"T0": "hostile"}
        scene = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
        )
        targets_entity = next(e for e in scene["entities"] if e["id"] == "targets")
        assert len(targets_entity["models"]) == 1
        assert len(targets_entity["spheres"]) == 0
        assert len(targets_entity["texts"]) == 1
        assert "hostile" in targets_entity["texts"][0]["text"]
        assert "data" in targets_entity["models"][0]
        assert targets_entity["models"][0]["data"] != ""
        assert "media_type" in targets_entity["models"][0]

    def test_ground_plane_entity_present(self):
        """Ground plane entity should be present when grid_extent is set."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        scene = self._build_scene(
            bridge, drone_positions=positions, grid_extent=2000.0,
        )
        ground = next(e for e in scene["entities"] if e["id"] == "ground_plane")
        assert len(ground["cubes"]) == 1

    def test_all_entities_have_required_fields(self):
        """Every entity must have the required Foxglove SceneEntity fields."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        tgt_pos = {"T0": np.array([100.0, 100.0, 80.0])}
        tgt_cls = {"T0": "friendly"}
        scene = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
            grid_extent=2000.0,
        )
        required = {
            "timestamp", "frame_id", "id", "lifetime", "frame_locked",
            "metadata", "arrows", "cubes", "spheres", "cylinders",
            "lines", "triangles", "texts", "models",
        }
        for entity in scene["entities"]:
            missing = required - set(entity.keys())
            assert not missing, f"Entity '{entity['id']}' missing fields: {missing}"

    def test_scene_mcap_roundtrip_with_targets(self, tmp_path):
        """Scene with drones, targets, and ground plane can be recorded and read back."""
        from mcap.reader import make_reader

        filepath = str(tmp_path / "scene_roundtrip.mcap")
        with MCAPRecorder(filepath) as recorder:
            positions = np.array([[100.0, 200.0, 50.0], [300.0, 400.0, 60.0]])
            batteries = np.array([4500.0, 3000.0])
            tgt_pos = {
                "T0": np.array([500.0, 500.0, 80.0]),
                "T1": np.array([600.0, 600.0, 90.0]),
            }
            tgt_cls = {"T0": "hostile", "T1": "friendly"}
            for step in range(5):
                ts = int(time.time() * 1e9) + step * 100_000_000
                recorder.record_scene(
                    drone_positions=positions,
                    drone_batteries=batteries,
                    target_positions=tgt_pos,
                    target_classifications=tgt_cls,
                    timestamp_ns=ts,
                )
                recorder.record_metrics(
                    {"step": step, "coverage": 0.5, "active_drones": 2,
                     "total_drones": 2, "collisions": 0, "avg_battery": 3750.0},
                    reward=0.1,
                    timestamp_ns=ts,
                )

        # Verify MCAP is readable
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            assert summary is not None
            assert len(summary.channels) >= 2
            # Verify scene and metrics channels exist
            topics = {ch.topic for ch in summary.channels.values()}
            assert "/swarm/scene" in topics
            assert "/swarm/metrics" in topics

    def test_scene_schema_arrays_have_items(self):
        """Every 'array' property in _SCENE_UPDATE_SCHEMA must define 'items'.

        Foxglove Studio's JSON Schema parser accesses ``items.type`` when
        traversing the schema.  A missing ``items`` causes:
        "Cannot read properties of undefined (reading 'type')".
        """

        def _check(schema, path=""):
            if isinstance(schema, dict):
                if schema.get("type") == "array":
                    assert "items" in schema, (
                        f"Array at '{path}' missing 'items' — "
                        "Foxglove will fail to parse this schema"
                    )
                for key, value in schema.items():
                    _check(value, f"{path}.{key}")

        _check(_SCENE_UPDATE_SCHEMA, "SceneUpdate")


# ---------------------------------------------------------------------------
# Model loading and optimization tests
# ---------------------------------------------------------------------------

class TestLocalModels:
    """Tests for local model loading and scene entity data embedding."""

    def _build_scene(self, bridge, **kwargs):
        """Capture the scene dict by calling publish_scene on an unstarted bridge."""
        captured = {}

        def _capture(channel_key, data, timestamp_ns):
            captured[channel_key] = data

        bridge._send = _capture
        bridge._started = True
        bridge.publish_scene(**kwargs)
        bridge._started = False
        return captured.get("scene")

    def test_local_model_files_exist(self):
        """All local .glb model files must exist on disk."""
        assert DRONE_MODEL_PATH.exists(), f"Missing: {DRONE_MODEL_PATH}"
        for cls_name, path in TARGET_MODEL_PATHS.items():
            assert path.exists(), f"Missing {cls_name} model: {path}"

    def test_local_model_files_are_valid_glb(self):
        """Local model files must start with the glTF magic bytes."""
        for path in [DRONE_MODEL_PATH] + list(TARGET_MODEL_PATHS.values()):
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == b"glTF", f"{path.name} is not a valid glTF binary"

    def test_get_drone_model_data_returns_nonempty(self):
        """get_drone_model_data returns non-empty base64 string."""
        data = get_drone_model_data()
        assert isinstance(data, str)
        assert len(data) > 0

    def test_get_target_model_data_returns_nonempty(self):
        """get_target_model_data returns non-empty base64 string for each class."""
        for cls_name in ("hostile", "friendly", "unknown"):
            data = get_target_model_data(cls_name)
            assert isinstance(data, str)
            assert len(data) > 0

    def test_get_target_model_data_unknown_falls_back(self):
        """Unknown classification falls back to 'unknown' model."""
        data = get_target_model_data("nonexistent_class")
        expected = get_target_model_data("unknown")
        assert data == expected

    def test_drone_models_use_embedded_data(self):
        """Drone models should use 'data' field with embedded model bytes."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        scene = self._build_scene(bridge, drone_positions=positions)
        model = scene["entities"][0]["models"][0]
        assert "data" in model
        assert model["data"] == get_drone_model_data()
        assert "url" not in model

    def test_drone_models_consistent_across_frames(self):
        """Drone models should use the same data on every frame."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        scene1 = self._build_scene(bridge, drone_positions=positions)
        scene2 = self._build_scene(bridge, drone_positions=positions)
        model1 = scene1["entities"][0]["models"][0]
        model2 = scene2["entities"][0]["models"][0]
        assert model1["data"] == model2["data"]

    def test_target_models_use_embedded_data(self):
        """Target models should use 'data' field with embedded model bytes."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        tgt_pos = {"T0": np.array([100.0, 100.0, 80.0])}
        tgt_cls = {"T0": "hostile"}
        scene = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
        )
        targets_entity = next(e for e in scene["entities"] if e["id"] == "targets")
        model = targets_entity["models"][0]
        assert "data" in model
        assert model["data"] == get_target_model_data("hostile")
        assert "url" not in model

    def test_target_models_consistent_across_frames(self):
        """Target models should use the same data on every frame."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        tgt_pos = {"T0": np.array([100.0, 100.0, 80.0])}
        tgt_cls = {"T0": "hostile"}
        scene1 = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
        )
        scene2 = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
        )
        targets1 = next(e for e in scene1["entities"] if e["id"] == "targets")
        targets2 = next(e for e in scene2["entities"] if e["id"] == "targets")
        assert targets1["models"][0]["data"] == targets2["models"][0]["data"]

    def test_target_model_data_matches_classification(self):
        """Target model data should match the target classification."""
        bridge = FoxgloveBridge()
        positions = np.array([[50.0, 50.0, 50.0]])
        tgt_pos = {
            "T0": np.array([100.0, 100.0, 80.0]),
            "T1": np.array([200.0, 200.0, 90.0]),
            "T2": np.array([300.0, 300.0, 70.0]),
        }
        tgt_cls = {"T0": "hostile", "T1": "friendly", "T2": "unknown"}
        scene = self._build_scene(
            bridge,
            drone_positions=positions,
            target_positions=tgt_pos,
            target_classifications=tgt_cls,
        )
        targets_entity = next(e for e in scene["entities"] if e["id"] == "targets")
        models = targets_entity["models"]
        data_values = [m["data"] for m in models]
        assert get_target_model_data("hostile") in data_values
        assert get_target_model_data("friendly") in data_values
        assert get_target_model_data("unknown") in data_values


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Test that new modules are importable from the package."""

    def test_import_foxglove_bridge(self):
        """FoxgloveBridge is importable from utils."""
        from isr_rl_dmpc.utils import FoxgloveBridge
        assert FoxgloveBridge is not None

    def test_import_mcap_recorder(self):
        """MCAPRecorder is importable from utils."""
        from isr_rl_dmpc.utils import MCAPRecorder
        assert MCAPRecorder is not None

    def test_import_extract_targets_from_obs(self):
        """extract_targets_from_obs is importable from utils."""
        from isr_rl_dmpc.utils import extract_targets_from_obs
        assert extract_targets_from_obs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
