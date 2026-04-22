from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("cvxpy")
pytest.importorskip("scipy")
pytest.importorskip("osqp")

from isr_rl_dmpc.gym_env.belief_coverage_env import BeliefCoverageEnv, _wrap_angle


def _block_cells(env: BeliefCoverageEnv, row_start: int, col_start: int) -> np.ndarray:
    mask = (
        (env.global_belief.rows >= row_start)
        & (env.global_belief.rows < row_start + 2)
        & (env.global_belief.cols >= col_start)
        & (env.global_belief.cols < col_start + 2)
    )
    return np.flatnonzero(mask).astype(np.int32)


def _force_patch(
    env: BeliefCoverageEnv,
    *,
    row_start: int,
    col_start: int,
    base_eta_steps: int = 999,
) -> np.ndarray:
    patch = _block_cells(env, row_start=row_start, col_start=col_start)
    assert patch.size == 4
    env._activate_threat_patch(
        patch,
        candidate_index=-1,
        count_as_spawn=False,
        base_eta_steps=base_eta_steps,
    )
    return patch


def _force_confirmed_patch(
    env: BeliefCoverageEnv,
    patch: np.ndarray,
    *,
    belief_level: float = 0.95,
) -> None:
    env.global_belief.anomaly_score.fill(0.0)
    env.global_belief.anomaly_score[patch] = belief_level
    env._threat_persistence_score.fill(0.0)
    env._threat_persistence_score[patch] = 1.0
    env._active_threat_confirmation_level = 1.0
    env._threat_confirmed = True
    env._confirmed_threat_mask.fill(False)
    env._confirmed_threat_mask[patch] = True
    env._central_command_notified = True


def test_belief_coverage_env_reset_and_step_smoke() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        fixed_altitude=30.0,
        sensor_range=60.0,
        global_sync_steps=2,
        suspicious_zones=[
            {"center": [40.0, 40.0], "radius": 20.0, "score": 0.8},
        ],
    )

    observation, info = env.reset(seed=7)

    assert observation.shape == env.observation_space.shape
    assert len(env.local_beliefs) == env.num_drones
    assert env.global_belief.n_cells == env.action_space.nvec[0]
    assert "observation_dict" in info

    action = env.select_greedy_action(unique=True)
    next_observation, reward, terminated, truncated, step_info = env.step(action)

    assert next_observation.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "mean_uncertainty" in step_info
    assert "connectivity_state" in step_info
    assert "anomaly_counts" in step_info
    assert "truth_risk_counts" in step_info
    assert step_info["selected_target_cells"].shape == (env.num_drones,)
    assert "revisit_bonus_norm" not in step_info
    assert "anomaly_revisit_bonus" not in step_info["reward_components"]

    env.close()


def test_neighbor_sharing_reduces_local_belief_divergence() -> None:
    def _run(enable_neighbor_sharing: bool) -> tuple[BeliefCoverageEnv, dict]:
        env = BeliefCoverageEnv(
            num_drones=2,
            mission_duration=10,
            horizon=5,
            dt=0.05,
            area_size=(80.0, 80.0),
            grid_resolution=20.0,
            sensor_range=40.0,
            communication_range=200.0,
            enable_neighbor_sharing=enable_neighbor_sharing,
        )
        env.reset(seed=0)
        env.global_belief.reset()
        for local in env.local_beliefs:
            local.reset()
            local.clear_recent_changes()
        env._simulator.set_drone_initial_state(0, np.array([10.0, 10.0, 30.0]))
        env._simulator.set_drone_initial_state(1, np.array([70.0, 70.0, 30.0]))
        env._sync_states_from_sim()
        connectivity = env._compute_connectivity()
        metrics = env._update_beliefs(
            step=1,
            connectivity=connectivity,
            grow=False,
            apply_global_sync=False,
        )
        return env, metrics

    env_no_share, metrics_no_share = _run(enable_neighbor_sharing=False)
    env_share, metrics_share = _run(enable_neighbor_sharing=True)

    diff_no_share = np.sum(
        np.abs(env_no_share.local_beliefs[0].uncertainty - env_no_share.local_beliefs[1].uncertainty) > 1e-9
    )
    diff_share = np.sum(
        np.abs(env_share.local_beliefs[0].uncertainty - env_share.local_beliefs[1].uncertainty) > 1e-9
    )

    assert diff_share < diff_no_share
    assert metrics_no_share["shared_cell_count"] == 0
    assert metrics_share["shared_cell_count"] > 0

    env_no_share.close()
    env_share.close()


def test_global_sync_does_not_cross_disconnected_components() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        sensor_range=40.0,
        communication_range=30.0,
        enable_neighbor_sharing=False,
        global_sync_steps=1,
    )
    env.reset(seed=0)
    env.global_belief.reset()
    for local in env.local_beliefs:
        local.reset()
        local.clear_recent_changes()

    env._simulator.set_drone_initial_state(0, np.array([10.0, 10.0, 30.0]))
    env._simulator.set_drone_initial_state(1, np.array([130.0, 130.0, 30.0]))
    env._sync_states_from_sim()

    connectivity = env._compute_connectivity()
    assert int(connectivity["component_count"]) == 2

    metrics = env._update_beliefs(
        step=1,
        connectivity=connectivity,
        grow=False,
        apply_global_sync=True,
    )
    local_diff = np.sum(
        np.abs(env.local_beliefs[0].uncertainty - env.local_beliefs[1].uncertainty) > 1e-9
    )

    assert metrics["global_sync_applied"] is False
    assert local_diff > 0

    env.close()


def test_goal_projection_rewrites_isolating_target_choice() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        communication_range=25.0,
    )
    env.reset(seed=0)

    requested = np.array([env.n_cells - 1, 0], dtype=np.int32)
    projected = env._project_action_cells(requested)

    assert projected[0] != requested[0]
    assert env._goal_preserves_link(
        0,
        env.cell_centers_xy[int(projected[0])],
        env._drone_states[:, :2],
    )

    env.close()


def test_heading_aligns_with_selected_target_cell() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=60.0,
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([15], dtype=np.int32))
    selected = int(info["selected_target_cells"][0])
    target = env.cell_centers_xy[selected]
    position = env._drone_states[0, :2]
    expected_yaw = np.arctan2(target[1] - position[1], target[0] - position[0])
    yaw_error = abs(float(_wrap_angle(expected_yaw - env._drone_states[0, 9])))

    assert yaw_error < 1e-6
    assert float(info["heading_error_rad"][0]) < 1e-6

    env.close()


def test_suspicious_zone_updates_anomaly_belief() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=100.0,
        suspicious_zones=[
            {"center": [70.0, 70.0], "radius": 25.0, "score": 0.8},
        ],
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([15], dtype=np.int32))

    assert info["anomaly_counts"]["gt_0_1"] > 0
    assert info["mean_anomaly_score"] > 0.0

    env.close()


def test_home_subregions_cover_grid_and_launch_is_centered() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        enable_persistent_threats=False,
    )
    env.reset(seed=0)

    home_regions = [env.get_home_cell_indices(i) for i in range(env.num_drones)]
    concatenated = np.concatenate(home_regions)
    assert np.array_equal(np.sort(concatenated), np.arange(env.n_cells, dtype=np.int32))
    for i in range(env.num_drones):
        for j in range(i + 1, env.num_drones):
            assert np.intersect1d(home_regions[i], home_regions[j]).size == 0

    positions = env._drone_states[:, :2]
    assert np.allclose(np.mean(positions, axis=0), env.base_station, atol=1e-6)

    env.close()


def test_patrol_policy_stays_in_home_region_when_local_risk_is_high() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        enable_persistent_threats=False,
    )
    env.reset(seed=0)
    env.global_belief.uncertainty.fill(0.9)
    env.global_belief.anomaly_score.fill(0.0)

    action = env.select_patrol_action()

    for drone_idx, cell_idx in enumerate(action):
        assert int(cell_idx) in set(env.get_home_cell_indices(drone_idx).tolist())
        assert int(cell_idx) not in set(env.get_assist_cell_indices(drone_idx).tolist())

    env.close()


def test_patrol_policy_takes_local_detour_then_returns_home() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        enable_persistent_threats=False,
    )
    env.reset(seed=0)
    env.global_belief.uncertainty.fill(0.2)
    env.global_belief.anomaly_score.fill(0.0)

    assist_cells = env.get_assist_cell_indices(0)
    assert assist_cells.size > 0
    assist_target = int(assist_cells[0])
    env.global_belief.uncertainty[assist_target] = 0.95

    detour_action = env.select_patrol_action()
    assert int(detour_action[0]) == assist_target
    assert env._patrol_detour_targets[0] == assist_target

    env.global_belief.uncertainty[assist_target] = 0.2
    return_action = env.select_patrol_action()
    assert int(return_action[0]) in set(env.get_home_cell_indices(0).tolist())
    assert env._patrol_detour_targets[0] == -1

    env.close()


def test_patrol_policy_does_not_collapse_into_far_away_helping() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        enable_persistent_threats=False,
    )
    env.reset(seed=0)
    env.global_belief.uncertainty.fill(0.2)
    env.global_belief.anomaly_score.fill(0.0)

    remote_owner = env.num_drones - 1
    remote_home = env.get_home_cell_indices(remote_owner)
    remote_assist_others = np.concatenate(
        [env.get_assist_cell_indices(i) for i in range(env.num_drones - 1)]
    )
    far_candidates = np.setdiff1d(remote_home, remote_assist_others, assume_unique=False)
    assert far_candidates.size > 0
    remote_target = int(far_candidates[0])
    env.global_belief.uncertainty[remote_target] = 0.98

    action = env.select_patrol_action()

    for drone_idx in range(env.num_drones - 1):
        assert int(action[drone_idx]) != remote_target
        assert int(action[drone_idx]) in set(env.get_home_cell_indices(drone_idx).tolist())
    assert int(action[remote_owner]) in set(env.get_home_cell_indices(remote_owner).tolist())

    env.close()


def test_build_info_does_not_advance_patrol_route_state() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
    )
    env.reset(seed=0)
    before = env._patrol_route_indices.copy()

    env._build_info(
        connectivity=env._compute_connectivity(),
        reward=0.0,
        reward_components={},
        solve_times=np.zeros(env.num_drones, dtype=np.float64),
    )

    assert np.array_equal(before, env._patrol_route_indices)

    env.close()


def test_truth_risk_and_rasterized_belief_views_are_available() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=80.0,
        suspicious_zones=[
            {"center": [60.0, 60.0], "radius": 25.0, "score": 0.75},
        ],
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([15], dtype=np.int32))

    truth_scores = env.get_truth_risk_scores()
    belief_scores = env.get_belief_risk_scores()
    truth_grid = env.rasterize_cell_values(truth_scores)
    belief_grid = env.rasterize_cell_values(belief_scores)
    boundaries = env.get_home_strip_boundaries()

    assert truth_scores.shape == (env.n_cells,)
    assert belief_scores.shape == (env.n_cells,)
    assert truth_grid.shape == belief_grid.shape
    assert truth_grid.shape == (len(env._row_values), len(env._col_values))
    assert boundaries["axis"] in {"x", "y"}
    assert info["risk_mismatch_mean"] >= 0.0

    env.close()


def test_env_exposes_pomdp_style_world_and_belief_views() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=10,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=80.0,
        suspicious_zones=[
            {"center": [60.0, 60.0], "radius": 25.0, "score": 0.75},
        ],
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([15, 0], dtype=np.int32))

    world_state = env.get_hidden_world_state()
    belief_state = env.get_belief_state()

    assert world_state.patrol_risk_state.shape == (env.n_cells,)
    assert world_state.persistent_threat_state.shape == (env.n_cells,)
    assert world_state.drone_positions.shape == (env.num_drones, 2)
    assert belief_state.global_patrol_risk_belief.shape == (env.n_cells,)
    assert belief_state.global_threat_belief.shape == (env.n_cells,)
    assert np.allclose(world_state.combined_risk_state, env.get_truth_risk_scores())
    assert np.allclose(belief_state.combined_global_risk_belief, env.get_belief_risk_scores())
    assert "mean_patrol_risk_belief" in info
    assert "mean_threat_belief" in info
    assert "mean_persistent_threat_state" in info
    assert info["policy_name"] == env.policy_name

    env.close()


def test_visualization_script_generates_diagnostic_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("yaml")

    script_path = Path(
        "/Users/apple/Desktop/cornerstone/isr-rl-dmpc/scripts/visualize_belief_coverage.py"
    )
    spec = importlib.util.spec_from_file_location("visualize_belief_coverage", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=12,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=60.0,
        suspicious_zones=[
            {"center": [60.0, 60.0], "radius": 20.0, "score": 0.75},
        ],
    )
    history, info = module._run_rollout(env, steps=6, seed=0)
    output_path = tmp_path / "phase1_patrol_risk.png"
    module._plot_rollout_diagnostics(
        env,
        history,
        info,
        output_path=output_path,
        show=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    env.close()


@pytest.mark.parametrize("num_drones", [4, 5, 6])
def test_patrol_baseline_scales_to_medium_swarms(num_drones: int) -> None:
    env = BeliefCoverageEnv(
        num_drones=num_drones,
        mission_duration=260,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        enable_persistent_threats=False,
    )
    env.reset(seed=0)

    final_info = None
    terminated = False
    truncated = False
    for _ in range(200):
        _, _, terminated, truncated, final_info = env.step(env.select_patrol_action())
        if terminated or truncated:
            break

    assert final_info is not None
    assert not terminated
    assert not truncated
    assert int(np.sum(final_info["in_home_region"])) == num_drones
    assert float(final_info["never_observed_fraction"]) < 0.3
    assert float(final_info["low_risk_fraction"]) > 0.15
    assert float(np.min(final_info["home_coverage_fraction"])) > 0.6

    env.close()


def test_persistent_threat_requires_repeated_observation_for_confirmation() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=20,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=120.0,
        max_threat_cycles=1,
    )
    env.reset(seed=0)
    patch = _force_patch(env, row_start=2, col_start=2)
    env._simulator.set_drone_initial_state(0, np.array([52.0, 44.0, 30.0]))
    env._sync_states_from_sim()

    confirmations = []
    confirmation_scores = []
    threat_beliefs = []
    final_info = None
    for _ in range(8):
        _, _, _, _, final_info = env.step(np.array([int(np.max(patch))], dtype=np.int32))
        confirmations.append(bool(final_info["threat_confirmed"]))
        confirmation_scores.append(float(final_info["active_threat_confirmation_score"]))
        threat_beliefs.append(float(final_info["mean_threat_belief"]))

    assert final_info is not None
    assert confirmations[0] is False
    assert any(confirmations)
    assert max(confirmation_scores) >= final_info["threat_confirmation_threshold"]
    assert float(np.mean(env.get_patrol_risk_belief_scores()[patch])) < 0.5
    assert max(threat_beliefs) > 0.2

    env.close()


def test_confirmed_threat_biases_only_small_nearby_subset() -> None:
    env = BeliefCoverageEnv(
        num_drones=6,
        mission_duration=30,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
    )
    env.reset(seed=0)
    patch = _force_patch(env, row_start=0, col_start=0)
    _force_confirmed_patch(env, patch)
    env.global_belief.uncertainty.fill(0.2)
    env._tracking_bias_drones = env._choose_tracking_bias_drones()

    action = env.select_patrol_action()
    trackers = np.flatnonzero(env._tracking_bias_drones)

    assert 1 <= trackers.size <= min(env.THREAT_MAX_TRACKERS, max(1, env.num_drones // 3))
    for drone_idx in trackers:
        assert int(action[drone_idx]) in set(patch.tolist())
    for drone_idx in range(env.num_drones):
        if drone_idx in set(trackers.tolist()):
            continue
        assert int(action[drone_idx]) in set(env.get_home_cell_indices(drone_idx).tolist())

    env.close()


def test_interceptor_dispatch_and_collision_remove_persistent_threat() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=30,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
    )
    env.reset(seed=0)
    patch = _force_patch(env, row_start=6, col_start=6)
    _force_confirmed_patch(env, patch)

    first_metrics = env._advance_interceptor_and_threat()
    assert first_metrics["interceptor_dispatched"] is True
    assert env._interceptor_active is True

    env._interceptor_position = env.cell_centers_xy[int(patch[0])].copy()
    second_metrics = env._advance_interceptor_and_threat()

    assert second_metrics["threat_removed"] is True
    assert env._threat_cycles_completed == 1
    assert env._active_threat_exists() is False

    env.close()


def test_persistent_threat_respawns_for_multiple_cycles() -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=40,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        max_threat_cycles=3,
    )
    env.reset(seed=0)
    first_patch = env._active_threat_cells.copy()
    assert first_patch.size == 4

    seen_patches = {tuple(first_patch.tolist())}
    for cycle_idx in range(3):
        patch = env._active_threat_cells.copy()
        _force_confirmed_patch(env, patch)
        env._interceptor_active = True
        env._interceptor_position = env.cell_centers_xy[int(patch[0])].copy()
        env._advance_interceptor_and_threat()
        if cycle_idx < 2:
            assert env._active_threat_exists()
            seen_patches.add(tuple(env._active_threat_cells.tolist()))
        else:
            assert not env._active_threat_exists()

    assert env._threat_cycles_completed == 3
    assert len(seen_patches) >= 2

    env.close()


def test_mission_fail_when_persistent_threat_reaches_base() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=20,
        horizon=5,
        dt=0.05,
        area_size=(80.0, 80.0),
        grid_resolution=20.0,
        sensor_range=80.0,
        max_threat_cycles=1,
    )
    env.reset(seed=0)
    _force_patch(env, row_start=2, col_start=2, base_eta_steps=1)

    _, _, terminated, truncated, info = env.step(np.array([0], dtype=np.int32))

    assert terminated is True
    assert truncated is False
    assert info["mission_failed"] is True
    assert info["mission_fail_reason"] == "persistent_threat_reached_base"

    env.close()


@pytest.mark.parametrize("speed_case", ["slow", "medium", "fast"])
def test_moving_threat_progresses_toward_base(speed_case: str) -> None:
    env = BeliefCoverageEnv(
        num_drones=2,
        mission_duration=20,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
        persistent_threat_speed_case=speed_case,
    )
    env.reset(seed=0)
    _force_patch(env, row_start=6, col_start=0, base_eta_steps=999)

    start_position = env._active_threat_centroid()
    assert start_position is not None
    start_distance = float(np.linalg.norm(start_position - env.base_station))
    metrics = env._advance_moving_threat()
    end_position = env._active_threat_centroid()
    assert end_position is not None
    end_distance = float(np.linalg.norm(end_position - env.base_station))

    assert metrics["threat_moved"] is True
    assert metrics["threat_step_distance"] > 0.0
    assert end_distance < start_distance
    assert abs(float(end_position[1] - start_position[1])) > 1e-6

    env.close()


def test_moving_threat_confirmation_survives_motion() -> None:
    env = BeliefCoverageEnv(
        num_drones=1,
        mission_duration=30,
        horizon=5,
        dt=0.05,
        area_size=(120.0, 120.0),
        grid_resolution=20.0,
        sensor_range=140.0,
        max_threat_cycles=1,
        persistent_threat_speed_case="medium",
    )
    env.reset(seed=0)
    _force_patch(env, row_start=4, col_start=0, base_eta_steps=999)
    env._simulator.set_drone_initial_state(0, np.array([40.0, 80.0, 30.0]))
    env._sync_states_from_sim()

    confirmation_scores = []
    confirmations = []
    for _ in range(10):
        action = np.array([int(np.max(env._active_threat_cells))], dtype=np.int32)
        _, _, terminated, truncated, info = env.step(action)
        confirmation_scores.append(float(info["active_threat_confirmation_score"]))
        confirmations.append(bool(info["threat_confirmed"]))
        assert not terminated
        assert not truncated

    assert confirmations[0] is False
    assert any(confirmations)
    assert max(confirmation_scores) >= env.THREAT_CONFIRMATION_THRESHOLD

    env.close()


def test_interceptor_retargets_against_moving_threat() -> None:
    env = BeliefCoverageEnv(
        num_drones=3,
        mission_duration=30,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
        persistent_threat_speed_case="fast",
    )
    env.reset(seed=0)
    patch = _force_patch(env, row_start=6, col_start=1, base_eta_steps=999)
    _force_confirmed_patch(env, patch)

    first_metrics = env._advance_interceptor_and_threat()
    first_target = env._interceptor_target.copy()
    assert first_metrics["interceptor_dispatched"] is True
    env._advance_moving_threat()
    second_metrics = env._advance_interceptor_and_threat()
    second_target = env._interceptor_target.copy()

    assert second_metrics["mission_failed"] is False
    assert np.linalg.norm(second_target - first_target) > 1e-6

    env.close()


def test_interceptor_can_remove_moving_persistent_threat() -> None:
    env = BeliefCoverageEnv(
        num_drones=3,
        mission_duration=30,
        horizon=5,
        dt=0.05,
        area_size=(160.0, 160.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
        persistent_threat_speed_case="medium",
    )
    env.reset(seed=0)
    patch = _force_patch(env, row_start=6, col_start=1, base_eta_steps=999)
    _force_confirmed_patch(env, patch)
    env._advance_moving_threat()
    env._interceptor_active = True
    env._interceptor_position = env.cell_centers_xy[int(env._active_threat_cells[0])].copy()
    env._interceptor_target = env._active_threat_centroid().copy()

    metrics = env._advance_interceptor_and_threat()

    assert metrics["threat_removed"] is True
    assert env._active_threat_exists() is False
    assert env._threat_cycles_completed == 1

    env.close()


def test_moving_persistent_threat_preserves_patrol_elsewhere() -> None:
    env = BeliefCoverageEnv(
        num_drones=4,
        mission_duration=120,
        horizon=5,
        dt=0.05,
        area_size=(400.0, 400.0),
        grid_resolution=20.0,
        max_threat_cycles=1,
        suspicious_zones=[],
        persistent_threat_speed_case="medium",
    )
    env.reset(seed=7)

    final_info = None
    tracking_counts = []
    home_fractions = []
    for _ in range(180):
        _, _, terminated, truncated, final_info = env.step(env.select_patrol_action())
        tracking_counts.append(int(np.sum(final_info["tracking_bias_drones"])))
        home_fractions.append(float(np.mean(final_info["selected_in_home"])))
        if terminated or truncated:
            break

    assert final_info is not None
    assert max(tracking_counts) <= min(env.THREAT_MAX_TRACKERS, max(1, env.num_drones // 3))
    assert float(np.mean(home_fractions)) > 0.7
    assert float(final_info["never_observed_fraction"]) < 0.6

    env.close()
