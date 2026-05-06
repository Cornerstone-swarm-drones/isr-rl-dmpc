# Phase 1 Belief Coverage Handoff

This branch adds a parallel Phase 1 belief/risk coverage path for `area_surveillance`.
It does not replace the existing continuous `MARLDMPCEnv` Q/R-scaling interface.

## What Phase 1 Currently Does

- Runs `BeliefCoverageEnv`, a separate Gymnasium environment for cell/region-selection coverage.
- Maintains per-drone local belief grids and a fused global command belief grid.
- Uses a 135 degree forward FOV sensor model with range and angle quality decay.
- Treats Phase 1 scoring as patrol risk / threat belief, not a real classifier output.
- Assigns drones to home strips and runs a deterministic boustrophedon-style patrol baseline.
- Allows conservative, bounded local assist behavior without global clustering.
- Supports moving 2x2 persistent threat patches that travel toward the base.
- Confirms persistent threats through repeated evidence / persistence score, not one observation.
- Runs a shared single-target EKF track with delayed communication measurements.
- Dispatches a simple interceptor using either oracle or EKF guidance.
- Provides matplotlib diagnostics for patrol risk, threat belief, hidden threat state, EKF track, drone trajectories, and interceptor lifecycle.

## Canonical Demo Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 220 --seed 7 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy improved --max-threat-cycles 3 --output visualizations/phase1_handoff_demo/phase1_demo_6d_fast_ekf_improved.png
```

Output:

- `visualizations/phase1_handoff_demo/phase1_demo_6d_fast_ekf_improved.png`

Last verified result from this command:

- 6 drones
- fast moving threat
- EKF guidance
- improved response policy
- 2 threat eliminations
- no mission failure

## Canonical Validation Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_ekf_response_policy.py --steps 220 --num-drones-cases 4,5,6 --speed-cases slow,medium,fast --seeds 7 --output-dir visualizations/phase1_handoff_validation
```

Outputs:

- `visualizations/phase1_handoff_validation/phase1_validation_metrics.json`
- `visualizations/phase1_handoff_validation/phase1_validation_summary.csv`
- `visualizations/phase1_handoff_validation/phase1_validation_runs.csv`
- `visualizations/phase1_handoff_validation/ekf_response_policy_comparison_4d.png`
- `visualizations/phase1_handoff_validation/ekf_response_policy_comparison_5d.png`
- `visualizations/phase1_handoff_validation/ekf_response_policy_comparison_6d.png`
- representative rollout PNGs for each drone-count / speed / policy case

Last verified result from this command:

- 4, 5, and 6 drone cases ran for slow, medium, and fast threat speeds.
- Seed `7` had `base_compromise_rate = 0.0` and `intercept_success_rate = 1.0` for all reported cases.
- This is a deterministic handoff/demo check, not a statistical performance claim.

## Relevant Test Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest tests/unit/test_belief_grid.py tests/unit/test_sensor_model.py tests/unit/test_reduced_pomdp.py tests/unit/test_shared_track_ekf.py tests/integration/test_belief_coverage_env.py -q
```

Last verified result:

- `46 passed`

## Main Known Limitations

- Phase 1 is still a deterministic baseline plus diagnostics, not trained RL.
- The shared EKF is intentionally small and single-target only.
- The persistent threat model uses simple 2x2 patches and deterministic motion toward base.
- Interceptor dynamics are simple and not yet a full drone/controller model.
- PyBullet is installed locally, but the full Phase 1 belief/threat visualization is still matplotlib-based.
- Generated figures and metrics are ignored by git under `visualizations/`.
- The legacy field name `anomaly_score` still appears in some belief-grid internals, but in this Phase 1 path it should be interpreted as patrol-risk / threat-belief score.

## Suggested Next Phase 2 Directions

- Decide the exact Phase 2 boundary: richer threat response, shared tracking, interceptor handoff, or RL training integration.
- Replace or wrap the lightweight EKF with the repo's heavier tracking/state-estimation modules if their state model fits the moving-threat use case.
- Add more seeds and scenario variants for stronger evidence beyond the current deterministic handoff run.
- Add a video/animation export path for the matplotlib demo, or wire the Phase 1 state into PyBullet visualization.
- Introduce richer tracking quality and moving-threat behavior before any policy learning.
- Only start RL once the deterministic baseline metrics and failure modes are stable.
