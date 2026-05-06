# Phase 2 Base Defense Slice

This branch starts Phase 2 as an operational base-defense upgrade on top of the
working Phase 1 patrol-risk system. `BeliefCoverageEnv` remains parallel to
`MARLDMPCEnv`; no RL training path is changed by this slice.

## What Changed

- Added `response_policy="phase2"` as a comparison path alongside `baseline` and `improved`.
- Added an explicit central-command/base sensor model with range, noise, and delay.
- Fed base-sensor detections into the shared EKF as labeled measurements.
- Surfaced base-sensor detections, EKF updates, quality, range, and totals in `info`.
- Kept the lightweight shared EKF for this slice; the existing 11D target EKF is richer but mismatched to the current 2D patch-centroid + delayed-relay prototype.
- Added Phase 2 interceptor guidance that still uses estimated `[x, y, vx, vy]`, but moves with speed and acceleration limits instead of an instantaneous pure point/bullet step.
- Added deterministic threat-path weave for `phase2` to make the single moving 2x2 threat less centerline-like while preserving inspectability.
- Tuned Phase 2 guidance so lead prediction uses the constrained interceptor speed, and urgent base-sensor tracks can trigger pre-confirmation launch.

## Canonical Phase 2 Demo Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 220 --seed 7 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy phase2 --max-threat-cycles 3 --output visualizations/phase2_base_defense_demo/phase2_demo_6d_fast_ekf_base_defense.png
```

Output:

- `visualizations/phase2_base_defense_demo/phase2_demo_6d_fast_ekf_base_defense.png`

Last verified result:

- 6 drones
- fast moving threat
- EKF guidance
- Phase 2 response policy
- 2 threat eliminations
- no mission failure
- base sensor contributed 54 detections and 52 EKF updates

## Canonical Phase 2 Validation Command

The full DMPC-backed rollout path is still available, but broad evidence
sweeps should use the deterministic fast planar validation mode:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_ekf_response_policy.py --steps 140 --num-drones-cases 4,5,6 --speed-cases slow,medium,fast --seeds 7,11 --policies improved,phase2 --skip-diagnostic-plots --validation-dynamics fast_planar --output-dir visualizations/phase2_fast_planar_broad_smoke
```

Outputs:

- `visualizations/phase2_fast_planar_broad_smoke/response_policy_validation_metrics.json`
- `visualizations/phase2_fast_planar_broad_smoke/response_policy_validation_summary.csv`
- `visualizations/phase2_fast_planar_broad_smoke/response_policy_validation_runs.csv`

Last verified result:

- The fast planar mode ran 36 rollouts covering 4, 5, and 6 drones, slow/medium/fast threats, two seeds, and two policies in about 13 seconds.
- In fast-threat cases, `phase2` had `base_compromise_rate = 0.0` and `intercept_success_rate = 1.0` for 4, 5, and 6 drones.
- The full-vs-fast representative check for 4 drones / fast / seed `7` preserved the same success and base-failure outcome, with launch/intercept within two steps.
- Internal rollout runtime dropped from `3.31s` in full mode to `0.24s` in fast planar mode for that representative case.

For a direct full-vs-fast sanity check, use the same command with
`--validation-dynamics full` and `--validation-dynamics fast_planar` on a
single drone-count / speed / seed case.

## Relevant Tests

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest tests/unit/test_belief_grid.py tests/unit/test_sensor_model.py tests/unit/test_reduced_pomdp.py tests/unit/test_shared_track_ekf.py tests/integration/test_belief_coverage_env.py -q
```

Last verified result:

- `48 passed`

## Known Limitations

- This is still a single-threat, single-shared-track prototype.
- Base sensing is modeled as a simple range/noise/delay sensor, not a full radar/camera stack.
- The Phase 2 interceptor is acceleration-limited but still abstract.
- The fast planar validation mode skips DMPC/ADMM solver fidelity, so final claims should still be spot-checked with full mode.
- More seeds are needed before making statistical claims about base-compromise rate.
