# Phase 2 Base Defense / Communication Realism Handoff

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
- Tuned Phase 2 guidance so lead prediction uses the constrained interceptor speed.
- Added threat-belief communication realism modes:
  - `shared`: original central/shared threat response path.
  - `limited`: local drone confirmation with delayed base confirmation, but still allows urgent base-track pre-confirmation launch.
  - `limited_strict`: recommended Phase 2 evaluation mode. Local/base confirmation is delayed, and base tracks cannot launch the interceptor until repeated local/base evidence confirms the threat.

## Recommended Phase 2 Mode

Use `response_policy="phase2"`, `interceptor_guidance_mode="ekf"`,
`validation_dynamics="fast_planar"` for broad sweeps, and
`threat_belief_mode="limited_strict"` for the default Phase 2 evaluation mode.

`limited_strict` makes communication realism matter operationally: launch waits
for base-confirmed evidence rather than a single fresh base/EKF track. In the
latest 5-seed medium/fast fast-planar sweep, this delayed launches while keeping
`base_compromise_rate = 0.0` in all tested 4-, 5-, and 6-drone cases.

## Canonical Phase 2 Demo Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 220 --seed 7 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy phase2 --threat-belief-mode limited_strict --validation-dynamics fast_planar --max-threat-cycles 1 --output visualizations/phase2_final_demo/phase2_limited_strict_6d_fast.png
```

Output:

- `visualizations/phase2_final_demo/phase2_limited_strict_6d_fast.png`

Last verified result:

- 6 drones
- fast moving threat
- EKF guidance
- Phase 2 response policy
- `limited_strict` threat-belief mode
- fast planar validation dynamics
- 1 threat elimination
- no mission failure

## Canonical Phase 2 Validation Command

The full DMPC-backed rollout path is still available, but broad evidence
sweeps should use the deterministic fast planar validation mode:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_ekf_response_policy.py --steps 140 --num-drones-cases 4,5,6 --speed-cases medium,fast --seeds 7,11,19,23,29 --policies phase2 --skip-diagnostic-plots --validation-dynamics fast_planar --threat-belief-mode limited_strict --output-dir visualizations/phase2_final_limited_strict
```

Outputs:

- `visualizations/phase2_final_limited_strict/response_policy_validation_metrics.json`
- `visualizations/phase2_final_limited_strict/response_policy_validation_summary.csv`
- `visualizations/phase2_final_limited_strict/response_policy_validation_runs.csv`

Last verified result:

- The fast planar mode ran 30 `limited_strict` rollouts covering 4, 5, and 6 drones, medium/fast threats, and five seeds.
- `base_compromise_rate = 0.0` for all tested cases.
- `intercept_success_rate = 0.8` for all tested cases. The missed intercepts were seed `29`, where no mode (`shared`, `limited`, or `limited_strict`) observed/confirmed/launched within the validation horizon and no mission failure occurred.
- `limited_strict` produced meaningful launch delay relative to `shared`/`limited` while preserving catastrophic safety in the tested set.
- Representative full-dynamics spot checks:
  - 4 drones / fast / seed `7` / 140 steps: success, no base failure, launch step `48`, intercept step `111`.
  - 6 drones / medium / seed `7` / 220 steps: success, no base failure, launch step `49`, intercept step `177`.

For a direct full-vs-fast sanity check, use the same command with
`--validation-dynamics full` and `--validation-dynamics fast_planar` on a
single drone-count / speed / seed case.

## Relevant Tests

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest tests/unit/test_belief_grid.py tests/unit/test_sensor_model.py tests/unit/test_reduced_pomdp.py tests/unit/test_shared_track_ekf.py tests/integration/test_belief_coverage_env.py -q
```

Last verified result:

- `54 passed`

## Known Limitations

- This is still a single-threat, single-shared-track prototype.
- Base sensing is modeled as a simple range/noise/delay sensor, not a full radar/camera stack.
- The Phase 2 interceptor is acceleration-limited but still abstract.
- The fast planar validation mode skips DMPC/ADMM solver fidelity, so final claims should still be spot-checked with full mode.
- Seed `29` in the latest sweep exposed an observability gap where the threat is not confirmed or intercepted by any tested belief mode within 140/220 fast-planar steps, but it also does not reach the base.
- More seeds are needed before making statistical claims about base-compromise rate.

## Deferred To Phase 3

- Multi-target threat management.
- Moving-target shared tracking beyond one EKF track.
- Richer communication realism, including packet loss and bandwidth limits.
- Better sensor models for base and drones.
- Full PyBullet visualization/integration.
- RL training or learned belief-space policies.
- Interceptor handoff/assignment policies beyond the current single abstract interceptor.
