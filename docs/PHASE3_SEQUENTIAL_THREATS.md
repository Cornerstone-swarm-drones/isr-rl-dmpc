# Phase 3 Sequential Threat Slice

This slice starts Phase 3 by extending the completed Phase 2 base-defense path
from a single isolated threat incident to bounded sequential incidents.
`BeliefCoverageEnv` remains parallel to `MARLDMPCEnv`; no RL/training path is
changed.

## Scope

- Preserves the Phase 2 default evaluation path:
  `response_policy="phase2"`, `interceptor_guidance_mode="ekf"`,
  `validation_dynamics="fast_planar"`, and
  `threat_belief_mode="limited_strict"`.
- Keeps one active threat focus, one shared EKF, and one interceptor at a time.
- Adds an opt-in pending threat queue for the next incident.
- A pending threat can appear in diagnostics before the active threat is
  resolved, then is promoted after the current threat is intercepted.
- Visible pending threats now act as bounded watchlist cues:
  - up to two non-active-tracking drones may take a local watchlist detour
  - real pre-promotion observations accumulate a pending persistence score
  - the promoted threat carries only sub-threshold/local belief memory unless
    repeated observations already made it persistent
- Does not implement full simultaneous multi-target task allocation.

## Canonical Phase 3 Demo Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 260 --seed 11 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy phase2 --threat-belief-mode limited_strict --validation-dynamics fast_planar --max-threat-cycles 2 --enable-sequential-pending-threats --pending-threat-delay-steps 12 --output visualizations/phase3_sequential_demo/phase3_sequential_final_6d_fast_seed11.png
```

Output:

- `visualizations/phase3_sequential_demo/phase3_sequential_final_6d_fast_seed11.png`

Last verified result:

- 6 drones, fast threat, seed 11, 260 steps.
- Pending second threat appeared at step 12.
- Threat removals occurred at steps 103 and 131.
- Threat eliminations: 2.
- Mission failed: false.

## Canonical Phase 3 Validation Command

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_sequential_threats.py --steps 260 --num-drones-cases 4,5,6 --speed-cases medium,fast --seeds 7,11 --max-threat-cycles 2 --pending-threat-delay-steps 12 --validation-dynamics fast_planar --threat-belief-mode limited_strict --output-dir visualizations/phase3_sequential_final_preconfirm_260
```

Outputs:

- `visualizations/phase3_sequential_final_preconfirm_260/sequential_threat_metrics.json`
- `visualizations/phase3_sequential_final_preconfirm_260/sequential_threat_summary.csv`
- `visualizations/phase3_sequential_final_preconfirm_260/sequential_threat_runs.csv`

Last verified result:

- 12 fast-planar rollouts.
- `mission_fail_rate = 0.0` for all tested 4-, 5-, and 6-drone medium/fast cases.
- The pending second threat appeared in every tested run.
- Both threats were cleared in 50% of 260-step runs.
- Successful second-threat cases now confirm and relaunch one step after
  promotion because repeated pending observations are retained.
- The best 6-drone fast seed-11 case improved from second removal step 167 to
  131.
- The seed-7 cases still miss the second threat inside 260 steps; the 6-drone
  fast seed-7 longer 360-step spot check still clears both threats with no base
  failure.

## Relevant Tests

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest tests/unit/test_belief_grid.py tests/unit/test_sensor_model.py tests/unit/test_reduced_pomdp.py tests/unit/test_shared_track_ekf.py tests/integration/test_belief_coverage_env.py -q
```

Last verified result:

- `56 passed`

## Deferred

- True simultaneous multi-target tracking and assignment.
- Multiple active interceptors.
- Prioritization between two confirmed moving threats.
- Richer communication constraints under sequential incidents.
- Full PyBullet visualization and RL training.
