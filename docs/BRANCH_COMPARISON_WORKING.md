# Branch Comparison: `phase1-belief-coverage` vs `working`

This note records the consolidation/comparison pass between the local Phase 1-3
belief-coverage branch and the classmates' `working` branch fetched from
GitHub. It is intentionally evidence-first: results below are from git
inspection, local tests, and local demo/validation commands.

## Branch State

Commands run from `/Users/apple/Desktop/cornerstone/isr-rl-dmpc`:

```bash
git status --short --branch
git fetch origin working:refs/remotes/origin/working
git rev-list --left-right --count HEAD...origin/working
git merge-base HEAD origin/working
git diff --name-status origin/working..HEAD
git diff --name-status HEAD..origin/working
```

Observed state:

- Current branch: `phase1-belief-coverage`.
- Local branch is clean and ahead of `origin/phase1-belief-coverage` by 2
  commits before this comparison report.
- Current local HEAD before this report: `2d7b9fe Add presentation demo generation guide`.
- Fetched `origin/working`: `b532f3c Fix Search and Track`.
- Divergence from `origin/working`: `5 14` from `git rev-list --left-right --count HEAD...origin/working`.
- Merge base: `e68471bb517c83f7336c22ba7263f2df27dcbe51`.

Recent local belief-coverage commits:

- `df92316 Add bounded sequential threat handling`
- `2d7b9fe Add presentation demo generation guide`

## Phase Branch Additions

The `phase1-belief-coverage` branch adds or preserves the Phase 1-3
belief-coverage system:

- `BeliefCoverageEnv` remains parallel to `MARLDMPCEnv`.
- Phase 1 patrol-risk belief grid, local/global belief split, deterministic
  home-region patrol, risk visualizations, and validation tests.
- Phase 2 moving persistent threat loop, explicit base sensor, shared EKF
  tracking, constrained predictive interceptor, `fast_planar` validation mode,
  and communication realism modes including `limited_strict`.
- Phase 3 bounded sequential threat handling with one active focus, one pending
  threat, promotion after active-threat removal, and sequential validation.
- Presentation generator and teammate-facing presentation notes:
  `scripts/generate_presentation_outputs.py` and `docs/PRESENTATION_DEMOS.md`.

Files present on this branch that are absent from `working` include:

- `docs/PHASE3_SEQUENTIAL_THREATS.md`
- `docs/PRESENTATION_DEMOS.md`
- `scripts/experiment_sequential_threats.py`
- `scripts/generate_presentation_outputs.py`

## Working Branch Additions

The `working` branch contains useful presentation/PyBullet work that is not
currently present on the belief branch:

- `pybullet_sim/pybullet_recorder.py`
- `src/isr_rl_dmpc/models/hector_quadrotor/quadrotor.urdf`
- `docs/presentation/isr_rl_dmpc_presentation.pptx`
- Larger PyBullet visualization changes in `pybullet_sim/swarm_pybullet_sim.py`
- Staggered launch / spawn-interval behavior in `MARLDMPCEnv`, `run_dmpc.py`,
  `run_dmpc_rl.py`, and simulator launch-mask handling.
- README, gym-design, and troubleshooting updates focused on DMPC/PyBullet demo
  flows.

These are worth reviewing for integration, especially the PyBullet recorder and
URDF assets. The `MARLDMPCEnv` and simulator behavior changes should not be
merged blindly because the belief branch deliberately kept the existing
continuous-control interface stable.

## Validation Evidence

### Belief Branch

Command run:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest \
  tests/unit/test_belief_grid.py \
  tests/unit/test_sensor_model.py \
  tests/unit/test_reduced_pomdp.py \
  tests/unit/test_shared_track_ekf.py \
  tests/integration/test_belief_coverage_env.py -q
```

Observed result:

- `56 passed`

Presentation output command run:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src \
python3 scripts/generate_presentation_outputs.py --output-dir visualizations/presentation
```

Observed generated outputs:

- `visualizations/presentation/phase1_patrol_risk.png`
- `visualizations/presentation/phase2_moving_threat_ekf.png`
- `visualizations/presentation/phase3_sequential_threats.png`
- `visualizations/presentation/phase2_policy_metrics/response_policy_validation_metrics.json`
- `visualizations/presentation/phase2_policy_metrics/response_policy_validation_runs.csv`
- `visualizations/presentation/phase2_policy_metrics/response_policy_validation_summary.csv`
- `visualizations/presentation/phase3_sequential_metrics/sequential_threat_metrics.json`
- `visualizations/presentation/phase3_sequential_metrics/sequential_threat_runs.csv`
- `visualizations/presentation/phase3_sequential_metrics/sequential_threat_summary.csv`
- `visualizations/presentation/presentation_manifest.json`

The `visualizations/` tree is ignored by `.gitignore`, so these are local demo
artifacts rather than committed source files.

### Working Branch

The `working` branch was inspected in a detached worktree at:

```text
/tmp/isr-working-compare.JNgQW7
```

Command run:

```bash
PYTHONPATH=src pytest \
  tests/unit/test_belief_grid.py \
  tests/unit/test_sensor_model.py \
  tests/unit/test_reduced_pomdp.py \
  tests/unit/test_shared_track_ekf.py \
  tests/integration/test_belief_coverage_env.py -q
```

Observed result:

- `52 passed`

Command run:

```bash
python3 -m py_compile \
  scripts/run_dmpc.py \
  scripts/run_dmpc_rl.py \
  scripts/visualize_belief_coverage.py \
  scripts/experiment_ekf_response_policy.py \
  pybullet_sim/swarm_pybullet_sim.py \
  pybullet_sim/pybullet_recorder.py \
  src/isr_rl_dmpc/gym_env/marl_env.py \
  src/isr_rl_dmpc/gym_env/belief_coverage_env.py \
  src/isr_rl_dmpc/gym_env/simulator.py
```

Observed result:

- Exit code `0`.

Command run:

```bash
PYTHONPATH=src python3 scripts/visualize_belief_coverage.py \
  --steps 20 \
  --seed 7 \
  --num-drones 4 \
  --validation-dynamics fast_planar \
  --threat-belief-mode limited_strict \
  --interceptor-guidance-mode ekf \
  --response-policy phase2 \
  --max-threat-cycles 1 \
  --output /tmp/working_belief_smoke.png
```

Observed result:

- Exit code `2`.
- The CLI rejected `limited_strict`: valid choices on `working` are only
  `shared` and `limited`.

Compatible working-branch command run:

```bash
PYTHONPATH=src python3 scripts/visualize_belief_coverage.py \
  --steps 20 \
  --seed 7 \
  --num-drones 4 \
  --validation-dynamics fast_planar \
  --threat-belief-mode limited \
  --interceptor-guidance-mode ekf \
  --response-policy phase2 \
  --max-threat-cycles 1 \
  --output /tmp/working_belief_smoke.png
```

Observed result:

- Exit code `0`.
- Smoke run completed 20 steps with no threat confirmation/launch/intercept
  because the run was intentionally short.

PyBullet smoke command run on `working`:

```bash
PYTHONPATH=src python3 pybullet_sim/swarm_pybullet_sim.py --no-gui --max-steps 5
```

Observed result:

- Exit code `0`.
- The script fell back to headless mode because `pybullet` was not installed in
  the current shell.

DMPC script smoke command run on `working`:

```bash
PYTHONPATH=src python3 scripts/run_dmpc.py --scenario area_surveillance --max-steps 5
```

Observed result:

- Failed with `ModuleNotFoundError: No module named 'yaml'`.
- This is an environment dependency issue in the current shell, not direct proof
  of a branch code defect. `requirements/base.txt` includes `pyyaml`.

## Serious Differences and Issues

Evidence-backed issues found in `working`:

- `working` does not support `threat_belief_mode="limited_strict"`, while the
  completed Phase 2/3 belief branch recommends `limited_strict` as the default
  evaluation mode.
- `working` does not contain the Phase 3 sequential-threat implementation,
  sequential experiment script, Phase 3 handoff doc, or presentation generator.
- `working` modifies `MARLDMPCEnv` and the shared simulator for staggered
  launch behavior. That may be useful, but it is a higher-risk integration area
  because the belief branch intentionally preserved the existing continuous
  `Q/R`-scaling path.
- `working` contains useful PyBullet/recorder assets, but the current local
  shell did not have `pybullet` installed, so only the no-GUI fallback was
  validated here.
- Some docs on `working` appear stale relative to the latest belief branch:
  they do not describe `limited_strict`, bounded sequential threats, or the
  canonical presentation-output generator.

Judgment, based on the code comparison:

- `working` is not simply "broken"; its unit/integration tests passed and key
  scripts compile.
- `working` is behind the belief branch on Phase 2/3 behavior and cannot run the
  latest recommended belief-coverage evaluation commands as-is.
- The PyBullet additions in `working` look valuable, but the `MARLDMPCEnv` and
  simulator changes should be treated as integration candidates rather than
  automatically accepted wholesale.

## Integration Recommendation

Safest path: use `phase1-belief-coverage` as the base branch and selectively
port reviewed `working` features into it.

Recommended order:

1. Keep all Phase 1-3 belief-coverage source, tests, scripts, and docs from
   `phase1-belief-coverage`.
2. Port `working`'s presentation/PyBullet assets in a focused PR or commit:
   `pybullet_sim/pybullet_recorder.py`, the Hector quadrotor URDF, and selected
   non-invasive visualization improvements from `pybullet_sim/swarm_pybullet_sim.py`.
3. Review `working`'s `MARLDMPCEnv` staggered-launch changes separately. If they
   are needed, add regression tests proving the original continuous-control
   action path still behaves as expected.
4. Review simulator changes separately, especially lateral-only target detection
   and launch-mask handling, because they affect shared infrastructure.
5. Reconcile docs last, after code behavior is decided. Preserve the latest
   `PHASE2_BASE_DEFENSE.md`, `PHASE3_SEQUENTIAL_THREATS.md`, and
   `PRESENTATION_DEMOS.md` content from the belief branch.
6. Re-run the belief branch tests and canonical presentation generator after
   each integration step.

What should not be merged as-is:

- A full branch merge from `working` into `phase1-belief-coverage`, because it
  would remove or downgrade `limited_strict`, sequential-threat handling, and
  current presentation-generation files.
- A full branch merge from `phase1-belief-coverage` into `working` without
  explicitly preserving the PyBullet recorder, URDF, and any teammate-owned demo
  assets.

## Canonical Presentation Commands

Generate the full presentation output bundle:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src \
python3 /Users/apple/Desktop/cornerstone/isr-rl-dmpc/scripts/generate_presentation_outputs.py \
  --output-dir /Users/apple/Desktop/cornerstone/isr-rl-dmpc/visualizations/presentation
```

Run the focused Phase 3 sequential validation:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src \
python3 /Users/apple/Desktop/cornerstone/isr-rl-dmpc/scripts/experiment_sequential_threats.py \
  --steps 260 \
  --num-drones-cases 4,5,6 \
  --speed-cases medium,fast \
  --seeds 7,11 \
  --max-threat-cycles 2 \
  --pending-threat-delay-steps 12 \
  --validation-dynamics fast_planar \
  --threat-belief-mode limited_strict \
  --output-dir /Users/apple/Desktop/cornerstone/isr-rl-dmpc/visualizations/phase3_sequential_final
```

Run the relevant test suite:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src pytest \
  /Users/apple/Desktop/cornerstone/isr-rl-dmpc/tests/unit/test_belief_grid.py \
  /Users/apple/Desktop/cornerstone/isr-rl-dmpc/tests/unit/test_sensor_model.py \
  /Users/apple/Desktop/cornerstone/isr-rl-dmpc/tests/unit/test_reduced_pomdp.py \
  /Users/apple/Desktop/cornerstone/isr-rl-dmpc/tests/unit/test_shared_track_ekf.py \
  /Users/apple/Desktop/cornerstone/isr-rl-dmpc/tests/integration/test_belief_coverage_env.py -q
```

## Remaining Risks

- The local branch is committed but still needs to be pushed to
  `origin/phase1-belief-coverage` if teammates should see the latest commits on
  GitHub.
- The current shell is missing some project dependencies (`pyyaml` and
  `pybullet` were observed as missing in branch smoke checks). A clean project
  environment should install `requirements/base.txt` and `requirements/dev.txt`.
- The Phase 3 sequential implementation is intentionally bounded: one active
  focus, one pending threat, one interceptor. True simultaneous multi-target
  tracking remains a future Phase 3/4 task.
- PyBullet presentation integration should be a follow-up integration slice,
  not mixed into the current Phase 1-3 belief behavior branch without tests.
