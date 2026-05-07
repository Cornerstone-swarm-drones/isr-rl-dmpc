# Phase 1-3 Presentation Demos

This note lists the canonical, reproducible demo artifacts for the current
belief-coverage branch. These outputs are meant for presentation and teammate
handoff, not as statistical proof of field performance.

## One-Command Demo Suite

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/generate_presentation_outputs.py --output-dir visualizations/presentation
```

This writes:

- `visualizations/presentation/phase1_patrol_risk.png`
- `visualizations/presentation/phase2_moving_threat_ekf.png`
- `visualizations/presentation/phase3_sequential_threats.png`
- `visualizations/presentation/phase2_policy_metrics/response_policy_validation_metrics.json`
- `visualizations/presentation/phase2_policy_metrics/response_policy_validation_summary.csv`
- `visualizations/presentation/phase3_sequential_metrics/sequential_threat_metrics.json`
- `visualizations/presentation/phase3_sequential_metrics/sequential_threat_summary.csv`
- `visualizations/presentation/presentation_manifest.json`

## Individual Commands

### Phase 1 Patrol/Risk

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 120 --seed 7 --num-drones 6 --disable-persistent-threats --validation-dynamics fast_planar --output visualizations/presentation/phase1_patrol_risk.png
```

Shows deterministic home-strip patrol, patrol-risk belief, fused risk fields,
drone positions, target choices, and home-focused coverage without active
persistent threats.

Claim supported: the branch has a runnable Phase 1 patrol-risk environment
parallel to `MARLDMPCEnv`.

### Phase 2 Moving Threat + EKF + Interceptor

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 220 --seed 7 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy phase2 --threat-belief-mode limited_strict --validation-dynamics fast_planar --max-threat-cycles 1 --output visualizations/presentation/phase2_moving_threat_ekf.png
```

Shows a moving persistent threat, limited-strict confirmation, shared EKF track,
base sensor contribution, and constrained interceptor response.

Claim supported: the branch has a runnable moving-threat/base-defense loop with
shared EKF guidance and explicit limited-belief mode.

### Phase 3 Bounded Sequential Threats

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/visualize_belief_coverage.py --steps 260 --seed 11 --num-drones 6 --threat-speed-case fast --interceptor-guidance-mode ekf --response-policy phase2 --threat-belief-mode limited_strict --validation-dynamics fast_planar --max-threat-cycles 2 --enable-sequential-pending-threats --pending-threat-delay-steps 12 --output visualizations/presentation/phase3_sequential_threats.png
```

Shows a second pending incident appearing before the first is resolved, bounded
watchlist behavior, promotion after the first intercept, and two total threat
eliminations in the representative seed.

Claim supported: the branch can handle bounded sequential incidents with one
active EKF/interceptor focus at a time.

## Validation Commands

Phase 2 policy comparison:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_ekf_response_policy.py --steps 140 --num-drones-cases 4,5,6 --speed-cases medium,fast --seeds 7,11 --policies improved,phase2 --skip-diagnostic-plots --validation-dynamics fast_planar --threat-belief-mode limited_strict --output-dir visualizations/presentation/phase2_policy_metrics
```

Phase 3 sequential-threat smoke:

```bash
PYTHONPATH=/Users/apple/Desktop/cornerstone/isr-rl-dmpc/src python3 scripts/experiment_sequential_threats.py --steps 260 --num-drones-cases 4,5,6 --speed-cases medium,fast --seeds 7,11 --max-threat-cycles 2 --pending-threat-delay-steps 12 --validation-dynamics fast_planar --threat-belief-mode limited_strict --output-dir visualizations/presentation/phase3_sequential_metrics
```

## Honest Limitations

- These are deterministic smoke/demo seeds, not broad statistical guarantees.
- `fast_planar` is a validation-speed approximation; full DMPC spot checks are
  still needed before strong claims.
- Phase 3 is bounded sequential handling, not full simultaneous multi-target
  tracking or assignment.
- `limited_strict` intentionally delays confirmation/launch compared with a
  globally shared/oracle belief model.
