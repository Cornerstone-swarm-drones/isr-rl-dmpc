from __future__ import annotations

import numpy as np

from isr_rl_dmpc.core import (
    QMDPPlanner,
    ReducedLocalPatrolPOMDP,
    ReducedPatrolAction,
    ReducedPatrolObservation,
)


def test_reduced_pomdp_transition_and_observation_models_are_normalized() -> None:
    pomdp = ReducedLocalPatrolPOMDP()

    assert np.allclose(np.sum(pomdp.transition_matrix, axis=-1), 1.0)
    assert np.allclose(np.sum(pomdp.observation_matrix, axis=-1), 1.0)


def test_focused_revisit_belief_update_disambiguates_neglect_from_persistent_threat() -> None:
    pomdp = ReducedLocalPatrolPOMDP()
    prior = pomdp.factorized_belief(threat_probability=0.4, neglect_probability=0.7)
    prior_threat = float(prior[2] + prior[3])

    quiet_posterior = pomdp.belief_update(
        prior,
        ReducedPatrolAction.FOCUSED_REVISIT,
        ReducedPatrolObservation.QUIET,
    )
    persistent_posterior = pomdp.belief_update(
        prior,
        ReducedPatrolAction.FOCUSED_REVISIT,
        ReducedPatrolObservation.PERSISTENT,
    )

    assert np.isclose(np.sum(quiet_posterior), 1.0)
    assert np.isclose(np.sum(persistent_posterior), 1.0)
    assert float(quiet_posterior[2] + quiet_posterior[3]) < prior_threat
    assert float(persistent_posterior[2] + persistent_posterior[3]) > prior_threat


def test_qmdp_policy_matches_interpretable_belief_regimes() -> None:
    pomdp = ReducedLocalPatrolPOMDP()
    planner = QMDPPlanner(pomdp)

    calm = pomdp.factorized_belief(threat_probability=0.05, neglect_probability=0.10)
    likely_neglect = pomdp.factorized_belief(threat_probability=0.10, neglect_probability=0.80)
    likely_threat = pomdp.factorized_belief(threat_probability=0.85, neglect_probability=0.55)

    assert planner.select_action(calm) == ReducedPatrolAction.ROUTINE_PATROL
    assert planner.select_action(likely_neglect) == ReducedPatrolAction.FOCUSED_REVISIT
    assert planner.select_action(likely_threat) == ReducedPatrolAction.ESCALATE_AND_TRACK
