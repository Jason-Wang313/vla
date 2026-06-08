import numpy as np

from vla_best_of_n.deployment import VALID_GATES, deployment_gate
from vla_best_of_n.diagnostics import evaluate_selection
from vla_best_of_n.learned_vla import LearnedVLAScorer, RealUtilityCalibrator, model_features
from vla_best_of_n.scorers import hand_scorer_dict
from vla_best_of_n.vla_env import generate_pools


def test_vla_pool_has_separate_semantic_and_real_utility():
    pool = generate_pools([1], 1, 32, "controlled")[0]
    assert pool.language_vector.size > 0
    assert pool.observation_vector.size > 0
    assert pool.candidate_features.shape[0] == 32
    assert not np.allclose(pool.semantic_proxy, pool.real_utility)


def test_learned_scorer_uses_language_observation_and_action_features():
    pools = generate_pools([1], 3, 32, "learned_train")
    scorer = LearnedVLAScorer().fit(pools)
    scores = scorer.score_pool(pools[0])
    features = model_features(pools[0])
    assert features.shape[1] > pools[0].candidate_features.shape[1]
    assert scores.shape == (pools[0].size,)
    assert np.std(scores) > 0.001


def test_calibrator_improves_or_changes_real_utility_score_space():
    train = generate_pools([1], 4, 32, "learned_train")
    pilot = generate_pools([2], 2, 32, "learned_train")
    eval_pool = generate_pools([3], 1, 32, "learned_eval")[0]
    scorer = LearnedVLAScorer().fit(train)
    pilot_scores = scorer.score_pools(pilot)
    calibrator = RealUtilityCalibrator().fit(pilot, pilot_scores)
    raw = scorer.score_pool(eval_pool)
    calibrated = calibrator.score_pool(eval_pool, raw)
    assert not np.allclose(raw, calibrated)


def test_deployment_gate_returns_valid_value():
    decision = deployment_gate(
        {
            "high_N_regret": 0.35,
            "tail_rank_correlation": -0.2,
            "semantic_real_tail_gap": 0.5,
            "violation_rate": 0.6,
            "marginal_value_per_added_candidate": -0.01,
            "calibration_improvement": 0.0,
        }
    )
    assert decision in VALID_GATES
    assert decision == "block_high_n"


def test_diagnostics_emit_required_columns():
    pools = generate_pools([1], 2, 32, "controlled")
    result = evaluate_selection("unit", pools, hand_scorer_dict(pools), [1, 2, 4], 10)
    required = {
        "selected_real_utility",
        "selected_semantic_score",
        "semantic_real_tail_gap",
        "tail_rank_correlation",
        "high_N_regret",
        "violation_rate",
        "deployment_gate",
    }
    assert required.issubset(set(result.summary.columns))
