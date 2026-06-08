import numpy as np

from vla_best_of_n.learned_vla import LearnedVLAScorer
from vla_best_of_n.rendering import attach_visual_observation
from vla_best_of_n.robustness import NoisyVerifierConfig, noisy_physical_feasibility, robustness_scores
from vla_best_of_n.simulator import simulate_pool
from vla_best_of_n.vla_env import generate_pools


def test_noisy_verifier_is_seeded_and_imperfect():
    pool = simulate_pool(attach_visual_observation(generate_pools([1], 1, 64, "rendered_visual")[0]))
    config = NoisyVerifierConfig("unit", 0.2, 0.1, 0.05, 0.1, 7)
    a = noisy_physical_feasibility(pool, config)
    b = noisy_physical_feasibility(pool, config)
    assert np.allclose(a, b)
    assert a.shape == (pool.size,)
    assert np.any(np.abs(a - pool.physical_feasibility) > 1e-6)


def test_robustness_scores_include_noisy_and_budget_methods():
    train = [simulate_pool(attach_visual_observation(pool)) for pool in generate_pools([101], 4, 64, "learned_train")]
    pilot = [simulate_pool(attach_visual_observation(pool)) for pool in generate_pools([201], 2, 64, "learned_train")]
    eval_pools = [simulate_pool(attach_visual_observation(pool)) for pool in generate_pools([1], 2, 64, "rendered_visual")]
    scorer = LearnedVLAScorer().fit(train)
    eval_scores = scorer.score_pools(eval_pools)
    pilot_scores = scorer.score_pools(pilot)
    scores = robustness_scores(eval_pools, eval_scores, pilot, pilot_scores)
    expected = {
        "noisy_verifier_mild",
        "noisy_verifier_harsh",
        "calib_1pct_noise0",
        "calib_2pct_noise5",
        "calib_5pct_noise10",
        "calib_10pct_noise20",
    }
    assert expected.issubset(scores)
    for method in expected:
        assert len(scores[method]) == len(eval_pools)
        assert scores[method][0].shape == (eval_pools[0].size,)
