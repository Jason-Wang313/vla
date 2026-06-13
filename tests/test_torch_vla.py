import numpy as np
import pytest

from vla_tailguard_audit.rendering import attach_visual_observation
from vla_tailguard_audit.simulator import simulate_pool
from vla_tailguard_audit.torch_vla import TorchVLAScorer, torch_available
from vla_tailguard_audit.vla_env import generate_pools


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not available")
def test_torch_vla_scorer_trains_and_scores_visual_pools():
    train = [simulate_pool(attach_visual_observation(pool)) for pool in generate_pools([101], 3, 48, "learned_train")]
    eval_pool = train[0]
    scorer = TorchVLAScorer(seed=5, epochs=20, batch_size=64).fit(train)
    scores = scorer.score_pool(eval_pool)
    assert scores.shape == (eval_pool.size,)
    assert np.all(np.isfinite(scores))
    assert np.min(scores) >= 0.0
    assert np.max(scores) <= 1.0
    assert np.std(scores) > 1e-4
