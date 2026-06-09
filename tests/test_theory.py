import numpy as np
import pytest

from vla_best_of_n.theory import (
    anti_aligned_score_example,
    binary_expected_success,
    constant_utility_edge_case,
    expected_selected_value,
    oracle_score_edge_case,
    tie_aware_selection_probabilities,
)


def test_tie_aware_probabilities_sum_to_one():
    law = tie_aware_selection_probabilities([1.0, 2.0, 2.0, 0.0], 2)
    assert law.total_probability == pytest.approx(1.0)
    assert law.probabilities[1] == pytest.approx(law.probabilities[2])


def test_constant_utility_edge_case_is_constant():
    values = constant_utility_edge_case()
    assert set(round(v, 8) for v in values.values()) == {0.37}


def test_oracle_score_is_monotone():
    values = list(oracle_score_edge_case().values())
    assert values == sorted(values)


def test_anti_aligned_score_decreases():
    values = list(anti_aligned_score_example().values())
    assert values[-1] < values[0]


def test_semantic_scores_alone_cannot_certify_tail_utility():
    scores = [0.1, 0.4, 0.8, 1.0]
    good_tail_utility = [0.0, 0.2, 0.7, 1.0]
    bad_tail_utility = [1.0, 0.7, 0.2, 0.0]
    semantic_only_law = tie_aware_selection_probabilities(scores, 4)

    good_selected = expected_selected_value(scores, good_tail_utility, 4)
    bad_selected = expected_selected_value(scores, bad_tail_utility, 4)

    assert semantic_only_law.probabilities.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert good_selected == pytest.approx(1.0)
    assert bad_selected == pytest.approx(0.0)


def test_binary_expected_success_matches_value_law():
    scores = [0.1, 0.9, 0.8]
    success = [1, 0, 1]
    assert binary_expected_success(scores, success, 2) == pytest.approx(
        expected_selected_value(scores, success, 2)
    )


def test_invalid_n_rejected():
    with pytest.raises(ValueError):
        tie_aware_selection_probabilities([1.0, 2.0], 3)
