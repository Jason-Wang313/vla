"""Exact finite tie-aware Best-of-N laws.

The law here audits a fixed finite candidate pool. A subset of N candidates is
sampled uniformly without replacement; the candidate with the largest score is
selected; if several sampled candidates tie for largest score, the tie is
broken uniformly among the tied candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SelectionLawResult:
    """Tie-aware selection probabilities for a finite candidate pool."""

    scores: np.ndarray
    n: int
    probabilities: np.ndarray

    @property
    def total_probability(self) -> float:
        return float(np.sum(self.probabilities))


def _as_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def tie_aware_selection_probabilities(scores: Iterable[float], n: int) -> SelectionLawResult:
    """Return exact candidate-selection probabilities for finite Best-of-N.

    The formula is exact for a fixed finite pool with tied scores. For candidate
    i, it sums over how many same-score candidates are also sampled, excludes
    all higher-score candidates, and divides uniformly among sampled max-score
    ties.
    """

    score = _as_vector(scores, "scores")
    m = len(score)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > m:
        raise ValueError("n cannot exceed the finite pool size")

    denom = comb(m, n)
    probs = np.zeros(m, dtype=float)
    for s_i in np.unique(score):
        group = np.flatnonzero(score == s_i)
        group_size = len(group)
        lower = int(np.sum(score < s_i))
        group_mass = 0.0
        min_from_group = max(1, n - lower)
        max_from_group = min(group_size, n)
        for from_group in range(min_from_group, max_from_group + 1):
            lower_needed = n - from_group
            group_mass += comb(group_size, from_group) * comb(lower, lower_needed) / denom
        probs[group] = group_mass / group_size

    # Numerical drift can occur only after conversion from exact integers.
    probs = probs / probs.sum()
    return SelectionLawResult(scores=score, n=n, probabilities=probs)


def expected_selected_value(scores: Iterable[float], values: Iterable[float], n: int) -> float:
    """Expected selected value under the tie-aware finite Best-of-N law."""

    score = _as_vector(scores, "scores")
    value = _as_vector(values, "values")
    if len(score) != len(value):
        raise ValueError("scores and values must have the same length")
    law = tie_aware_selection_probabilities(score, n)
    return float(np.dot(law.probabilities, value))


def binary_expected_success(scores: Iterable[float], successes: Iterable[int], n: int) -> float:
    """Expected selected binary utility under the same tie-aware law."""

    success = _as_vector(successes, "successes")
    if not np.all((success == 0.0) | (success == 1.0)):
        raise ValueError("successes must be binary 0/1 values")
    return expected_selected_value(scores, success, n)


def tie_aware_argmax(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a tie-aware argmax index from a score vector."""

    max_score = np.max(scores)
    tied = np.flatnonzero(scores == max_score)
    return int(rng.choice(tied))


def monte_carlo_selected_values(
    scores: Iterable[float],
    values: Iterable[float],
    n: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo selected values for finite-pool Best-of-N."""

    score = _as_vector(scores, "scores")
    value = _as_vector(values, "values")
    if len(score) != len(value):
        raise ValueError("scores and values must have the same length")
    if trials < 1:
        raise ValueError("trials must be positive")
    m = len(score)
    if n > m:
        raise ValueError("n cannot exceed the finite pool size")

    out = np.zeros(trials, dtype=float)
    for t in range(trials):
        subset = rng.choice(m, size=n, replace=False)
        local = tie_aware_argmax(score[subset], rng)
        out[t] = value[subset[local]]
    return out


def exact_curve(scores: Iterable[float], values: Iterable[float], ns: Iterable[int]) -> dict[int, float]:
    """Expected selected values for multiple N values."""

    return {int(n): expected_selected_value(scores, values, int(n)) for n in ns}


def constant_utility_edge_case() -> dict[str, float]:
    """A constant real utility remains constant under any scorer and N."""

    scores = [0.1, 0.4, 0.4, 0.9]
    values = [0.37, 0.37, 0.37, 0.37]
    return {str(n): expected_selected_value(scores, values, n) for n in [1, 2, 3, 4]}


def oracle_score_edge_case() -> dict[str, float]:
    """Oracle score equals utility, so selected expected utility is monotone."""

    values = [0.0, 0.2, 0.7, 1.0]
    return {str(n): expected_selected_value(values, values, n) for n in [1, 2, 3, 4]}


def anti_aligned_score_example() -> dict[str, float]:
    """Anti-aligned score can make larger N select lower real utility."""

    utilities = np.asarray([0.0, 0.2, 0.7, 1.0], dtype=float)
    scores = -utilities
    return {str(n): expected_selected_value(scores, utilities, n) for n in [1, 2, 3, 4]}
