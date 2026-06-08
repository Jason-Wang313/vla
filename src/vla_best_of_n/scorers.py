"""Scorers for VLA-style Best-of-N selection."""

from __future__ import annotations

import numpy as np

from .vla_env import CandidatePool, FEATURE_NAMES


def semantic_score(pool: CandidatePool) -> np.ndarray:
    return pool.semantic_proxy.copy()


def physical_score(pool: CandidatePool) -> np.ndarray:
    return pool.physical_feasibility.copy()


def semantic_plus_physical_score(pool: CandidatePool) -> np.ndarray:
    """A verifier-style score that gates semantic plausibility by feasibility."""

    return np.clip(pool.semantic_proxy * (0.15 + 0.85 * pool.physical_feasibility), 0.0, 1.0)


def grounded_score(pool: CandidatePool) -> np.ndarray:
    """Combined semantic, physical, and object-binding score."""

    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    object_grounding = 0.5 * pool.candidate_features[:, idx["correct_object"]]
    object_grounding += 0.5 * pool.candidate_features[:, idx["correct_receptacle"]]
    score = 0.45 * pool.semantic_proxy + 0.35 * pool.physical_feasibility + 0.20 * object_grounding
    return np.clip(score, 0.0, 1.0)


def oracle_score(pool: CandidatePool) -> np.ndarray:
    return pool.real_utility.copy()


def random_score(pool: CandidatePool) -> np.ndarray:
    return pool.random_score.copy()


def hand_scorer_dict(pools: list[CandidatePool]) -> dict[str, list[np.ndarray]]:
    return {
        "raw_semantic": [semantic_score(pool) for pool in pools],
        "semantic_plus_physical": [semantic_plus_physical_score(pool) for pool in pools],
        "grounded_combined": [grounded_score(pool) for pool in pools],
        "random": [random_score(pool) for pool in pools],
        "oracle": [oracle_score(pool) for pool in pools],
    }
