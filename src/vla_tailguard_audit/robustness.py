"""Robustness scorers for imperfect grounding and pilot-label calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learned_vla import RealUtilityCalibrator
from .scorers import grounded_score, oracle_score, random_score
from .vla_env import CandidatePool


@dataclass(frozen=True)
class NoisyVerifierConfig:
    name: str
    false_positive_rate: float
    false_negative_rate: float
    score_noise: float
    hidden_constraint_rate: float
    seed_offset: int


def noisy_physical_feasibility(pool: CandidatePool, config: NoisyVerifierConfig) -> np.ndarray:
    """Perturb physical feasibility with seeded false positives/negatives."""

    rng = np.random.default_rng(pool.seed * 8191 + pool.state_id * 313 + config.seed_offset)
    feasible = pool.physical_feasibility >= 0.5
    true_feasible_score = max(0.60, 0.92 - 0.35 * config.false_negative_rate)
    false_positive_score = min(0.92, 0.72 + 0.75 * config.false_positive_rate)
    hidden_score = min(0.90, 0.70 + 0.75 * config.hidden_constraint_rate)
    noisy = np.where(feasible, true_feasible_score, 0.08 + 0.20 * config.false_positive_rate)
    false_negative = feasible & (rng.random(pool.size) < config.false_negative_rate)
    false_positive = (~feasible) & (rng.random(pool.size) < config.false_positive_rate)
    noisy[false_negative] = 0.18
    noisy[false_positive] = false_positive_score

    hidden = rng.random(pool.size) < config.hidden_constraint_rate
    # Hidden constraints are a verifier blind spot: some selected-tail violations
    # keep a high feasibility estimate despite physical failure.
    hidden_positive = hidden & (pool.flags["violation"] > 0.5)
    noisy[hidden_positive] = np.maximum(noisy[hidden_positive], hidden_score)

    score = noisy.astype(float)
    if config.score_noise > 0.0:
        score = np.clip(score + rng.normal(0.0, config.score_noise, size=pool.size), 0.0, 1.0)
    return score


def noisy_verifier_score(pool: CandidatePool, config: NoisyVerifierConfig) -> np.ndarray:
    feasibility = noisy_physical_feasibility(pool, config)
    return np.clip(pool.semantic_proxy * (0.15 + 0.85 * feasibility), 0.0, 1.0)


def robustness_scores(
    eval_pools: list[CandidatePool],
    learned_eval_scores: list[np.ndarray],
    pilot_pools: list[CandidatePool],
    learned_pilot_scores: list[np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Scores for imperfect verifier and calibration stress tests."""

    mild = NoisyVerifierConfig(
        name="noisy_verifier_mild",
        false_positive_rate=0.14,
        false_negative_rate=0.08,
        score_noise=0.08,
        hidden_constraint_rate=0.14,
        seed_offset=1,
    )
    harsh = NoisyVerifierConfig(
        name="noisy_verifier_harsh",
        false_positive_rate=0.22,
        false_negative_rate=0.14,
        score_noise=0.12,
        hidden_constraint_rate=0.20,
        seed_offset=2,
    )

    out: dict[str, list[np.ndarray]] = {
        "raw_semantic": learned_eval_scores,
        "ideal_verifier": [np.clip(pool.semantic_proxy * (0.15 + 0.85 * pool.physical_feasibility), 0.0, 1.0) for pool in eval_pools],
        mild.name: [noisy_verifier_score(pool, mild) for pool in eval_pools],
        harsh.name: [noisy_verifier_score(pool, harsh) for pool in eval_pools],
        "grounded_combined": [grounded_score(pool) for pool in eval_pools],
        "random": [random_score(pool) for pool in eval_pools],
        "oracle": [oracle_score(pool) for pool in eval_pools],
    }

    calibration_specs = [
        ("calib_1pct_noise0", 0.01, 0.0, 101),
        ("calib_2pct_noise5", 0.02, 0.05, 102),
        ("calib_5pct_noise10", 0.05, 0.10, 103),
        ("calib_10pct_noise20", 0.10, 0.20, 104),
    ]
    for name, fraction, noise, seed in calibration_specs:
        calibrator = RealUtilityCalibrator().fit(
            pilot_pools,
            learned_pilot_scores,
            sample_fraction=fraction,
            label_noise=noise,
            seed=seed,
        )
        out[name] = calibrator.score_pools(eval_pools, learned_eval_scores)
    return out
