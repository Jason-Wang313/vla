"""Learned VLA-style semantic scorer and calibrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vla_env import CandidatePool

def model_features(pool: CandidatePool, learned_semantic: np.ndarray | None = None) -> np.ndarray:
    """Concatenate language, visual/object observation, and action features."""

    repeated_lang = np.repeat(pool.language_vector[None, :], pool.size, axis=0)
    repeated_obs = np.repeat(pool.observation_vector[None, :], pool.size, axis=0)
    if pool.visual_vector is None:
        visual = np.zeros(8, dtype=np.float32)
    else:
        visual = np.asarray(pool.visual_vector, dtype=np.float32)
        if visual.size != 8:
            raise ValueError("visual_vector must have 8 entries")
    repeated_visual = np.repeat(visual[None, :], pool.size, axis=0)
    pieces = [repeated_lang, repeated_obs, repeated_visual, pool.candidate_features]
    if learned_semantic is not None:
        pieces.append(learned_semantic[:, None])
        pieces.append(pool.physical_feasibility[:, None])
    return np.hstack(pieces).astype(np.float32)


@dataclass
class LearnedVLAScorer:
    """A learned semantic/affordance scorer, not a real-utility model."""

    seed: int = 0
    epochs: int = 80
    lr: float = 0.01
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, pools: list[CandidatePool]) -> "LearnedVLAScorer":
        x = np.vstack([model_features(pool) for pool in pools])
        y = np.concatenate([pool.semantic_proxy for pool in pools]).astype(np.float32)
        # Compact nonlinear expansion gives a learned language-observation-action
        # scorer without importing a large deep-learning runtime.
        x = np.hstack([x, x[:, :6] * x[:, 6:12], x[:, 10:16] ** 2]).astype(np.float32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        x_norm = (x - self.mean) / self.std
        x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
        ridge = 2e-3 * np.eye(x_aug.shape[1])
        ridge[0, 0] = 0.0
        self.weights = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y)
        return self

    def score_pool(self, pool: CandidatePool) -> np.ndarray:
        x = model_features(pool)
        if self.mean is None or self.std is None:
            raise RuntimeError("LearnedVLAScorer must be fit before scoring")
        x = np.hstack([x, x[:, :6] * x[:, 6:12], x[:, 10:16] ** 2]).astype(np.float32)
        x_norm = (x - self.mean) / self.std
        if self.weights is None:
            raise RuntimeError("weights are missing")
        x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
        return np.clip(x_aug @ self.weights, 0.0, 1.0)

    def score_pools(self, pools: list[CandidatePool]) -> list[np.ndarray]:
        return [self.score_pool(pool) for pool in pools]


@dataclass
class RealUtilityCalibrator:
    """Small pilot-label calibrator for selected-tail real utility."""

    l2: float = 1e-2
    weights: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(
        self,
        pools: list[CandidatePool],
        learned_scores: list[np.ndarray],
        sample_fraction: float = 1.0,
        label_noise: float = 0.0,
        seed: int = 0,
    ) -> "RealUtilityCalibrator":
        x = np.vstack([model_features(pool, score) for pool, score in zip(pools, learned_scores)])
        y = np.concatenate([pool.real_utility for pool in pools])
        if not (0.0 < sample_fraction <= 1.0):
            raise ValueError("sample_fraction must be in (0, 1]")
        if label_noise < 0.0:
            raise ValueError("label_noise must be nonnegative")
        if sample_fraction < 1.0:
            rng = np.random.default_rng(seed)
            count = max(2, int(np.ceil(sample_fraction * len(y))))
            chosen = np.sort(rng.choice(len(y), size=count, replace=False))
            x = x[chosen]
            y = y[chosen]
        if label_noise > 0.0:
            rng = np.random.default_rng(seed + 10_007)
            y = np.clip(y + rng.normal(0.0, label_noise, size=len(y)), 0.0, 1.0)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        x_norm = (x - self.mean) / self.std
        x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
        ridge = self.l2 * np.eye(x_aug.shape[1])
        ridge[0, 0] = 0.0
        self.weights = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y)
        return self

    def score_pool(self, pool: CandidatePool, learned_score: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.std is None:
            raise RuntimeError("RealUtilityCalibrator must be fit before scoring")
        x = model_features(pool, learned_score)
        x_norm = (x - self.mean) / self.std
        x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
        return np.clip(x_aug @ self.weights, 0.0, 1.0)

    def score_pools(
        self,
        pools: list[CandidatePool],
        learned_scores: list[np.ndarray],
    ) -> list[np.ndarray]:
        return [self.score_pool(pool, score) for pool, score in zip(pools, learned_scores)]
