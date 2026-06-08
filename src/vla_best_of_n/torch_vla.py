"""PyTorch VLA-style semantic scorer.

The scorer is intentionally trained on semantic/affordance labels, not real
physical utility. It is used as a stronger learned VLA-style scorer while the
evaluation still measures real utility separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learned_vla import model_features
from .vla_env import CandidatePool

try:  # pragma: no cover - import availability is environment-specific
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None and nn is not None


@dataclass
class TorchVLAScorer:
    """A small MLP semantic/affordance scorer over VLA-style features."""

    seed: int = 0
    epochs: int = 24
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    model: object | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def _expanded_features(self, pool: CandidatePool) -> np.ndarray:
        x = model_features(pool)
        # Include lightweight cross terms so the MLP sees language/visual/action
        # interactions, not merely concatenated tabular features.
        return np.hstack([x, x[:, :8] * x[:, 8:16], x[:, -8:] ** 2]).astype(np.float32)

    def fit(self, pools: list[CandidatePool]) -> "TorchVLAScorer":
        if not torch_available():
            raise RuntimeError("PyTorch is not available")
        x = np.vstack([self._expanded_features(pool) for pool in pools]).astype(np.float32)
        y = np.concatenate([pool.semantic_proxy for pool in pools]).astype(np.float32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        x = (x - self.mean) / self.std

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        net = nn.Sequential(
            nn.Linear(x.shape[1], 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y[:, None], dtype=torch.float32)
        n = len(xt)
        epochs = max(1, int(self.epochs))
        for _ in range(epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                pred = net(xt[idx])
                loss = loss_fn(pred, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model = net.eval()
        return self

    def score_pool(self, pool: CandidatePool) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("TorchVLAScorer must be fit before scoring")
        x = self._expanded_features(pool)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            score = self.model(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1)
        return np.clip(score, 0.0, 1.0)

    def score_pools(self, pools: list[CandidatePool]) -> list[np.ndarray]:
        return [self.score_pool(pool) for pool in pools]
