"""Certified TailGuard-BoN: certified tail-calibrated Best-of-N control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .deployment import VALID_GATES
from .theory import expected_selected_value


CERTIFICATE_FAILURE_TYPES = (
    "reach_envelope",
    "swept_volume_collision",
    "receptacle_compatibility",
    "stability_margin",
    "fragile_heavy_handling",
    "tool_object_match",
    "blocked_path_constraints",
    "hidden_obstacle",
)


_CERTIFICATE_FLAG_MAP: Mapping[str, tuple[str, ...]] = {
    "reach_envelope": ("unreachable",),
    "swept_volume_collision": ("collision",),
    "receptacle_compatibility": ("receptacle_incompatible", "wrong_target"),
    "stability_margin": ("stability_failure",),
    "fragile_heavy_handling": ("fragile", "heavy_failure"),
    "tool_object_match": ("wrong_object", "tool_mismatch"),
    "blocked_path_constraints": ("blocked_path",),
    "hidden_obstacle": ("hidden_constraint",),
}


def _as_array(values: Iterable[float], name: str) -> np.ndarray:
    out = np.asarray(list(values), dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(out) == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _flatten(items: Sequence[np.ndarray] | np.ndarray, name: str) -> np.ndarray:
    if isinstance(items, np.ndarray):
        return _as_array(items, name)
    if not items:
        raise ValueError(f"{name} must be nonempty")
    return np.concatenate([_as_array(item, name) for item in items]).astype(float)


def _feature_matrix(semantic_scores: np.ndarray, physical_scores: np.ndarray | None = None) -> np.ndarray:
    sem = _as_array(semantic_scores, "semantic_scores")
    if physical_scores is None:
        phys = np.zeros_like(sem)
    else:
        phys = _as_array(physical_scores, "physical_scores")
        if len(phys) != len(sem):
            raise ValueError("physical_scores must match semantic_scores")
    return np.column_stack(
        [
            sem,
            phys,
            sem * phys,
            sem * sem,
            phys * phys,
            np.maximum(sem - phys, 0.0),
        ]
    )


@dataclass(frozen=True)
class CertificateResult:
    """Named first-principles certificate outcomes for one candidate pool."""

    pass_mask: np.ndarray
    failure_types: tuple[tuple[str, ...], ...]
    failure_counts: dict[str, int]

    @property
    def certified_candidate_count(self) -> int:
        return int(np.sum(self.pass_mask))

    @property
    def candidate_count(self) -> int:
        return int(len(self.pass_mask))

    def failure_types_for(self, index: int) -> tuple[str, ...]:
        if index < 0 or index >= len(self.failure_types):
            return tuple(CERTIFICATE_FAILURE_TYPES)
        return self.failure_types[index]


def certify_candidates(
    flags: Mapping[str, Iterable[float]],
    *,
    candidate_count: int | None = None,
    include_hidden_constraints: bool = True,
) -> CertificateResult:
    """Convert simulator/scene flags into hard physical certificates.

    The predicates are sound only for the modeled toy physics encoded by the
    flags. Missing flags are treated as no evidence of that failure type rather
    than as real-world safety evidence.
    """

    arrays: dict[str, np.ndarray] = {}
    inferred_count = candidate_count
    for name, values in flags.items():
        arr = _as_array(values, name)
        arrays[name] = arr
        if inferred_count is None:
            inferred_count = len(arr)
        elif len(arr) != inferred_count:
            raise ValueError("all certificate flag arrays must have the same length")
    if inferred_count is None:
        raise ValueError("candidate_count is required when no flags are provided")

    failures: list[tuple[str, ...]] = []
    counts = {name: 0 for name in CERTIFICATE_FAILURE_TYPES}
    enabled_types = set(CERTIFICATE_FAILURE_TYPES)
    if not include_hidden_constraints:
        enabled_types.discard("hidden_obstacle")

    for idx in range(int(inferred_count)):
        candidate_failures = []
        for failure_type in CERTIFICATE_FAILURE_TYPES:
            if failure_type not in enabled_types:
                continue
            flag_names = _CERTIFICATE_FLAG_MAP[failure_type]
            failed = any(float(arrays.get(flag_name, np.zeros(int(inferred_count)))[idx]) > 0.5 for flag_name in flag_names)
            if failed:
                candidate_failures.append(failure_type)
                counts[failure_type] += 1
        failures.append(tuple(candidate_failures))
    pass_mask = np.asarray([len(item) == 0 for item in failures], dtype=bool)
    return CertificateResult(pass_mask=pass_mask, failure_types=tuple(failures), failure_counts=counts)


@dataclass(frozen=True)
class TailGuardConfig:
    """Controller settings for Certified TailGuard-BoN."""

    n_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    margin: float = 0.03
    confidence_delta: float = 0.05
    min_pilot_labels: int = 16
    min_tail_labels: int = 4
    tail_quantile: float = 0.80
    l2: float = 1e-2
    uncertainty_collect_threshold: float = 0.30
    early_stop_tolerance: float = 0.01
    min_certified_candidates: int = 2
    min_certified_for_high_n: int = 8


@dataclass(frozen=True)
class TailCalibrator:
    """Bounded real-utility calibrator fit from pilot labels."""

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    residual_variance: float
    residual_max: float
    label_count: int
    tail_label_count: int
    confidence_delta: float
    tail_quantile: float

    def predict(self, semantic_scores: Iterable[float], physical_scores: Iterable[float] | None = None) -> np.ndarray:
        x = _feature_matrix(_as_array(semantic_scores, "semantic_scores"), None if physical_scores is None else _as_array(physical_scores, "physical_scores"))
        x_norm = (x - self.mean) / self.std
        x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
        return np.clip(x_aug @ self.weights, 0.0, 1.0)

    @property
    def confidence_radius(self) -> float:
        return empirical_bernstein_radius(
            max(1, self.tail_label_count),
            self.residual_variance,
            self.confidence_delta,
            value_range=1.0,
        )


@dataclass(frozen=True)
class TailGuardResult:
    """One Certified TailGuard-BoN decision for a candidate pool."""

    selected_n: int
    selected_index: int
    calibrated_score: float
    predicted_selected_utility_curve: dict[int, float]
    lower_confidence_bound_curve: dict[int, float]
    gate_decision: str
    reason_code: str
    baseline_n1: float
    random_baseline: float
    pilot_label_count: int
    tail_label_count: int
    confidence_radius: float
    certificate_pass: bool
    certificate_failure_types: tuple[str, ...]
    certified_candidate_count: int
    fallback_used: bool
    abstention_reason: str | None
    certified_selected_utility: float | None
    certified_violation_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_N": int(self.selected_n),
            "selected_index": int(self.selected_index),
            "calibrated_score": float(self.calibrated_score),
            "predicted_selected_utility_curve": {str(k): float(v) for k, v in self.predicted_selected_utility_curve.items()},
            "lower_confidence_bound_curve": {str(k): float(v) for k, v in self.lower_confidence_bound_curve.items()},
            "gate_decision": self.gate_decision,
            "reason_code": self.reason_code,
            "baseline_n1": float(self.baseline_n1),
            "random_baseline": float(self.random_baseline),
            "pilot_label_count": int(self.pilot_label_count),
            "tail_label_count": int(self.tail_label_count),
            "confidence_radius": float(self.confidence_radius),
            "certificate_pass": bool(self.certificate_pass),
            "certificate_failure_types": list(self.certificate_failure_types),
            "certified_candidate_count": int(self.certified_candidate_count),
            "fallback_used": bool(self.fallback_used),
            "abstention_reason": self.abstention_reason,
            "certified_selected_utility": self.certified_selected_utility,
            "certified_violation_rate": self.certified_violation_rate,
        }


def empirical_bernstein_radius(
    label_count: int,
    empirical_variance: float,
    delta: float,
    *,
    value_range: float = 1.0,
) -> float:
    """A bounded empirical-Bernstein-style one-sided confidence radius."""

    n = int(label_count)
    if n <= 1:
        return float(value_range)
    variance = float(np.clip(empirical_variance, 0.0, value_range * value_range))
    safe_delta = float(np.clip(delta, 1e-12, 0.5))
    log_term = float(np.log(2.0 / safe_delta))
    radius = np.sqrt(2.0 * variance * log_term / n) + 7.0 * value_range * log_term / (3.0 * (n - 1))
    return float(min(value_range, max(0.0, radius)))


def fit_tail_calibrator(
    semantic_scores: Sequence[np.ndarray] | np.ndarray,
    real_utilities: Sequence[np.ndarray] | np.ndarray,
    physical_scores: Sequence[np.ndarray] | np.ndarray | None = None,
    *,
    config: TailGuardConfig | None = None,
    sample_fraction: float = 1.0,
    label_noise: float = 0.0,
    seed: int = 0,
) -> TailCalibrator:
    """Fit a bounded real-utility model from pilot labels."""

    cfg = config or TailGuardConfig()
    sem = _flatten(semantic_scores, "semantic_scores")
    util = _flatten(real_utilities, "real_utilities")
    phys = None if physical_scores is None else _flatten(physical_scores, "physical_scores")
    if len(sem) != len(util):
        raise ValueError("semantic_scores and real_utilities must have the same length")
    if phys is not None and len(phys) != len(sem):
        raise ValueError("physical_scores must match semantic_scores")
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be in (0, 1]")
    if label_noise < 0.0:
        raise ValueError("label_noise must be nonnegative")

    rng = np.random.default_rng(seed)
    label_count = max(2, int(np.ceil(sample_fraction * len(sem))))
    if label_count < len(sem):
        chosen = np.sort(rng.choice(len(sem), size=label_count, replace=False))
        sem = sem[chosen]
        util = util[chosen]
        phys = None if phys is None else phys[chosen]
    if label_noise > 0.0:
        util = np.clip(util + rng.normal(0.0, label_noise, size=len(util)), 0.0, 1.0)

    x = _feature_matrix(sem, phys)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    x_norm = (x - mean) / std
    x_aug = np.hstack([np.ones((len(x_norm), 1)), x_norm])
    ridge = cfg.l2 * np.eye(x_aug.shape[1])
    ridge[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ util)
    pred = np.clip(x_aug @ weights, 0.0, 1.0)
    residual = util - pred
    tail_threshold = float(np.quantile(sem, cfg.tail_quantile))
    tail_label_count = int(np.sum(sem >= tail_threshold))
    return TailCalibrator(
        weights=weights,
        mean=mean,
        std=std,
        residual_variance=float(np.var(residual)) if len(residual) else 0.25,
        residual_max=float(np.max(np.abs(residual))) if len(residual) else 1.0,
        label_count=int(len(util)),
        tail_label_count=tail_label_count,
        confidence_delta=cfg.confidence_delta,
        tail_quantile=cfg.tail_quantile,
    )


def predict_selected_tail_curve(
    selection_scores: Iterable[float],
    predicted_utilities: Iterable[float],
    n_grid: Iterable[int],
) -> dict[int, float]:
    """Predict selected utility for each N under the exact finite law."""

    scores = _as_array(selection_scores, "selection_scores")
    utilities = _as_array(predicted_utilities, "predicted_utilities")
    if len(scores) != len(utilities):
        raise ValueError("selection_scores and predicted_utilities must have the same length")
    out: dict[int, float] = {}
    for n in n_grid:
        n_int = int(n)
        if n_int < 1 or n_int > len(scores):
            continue
        out[n_int] = expected_selected_value(scores, utilities, n_int)
    if not out:
        raise ValueError("n_grid contains no valid N for this candidate pool")
    return out


def _choose_lowest_argmax(scores: np.ndarray) -> int:
    return int(np.flatnonzero(scores == np.max(scores))[0])


def _all_pass_certificate(count: int) -> CertificateResult:
    return CertificateResult(
        pass_mask=np.ones(int(count), dtype=bool),
        failure_types=tuple(tuple() for _ in range(int(count))),
        failure_counts={name: 0 for name in CERTIFICATE_FAILURE_TYPES},
    )


def _gate_from_bounds(
    curve: dict[int, float],
    lower: dict[int, float],
    baseline_n1: float,
    random_baseline: float,
    calibrator: TailCalibrator,
    cfg: TailGuardConfig,
) -> tuple[int, str, str]:
    min_n = min(curve)
    max_n = max(curve)
    best_n = max(lower, key=lambda n: (lower[n], -n))
    best_lcb = lower[best_n]
    high_lcb = lower[max_n]
    required = max(baseline_n1, random_baseline) + cfg.margin

    if calibrator.label_count < cfg.min_pilot_labels or calibrator.tail_label_count < cfg.min_tail_labels:
        return min_n, "collect_pilot_labels", "insufficient_pilot_labels"
    if calibrator.confidence_radius >= cfg.uncertainty_collect_threshold and high_lcb < required:
        return min_n, "collect_pilot_labels", "tail_bound_too_wide"
    if high_lcb >= required:
        return max_n if lower[max_n] >= best_lcb - cfg.early_stop_tolerance else best_n, "allow_high_n", "tail_lcb_beats_baselines"
    if high_lcb + cfg.margin < max(baseline_n1, random_baseline):
        return min_n, "block_high_n", "tail_lcb_below_baselines"
    if best_n == min_n or best_lcb <= lower[min_n] + cfg.early_stop_tolerance:
        return min_n, "stop_early", "high_n_not_worth_margin"
    return best_n, "stop_early", "intermediate_n_has_best_lcb"


def tailguard_select(
    semantic_scores: Iterable[float],
    *,
    calibrator: TailCalibrator,
    physical_scores: Iterable[float] | None = None,
    certificate: CertificateResult | None = None,
    real_utilities: Iterable[float] | None = None,
    violation_flags: Iterable[float] | None = None,
    n_grid: Iterable[int] | None = None,
    config: TailGuardConfig | None = None,
) -> TailGuardResult:
    """Select N and an action under Certified TailGuard-BoN."""

    cfg = config or TailGuardConfig()
    sem = _as_array(semantic_scores, "semantic_scores")
    phys = None if physical_scores is None else _as_array(physical_scores, "physical_scores")
    if phys is not None and len(phys) != len(sem):
        raise ValueError("physical_scores must match semantic_scores")
    grid = tuple(int(n) for n in (n_grid if n_grid is not None else cfg.n_grid) if 1 <= int(n) <= len(sem))
    if not grid:
        raise ValueError("n_grid contains no valid N")
    cert = certificate or _all_pass_certificate(len(sem))
    if len(cert.pass_mask) != len(sem):
        raise ValueError("certificate pass mask must match semantic_scores")
    utilities = None if real_utilities is None else _as_array(real_utilities, "real_utilities")
    violations = None if violation_flags is None else _as_array(violation_flags, "violation_flags")
    if utilities is not None and len(utilities) != len(sem):
        raise ValueError("real_utilities must match semantic_scores")
    if violations is not None and len(violations) != len(sem):
        raise ValueError("violation_flags must match semantic_scores")

    calibrated = calibrator.predict(sem, phys)
    certified_indices = np.flatnonzero(cert.pass_mask)
    certified_count = int(len(certified_indices))

    if certified_count == 0:
        curve = predict_selected_tail_curve(calibrated, calibrated, grid)
        radius = calibrator.confidence_radius
        lower = {n: max(0.0, curve[n] - radius) for n in curve}
        baseline_n1 = curve.get(1, predict_selected_tail_curve(calibrated, calibrated, [1])[1])
        random_baseline = float(np.mean(calibrated))
        return TailGuardResult(
            selected_n=min(grid),
            selected_index=-1,
            calibrated_score=0.0,
            predicted_selected_utility_curve=curve,
            lower_confidence_bound_curve=lower,
            gate_decision="collect_pilot_labels",
            reason_code="no_certified_candidates",
            baseline_n1=float(baseline_n1),
            random_baseline=random_baseline,
            pilot_label_count=calibrator.label_count,
            tail_label_count=calibrator.tail_label_count,
            confidence_radius=radius,
            certificate_pass=False,
            certificate_failure_types=tuple(CERTIFICATE_FAILURE_TYPES),
            certified_candidate_count=0,
            fallback_used=True,
            abstention_reason="no_certified_candidates",
            certified_selected_utility=None,
            certified_violation_rate=None,
        )

    certified_scores = calibrated[certified_indices]
    certified_grid = tuple(n for n in grid if n <= certified_count)
    if not certified_grid:
        certified_grid = (1,)
    curve = predict_selected_tail_curve(certified_scores, certified_scores, certified_grid)
    radius = calibrator.confidence_radius
    lower = {n: max(0.0, curve[n] - radius) for n in curve}
    baseline_n1 = curve.get(1, predict_selected_tail_curve(calibrated, calibrated, [1])[1])
    random_baseline = float(np.mean(certified_scores))
    selected_n, gate, reason = _gate_from_bounds(curve, lower, baseline_n1, random_baseline, calibrator, cfg)

    abstention_reason: str | None = None
    if certified_count < cfg.min_certified_candidates:
        selected_n = min(certified_grid)
        gate = "collect_pilot_labels"
        reason = "insufficient_certified_candidates"
        abstention_reason = reason
    elif certified_count < cfg.min_certified_for_high_n and max(grid) > min(certified_grid):
        selected_n = min(certified_grid)
        gate = "block_high_n"
        reason = "certified_candidate_count_too_low"

    if gate not in VALID_GATES:
        raise RuntimeError(f"invalid TailGuard gate: {gate}")

    selected_local = _choose_lowest_argmax(certified_scores)
    selected_index = int(certified_indices[selected_local])
    if gate == "collect_pilot_labels":
        abstention_reason = abstention_reason or reason
    max_requested_n = max(grid)
    fallback_used = bool(gate != "allow_high_n" or selected_n < min(max_requested_n, certified_count))
    selected_utility = None if utilities is None or abstention_reason else float(utilities[selected_index])
    selected_violation = None if violations is None or abstention_reason else float(violations[selected_index])
    return TailGuardResult(
        selected_n=int(selected_n),
        selected_index=selected_index,
        calibrated_score=float(calibrated[selected_index]),
        predicted_selected_utility_curve=curve,
        lower_confidence_bound_curve=lower,
        gate_decision=gate,
        reason_code=reason,
        baseline_n1=float(baseline_n1),
        random_baseline=random_baseline,
        pilot_label_count=calibrator.label_count,
        tail_label_count=calibrator.tail_label_count,
        confidence_radius=radius,
        certificate_pass=bool(cert.pass_mask[selected_index]),
        certificate_failure_types=cert.failure_types_for(selected_index),
        certified_candidate_count=certified_count,
        fallback_used=fallback_used,
        abstention_reason=abstention_reason,
        certified_selected_utility=selected_utility,
        certified_violation_rate=selected_violation,
    )
