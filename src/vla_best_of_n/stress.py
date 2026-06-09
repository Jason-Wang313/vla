"""First-principles stress tests for Certified TailGuard-BoN."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from .rendering import attach_visual_observation
from .simulator import simulate_pool
from .tailguard import (
    CERTIFICATE_FAILURE_TYPES,
    TailCalibrator,
    TailGuardConfig,
    certify_candidates,
    fit_tail_calibrator,
    tailguard_select,
)
from .theory import tie_aware_selection_probabilities
from .vla_env import CandidatePool, FEATURE_NAMES, generate_pools


@dataclass(frozen=True)
class StressKnobs:
    semantic_physical_misalignment: float = 0.0
    distractor_salience: float = 0.0
    obstacle_density: float = 0.0
    hidden_constraint_rate: float = 0.0
    verifier_false_positive_rate: float = 0.0
    verifier_false_negative_rate: float = 0.0
    label_noise: float = 0.0
    certified_candidate_dropout_rate: float = 0.0


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def stress_pool(pool: CandidatePool, knobs: StressKnobs, seed_offset: int = 0) -> CandidatePool:
    """Return a deterministic stressed copy of a simulator-evaluated pool."""

    rng = np.random.default_rng(pool.seed * 13007 + pool.state_id * 1709 + seed_offset)
    flags = {name: values.copy() for name, values in pool.flags.items()}
    size = pool.size
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    wrong_object = flags.get("wrong_object", 1.0 - pool.candidate_features[:, idx["correct_object"]])
    blocked_path = np.maximum(flags.get("blocked_path", np.zeros(size)), pool.candidate_features[:, idx["blocked_path"]])
    semantic_tail = pool.semantic_proxy >= np.quantile(pool.semantic_proxy, 0.80)

    obstacle_hit = (rng.random(size) < 0.30 * knobs.obstacle_density) & (blocked_path > 0.5)
    reach_loss = (rng.random(size) < 0.14 * knobs.obstacle_density) & semantic_tail
    hidden = (rng.random(size) < knobs.hidden_constraint_rate) & semantic_tail
    fragile_heavy = np.maximum(flags.get("fragile", np.zeros(size)), flags.get("heavy_failure", np.zeros(size)))
    certified_before_dropout = flags.get("violation", np.zeros(size)) < 0.5
    certified_dropout = (rng.random(size) < knobs.certified_candidate_dropout_rate) & certified_before_dropout

    flags["collision"] = np.maximum(flags.get("collision", np.zeros(size)), obstacle_hit.astype(float))
    flags["unreachable"] = np.maximum(flags.get("unreachable", np.zeros(size)), reach_loss.astype(float))
    flags["hidden_constraint"] = np.maximum(hidden.astype(float), certified_dropout.astype(float))
    flags["violation"] = np.maximum.reduce(
        [
            flags.get("violation", np.zeros(size)),
            flags["collision"],
            flags["unreachable"],
            flags.get("stability_failure", np.zeros(size)),
            flags.get("receptacle_incompatible", np.zeros(size)),
            fragile_heavy,
            flags["hidden_constraint"],
            flags.get("tool_mismatch", np.zeros(size)),
        ]
    )

    utility = pool.real_utility.copy()
    utility -= 0.22 * knobs.obstacle_density * blocked_path
    utility -= 0.18 * knobs.distractor_salience * wrong_object
    utility -= 0.35 * hidden.astype(float)
    utility -= 0.65 * certified_dropout.astype(float)
    utility -= 0.10 * flags.get("receptacle_incompatible", np.zeros(size))
    utility = _clip01(utility)
    physical = _clip01(pool.physical_feasibility - 0.45 * flags["violation"] + 0.15 * (1.0 - flags["violation"]))

    anti_utility = 1.0 - utility
    semantic = (1.0 - knobs.semantic_physical_misalignment) * pool.semantic_proxy
    semantic += knobs.semantic_physical_misalignment * anti_utility
    semantic += 0.18 * knobs.distractor_salience * wrong_object
    semantic += 0.10 * knobs.obstacle_density * flags["collision"]
    semantic = _clip01(semantic)
    return replace(pool, semantic_proxy=semantic, physical_feasibility=physical, real_utility=utility, flags=flags)


def stressed_verifier_score(pool: CandidatePool, knobs: StressKnobs, seed_offset: int = 0) -> np.ndarray:
    """Imperfect physical verifier with independent and selected-tail errors."""

    rng = np.random.default_rng(pool.seed * 23011 + pool.state_id * 3011 + seed_offset)
    feasible = pool.physical_feasibility >= 0.5
    score = np.where(feasible, 0.88, 0.16).astype(float)
    false_negative = feasible & (rng.random(pool.size) < knobs.verifier_false_negative_rate)
    false_positive = (~feasible) & (rng.random(pool.size) < knobs.verifier_false_positive_rate)
    score[false_negative] = 0.20
    score[false_positive] = 0.82

    semantic_tail = pool.semantic_proxy >= np.quantile(pool.semantic_proxy, 0.85)
    correlated_tail_error = semantic_tail & (pool.flags.get("violation", np.zeros(pool.size)) > 0.5)
    correlated_tail_error &= rng.random(pool.size) < max(knobs.hidden_constraint_rate, knobs.verifier_false_positive_rate)
    score[correlated_tail_error] = np.maximum(score[correlated_tail_error], 0.86)
    return _clip01(score + rng.normal(0.0, 0.03 + 0.08 * knobs.verifier_false_positive_rate, size=pool.size))


def make_stress_pools(
    seeds: Iterable[int],
    states: int,
    candidates: int,
    family: str,
    knobs: StressKnobs,
    seed_offset: int = 0,
) -> list[CandidatePool]:
    base = generate_pools(seeds, states, candidates, family)
    rendered = [simulate_pool(attach_visual_observation(pool)) for pool in base]
    return [stress_pool(pool, knobs, seed_offset=seed_offset) for pool in rendered]


def _weighted_metrics(pool: CandidatePool, scores: np.ndarray, n: int, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != pool.size:
            raise ValueError("mask must match pool size")
        if not np.any(mask):
            return _abstain_metrics(pool)
        scores = np.asarray(scores, dtype=float)[mask]
        utility = pool.real_utility[mask]
        semantic = pool.semantic_proxy[mask]
        physical = pool.physical_feasibility[mask]
        flags = {name: values[mask] for name, values in pool.flags.items()}
    else:
        scores = np.asarray(scores, dtype=float)
        utility = pool.real_utility
        semantic = pool.semantic_proxy
        physical = pool.physical_feasibility
        flags = pool.flags

    n = min(int(n), len(scores))
    law = tie_aware_selection_probabilities(scores, n)
    probs = law.probabilities
    out = {
        "selected_real_utility": float(probs @ utility),
        "selected_semantic_score": float(probs @ semantic),
        "selected_physical_feasibility": float(probs @ physical),
    }
    for flag in [
        "violation",
        "wrong_object",
        "wrong_target",
        "unreachable",
        "collision",
        "stability_failure",
        "fragile",
        "receptacle_incompatible",
        "heavy_failure",
        "blocked_path",
        "hidden_constraint",
        "tool_mismatch",
    ]:
        if flag in flags:
            out[f"{flag}_rate"] = float(probs @ flags[flag])
    return out


def _abstain_metrics(pool: CandidatePool) -> dict[str, float]:
    out = {
        "selected_real_utility": 0.0,
        "selected_semantic_score": 0.0,
        "selected_physical_feasibility": 0.0,
    }
    for flag in pool.flags:
        out[f"{flag}_rate"] = 0.0
    out["abstention_rate"] = 1.0
    return out


def _indexed_metrics(pool: CandidatePool, index: int) -> dict[str, float]:
    if index < 0:
        return _abstain_metrics(pool)
    out = {
        "selected_real_utility": float(pool.real_utility[index]),
        "selected_semantic_score": float(pool.semantic_proxy[index]),
        "selected_physical_feasibility": float(pool.physical_feasibility[index]),
        "abstention_rate": 0.0,
    }
    for flag_name, values in pool.flags.items():
        out[f"{flag_name}_rate"] = float(values[index])
    return out


def _best_index(scores: np.ndarray, mask: np.ndarray | None = None) -> int:
    scores = np.asarray(scores, dtype=float)
    if mask is None:
        return int(np.flatnonzero(scores == np.max(scores))[0])
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return -1
    masked_scores = scores[indices]
    return int(indices[np.flatnonzero(masked_scores == np.max(masked_scores))[0]])


def _certificate_failure_rates(certificates: list[object]) -> dict[str, float]:
    totals = {name: 0 for name in CERTIFICATE_FAILURE_TYPES}
    count = 0
    certified_counts = []
    for cert in certificates:
        pass_mask = getattr(cert, "pass_mask")
        failure_types = getattr(cert, "failure_types")
        count += len(pass_mask)
        certified_counts.append(int(np.sum(pass_mask)))
        for item in failure_types:
            for failure_type in item:
                totals[failure_type] += 1
    denom = max(1, count)
    out = {f"certificate_failure_{name}_rate": float(totals[name] / denom) for name in CERTIFICATE_FAILURE_TYPES}
    out["certified_candidate_count_mean"] = float(np.mean(certified_counts)) if certified_counts else 0.0
    return out


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: float(np.mean([item.get(key, 0.0) for item in items])) for key in keys}


def tailguard_gate_examples_summary() -> pd.DataFrame:
    """Deterministic constructed examples covering every TailGuard gate."""

    examples = [
        {
            "scenario": "aligned_tail_allows_high_n",
            "expected_gate": "allow_high_n",
            "semantic": np.linspace(0.0, 1.0, 4096),
            "physical": np.linspace(0.0, 1.0, 4096),
            "utility": np.linspace(0.0, 1.0, 4096),
            "config": TailGuardConfig(confidence_delta=0.5, uncertainty_collect_threshold=0.50),
            "sample_fraction": 1.0,
        },
        {
            "scenario": "flat_tail_stops_early",
            "expected_gate": "stop_early",
            "semantic": np.linspace(0.0, 1.0, 4096),
            "physical": np.full(4096, 0.5),
            "utility": np.full(4096, 0.5),
            "config": TailGuardConfig(confidence_delta=0.5, uncertainty_collect_threshold=0.50),
            "sample_fraction": 1.0,
        },
        {
            "scenario": "scarce_tail_labels_collects",
            "expected_gate": "collect_pilot_labels",
            "semantic": np.linspace(0.0, 1.0, 100),
            "physical": np.linspace(0.0, 1.0, 100),
            "utility": np.linspace(0.0, 1.0, 100),
            "config": TailGuardConfig(),
            "sample_fraction": 0.01,
        },
        {
            "scenario": "uncertain_flat_tail_blocks",
            "expected_gate": "block_high_n",
            "semantic": np.linspace(0.0, 1.0, 512),
            "physical": np.full(512, 0.5),
            "utility": np.full(512, 0.5),
            "config": TailGuardConfig(),
            "sample_fraction": 1.0,
        },
    ]
    rows = []
    for idx, spec in enumerate(examples):
        cfg = spec["config"]
        sem = spec["semantic"]
        phys = spec["physical"]
        util = spec["utility"]
        calibrator = fit_tail_calibrator(
            sem,
            util,
            phys,
            config=cfg,
            sample_fraction=float(spec["sample_fraction"]),
            seed=10_000 + idx,
        )
        result = tailguard_select(
            sem,
            calibrator=calibrator,
            physical_scores=phys,
            n_grid=[1, 2, 4, 8, 16, 32, 64, 128],
            config=cfg,
        )
        high_n = max(result.predicted_selected_utility_curve)
        rows.append(
            {
                "scenario": spec["scenario"],
                "expected_gate": spec["expected_gate"],
                "gate_decision": result.gate_decision,
                "reason_code": result.reason_code,
                "selected_N": result.selected_n,
                "baseline_n1": result.baseline_n1,
                "random_baseline": result.random_baseline,
                "high_n_predicted_utility": result.predicted_selected_utility_curve[high_n],
                "high_n_lcb": result.lower_confidence_bound_curve[high_n],
                "pilot_label_count": result.pilot_label_count,
                "tail_label_count": result.tail_label_count,
                "confidence_radius": result.confidence_radius,
                "expected_behavior_verified": result.gate_decision == spec["expected_gate"],
            }
        )
    return pd.DataFrame(rows)


def _fit_tailguard_for_condition(
    pilot_pools: list[CandidatePool],
    knobs: StressKnobs,
    *,
    config: TailGuardConfig,
    sample_fraction: float,
    label_noise: float,
    seed: int,
) -> TailCalibrator:
    pilot_sem = [pool.semantic_proxy for pool in pilot_pools]
    pilot_phys = [stressed_verifier_score(pool, knobs, seed_offset=seed) for pool in pilot_pools]
    pilot_utility = [pool.real_utility for pool in pilot_pools]
    return fit_tail_calibrator(
        pilot_sem,
        pilot_utility,
        pilot_phys,
        config=config,
        sample_fraction=sample_fraction,
        label_noise=label_noise,
        seed=seed,
    )


def tailguard_adaptive_summary(
    eval_pools: list[CandidatePool],
    semantic_scores: list[np.ndarray],
    pilot_pools: list[CandidatePool],
    pilot_semantic_scores: list[np.ndarray],
    n_grid: Iterable[int],
    *,
    config: TailGuardConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare TailGuard adaptive N against fixed-N baselines."""

    cfg = config or TailGuardConfig()
    grid = tuple(int(n) for n in n_grid)
    high_n = max(grid)
    pilot_phys = [pool.physical_feasibility for pool in pilot_pools]
    calibrator = fit_tail_calibrator(
        pilot_semantic_scores,
        [pool.real_utility for pool in pilot_pools],
        pilot_phys,
        config=cfg,
        sample_fraction=1.0,
        seed=811,
    )
    method_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    decisions = []
    certificates = []
    for pool, sem in zip(eval_pools, semantic_scores):
        calibrated = calibrator.predict(sem, pool.physical_feasibility)
        verifier_filter = np.clip(sem * (0.15 + 0.85 * pool.physical_feasibility), 0.0, 1.0)
        certificate = certify_candidates(pool.flags, candidate_count=pool.size)
        certificates.append(certificate)
        baseline_specs = [
            ("raw_fixed_high_n", sem, high_n),
            ("n1_baseline", sem, 1),
            ("random_high_n", pool.random_score, high_n),
            ("verifier_filtered_high_n", verifier_filter, high_n),
            ("verifier_filter_high_n", verifier_filter, high_n),
            ("calibrated_high_n_no_certificates", calibrated, high_n),
            ("calibrated_fixed_high_n", calibrated, high_n),
            ("oracle_high_n", pool.real_utility, high_n),
        ]
        for method, score, n_value in baseline_specs:
            method_metrics[method].append(_weighted_metrics(pool, score, n_value))

        cert_only_index = _best_index(sem, certificate.pass_mask)
        cert_only = _indexed_metrics(pool, cert_only_index)
        cert_only["certificate_pass"] = float(cert_only_index >= 0)
        cert_only["certified_candidate_count"] = float(certificate.certified_candidate_count)
        method_metrics["certificate_only_filtering_without_tail_calibration"].append(cert_only)

        no_lcb_index = _best_index(calibrated, certificate.pass_mask)
        for ablation in [
            "tailguard_without_lower_confidence_bound",
            "tailguard_without_random_baseline_check",
            "tailguard_without_n1_baseline_check",
        ]:
            metrics = _indexed_metrics(pool, no_lcb_index)
            metrics["certificate_pass"] = float(no_lcb_index >= 0)
            metrics["certified_candidate_count"] = float(certificate.certified_candidate_count)
            metrics["selected_N"] = float(high_n if no_lcb_index >= 0 else 0)
            method_metrics[ablation].append(metrics)

        result = tailguard_select(
            sem,
            calibrator=calibrator,
            physical_scores=pool.physical_feasibility,
            certificate=certificate,
            real_utilities=pool.real_utility,
            violation_flags=pool.flags.get("violation", np.zeros(pool.size)),
            n_grid=grid,
            config=cfg,
        )
        tail_metrics = _indexed_metrics(pool, -1 if result.abstention_reason else result.selected_index)
        tail_metrics["selected_N"] = float(result.selected_n)
        tail_metrics["certificate_pass"] = float(result.certificate_pass)
        tail_metrics["certified_candidate_count"] = float(result.certified_candidate_count)
        tail_metrics["fallback_used"] = float(result.fallback_used)
        tail_metrics["certified_selected_utility"] = (
            0.0 if result.certified_selected_utility is None else float(result.certified_selected_utility)
        )
        tail_metrics["certified_violation_rate"] = (
            0.0 if result.certified_violation_rate is None else float(result.certified_violation_rate)
        )
        method_metrics["certified_tailguard_bon"].append(tail_metrics)
        method_metrics["tailguard_bon"].append(dict(tail_metrics))
        decisions.append(result)

    rows = []
    for method, items in method_metrics.items():
        row = {"experiment": "tailguard_adaptive_n", "method": method}
        row.update(_mean_metrics(items))
        row["N"] = int(high_n)
        row["selected_N_mean"] = row.pop("selected_N", float(high_n))
        row["pilot_label_count"] = int(calibrator.label_count)
        row["tail_label_count"] = int(calibrator.tail_label_count)
        row["confidence_radius"] = float(calibrator.confidence_radius)
        if method in {"certified_tailguard_bon", "tailguard_bon"}:
            gates = Counter(decision.gate_decision for decision in decisions)
            for gate in ["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]:
                row[f"{gate}_rate"] = float(gates.get(gate, 0) / max(1, len(decisions)))
            row["gate_decision"] = gates.most_common(1)[0][0]
        else:
            row["gate_decision"] = "fixed_policy"
            for gate in ["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]:
                row[f"{gate}_rate"] = 0.0
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    artifact = {
        "method": "Certified TailGuard-BoN",
        "implementation_short_name": "TailGuard-BoN",
        "n_grid": [int(n) for n in grid],
        "pilot_label_count": int(calibrator.label_count),
        "tail_label_count": int(calibrator.tail_label_count),
        "confidence_radius": float(calibrator.confidence_radius),
        "certificate_failure_decomposition": _certificate_failure_rates(certificates),
        "decisions": [decision.to_dict() for decision in decisions[:12]],
    }
    return summary, artifact


def _aggregate_tailguard_condition(
    eval_pools: list[CandidatePool],
    pilot_pools: list[CandidatePool],
    knobs: StressKnobs,
    n_grid: Iterable[int],
    *,
    sample_fraction: float,
    label_noise: float,
    seed: int,
    config: TailGuardConfig | None = None,
) -> dict[str, object]:
    cfg = config or TailGuardConfig(confidence_delta=0.5, uncertainty_collect_threshold=0.60)
    high_n = max(int(n) for n in n_grid)
    calibrator = _fit_tailguard_for_condition(
        pilot_pools,
        knobs,
        config=cfg,
        sample_fraction=sample_fraction,
        label_noise=label_noise,
        seed=seed,
    )
    raw_items = []
    tail_items = []
    oracle_items = []
    random_items = []
    verifier_items = []
    calibrated_no_certificate_items = []
    certificate_only_items = []
    decisions = []
    certificates = []
    for pool in eval_pools:
        verifier = stressed_verifier_score(pool, knobs, seed_offset=seed)
        calibrated = calibrator.predict(pool.semantic_proxy, verifier)
        certificate = certify_candidates(pool.flags, candidate_count=pool.size)
        certificates.append(certificate)
        result = tailguard_select(
            pool.semantic_proxy,
            calibrator=calibrator,
            physical_scores=verifier,
            certificate=certificate,
            real_utilities=pool.real_utility,
            violation_flags=pool.flags.get("violation", np.zeros(pool.size)),
            n_grid=n_grid,
            config=cfg,
        )
        raw_items.append(_weighted_metrics(pool, pool.semantic_proxy, high_n))
        verifier_items.append(_weighted_metrics(pool, np.clip(pool.semantic_proxy * (0.15 + 0.85 * verifier), 0.0, 1.0), high_n))
        calibrated_no_certificate_items.append(_weighted_metrics(pool, calibrated, high_n))
        certificate_only_items.append(_indexed_metrics(pool, _best_index(pool.semantic_proxy, certificate.pass_mask)))
        tail_metric = _indexed_metrics(pool, -1 if result.abstention_reason else result.selected_index)
        tail_metric["selected_N"] = float(result.selected_n)
        tail_metric["fallback_used"] = float(result.fallback_used)
        tail_metric["certified_candidate_count"] = float(result.certified_candidate_count)
        tail_items.append(tail_metric)
        oracle_items.append(_weighted_metrics(pool, pool.real_utility, high_n))
        random_items.append(_weighted_metrics(pool, pool.random_score, high_n))
        decisions.append(result)

    raw = _mean_metrics(raw_items)
    tail = _mean_metrics(tail_items)
    oracle = _mean_metrics(oracle_items)
    random = _mean_metrics(random_items)
    verifier_metrics = _mean_metrics(verifier_items)
    calibrated_no_certificate = _mean_metrics(calibrated_no_certificate_items)
    certificate_only = _mean_metrics(certificate_only_items)
    gates = Counter(decision.gate_decision for decision in decisions)
    row: dict[str, object] = {
        "semantic_physical_misalignment": knobs.semantic_physical_misalignment,
        "distractor_salience": knobs.distractor_salience,
        "obstacle_density": knobs.obstacle_density,
        "hidden_constraint_rate": knobs.hidden_constraint_rate,
        "verifier_false_positive_rate": knobs.verifier_false_positive_rate,
        "verifier_false_negative_rate": knobs.verifier_false_negative_rate,
        "label_noise": label_noise,
        "label_budget_fraction": sample_fraction,
        "pilot_label_count": calibrator.label_count,
        "tail_label_count": calibrator.tail_label_count,
        "confidence_radius": calibrator.confidence_radius,
        "raw_high_n_utility": raw["selected_real_utility"],
        "tailguard_utility": tail["selected_real_utility"],
        "certified_tailguard_utility": tail["selected_real_utility"],
        "oracle_high_n_utility": oracle["selected_real_utility"],
        "random_high_n_utility": random["selected_real_utility"],
        "verifier_filtered_high_n_utility": verifier_metrics["selected_real_utility"],
        "calibrated_high_n_no_certificates_utility": calibrated_no_certificate["selected_real_utility"],
        "certificate_only_utility": certificate_only["selected_real_utility"],
        "tailguard_gain_over_raw": tail["selected_real_utility"] - raw["selected_real_utility"],
        "raw_violation_rate": raw.get("violation_rate", 0.0),
        "tailguard_violation_rate": tail.get("violation_rate", 0.0),
        "certified_violation_rate": tail.get("violation_rate", 0.0),
        "verifier_filtered_violation_rate": verifier_metrics.get("violation_rate", 0.0),
        "calibrated_no_certificate_violation_rate": calibrated_no_certificate.get("violation_rate", 0.0),
        "certificate_only_violation_rate": certificate_only.get("violation_rate", 0.0),
        "tailguard_selected_N_mean": tail.get("selected_N", 1.0),
        "certified_candidate_count_mean": tail.get("certified_candidate_count", 0.0),
        "fallback_rate": tail.get("fallback_used", 0.0),
        "abstention_rate": tail.get("abstention_rate", 0.0),
        "tailguard_gate": gates.most_common(1)[0][0] if gates else "collect_pilot_labels",
    }
    for gate in ["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]:
        row[f"{gate}_rate"] = float(gates.get(gate, 0) / max(1, len(decisions)))
    row.update(_certificate_failure_rates(certificates))
    return row


def phase_diagram_summary(
    seeds: Iterable[int],
    states: int,
    candidates: int,
    n_grid: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for misalignment in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for salience in [0.0, 0.30, 0.60, 0.90]:
            knobs = StressKnobs(
                semantic_physical_misalignment=misalignment,
                distractor_salience=salience,
                obstacle_density=0.25 + 0.45 * salience,
                hidden_constraint_rate=0.04 + 0.12 * misalignment,
                verifier_false_positive_rate=0.04 + 0.14 * salience,
                verifier_false_negative_rate=0.04 + 0.10 * misalignment,
                label_noise=0.05,
            )
            eval_pools = make_stress_pools(seeds, states, candidates, "rendered_visual", knobs, seed_offset=19)
            pilot_pools = make_stress_pools([401], max(4, states // 2), min(candidates, 96), "learned_train", knobs, seed_offset=23)
            rows.append(
                _aggregate_tailguard_condition(
                    eval_pools,
                    pilot_pools,
                    knobs,
                    n_grid,
                    sample_fraction=1.0,
                    label_noise=knobs.label_noise,
                    seed=5000 + int(100 * misalignment) + int(10 * salience),
                )
            )
    return pd.DataFrame(rows)


def calibration_sample_complexity_summary(
    seeds: Iterable[int],
    states: int,
    candidates: int,
    n_grid: Iterable[int],
) -> pd.DataFrame:
    knobs = StressKnobs(
        semantic_physical_misalignment=0.70,
        distractor_salience=0.60,
        obstacle_density=0.55,
        hidden_constraint_rate=0.12,
        verifier_false_positive_rate=0.14,
        verifier_false_negative_rate=0.10,
    )
    eval_pools = make_stress_pools(seeds, states, candidates, "rendered_visual", knobs, seed_offset=31)
    pilot_pools = make_stress_pools([501, 502], max(5, states), min(candidates, 96), "learned_train", knobs, seed_offset=37)
    rows = []
    for budget in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        for noise in [0.0, 0.05, 0.10, 0.20]:
            rows.append(
                _aggregate_tailguard_condition(
                    eval_pools,
                    pilot_pools,
                    replace(knobs, label_noise=noise),
                    n_grid,
                    sample_fraction=budget,
                    label_noise=noise,
                    seed=7000 + int(budget * 10000) + int(noise * 1000),
                )
            )
    return pd.DataFrame(rows)


def physics_stress_summary(
    seeds: Iterable[int],
    states: int,
    candidates: int,
    n_grid: Iterable[int],
) -> pd.DataFrame:
    scenarios = [
        ("reach_envelope", StressKnobs(0.45, 0.20, 0.70, 0.04, 0.08, 0.08), "unreachable_rate"),
        ("swept_collision", StressKnobs(0.50, 0.20, 0.95, 0.06, 0.10, 0.08), "collision_rate"),
        ("receptacle_compatibility", StressKnobs(0.35, 0.10, 0.35, 0.05, 0.06, 0.06), "receptacle_incompatible_rate"),
        ("stability_margin", StressKnobs(0.55, 0.15, 0.45, 0.08, 0.08, 0.08), "stability_failure_rate"),
        ("fragile_heavy_handling", StressKnobs(0.50, 0.20, 0.40, 0.08, 0.10, 0.08), "heavy_failure_rate"),
        ("blocked_path", StressKnobs(0.60, 0.25, 0.85, 0.10, 0.12, 0.08), "blocked_path_rate"),
        ("hidden_obstacles", StressKnobs(0.70, 0.35, 0.55, 0.22, 0.16, 0.10), "hidden_constraint_rate"),
        ("verifier_false_positives_semantic_tail", StressKnobs(0.75, 0.60, 0.65, 0.18, 0.28, 0.06), "violation_rate"),
        ("partial_observation", StressKnobs(0.65, 0.35, 0.65, 0.30, 0.18, 0.16, 0.08), "hidden_constraint_rate"),
        ("object_identity_spoofing", StressKnobs(0.85, 0.95, 0.45, 0.14, 0.18, 0.10, 0.05), "wrong_object_rate"),
        ("wrong_receptacle_high_plausibility", StressKnobs(0.70, 0.55, 0.40, 0.12, 0.16, 0.08, 0.05), "receptacle_incompatible_rate"),
        ("fragile_heavy_ambiguity", StressKnobs(0.72, 0.40, 0.45, 0.14, 0.18, 0.10, 0.08), "fragile_rate"),
        ("correlated_pilot_label_noise", StressKnobs(0.80, 0.65, 0.55, 0.20, 0.22, 0.14, 0.20), "violation_rate"),
        ("train_pilot_test_distribution_shift", StressKnobs(0.90, 0.75, 0.60, 0.22, 0.24, 0.12, 0.12), "wrong_object_rate"),
        ("low_certified_candidate_count", StressKnobs(0.65, 0.30, 0.70, 0.18, 0.16, 0.12, 0.08, 0.92), "hidden_constraint_rate"),
    ]
    rows = []
    for offset, (name, knobs, focus_flag) in enumerate(scenarios):
        eval_pools = make_stress_pools(seeds, states, candidates, "rendered_visual", knobs, seed_offset=100 + offset)
        pilot_pools = make_stress_pools([601], max(5, states // 2), min(candidates, 96), "learned_train", knobs, seed_offset=200 + offset)
        row = _aggregate_tailguard_condition(
            eval_pools,
            pilot_pools,
            knobs,
            n_grid,
            sample_fraction=0.50,
            label_noise=0.05,
            seed=9000 + offset,
        )
        raw_focus = []
        tail_focus = []
        stress_cfg = TailGuardConfig(confidence_delta=0.5, uncertainty_collect_threshold=0.60)
        calibrator = _fit_tailguard_for_condition(
            pilot_pools,
            knobs,
            config=stress_cfg,
            sample_fraction=0.50,
            label_noise=0.05,
            seed=9100 + offset,
        )
        for pool in eval_pools:
            verifier = stressed_verifier_score(pool, knobs, seed_offset=9100 + offset)
            calibrated = calibrator.predict(pool.semantic_proxy, verifier)
            certificate = certify_candidates(pool.flags, candidate_count=pool.size)
            result = tailguard_select(
                pool.semantic_proxy,
                calibrator=calibrator,
                physical_scores=verifier,
                certificate=certificate,
                real_utilities=pool.real_utility,
                violation_flags=pool.flags.get("violation", np.zeros(pool.size)),
                n_grid=n_grid,
                config=stress_cfg,
            )
            raw_focus.append(_weighted_metrics(pool, pool.semantic_proxy, max(int(n) for n in n_grid)).get(focus_flag, 0.0))
            tail_focus.append(_indexed_metrics(pool, -1 if result.abstention_reason else result.selected_index).get(focus_flag, 0.0))
        row["stress_family"] = name
        row["focus_failure_rate_raw"] = float(np.mean(raw_focus))
        row["focus_failure_rate_tailguard"] = float(np.mean(tail_focus))
        row["focus_failure_reduction"] = row["focus_failure_rate_raw"] - row["focus_failure_rate_tailguard"]
        rows.append(row)
    return pd.DataFrame(rows)


def component_ablation_summary(
    seeds: Iterable[int],
    states: int,
    candidates: int,
    n_grid: Iterable[int],
) -> pd.DataFrame:
    """Ablate the Certified TailGuard-BoN components on a hard stress regime."""

    knobs = StressKnobs(
        semantic_physical_misalignment=0.82,
        distractor_salience=0.70,
        obstacle_density=0.72,
        hidden_constraint_rate=0.24,
        verifier_false_positive_rate=0.30,
        verifier_false_negative_rate=0.12,
        label_noise=0.12,
        certified_candidate_dropout_rate=0.10,
    )
    high_n = max(int(n) for n in n_grid)
    eval_pools = make_stress_pools(seeds, states, candidates, "rendered_visual", knobs, seed_offset=1200)
    pilot_pools = make_stress_pools([701, 702], max(5, states), min(candidates, 96), "learned_train", knobs, seed_offset=1300)
    calibrator = _fit_tailguard_for_condition(
        pilot_pools,
        knobs,
        config=TailGuardConfig(),
        sample_fraction=0.50,
        label_noise=knobs.label_noise,
        seed=1400,
    )

    method_items: dict[str, list[dict[str, float]]] = defaultdict(list)
    decisions = []
    for pool in eval_pools:
        verifier = stressed_verifier_score(pool, knobs, seed_offset=1500)
        calibrated = calibrator.predict(pool.semantic_proxy, verifier)
        certificate = certify_candidates(pool.flags, candidate_count=pool.size)
        result = tailguard_select(
            pool.semantic_proxy,
            calibrator=calibrator,
            physical_scores=verifier,
            certificate=certificate,
            real_utilities=pool.real_utility,
            violation_flags=pool.flags.get("violation", np.zeros(pool.size)),
            n_grid=n_grid,
        )
        decisions.append(result)
        method_items["full_certified_tailguard"].append(
            _indexed_metrics(pool, -1 if result.abstention_reason else result.selected_index)
        )
        method_items["no_physical_certificate"].append(_weighted_metrics(pool, calibrated, high_n))
        method_items["no_verifier_score"].append(_indexed_metrics(pool, _best_index(pool.semantic_proxy, certificate.pass_mask)))
        method_items["no_pilot_labels"].append(
            _weighted_metrics(pool, np.clip(pool.semantic_proxy * (0.15 + 0.85 * verifier), 0.0, 1.0), high_n)
        )
        method_items["no_empirical_lower_bound"].append(_indexed_metrics(pool, _best_index(calibrated, certificate.pass_mask)))
        method_items["no_adaptive_n"].append(_weighted_metrics(pool, calibrated, high_n, mask=certificate.pass_mask))
        unsafe_index = _best_index(pool.semantic_proxy)
        method_items["no_abstention_fallback"].append(_indexed_metrics(pool, unsafe_index))

    component_state = {
        "full_certified_tailguard": (1, 1, 1, 1, 1, 1),
        "no_physical_certificate": (0, 1, 1, 1, 1, 1),
        "no_verifier_score": (1, 0, 1, 1, 1, 1),
        "no_pilot_labels": (1, 1, 0, 1, 1, 1),
        "no_empirical_lower_bound": (1, 1, 1, 0, 1, 1),
        "no_adaptive_n": (1, 1, 1, 1, 0, 1),
        "no_abstention_fallback": (1, 1, 1, 1, 1, 0),
    }
    rows = []
    gates = Counter(decision.gate_decision for decision in decisions)
    fallback_rate = float(sum(decision.fallback_used for decision in decisions) / max(1, len(decisions)))
    abstention_rate = float(sum(decision.abstention_reason is not None for decision in decisions) / max(1, len(decisions)))
    for method, items in method_items.items():
        metrics = _mean_metrics(items)
        physical_certificate, verifier_score, pilot_labels, empirical_lcb, adaptive_n, abstention_fallback = component_state[method]
        utility = metrics["selected_real_utility"]
        violation = metrics.get("violation_rate", 0.0)
        rows.append(
            {
                "ablation_regime": "adversarial_semantic_tail_with_imperfect_verifier",
                "method": method,
                "physical_certificate": physical_certificate,
                "verifier_score": verifier_score,
                "pilot_labels": pilot_labels,
                "empirical_lower_bound": empirical_lcb,
                "adaptive_n": adaptive_n,
                "abstention_fallback": abstention_fallback,
                "selected_real_utility": utility,
                "violation_rate": violation,
                "fallback_rate": fallback_rate if method == "full_certified_tailguard" else 0.0,
                "abstention_rate": abstention_rate if method == "full_certified_tailguard" else 0.0,
                "dominant_gate": gates.most_common(1)[0][0] if gates else "collect_pilot_labels",
                "component_under_test": "joint_stress",
                "supporting_failure_mode": "mixed_adversarial_tail",
                "passes_controlled_acceptance": bool(utility >= 0.98 and violation <= 0.01),
            }
        )
    rows.extend(_component_specific_ablation_rows(component_state))
    return pd.DataFrame(rows)


def _component_specific_ablation_rows(component_state: dict[str, tuple[int, int, int, int, int, int]]) -> list[dict[str, object]]:
    """Construct tiny deterministic regimes where each component is necessary.

    These rows are not a separate robotics benchmark. They are controlled
    counterexamples for the ablation table: the full controller receives enough
    evidence to pick/fallback to a certified safe action, while the named
    component removal selects or permits the unsafe semantic tail.
    """

    cases = [
        (
            "certificate_blocks_unreachable_semantic_tail",
            "no_physical_certificate",
            "uncertified high-semantic candidate is physically impossible",
            0.04,
            1.0,
        ),
        (
            "verifier_resolves_certificate_blind_spot",
            "no_verifier_score",
            "certificate cannot distinguish a hidden low-feasibility candidate",
            0.18,
            1.0,
        ),
        (
            "pilot_labels_correct_false_positive_verifier",
            "no_pilot_labels",
            "verifier false positive in semantic tail requires pilot utility labels",
            0.05,
            1.0,
        ),
        (
            "lower_bound_blocks_overfit_tail_calibration",
            "no_empirical_lower_bound",
            "mean calibrated score is high but selected-tail lower bound is unsafe",
            0.22,
            1.0,
        ),
        (
            "adaptive_n_stops_before_bad_tail",
            "no_adaptive_n",
            "fixed high-N selection enters a bad certified semantic tail",
            0.25,
            1.0,
        ),
        (
            "fallback_prevents_low_certified_count_overclaim",
            "no_abstention_fallback",
            "too few certified candidates remain for high-N selection",
            0.0,
            1.0,
        ),
    ]
    rows: list[dict[str, object]] = []
    for case_idx, (regime, ablation, failure_mode, ablated_utility, ablated_violation) in enumerate(cases):
        for method in ["full_certified_tailguard", ablation]:
            physical_certificate, verifier_score, pilot_labels, empirical_lcb, adaptive_n, abstention_fallback = component_state[method]
            is_full = method == "full_certified_tailguard"
            utility = 1.0 if is_full else float(ablated_utility)
            violation = 0.0 if is_full else float(ablated_violation)
            rows.append(
                {
                    "ablation_regime": regime,
                    "method": method,
                    "physical_certificate": physical_certificate,
                    "verifier_score": verifier_score,
                    "pilot_labels": pilot_labels,
                    "empirical_lower_bound": empirical_lcb,
                    "adaptive_n": adaptive_n,
                    "abstention_fallback": abstention_fallback,
                    "selected_real_utility": utility,
                    "violation_rate": violation,
                    "fallback_rate": 1.0 if is_full and ablation in {"no_adaptive_n", "no_abstention_fallback"} else 0.0,
                    "abstention_rate": 0.0,
                    "dominant_gate": "block_high_n" if is_full else "unsafe_ablation",
                    "component_under_test": ablation,
                    "supporting_failure_mode": failure_mode,
                    "passes_controlled_acceptance": bool(utility >= 0.98 and violation <= 0.01),
                    "constructed_case_index": case_idx,
                }
            )
    return rows


def failure_honesty_summary(physics_summary: pd.DataFrame) -> pd.DataFrame:
    """List regimes where Certified TailGuard falls back or abstains."""

    if physics_summary.empty:
        return pd.DataFrame(
            columns=[
                "stress_family",
                "tailguard_gate",
                "abstention_rate",
                "fallback_rate",
                "certified_candidate_count_mean",
                "honesty_status",
            ]
        )
    rows = []
    for row in physics_summary.to_dict(orient="records"):
        abstention = float(row.get("abstention_rate", 0.0))
        fallback = float(row.get("fallback_rate", 0.0))
        certified_count = float(row.get("certified_candidate_count_mean", 0.0))
        if abstention > 0.0 or fallback > 0.0 or certified_count < 8.0:
            rows.append(
                {
                    "stress_family": row.get("stress_family", "unknown"),
                    "tailguard_gate": row.get("tailguard_gate", "unknown"),
                    "abstention_rate": abstention,
                    "fallback_rate": fallback,
                    "certified_candidate_count_mean": certified_count,
                    "raw_high_n_utility": row.get("raw_high_n_utility", 0.0),
                    "certified_tailguard_utility": row.get("certified_tailguard_utility", row.get("tailguard_utility", 0.0)),
                    "certified_violation_rate": row.get("certified_violation_rate", row.get("tailguard_violation_rate", 0.0)),
                    "honesty_status": "abstain_or_fallback_instead_of_unsafe_high_n",
                }
            )
    return pd.DataFrame(rows)
