"""Tail-aware deployment gate for high-N VLA selection."""

from __future__ import annotations


VALID_GATES = {"allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"}


def deployment_gate(metrics: dict[str, float]) -> str:
    """Return exactly one deterministic high-N gate decision.

    The gate is intentionally conservative: high selected-tail regret plus high
    physical violation blocks high-N; ambiguous tail ranking asks for pilot
    labels; low-risk saturation stops early; otherwise high-N is allowed.
    """

    high_n_regret = float(metrics.get("high_N_regret", metrics.get("oracle_minus_method_gap", 0.0)))
    tail_corr = float(metrics.get("tail_rank_correlation", 0.0))
    tail_gap = float(metrics.get("semantic_real_tail_gap", 0.0))
    violation = float(metrics.get("violation_rate", 0.0))
    marginal = float(metrics.get("marginal_value_per_added_candidate", 0.0))
    improvement = float(metrics.get("calibration_improvement", 0.0))
    utility = float(metrics.get("selected_real_utility", 0.0))

    if high_n_regret >= 0.28 and violation >= 0.45 and tail_corr <= 0.12:
        return "block_high_n"
    if high_n_regret >= 0.18 or tail_corr < -0.05 or tail_gap >= 0.42:
        if improvement >= 0.08 and violation < 0.35:
            return "allow_high_n"
        return "collect_pilot_labels"
    if utility >= 0.78 and abs(marginal) <= 0.006:
        return "stop_early"
    return "allow_high_n"
