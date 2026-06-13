"""Diagnostics for selected-tail VLA score-tail behavior."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .deployment import deployment_gate
from .theory import monte_carlo_selected_values, tie_aware_selection_probabilities
from .vla_env import CandidatePool


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    if len(xv) < 2 or np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return 0.0
    rx = _rankdata(xv)
    ry = _rankdata(yv)
    corr = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def tail_stats(scores: np.ndarray, semantic: np.ndarray, utility: np.ndarray, n: int) -> dict[str, float]:
    top_count = max(5, int(np.ceil(len(scores) / max(1, n))))
    top_count = min(top_count, len(scores))
    top = np.argsort(scores)[-top_count:]
    return {
        "semantic_real_tail_gap": float(np.mean(semantic[top]) - np.mean(utility[top])),
        "top_score_tail_real_utility": float(np.mean(utility[top])),
        "tail_rank_correlation": spearman(scores[top], utility[top]),
    }


@dataclass
class EvaluationResult:
    summary: pd.DataFrame
    seed_level: pd.DataFrame


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def evaluate_selection(
    experiment: str,
    pools: list[CandidatePool],
    scores_by_method: dict[str, list[np.ndarray]],
    ns: list[int],
    mc_trials: int,
    rng_seed: int = 0,
) -> EvaluationResult:
    """Evaluate exact law, Monte Carlo estimates, and VLA diagnostics."""

    rows: list[dict[str, float | str | int]] = []
    seed_rows: list[dict[str, float | str | int]] = []
    oracle_values_by_n: dict[int, list[float]] = defaultdict(list)
    for n in ns:
        for pool in pools:
            probs = tie_aware_selection_probabilities(pool.real_utility, n).probabilities
            oracle_values_by_n[n].append(float(probs @ pool.real_utility))
    oracle_mean = {n: float(np.mean(vals)) for n, vals in oracle_values_by_n.items()}

    rng = np.random.default_rng(rng_seed)
    previous_by_method: dict[str, float] = {}
    for method, score_lists in scores_by_method.items():
        if len(score_lists) != len(pools):
            raise ValueError(f"method {method} has {len(score_lists)} score arrays for {len(pools)} pools")
        for n in ns:
            exact_utility = []
            exact_semantic = []
            exact_physical = []
            exact_flags: dict[str, list[float]] = defaultdict(list)
            overall_corr = []
            tail_gap = []
            tail_real = []
            tail_corr = []
            mc_values = []
            by_seed: dict[int, list[float]] = defaultdict(list)
            by_seed_violation: dict[int, list[float]] = defaultdict(list)
            for pool, scores in zip(pools, score_lists):
                probs = tie_aware_selection_probabilities(scores, n).probabilities
                util = float(probs @ pool.real_utility)
                sem = float(probs @ pool.semantic_proxy)
                phys = float(probs @ pool.physical_feasibility)
                exact_utility.append(util)
                exact_semantic.append(sem)
                exact_physical.append(phys)
                by_seed[pool.seed].append(util)
                by_seed_violation[pool.seed].append(float(probs @ pool.flags["violation"]))
                for flag_name, flag_values in pool.flags.items():
                    exact_flags[flag_name].append(float(probs @ flag_values))
                overall_corr.append(spearman(scores, pool.real_utility))
                ts = tail_stats(scores, pool.semantic_proxy, pool.real_utility, n)
                tail_gap.append(ts["semantic_real_tail_gap"])
                tail_real.append(ts["top_score_tail_real_utility"])
                tail_corr.append(ts["tail_rank_correlation"])
                mc = monte_carlo_selected_values(scores, pool.real_utility, n, mc_trials, rng)
                mc_values.extend(mc.tolist())

            selected_utility = float(np.mean(exact_utility))
            mc_mean, mc_low, mc_high = _mean_ci(np.asarray(mc_values, dtype=float))
            marginal = selected_utility - previous_by_method.get(method, selected_utility)
            previous_by_method[method] = selected_utility
            row = {
                "experiment": experiment,
                "method": method,
                "N": int(n),
                "selected_real_utility": selected_utility,
                "selected_semantic_score": float(np.mean(exact_semantic)),
                "selected_physical_feasibility": float(np.mean(exact_physical)),
                "exact_selected_utility": selected_utility,
                "mc_selected_utility_mean": mc_mean,
                "mc_ci_low": mc_low,
                "mc_ci_high": mc_high,
                "exact_law_prediction_error": abs(mc_mean - selected_utility),
                "semantic_real_tail_gap": float(np.mean(tail_gap)),
                "top_score_tail_real_utility": float(np.mean(tail_real)),
                "overall_rank_correlation": float(np.mean(overall_corr)),
                "tail_rank_correlation": float(np.mean(tail_corr)),
                "high_N_regret": oracle_mean[n] - selected_utility,
                "oracle_minus_method_gap": oracle_mean[n] - selected_utility,
                "violation_rate": float(np.mean(exact_flags["violation"])),
                "wrong_object_rate": float(np.mean(exact_flags["wrong_object"])),
                "wrong_target_rate": float(np.mean(exact_flags["wrong_target"])),
                "collision_rate": float(np.mean(exact_flags["collision"])),
                "unreachable_rate": float(np.mean(exact_flags["unreachable"])),
                "fragile_rate": float(np.mean(exact_flags["fragile"])),
                "stability_failure_rate": float(np.mean(exact_flags["stability_failure"])),
                "instruction_satisfaction_proxy": float(np.mean(exact_semantic)),
                "physical_success": selected_utility,
                "calibration_improvement": 0.0,
                "marginal_value_per_added_candidate": marginal,
                "deployment_gate": "allow_high_n",
                "seed_count": int(len(set(pool.seed for pool in pools))),
                "pool_count": int(len(pools)),
            }
            row["deployment_gate"] = deployment_gate(row)
            rows.append(row)
            for seed, vals in sorted(by_seed.items()):
                seed_rows.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "N": int(n),
                        "seed": int(seed),
                        "selected_real_utility": float(np.mean(vals)),
                        "violation_rate": float(np.mean(by_seed_violation[seed])),
                    }
                )

    summary = pd.DataFrame(rows)
    summary = annotate_calibration_improvement(summary)
    for idx, row in summary.iterrows():
        summary.loc[idx, "deployment_gate"] = deployment_gate(row.to_dict())
    return EvaluationResult(summary=summary, seed_level=pd.DataFrame(seed_rows))


def annotate_calibration_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    """Add method improvement relative to raw semantic at each N."""

    out = summary.copy()
    raw = out[out["method"] == "raw_semantic"][["N", "selected_real_utility"]].rename(
        columns={"selected_real_utility": "raw_selected_real_utility"}
    )
    if raw.empty:
        return out
    out = out.merge(raw, on="N", how="left")
    out["calibration_improvement"] = out["selected_real_utility"] - out["raw_selected_real_utility"]
    out = out.drop(columns=["raw_selected_real_utility"])
    return out


def seed_variance(seed_level: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_level.groupby(["experiment", "method", "N"], as_index=False)
        .agg(seed_level_variance=("selected_real_utility", "var"))
        .fillna({"seed_level_variance": 0.0})
    )
