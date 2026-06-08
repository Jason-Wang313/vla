"""Run controlled and learned VLA-style Best-of-N experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vla_best_of_n.diagnostics import evaluate_selection, seed_variance
from vla_best_of_n.io import ensure_dir, write_csv, write_json
from vla_best_of_n.learned_vla import LearnedVLAScorer, RealUtilityCalibrator
from vla_best_of_n.optional_vla import write_optional_vla_status
from vla_best_of_n.plotting import create_figures
from vla_best_of_n.rendering import attach_visual_observation, create_render_montage, render_pools
from vla_best_of_n.robustness import robustness_scores
from vla_best_of_n.scorers import (
    grounded_score,
    hand_scorer_dict,
    oracle_score,
    random_score,
    semantic_plus_physical_score,
)
from vla_best_of_n.simulator import simulate_pool
from vla_best_of_n.torch_vla import TorchVLAScorer, torch_available
from vla_best_of_n.vla_env import CandidatePool, generate_pools


def _ns_for(candidates: int) -> list[int]:
    ns = [1, 2, 4, 8, 16, 32, 64]
    if candidates >= 128:
        ns.append(128)
    return [n for n in ns if n <= candidates]


def _artifact_from(summary: pd.DataFrame, method: str, path: Path) -> dict:
    curve = summary[summary["method"] == method].sort_values("N")
    if curve.empty:
        raise ValueError(f"missing method {method}")
    high = curve.iloc[-1].to_dict()
    artifact = {
        "path": str(path),
        "method": method,
        "N_values": [int(n) for n in curve["N"].tolist()],
        "selected_semantic_score_vs_N": {str(int(r.N)): float(r.selected_semantic_score) for r in curve.itertuples()},
        "selected_real_utility_vs_N": {str(int(r.N)): float(r.selected_real_utility) for r in curve.itertuples()},
        "semantic_real_tail_gap": float(high["semantic_real_tail_gap"]),
        "tail_rank_correlation": float(high["tail_rank_correlation"]),
        "high_N_regret": float(high["high_N_regret"]),
        "violation_rate": float(high["violation_rate"]),
        "deployment_gate": str(high["deployment_gate"]),
    }
    return artifact


def _write_seed_files(root: Path, name: str, seed_df: pd.DataFrame) -> None:
    seed_dir = ensure_dir(root / "seed_level")
    for seed, group in seed_df.groupby("seed"):
        write_json(
            seed_dir / f"{name}_seed_{int(seed):03d}.json",
            {
                "experiment": name,
                "seed": int(seed),
                "rows": group.sort_values(["method", "N"]).to_dict(orient="records"),
            },
        )


def _learned_scores(
    eval_pools: list[CandidatePool],
    train_pools: list[CandidatePool],
    pilot_pools: list[CandidatePool],
    epochs: int,
) -> dict[str, list[np.ndarray]]:
    scorer = LearnedVLAScorer(seed=17, epochs=epochs).fit(train_pools)
    learned_eval = scorer.score_pools(eval_pools)
    learned_pilot = scorer.score_pools(pilot_pools)
    calibrator = RealUtilityCalibrator().fit(pilot_pools, learned_pilot)
    calibrated_eval = calibrator.score_pools(eval_pools, learned_eval)
    scores = {
        "raw_semantic": learned_eval,
        "semantic_plus_physical": [semantic_plus_physical_score(pool) for pool in eval_pools],
        "calibrated": calibrated_eval,
        "grounded_combined": [grounded_score(pool) for pool in eval_pools],
        "random": [random_score(pool) for pool in eval_pools],
        "oracle": [oracle_score(pool) for pool in eval_pools],
    }
    if torch_available():
        torch_scorer = TorchVLAScorer(seed=37, epochs=max(8, int(epochs) * 12)).fit(train_pools)
        torch_eval = torch_scorer.score_pools(eval_pools)
        torch_pilot = torch_scorer.score_pools(pilot_pools)
        torch_calibrator = RealUtilityCalibrator().fit(pilot_pools, torch_pilot)
        scores["torch_semantic"] = torch_eval
        scores["torch_calibrated"] = torch_calibrator.score_pools(eval_pools, torch_eval)
    return scores


def run(mode: str, results_dir: Path, seeds: list[int], states: int, candidates: int, mc_trials: int, epochs: int) -> dict:
    ensure_dir(results_dir)
    ensure_dir(results_dir / "figures")
    ensure_dir(results_dir / "seed_level")
    ns = _ns_for(candidates)

    controlled_pools = generate_pools(seeds, states, candidates, "controlled")
    controlled_eval = evaluate_selection(
        "controlled_semantic_overselection",
        controlled_pools,
        hand_scorer_dict(controlled_pools),
        ns,
        mc_trials,
        rng_seed=11,
    )
    write_csv(results_dir / "controlled_summary.csv", controlled_eval.summary)
    write_csv(results_dir / "controlled_seed_metrics.csv", controlled_eval.seed_level)
    write_csv(results_dir / "controlled_seed_variance.csv", seed_variance(controlled_eval.seed_level))
    _write_seed_files(results_dir, "controlled", controlled_eval.seed_level)

    train_states = max(10, states * 2)
    pilot_states = max(5, states // 2)
    train_pools = generate_pools([101, 102, 103], train_states, min(candidates, 96), "learned_train")
    pilot_pools = generate_pools([201], pilot_states, min(candidates, 96), "learned_train")
    learned_pools = generate_pools(seeds, states, candidates, "learned_eval")
    learned_scores = _learned_scores(learned_pools, train_pools, pilot_pools, epochs)
    learned_eval = evaluate_selection("learned_vla_style", learned_pools, learned_scores, ns, mc_trials, rng_seed=23)
    write_csv(results_dir / "learned_summary.csv", learned_eval.summary)
    write_csv(results_dir / "learned_seed_metrics.csv", learned_eval.seed_level)
    write_csv(results_dir / "learned_seed_variance.csv", seed_variance(learned_eval.seed_level))
    _write_seed_files(results_dir, "learned", learned_eval.seed_level)

    distractor_pools = generate_pools(seeds, states, candidates, "distractor")
    distractor_scores = hand_scorer_dict(distractor_pools)
    distractor_eval = evaluate_selection(
        "semantic_distractors_object_binding",
        distractor_pools,
        distractor_scores,
        ns,
        mc_trials,
        rng_seed=31,
    )
    write_csv(results_dir / "distractor_summary.csv", distractor_eval.summary)
    write_csv(results_dir / "distractor_seed_metrics.csv", distractor_eval.seed_level)
    write_csv(results_dir / "distractor_seed_variance.csv", seed_variance(distractor_eval.seed_level))
    _write_seed_files(results_dir, "distractor", distractor_eval.seed_level)

    repair_eval = evaluate_selection("grounding_calibration_repair", learned_pools, learned_scores, ns, mc_trials, rng_seed=41)
    write_csv(results_dir / "repair_summary.csv", repair_eval.summary)
    write_csv(results_dir / "repair_seed_metrics.csv", repair_eval.seed_level)
    write_csv(results_dir / "repair_seed_variance.csv", seed_variance(repair_eval.seed_level))
    _write_seed_files(results_dir, "repair", repair_eval.seed_level)

    rendered_base = generate_pools(seeds, states, candidates, "rendered_visual")
    rendered_visual, rendered_metadata = render_pools(
        rendered_base,
        results_dir / "rendered",
        max_images=min(12, len(rendered_base)),
    )
    rendered_sim = [simulate_pool(pool) for pool in rendered_visual]
    write_csv(results_dir / "rendered_visual_metadata.csv", rendered_metadata)
    visual_train = [simulate_pool(attach_visual_observation(pool)) for pool in train_pools]
    visual_pilot = [simulate_pool(attach_visual_observation(pool)) for pool in pilot_pools]
    rendered_scores = _learned_scores(rendered_sim, visual_train, visual_pilot, epochs)
    rendered_eval = evaluate_selection(
        "rendered_visual_simulator",
        rendered_sim,
        rendered_scores,
        ns,
        mc_trials,
        rng_seed=53,
    )
    write_csv(results_dir / "rendered_summary.csv", rendered_eval.summary)
    write_csv(results_dir / "rendered_seed_metrics.csv", rendered_eval.seed_level)
    write_csv(results_dir / "rendered_seed_variance.csv", seed_variance(rendered_eval.seed_level))
    _write_seed_files(results_dir, "rendered", rendered_eval.seed_level)

    robustness_score_dict = robustness_scores(
        rendered_sim,
        rendered_scores["raw_semantic"],
        visual_pilot,
        LearnedVLAScorer(seed=17, epochs=epochs).fit(visual_train).score_pools(visual_pilot),
    )
    robustness_eval = evaluate_selection(
        "noisy_verifier_calibration_robustness",
        rendered_sim,
        robustness_score_dict,
        ns,
        mc_trials,
        rng_seed=67,
    )
    write_csv(results_dir / "robustness_summary.csv", robustness_eval.summary)
    write_csv(results_dir / "robustness_seed_metrics.csv", robustness_eval.seed_level)
    write_csv(results_dir / "robustness_seed_variance.csv", seed_variance(robustness_eval.seed_level))
    _write_seed_files(results_dir, "robustness", robustness_eval.seed_level)

    controlled_artifact = _artifact_from(controlled_eval.summary, "raw_semantic", results_dir / "controlled_summary.csv")
    learned_artifact = _artifact_from(learned_eval.summary, "raw_semantic", results_dir / "learned_summary.csv")
    repair_raw = _artifact_from(repair_eval.summary, "raw_semantic", results_dir / "repair_summary.csv")
    repair_calibrated = _artifact_from(repair_eval.summary, "calibrated", results_dir / "repair_summary.csv")
    high_n = max(ns)
    raw_high = repair_eval.summary[(repair_eval.summary["method"] == "raw_semantic") & (repair_eval.summary["N"] == high_n)].iloc[0]
    calibrated_high = repair_eval.summary[(repair_eval.summary["method"] == "calibrated") & (repair_eval.summary["N"] == high_n)].iloc[0]
    oracle_high = repair_eval.summary[(repair_eval.summary["method"] == "oracle") & (repair_eval.summary["N"] == high_n)].iloc[0]
    repair_artifact = {
        "path": str(results_dir / "repair_summary.csv"),
        "N": int(high_n),
        "raw_semantic": repair_raw,
        "calibrated": repair_calibrated,
        "real_utility_improvement": float(calibrated_high["selected_real_utility"] - raw_high["selected_real_utility"]),
        "raw_ci": [float(raw_high["mc_ci_low"]), float(raw_high["mc_ci_high"])],
        "calibrated_ci": [float(calibrated_high["mc_ci_low"]), float(calibrated_high["mc_ci_high"])],
        "oracle_gap_raw": float(oracle_high["selected_real_utility"] - raw_high["selected_real_utility"]),
        "oracle_gap_calibrated": float(oracle_high["selected_real_utility"] - calibrated_high["selected_real_utility"]),
        "violation_reduction": float(raw_high["violation_rate"] - calibrated_high["violation_rate"]),
        "deployment_gate_change": f"{raw_high['deployment_gate']} -> {calibrated_high['deployment_gate']}",
    }
    write_json(results_dir / "controlled_semantic_overselection_artifact.json", controlled_artifact)
    write_json(results_dir / "learned_vla_artifact.json", learned_artifact)
    write_json(results_dir / "repair_artifact.json", repair_artifact)
    rendered_artifact = _artifact_from(rendered_eval.summary, "raw_semantic", results_dir / "rendered_summary.csv")
    write_json(results_dir / "rendered_visual_simulator_artifact.json", rendered_artifact)
    robustness_artifact = {
        "path": str(results_dir / "robustness_summary.csv"),
        "N": int(high_n),
        "methods": robustness_eval.summary[robustness_eval.summary["N"] == high_n]
        .sort_values("method")
        .to_dict(orient="records"),
    }
    write_json(results_dir / "robustness_artifact.json", robustness_artifact)

    figure_paths = create_figures(results_dir)
    montage_inputs = [p for p in rendered_metadata["render_path"].dropna().tolist() if p][:8]
    if montage_inputs:
        montage = create_render_montage(montage_inputs, results_dir / "figures" / "figure7_rendered_scene_examples.png")
        figure_paths.append(str(montage))
    optional_vla_status = write_optional_vla_status(results_dir)
    optional_artifacts = {
        "optional_vla_status": str(results_dir / "optional_vla" / "adapter_status.json"),
    }
    optional_probe_path = results_dir / "optional_vla" / "inference_probe.json"
    if optional_probe_path.exists():
        optional_artifacts["optional_vla_inference_probe"] = str(optional_probe_path)
    manifest = {
        "mode": mode,
        "N_values": ns,
        "seeds": seeds,
        "states_per_seed": states,
        "candidates_per_state": candidates,
        "mc_trials": mc_trials,
        "epochs": epochs,
        "figures": figure_paths,
        "artifacts": {
            "controlled": str(results_dir / "controlled_semantic_overselection_artifact.json"),
            "learned": str(results_dir / "learned_vla_artifact.json"),
            "repair": str(results_dir / "repair_artifact.json"),
            "rendered_visual_simulator": str(results_dir / "rendered_visual_simulator_artifact.json"),
            "robustness": str(results_dir / "robustness_artifact.json"),
            **optional_artifacts,
        },
        "optional_vla_status": optional_vla_status["status"],
    }
    write_json(results_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--results-dir", default=os.environ.get("VLA_BON_RESULTS_DIR", "results"))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--states", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--mc-trials", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    if args.mode == "smoke":
        seeds = args.seeds or [1, 2]
        states = args.states or 6
        candidates = args.candidates or 64
        mc_trials = args.mc_trials or 80
        epochs = args.epochs or 35
    else:
        seeds = args.seeds or [1, 2, 3, 4]
        states = args.states or 12
        candidates = args.candidates or 128
        mc_trials = args.mc_trials or 180
        epochs = args.epochs or 70
    run(args.mode, Path(args.results_dir), seeds, states, candidates, mc_trials, epochs)


if __name__ == "__main__":
    main()
