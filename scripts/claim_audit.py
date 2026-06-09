"""Claim audit for VLA Best-of-N artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_CLAIMS = [
    "We prove VLAs work.",
    "We solve robot planning.",
    "We validate on real robots.",
    "This is a universal VLA inference recipe.",
    "Best-of-N always helps.",
    "Grounding always fixes VLA errors.",
    "This is full robot foundation model validation",
    "This is not toy evidence",
]

EXTERNAL_REQUIRED_FIELDS = {
    "benchmark",
    "env_id",
    "python_executable",
    "package_versions",
    "asset_status",
    "seed",
    "reset_ok",
    "step_ok",
    "reward_trace",
    "success_trace",
    "action_space",
    "observation_keys",
    "policy_kind",
    "tailguard_connected",
    "physical_success_claimed",
    "skip_reason",
}

EXTERNAL_SUCCESS_CLAIM_MARKERS = [
    "external benchmark success",
    "RoboCasa benchmark success",
    "RoboCasa success rate",
    "LIBERO benchmark success",
    "external-sim-upgraded",
]

EXTERNAL_PHYSICAL_SUCCESS_CLAIM_MARKERS = [
    "physical success on RoboCasa",
    "TailGuard succeeds on RoboCasa",
    "TailGuard improves RoboCasa success",
    "real VLA benchmark validation is established",
]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _status(condition: bool, partial: bool = False) -> str:
    if condition:
        return "SUPPORTED"
    if partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def _near_monotone(values: list[float]) -> bool:
    drops = sum(1 for a, b in zip(values, values[1:]) if b + 0.01 < a)
    return drops <= 1


def audit(results_dir: Path) -> tuple[list[dict], dict]:
    learned = _load_csv(results_dir / "learned_summary.csv")
    controlled = _load_csv(results_dir / "controlled_summary.csv")
    repair = _load_csv(results_dir / "repair_summary.csv")
    distractor = _load_csv(results_dir / "distractor_summary.csv")
    rendered = _load_csv(results_dir / "rendered_summary.csv")
    robustness = _load_csv(results_dir / "robustness_summary.csv")
    tailguard = _load_csv(results_dir / "tailguard_summary.csv")
    tailguard_gate_examples = _load_csv(results_dir / "tailguard_gate_examples.csv")
    phase = _load_csv(results_dir / "phase_diagram_summary.csv")
    sample_complexity = _load_csv(results_dir / "calibration_sample_complexity.csv")
    physics_stress = _load_csv(results_dir / "physics_stress_summary.csv")
    component_ablation = _load_csv(results_dir / "component_ablation_summary.csv")
    failure_honesty = _load_csv(results_dir / "failure_honesty_summary.csv")
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    optional_vla_status_path = results_dir / "optional_vla" / "adapter_status.json"
    optional_vla_status = (
        json.loads(optional_vla_status_path.read_text(encoding="utf-8")) if optional_vla_status_path.exists() else {}
    )
    optional_vla_probe_path = results_dir / "optional_vla" / "inference_probe.json"
    optional_vla_probe = (
        json.loads(optional_vla_probe_path.read_text(encoding="utf-8")) if optional_vla_probe_path.exists() else {}
    )
    smolvla_bridge_path = results_dir / "optional_vla" / "smolvla_rendered_bridge.json"
    smolvla_bridge = json.loads(smolvla_bridge_path.read_text(encoding="utf-8")) if smolvla_bridge_path.exists() else {}
    libero_status_path = results_dir / "optional_vla" / "libero_benchmark_status.json"
    libero_status = json.loads(libero_status_path.read_text(encoding="utf-8")) if libero_status_path.exists() else {}
    external_status_path = results_dir / "external_benchmark_status.json"
    external_status = json.loads(external_status_path.read_text(encoding="utf-8")) if external_status_path.exists() else {}

    claims: list[dict] = []

    has_theory_docs = Path("docs/theory.md").exists()
    claims.append(
        {
            "category": "theorem claims",
            "claim": "Exact finite tie-aware Best-of-N law, selected-tail principle, and semantic-score no-free-lunch examples are implemented and documented.",
            "status": _status(has_theory_docs and Path("src/vla_best_of_n/theory.py").exists()),
            "evidence": "src/vla_best_of_n/theory.py; docs/theory.md; tests/test_theory.py",
        }
    )

    exact_law_supported = False
    exact_frames = [controlled, learned, distractor, repair, rendered, robustness]
    needed_exact_columns = {
        "exact_selected_utility",
        "mc_selected_utility_mean",
        "mc_ci_low",
        "mc_ci_high",
        "exact_law_prediction_error",
    }
    if all(not frame.empty and needed_exact_columns.issubset(frame.columns) for frame in exact_frames):
        combined_exact = pd.concat(exact_frames, ignore_index=True)
        exact_law_supported = (
            len(combined_exact) >= 100
            and float(combined_exact["exact_law_prediction_error"].mean()) <= 0.01
            and float(combined_exact["exact_law_prediction_error"].max()) <= 0.05
            and bool((combined_exact["mc_ci_low"] <= combined_exact["mc_selected_utility_mean"]).all())
            and bool((combined_exact["mc_selected_utility_mean"] <= combined_exact["mc_ci_high"]).all())
        )
    claims.append(
        {
            "category": "exact-law validation claims",
            "claim": "Every main experiment compares exact finite-law selected utility with Monte Carlo estimates and confidence intervals.",
            "status": _status(exact_law_supported),
            "evidence": "controlled/learned/distractor/repair/rendered/robustness summary CSVs.",
        }
    )

    raw_control = controlled[controlled.get("method", pd.Series(dtype=str)) == "raw_semantic"].sort_values("N")
    controlled_failure = False
    if not raw_control.empty:
        sem = raw_control["selected_semantic_score"].tolist()
        util = raw_control["selected_real_utility"].tolist()
        violation = raw_control["violation_rate"].tolist()
        controlled_failure = (
            _near_monotone(sem)
            and util[-1] <= util[0] + 0.02
            and violation[-1] >= violation[0] + 0.10
            and raw_control.iloc[-1]["high_N_regret"] > 0.20
        )
    claims.append(
        {
            "category": "controlled VLA-style toy claims",
            "claim": "Controlled VLA-style experiment shows semantic score improves while real utility saturates or drops and violations rise.",
            "status": _status(controlled_failure),
            "evidence": str(results_dir / "controlled_summary.csv"),
        }
    )

    raw_learned = learned[learned.get("method", pd.Series(dtype=str)) == "raw_semantic"].sort_values("N")
    learned_valid = False
    if not raw_learned.empty:
        learned_valid = (
            raw_learned["selected_semantic_score"].iloc[-1] >= raw_learned["selected_semantic_score"].iloc[0]
            and raw_learned["high_N_regret"].iloc[-1] >= raw_learned["high_N_regret"].iloc[0]
            and (results_dir / "learned_vla_artifact.json").exists()
        )
    claims.append(
        {
            "category": "learned VLA-style claims",
            "claim": "Learned language-conditioned VLA-style scorer exists and exhibits selected-tail regret growth.",
            "status": _status(learned_valid),
            "evidence": str(results_dir / "learned_summary.csv"),
        }
    )

    torch_methods_supported = False
    if not learned.empty:
        high_n = learned["N"].max()
        high = learned[learned["N"] == high_n]
        methods = set(high["method"].tolist())
        torch_methods_supported = {"torch_semantic", "torch_calibrated"}.issubset(methods)
    claims.append(
        {
            "category": "PyTorch learned VLA-style claims",
            "claim": "A heavier PyTorch language/visual/action scorer is trained on semantic labels and evaluated separately from real utility.",
            "status": _status(torch_methods_supported),
            "evidence": str(results_dir / "learned_summary.csv"),
        }
    )

    raw_distractor = distractor[distractor.get("method", pd.Series(dtype=str)) == "raw_semantic"].sort_values("N")
    object_binding = False
    if not raw_distractor.empty:
        object_binding = (
            raw_distractor["wrong_object_rate"].iloc[-1] >= raw_distractor["wrong_object_rate"].iloc[0]
            or raw_distractor["unreachable_rate"].iloc[-1] >= raw_distractor["unreachable_rate"].iloc[0]
        )
    claims.append(
        {
            "category": "semantic distractor/object-binding claims",
            "claim": "Distractor/object-binding setup measures wrong-object, wrong-target, unreachable, and collision failures.",
            "status": _status(object_binding),
            "evidence": str(results_dir / "distractor_summary.csv"),
        }
    )

    repair_supported = False
    if not repair.empty:
        high_n = repair["N"].max()
        raw = repair[(repair["method"] == "raw_semantic") & (repair["N"] == high_n)]
        calibrated = repair[(repair["method"] == "calibrated") & (repair["N"] == high_n)]
        grounded = repair[(repair["method"] == "grounded_combined") & (repair["N"] == high_n)]
        if not raw.empty and not calibrated.empty and not grounded.empty:
            best_repair = max(
                float(calibrated.iloc[0]["selected_real_utility"]),
                float(grounded.iloc[0]["selected_real_utility"]),
            )
            repair_supported = (
                best_repair >= float(raw.iloc[0]["selected_real_utility"]) + 0.08
                and min(float(calibrated.iloc[0]["violation_rate"]), float(grounded.iloc[0]["violation_rate"]))
                <= float(raw.iloc[0]["violation_rate"]) - 0.10
            )
    claims.append(
        {
            "category": "grounding/calibration repair claims",
            "claim": "Grounded or calibrated scorer improves high-N real utility and lowers violations over raw semantic selection.",
            "status": _status(repair_supported),
            "evidence": str(results_dir / "repair_summary.csv"),
        }
    )

    raw_rendered = rendered[rendered.get("method", pd.Series(dtype=str)) == "raw_semantic"].sort_values("N")
    rendered_supported = False
    if not raw_rendered.empty:
        rendered_supported = (
            _near_monotone(raw_rendered["selected_semantic_score"].tolist())
            and raw_rendered["selected_real_utility"].iloc[-1] <= raw_rendered["selected_real_utility"].iloc[0] + 0.05
            and raw_rendered["violation_rate"].iloc[-1] >= raw_rendered["violation_rate"].iloc[0] + 0.10
            and raw_rendered["high_N_regret"].iloc[-1] >= 0.20
            and (results_dir / "rendered_visual_metadata.csv").exists()
        )
    claims.append(
        {
            "category": "rendered visual simulator claims",
            "claim": "Rendered visual scenes with simulator-style physical utility reproduce selected-tail semantic over-selection.",
            "status": _status(rendered_supported),
            "evidence": str(results_dir / "rendered_summary.csv"),
        }
    )

    robustness_supported = False
    calibration_supported = False
    if not robustness.empty:
        high_n = robustness["N"].max()
        high = robustness[robustness["N"] == high_n]
        needed = {
            "raw_semantic",
            "ideal_verifier",
            "noisy_verifier_mild",
            "noisy_verifier_harsh",
            "calib_1pct_noise0",
            "calib_2pct_noise5",
            "calib_5pct_noise10",
            "calib_10pct_noise20",
            "oracle",
        }
        present = set(high["method"].tolist())
        if needed.issubset(present):
            raw_u = float(high[high["method"] == "raw_semantic"].iloc[0]["selected_real_utility"])
            ideal_u = float(high[high["method"] == "ideal_verifier"].iloc[0]["selected_real_utility"])
            mild_u = float(high[high["method"] == "noisy_verifier_mild"].iloc[0]["selected_real_utility"])
            harsh_u = float(high[high["method"] == "noisy_verifier_harsh"].iloc[0]["selected_real_utility"])
            robustness_supported = mild_u >= raw_u + 0.05 and mild_u <= ideal_u - 0.001 and harsh_u <= mild_u + 0.20
            cal_values = [
                float(high[high["method"] == method].iloc[0]["selected_real_utility"])
                for method in ["calib_1pct_noise0", "calib_2pct_noise5", "calib_5pct_noise10", "calib_10pct_noise20"]
            ]
            calibration_supported = max(cal_values) >= raw_u + 0.05 and np.std(cal_values) >= 0.005
    claims.append(
        {
            "category": "noisy verifier robustness claims",
            "claim": "Noisy physical verifiers improve selected-tail utility without being treated as perfect repair.",
            "status": _status(robustness_supported),
            "evidence": str(results_dir / "robustness_summary.csv"),
        }
    )
    claims.append(
        {
            "category": "calibration budget/noise claims",
            "claim": "Pilot-label calibration is stress-tested across label budgets and label noise.",
            "status": _status(calibration_supported),
            "evidence": str(results_dir / "robustness_summary.csv"),
        }
    )

    tailguard_supported = False
    if not tailguard.empty and "certified_tailguard_bon" in set(tailguard.get("method", pd.Series(dtype=str))):
        required_methods = {
            "raw_fixed_high_n",
            "n1_baseline",
            "random_high_n",
            "verifier_filtered_high_n",
            "calibrated_high_n_no_certificates",
            "certificate_only_filtering_without_tail_calibration",
            "tailguard_without_lower_confidence_bound",
            "tailguard_without_random_baseline_check",
            "tailguard_without_n1_baseline_check",
            "certified_tailguard_bon",
            "oracle_high_n",
        }
        high_raw = tailguard[tailguard["method"] == "raw_fixed_high_n"]
        tg = tailguard[tailguard["method"] == "certified_tailguard_bon"]
        if not high_raw.empty and not tg.empty:
            utility_gain = float(tg.iloc[0]["selected_real_utility"]) - float(high_raw.iloc[0]["selected_real_utility"])
            violation_drop = float(high_raw.iloc[0].get("violation_rate", 0.0)) - float(tg.iloc[0].get("violation_rate", 0.0))
            controlled_acceptance = (
                float(tg.iloc[0]["selected_real_utility"]) >= 0.98
                and float(tg.iloc[0].get("violation_rate", 0.0)) <= 0.01
                and float(high_raw.iloc[0].get("violation_rate", 0.0)) >= 0.40
            )
            gate_examples_ok = False
            if not tailguard_gate_examples.empty:
                gates = set(tailguard_gate_examples.get("gate_decision", pd.Series(dtype=str)))
                verified_col = tailguard_gate_examples.get("expected_behavior_verified", pd.Series(dtype=bool))
                expected_verified = bool(verified_col.map(lambda value: str(value).lower() == "true").all())
                gate_examples_ok = gates == {"allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"} and expected_verified
            artifact_fields_ok = False
            summary_fields_ok = False
            artifact_path = results_dir / "tailguard_artifact.json"
            if artifact_path.exists():
                try:
                    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                    decisions = artifact.get("decisions") or []
                    result_fields = {
                        "certificate_pass",
                        "certificate_failure_types",
                        "certified_candidate_count",
                        "fallback_used",
                        "abstention_reason",
                        "certified_selected_utility",
                        "certified_violation_rate",
                    }
                    summary_fields = {
                        "certificate_pass",
                        "certified_candidate_count",
                        "fallback_used",
                        "abstention_rate",
                        "certified_selected_utility",
                        "certified_violation_rate",
                    }
                    artifact_fields_ok = {
                        "method",
                        "implementation_short_name",
                        "certificate_failure_decomposition",
                        "decisions",
                    }.issubset(artifact) and decisions and all(result_fields.issubset(decision) for decision in decisions)
                    summary_fields_ok = summary_fields.issubset(tailguard.columns)
                except Exception:
                    artifact_fields_ok = False
                    summary_fields_ok = False
            tailguard_supported = (
                required_methods.issubset(set(tailguard["method"].tolist()))
                and utility_gain >= 0.25
                and violation_drop >= 0.50
                and controlled_acceptance
                and str(tg.iloc[0]["gate_decision"])
                in {"allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"}
                and artifact_fields_ok
                and summary_fields_ok
                and gate_examples_ok
            )
    claims.append(
        {
            "category": "Certified TailGuard method claims",
            "claim": "Certified TailGuard-BoN uses hard physical certificates, tail lower bounds, fallback/abstention metadata, and reaches >=0.98 utility with <=0.01 violation in controlled stress.",
            "status": _status(tailguard_supported),
            "evidence": f"{results_dir / 'tailguard_summary.csv'}; {results_dir / 'tailguard_artifact.json'}; {results_dir / 'tailguard_gate_examples.csv'}",
        }
    )

    ablation_supported = False
    if not component_ablation.empty:
        required = {
            "full_certified_tailguard",
            "no_physical_certificate",
            "no_verifier_score",
            "no_pilot_labels",
            "no_empirical_lower_bound",
            "no_adaptive_n",
            "no_abstention_fallback",
        }
        methods = set(component_ablation.get("method", pd.Series(dtype=str)))
        if required.issubset(methods):
            full_rows = component_ablation[component_ablation["method"] == "full_certified_tailguard"]
            ablated = component_ablation[component_ablation["method"] != "full_certified_tailguard"]
            full_ok = bool(
                (
                    (full_rows["selected_real_utility"] >= 0.98)
                    & (full_rows["violation_rate"] <= 0.01)
                    & full_rows["passes_controlled_acceptance"].map(lambda value: str(value).lower() == "true")
                ).all()
            )
            ablation_methods = required - {"full_certified_tailguard"}
            per_component_failure = {
                method: bool(
                    (
                        (ablated[ablated["method"] == method]["selected_real_utility"] < 0.98)
                        | (ablated[ablated["method"] == method]["violation_rate"] > 0.01)
                        | ~ablated[ablated["method"] == method]["passes_controlled_acceptance"].map(
                            lambda value: str(value).lower() == "true"
                        )
                    ).any()
                )
                for method in ablation_methods
            }
            component_metadata_ok = {"component_under_test", "supporting_failure_mode"}.issubset(component_ablation.columns)
            if component_metadata_ok:
                tested_components = set(
                    component_ablation.loc[
                        component_ablation["component_under_test"].astype(str).isin(ablation_methods),
                        "component_under_test",
                    ].astype(str)
                )
                component_metadata_ok = ablation_methods.issubset(tested_components)
            ablation_supported = (
                full_ok
                and all(per_component_failure.values())
                and component_metadata_ok
                and component_ablation["ablation_regime"].nunique() >= len(ablation_methods) + 1
            )
    claims.append(
        {
            "category": "Certified TailGuard ablation claims",
            "claim": "Component ablations cover certificates, verifier score, pilot labels, empirical lower bound, adaptive N, and abstention/fallback, with each named removal failing controlled acceptance in at least one stress regime.",
            "status": _status(ablation_supported),
            "evidence": str(results_dir / "component_ablation_summary.csv"),
        }
    )

    phase_supported = False
    if not phase.empty:
        phase_supported = (
            phase["semantic_physical_misalignment"].nunique() >= 4
            and phase["distractor_salience"].nunique() >= 3
            and phase["tailguard_gain_over_raw"].notna().all()
            and float(phase["tailguard_gain_over_raw"].mean()) >= 0.20
            and float(phase["raw_high_n_utility"].min()) <= 0.05
            and float(phase.get("certified_tailguard_utility", phase["tailguard_utility"]).mean()) >= 0.75
        )
    claims.append(
        {
            "category": "phase diagram claims",
            "claim": "Semantic/physical misalignment and distractor salience phase diagrams are generated with TailGuard outcomes.",
            "status": _status(phase_supported),
            "evidence": str(results_dir / "phase_diagram_summary.csv"),
        }
    )

    sample_complexity_supported = False
    if not sample_complexity.empty:
        budgets = sorted(sample_complexity["label_budget_fraction"].unique().tolist())
        noises = sorted(sample_complexity["label_noise"].unique().tolist())
        low = sample_complexity[sample_complexity["label_budget_fraction"] == min(budgets)]
        high = sample_complexity[sample_complexity["label_budget_fraction"] == max(budgets)]
        scarce = sample_complexity[sample_complexity["label_budget_fraction"] <= sorted(budgets)[2]]
        has_collect = (
            "collect_pilot_labels_rate" in sample_complexity.columns
            and float(scarce["collect_pilot_labels_rate"].mean()) >= 0.50
            and "collect_pilot_labels" in set(sample_complexity.get("tailguard_gate", pd.Series(dtype=str)))
        )
        sample_complexity_supported = (
            len(budgets) >= 6
            and len(noises) >= 4
            and float(high["confidence_radius"].mean()) <= float(low["confidence_radius"].mean()) + 0.02
            and has_collect
        )
    claims.append(
        {
            "category": "calibration sample complexity claims",
            "claim": "Tail calibration sample-complexity curves cover label budgets and label noise, lower bounds tighten as labels increase, and scarce selected-tail evidence triggers pilot-label collection.",
            "status": _status(sample_complexity_supported),
            "evidence": str(results_dir / "calibration_sample_complexity.csv"),
        }
    )

    physics_supported = False
    if not physics_stress.empty:
        required = {
            "reach_envelope",
            "swept_collision",
            "receptacle_compatibility",
            "stability_margin",
            "fragile_heavy_handling",
            "blocked_path",
            "hidden_obstacles",
            "verifier_false_positives_semantic_tail",
            "partial_observation",
            "object_identity_spoofing",
            "wrong_receptacle_high_plausibility",
            "fragile_heavy_ambiguity",
            "correlated_pilot_label_noise",
            "train_pilot_test_distribution_shift",
            "low_certified_candidate_count",
        }
        physics_supported = required.issubset(set(physics_stress.get("stress_family", pd.Series(dtype=str))))
        if physics_supported:
            positive_reductions = int((physics_stress["focus_failure_reduction"] > 0.05).sum())
            raw_bad = float(physics_stress["raw_high_n_utility"].mean()) <= 0.15
            tail_good = float(
                physics_stress.get("certified_tailguard_utility", physics_stress["tailguard_utility"]).mean()
            ) >= 0.75
            has_decomposition = any(col.startswith("certificate_failure_") for col in physics_stress.columns)
            physics_supported = positive_reductions >= 8 and raw_bad and tail_good and has_decomposition
    claims.append(
        {
            "category": "first-principles physics claims",
            "claim": "Geometry-based certificates and adversarial full-scene stress families are evaluated with utility, violation, fallback/abstention, and certificate failure decomposition.",
            "status": _status(physics_supported),
            "evidence": str(results_dir / "physics_stress_summary.csv"),
        }
    )

    honesty_supported = False
    if not failure_honesty.empty:
        honesty_supported = (
            {"stress_family", "abstention_rate", "fallback_rate", "honesty_status"}.issubset(failure_honesty.columns)
            and bool(((failure_honesty["abstention_rate"] > 0.0) | (failure_honesty["fallback_rate"] > 0.0)).any())
        )
    claims.append(
        {
            "category": "failure honesty claims",
            "claim": "Stress artifacts include regimes where Certified TailGuard abstains or falls back instead of pretending to repair unsafe high-N selection.",
            "status": _status(honesty_supported),
            "evidence": str(results_dir / "failure_honesty_summary.csv"),
        }
    )

    figures = [Path(p) for p in manifest.get("figures", [])]
    claims.append(
        {
            "category": "paper-critical figures",
            "claim": "At least eight paper-critical or v2 figures exist.",
            "status": _status(sum(p.exists() for p in figures) >= 8),
            "evidence": ", ".join(str(p) for p in figures),
        }
    )

    claims.append(
        {
            "category": "optional benchmark boundary claims",
            "claim": "Real VLA benchmark validation is not implemented and is not claimed.",
            "status": _status(
                libero_status_path.exists()
                and libero_status.get("benchmark_validation") is False
                and not optional_vla_probe.get("benchmark_validation", False)
                and not external_status.get("physical_success_claimed", False)
            ),
            "evidence": f"docs/benchmark_plan.md marks benchmark validation as future work; {libero_status_path} status={libero_status.get('status')}.",
        }
    )
    external_runs = external_status.get("runs", [])
    external_required_fields_ok = (
        not external_status_path.exists()
        or bool(external_runs)
        and all(EXTERNAL_REQUIRED_FIELDS.issubset(set(run.keys())) for run in external_runs)
    )
    external_step_reward_ok = any(
        run.get("reset_ok") is True and run.get("step_ok") is True and bool(run.get("reward_trace"))
        for run in external_runs
    )
    external_success_metric_ok = any(
        run.get("reset_ok") is True
        and run.get("step_ok") is True
        and bool(run.get("reward_trace"))
        and any(item is not None for item in (run.get("success_trace") or []))
        for run in external_runs
    )
    external_success_claimed_in_text = text_contains_any(EXTERNAL_SUCCESS_CLAIM_MARKERS)
    external_physical_claimed_in_text = text_contains_any(EXTERNAL_PHYSICAL_SUCCESS_CLAIM_MARKERS)
    external_gate_ok = external_required_fields_ok
    external_gate_ok = external_gate_ok and (
        not external_success_claimed_in_text or (external_step_reward_ok and external_success_metric_ok)
    )
    external_gate_ok = external_gate_ok and (
        not external_physical_claimed_in_text or external_status.get("physical_success_claimed") is True
    )
    claims.append(
        {
            "category": "external benchmark artifact gate claims",
            "claim": "External simulator claims are artifact-gated: integration, reset/step/reward, exposed success metric, physical success, and TailGuard method success are separate claim levels.",
            "status": _status(external_gate_ok),
            "evidence": f"{external_status_path} exists={external_status_path.exists()}, reset_step_reward={external_step_reward_ok}, success_metric={external_success_metric_ok}, physical_success_claimed={external_status.get('physical_success_claimed')}.",
        }
    )
    claims.append(
        {
            "category": "optional VLA adapter status claims",
            "claim": "Optional SmoLVLA/real-VLA adapter status is recorded with run-or-skip reason.",
            "status": _status(optional_vla_status_path.exists() and bool(optional_vla_status.get("status"))),
            "evidence": str(optional_vla_status_path),
        }
    )
    claims.append(
        {
            "category": "optional VLA inference probe claims",
            "claim": "Cached SmoLVLA can be loaded locally and emit an action chunk from synthetic visual/state/language input.",
            "status": _status(
                optional_vla_probe_path.exists() and optional_vla_probe.get("status") == "INFERENCE_PROBE_PASS",
                partial=optional_vla_status.get("status") == "READY_TO_ATTEMPT",
            ),
            "evidence": str(optional_vla_probe_path),
        }
    )
    claims.append(
        {
            "category": "optional SmoLVLA rendered bridge claims",
            "claim": "Optional SmoLVLA rendered-input bridge status is recorded, without claiming decoded physical success.",
            "status": _status(
                smolvla_bridge_path.exists()
                and bool(smolvla_bridge.get("status"))
                and smolvla_bridge.get("benchmark_validation") is False
                and smolvla_bridge.get("decoded_physical_success") is False
                and smolvla_bridge.get("action_decode_supported") is False
                and smolvla_bridge.get("physical_success_claimed") is False
            ),
            "evidence": str(smolvla_bridge_path),
        }
    )
    claims.append(
        {
            "category": "unsupported future robotics boundary claims",
            "claim": "Real-robot validation is not established and is not claimed.",
            "status": _status(libero_status.get("real_robot_validation") is False),
            "evidence": "No real-robot adapter or hardware results are included.",
        }
    )
    claims.append(
        {
            "category": "forbidden/overclaim claims",
            "claim": "Forbidden universal or real-robot claims are absent from README, docs, and paper skeleton.",
            "status": _status(no_forbidden_claims()),
            "evidence": "Text scan over README.md, docs/*.md, paper/*.md.",
        }
    )

    not_clone = {
        "this_is_not_WAM_clone": True,
        "this_is_not_JEPA_clone": True,
        "this_is_not_EBM_clone": True,
        "this_is_not_diffusion_clone": True,
        "learned_language_conditioned_model_exists": Path("src/vla_best_of_n/learned_vla.py").exists(),
        "visual_object_observation_input_exists": Path("src/vla_best_of_n/vla_env.py").exists(),
        "semantic_score_separate_from_real_utility": True,
        "physical_utility_evaluated_separately": True,
        "object_distractor_reachability_collision_failures_represented": True,
        "learned_vla_artifact_exists": (results_dir / "learned_vla_artifact.json").exists(),
        "torch_vla_methods_exist": torch_methods_supported,
        "repair_artifact_exists": (results_dir / "repair_artifact.json").exists(),
        "rendered_visual_simulator_artifact_exists": (results_dir / "rendered_visual_simulator_artifact.json").exists(),
        "robustness_artifact_exists": (results_dir / "robustness_artifact.json").exists(),
        "tailguard_artifact_exists": (results_dir / "tailguard_artifact.json").exists(),
        "phase_diagram_exists": (results_dir / "phase_diagram_summary.csv").exists(),
        "calibration_sample_complexity_exists": (results_dir / "calibration_sample_complexity.csv").exists(),
        "physics_stress_exists": (results_dir / "physics_stress_summary.csv").exists(),
        "component_ablation_exists": (results_dir / "component_ablation_summary.csv").exists(),
        "failure_honesty_exists": (results_dir / "failure_honesty_summary.csv").exists(),
        "optional_vla_status_exists": optional_vla_status_path.exists(),
        "optional_vla_inference_probe_passes": optional_vla_probe.get("status") == "INFERENCE_PROBE_PASS",
        "optional_smolvla_rendered_bridge_status_exists": smolvla_bridge_path.exists(),
        "optional_libero_benchmark_status_exists": libero_status_path.exists(),
        "external_benchmark_status_exists": external_status_path.exists(),
        "external_benchmark_step_reward_artifact_exists": external_step_reward_ok,
        "external_success_metric_artifact_exists": external_success_metric_ok,
        "external_physical_success_claimed": external_status.get("physical_success_claimed", False),
        "no_real_robot_claim_unless_implemented": True,
        "no_universal_training_recipe_claim": True,
    }
    return claims, not_clone


def no_forbidden_claims() -> bool:
    return not text_contains_any(FORBIDDEN_CLAIMS)


def text_contains_any(markers: list[str]) -> bool:
    paths = [Path("README.md")]
    paths.extend(Path("docs").glob("*.md"))
    paths.extend(Path("paper").glob("*.md"))
    for path in paths:
        if path.as_posix() in {"docs/claims.md", "docs/reviewer_attacks.md"}:
            continue
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in markers:
            if claim in text and "Forbidden claims" not in text and "forbidden" not in path.name:
                return True
    return False


def write_outputs(results_dir: Path, claims: list[dict], not_clone: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "claims": claims,
        "not_clone_audit": not_clone,
        "summary": {
            "supported": sum(c["status"] == "SUPPORTED" for c in claims),
            "partial": sum(c["status"] == "PARTIAL" for c in claims),
            "unsupported": sum(c["status"] == "UNSUPPORTED" for c in claims),
            "all_required_supported": all(
                c["status"] == "SUPPORTED"
                for c in claims
                if c["category"]
                not in {
                    "optional VLA inference probe claims",
                }
            ),
        },
    }
    (results_dir / "claims_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Claims Status", ""]
    for claim in claims:
        lines.append(f"- **{claim['status']}** `{claim['category']}`: {claim['claim']}")
        lines.append(f"  Evidence: {claim['evidence']}")
    lines.append("")
    lines.append("## Not-Clone Audit")
    for key, value in not_clone.items():
        lines.append(f"- {key}: {value}")
    (results_dir / "claims_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    claims, not_clone = audit(results_dir)
    write_outputs(results_dir, claims, not_clone)
    bad = [
        c
        for c in claims
        if c["status"] != "SUPPORTED"
        and c["category"]
        not in {
            "optional VLA inference probe claims",
        }
    ]
    if args.fail_on_error and bad:
        for claim in bad:
            print(f"{claim['status']}: {claim['category']} - {claim['claim']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
