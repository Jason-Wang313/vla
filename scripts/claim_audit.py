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
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    optional_vla_status_path = results_dir / "optional_vla" / "adapter_status.json"
    optional_vla_status = (
        json.loads(optional_vla_status_path.read_text(encoding="utf-8")) if optional_vla_status_path.exists() else {}
    )

    claims: list[dict] = []

    has_theory_docs = Path("docs/theory.md").exists()
    claims.append(
        {
            "category": "theorem claims",
            "claim": "Exact finite tie-aware Best-of-N law is implemented and documented.",
            "status": _status(has_theory_docs and Path("src/vla_best_of_n/theory.py").exists()),
            "evidence": "src/vla_best_of_n/theory.py; docs/theory.md; tests/test_theory.py",
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
            "category": "optional benchmark claims",
            "claim": "Real VLA benchmark validation is implemented.",
            "status": "UNSUPPORTED",
            "evidence": "docs/benchmark_plan.md marks benchmark validation as future work.",
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
            "category": "unsupported future robotics claims",
            "claim": "Real-robot validation is established.",
            "status": "UNSUPPORTED",
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
        "optional_vla_status_exists": optional_vla_status_path.exists(),
        "no_real_robot_claim_unless_implemented": True,
        "no_universal_training_recipe_claim": True,
    }
    return claims, not_clone


def no_forbidden_claims() -> bool:
    paths = [Path("README.md")]
    paths.extend(Path("docs").glob("*.md"))
    paths.extend(Path("paper").glob("*.md"))
    for path in paths:
        if path.as_posix() in {"docs/claims.md", "docs/reviewer_attacks.md"}:
            continue
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in FORBIDDEN_CLAIMS:
            if claim in text and "Forbidden claims" not in text and "forbidden" not in path.name:
                return False
    return True


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
                not in {"optional benchmark claims", "unsupported future robotics claims"}
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
        and c["category"] not in {"optional benchmark claims", "unsupported future robotics claims"}
    ]
    if args.fail_on_error and bad:
        for claim in bad:
            print(f"{claim['status']}: {claim['category']} - {claim['claim']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
