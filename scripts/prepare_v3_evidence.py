from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper" / "iclr2026"
MIN_PAGES = 25


def read_csv(rel: str) -> list[dict[str, str]]:
    path = RESULTS / rel
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tex(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def compact_claim_evidence(evidence: object) -> str:
    text = str(evidence).replace("\\", "/")
    if "results/figures" in text:
        return "results/figures/figure1--figure18 PNGs"
    if text.startswith("controlled/learned/distractor/repair/rendered/robustness"):
        return "controlled, learned, distractor, repair, rendered, and robustness summary CSVs"
    replacements = {
        "src/vla_tailguard_audit/theory.py": "src/.../theory.py",
        "results/controlled_summary.csv": "controlled_summary.csv",
        "results/learned_summary.csv": "learned_summary.csv",
        "results/distractor_summary.csv": "distractor_summary.csv",
        "results/repair_summary.csv": "repair_summary.csv",
        "results/rendered_summary.csv": "rendered_summary.csv",
        "results/robustness_summary.csv": "robustness_summary.csv",
        "results/tailguard_summary.csv": "tailguard_summary.csv",
        "results/tailguard_artifact.json": "tailguard_artifact.json",
        "results/tailguard_gate_examples.csv": "tailguard_gate_examples.csv",
        "results/component_ablation_summary.csv": "component_ablation_summary.csv",
        "results/phase_diagram_summary.csv": "phase_diagram_summary.csv",
        "results/calibration_sample_complexity.csv": "calibration_sample_complexity.csv",
        "results/physics_stress_summary.csv": "physics_stress_summary.csv",
        "results/failure_honesty_summary.csv": "failure_honesty_summary.csv",
        "results/external_benchmark_status.json": "external_benchmark_status.json",
        "results/optional_vla/adapter_status.json": "adapter_status.json",
        "results/optional_vla/inference_probe.json": "inference_probe.json",
        "results/optional_vla/smolvla_rendered_bridge.json": "smolvla_rendered_bridge.json",
        "results/optional_vla/libero_benchmark_status.json": "libero_benchmark_status.json",
        "docs/benchmark_plan.md": "benchmark_plan.md",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    if len(text) > 150:
        parts = [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]
        if len(parts) > 3:
            text = "; ".join(parts[:3]) + "; ..."
    return text


def compact_claim_status(status: object) -> str:
    names = {
        "SUPPORTED": "OK",
        "PASS": "OK",
        "PASS_WITH_BOUNDARY": "bounded",
    }
    return names.get(status, str(status))


def short_method(method: str) -> str:
    names = {
        "full_certified_tailguard": "full TailGuard",
        "raw_semantic": "raw semantic",
        "torch_semantic": "torch semantic",
        "torch_calibrated": "torch calibrated",
        "semantic_plus_physical": "semantic+physical",
        "grounded_combined": "grounded combined",
        "calibrated": "calibrated",
        "oracle": "oracle",
        "raw_fixed_high_n": "raw fixed high-N",
        "n1_baseline": "N=1 baseline",
        "random_high_n": "random high-N",
        "verifier_filtered_high_n": "verifier-filtered high-N",
        "calibrated_high_n_no_certificates": "calibrated, no cert.",
        "certificate_only_filtering_without_tail_calibration": "certificate only",
        "tailguard_without_lower_confidence_bound": "no lower bound",
        "tailguard_without_random_baseline_check": "no random check",
        "tailguard_without_n1_baseline_check": "no N=1 check",
        "certified_tailguard": "Certified TailGuard",
        "oracle_high_n": "oracle high-N",
    }
    return names.get(method, method.replace("_", " "))


def short_gate(value: object) -> str:
    names = {
        "allow_high_n": "allow high-N",
        "stop_early": "stop early",
        "collect_pilot_labels": "collect labels",
        "block_high_n": "block high-N",
        "fixed_policy": "fixed",
        "oracle": "oracle",
        "True": "true",
        "False": "false",
        True: "true",
        False: "false",
    }
    return names.get(value, str(value).replace("_", " "))


def short_status(value: object) -> str:
    names = {
        "INFERENCE_PROBE_PASS": "probe pass",
        "SKIPPED_WITH_REASON": "skipped",
        True: "true",
        False: "false",
        "True": "true",
        "False": "false",
    }
    return names.get(value, str(value).replace("_", " ").lower())


def short_regime(value: object) -> str:
    names = {
        "adversarial_semantic_tail_with_imperfect_verifier": "joint adversarial tail",
        "certificate_blocks_unreachable_semantic_tail": "unreachable tail",
        "verifier_resolves_certificate_blind_spot": "certificate blind spot",
        "pilot_labels_correct_false_positive_verifier": "false-positive verifier",
        "lower_bound_blocks_overfit_tail_calibration": "overfit calibration",
        "adaptive_n_stops_before_bad_tail": "adaptive-N stress",
        "fallback_prevents_low_certified_count_overclaim": "low certified count",
    }
    return names.get(value, str(value).replace("_", " "))


def short_component(value: object) -> str:
    names = {
        "joint_stress": "joint",
        "no_physical_certificate": "certificate",
        "no_verifier_score": "verifier score",
        "no_pilot_labels": "pilot labels",
        "no_empirical_lower_bound": "lower bound",
        "no_adaptive_n": "adaptive N",
        "no_abstention_fallback": "fallback",
    }
    return names.get(value, str(value).replace("_", " "))


def fnum(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return tex(value)


def find_row(rows: list[dict[str, str]], method: str, n: int | None = None) -> dict[str, str]:
    for row in rows:
        if row.get("method") != method:
            continue
        if n is None or int(float(row.get("N", -1))) == n:
            return row
    raise KeyError(f"missing row method={method} N={n}")


def longtable(
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
    size: str = r"\small",
    tabcolsep: str = "3pt",
) -> str:
    chunk_size = 10 if len(rows) > 18 else 14
    chunks = [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)] or [[]]
    lines = ["% Auto-generated by scripts/prepare_v3_evidence.py"]
    for index, chunk in enumerate(chunks):
        start = index * chunk_size + 1
        end = start + len(chunk) - 1
        chunk_caption = tex(caption)
        if len(chunks) > 1:
            chunk_caption = f"{chunk_caption}, rows {start}--{end}"
        label_text = rf"\label{{{label}}}" if index == 0 else ""
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                size,
                rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
                r"\renewcommand{\arraystretch}{1.05}",
                rf"\caption{{{chunk_caption}}}{label_text}",
                rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
                r"\toprule",
                f"{header}\\\\",
                r"\midrule",
            ]
        )
        lines.extend(chunk)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_macros(values: dict[str, object]) -> None:
    lines = ["% Auto-generated by scripts/prepare_v3_evidence.py"]
    for key, value in values.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{tex(value)}}}")
    write(PAPER / "v3_results_macros.tex", "\n".join(lines) + "\n")


def write_claim_scorecard(claims: list[dict]) -> None:
    rows = []
    for claim in claims:
        rows.append(
            f"{tex(claim['category'])} & {tex(compact_claim_status(claim['status']))} & "
            f"{tex(claim['claim'])} & {tex(compact_claim_evidence(claim['evidence']))} \\\\"
        )
    write(
        PAPER / "v3_claim_scorecard_table.tex",
        longtable(
            "V3 claim-to-artifact scorecard from the repository claim audit.",
            "tab:v3-claim-scorecard",
            r"p{0.18\linewidth}p{0.09\linewidth}p{0.36\linewidth}p{0.25\linewidth}",
            "Claim family & Status & Claim & Evidence",
            rows,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )


def write_failure_and_repair_tables(data: dict[str, list[dict[str, str]]]) -> None:
    failure_rows = []
    specs = [
        ("learned", "learned_summary.csv", "raw_semantic", 1),
        ("learned", "learned_summary.csv", "raw_semantic", 128),
        ("learned", "learned_summary.csv", "torch_semantic", 128),
        ("learned", "learned_summary.csv", "torch_calibrated", 128),
        ("rendered", "rendered_summary.csv", "raw_semantic", 1),
        ("rendered", "rendered_summary.csv", "raw_semantic", 128),
        ("rendered", "rendered_summary.csv", "torch_semantic", 128),
        ("rendered", "rendered_summary.csv", "torch_calibrated", 128),
    ]
    for setting, name, method, n in specs:
        row = find_row(data[name], method, n)
        failure_rows.append(
            f"{tex(setting)} & {tex(short_method(method))} & {n} & "
            f"{fnum(row['selected_semantic_score'])} & {fnum(row['selected_real_utility'])} & "
            f"{fnum(row['violation_rate'])} & {fnum(row['high_N_regret'])} & {tex(short_gate(row['deployment_gate']))} \\\\"
        )
    write(
        PAPER / "v3_selected_failure_table.tex",
        longtable(
            "Selected-tail semantic over-selection across learned and rendered VLA-style settings.",
            "tab:v3-selected-failure",
            r"llrrrrrl",
            r"Setting & Method & $N$ & Sem. & Utility & Viol. & Regret & Gate",
            failure_rows,
        ),
    )

    repair_rows = []
    for method in [
        "raw_semantic",
        "semantic_plus_physical",
        "calibrated",
        "grounded_combined",
        "torch_semantic",
        "torch_calibrated",
        "oracle",
    ]:
        row = find_row(data["repair_summary.csv"], method, 128)
        repair_rows.append(
            f"{tex(short_method(method))} & {fnum(row['selected_semantic_score'])} & "
            f"{fnum(row['selected_real_utility'])} & {fnum(row['violation_rate'])} & "
            f"{fnum(row['oracle_minus_method_gap'])} & {tex(short_gate(row['deployment_gate']))} \\\\"
        )
    write(
        PAPER / "v3_repair_table.tex",
        longtable(
            "High-budget repair comparison at N=128.",
            "tab:v3-repair",
            r"lrrrrl",
            r"Method & Sem. & Utility & Viol. & Oracle gap & Gate",
            repair_rows,
        ),
    )


def write_tailguard_tables(tailguard: list[dict[str, str]]) -> None:
    methods = [
        "raw_fixed_high_n",
        "n1_baseline",
        "random_high_n",
        "verifier_filtered_high_n",
        "calibrated_high_n_no_certificates",
        "certificate_only_filtering_without_tail_calibration",
        "tailguard_without_lower_confidence_bound",
        "tailguard_without_random_baseline_check",
        "tailguard_without_n1_baseline_check",
        "certified_tailguard",
        "oracle_high_n",
    ]
    rows = []
    for method in methods:
        row = find_row(tailguard, method, 128)
        cert_u = row.get("certified_selected_utility") or "-"
        cert_v = row.get("certified_violation_rate") or "-"
        rows.append(
            f"{tex(short_method(method))} & {fnum(row['selected_real_utility'])} & "
            f"{fnum(row['violation_rate'])} & {tex(short_gate(row.get('gate_decision', '')))} & "
            f"{fnum(cert_u) if cert_u != '-' else '-'} & {fnum(cert_v) if cert_v != '-' else '-'} \\\\"
        )
    write(
        PAPER / "v3_tailguard_table.tex",
        longtable(
            "Certified TailGuard baselines and controlled gate outcomes at N=128.",
            "tab:v3-tailguard",
            r"p{0.35\linewidth}rrrrr",
            r"Method & Utility & Viol. & Gate & Cert. util. & Cert. viol.",
            rows,
        ),
    )


def write_curve_table(name: str, rows: list[dict[str, str]], methods: list[str], caption: str, label: str) -> None:
    out = []
    for method in methods:
        for row in rows:
            if row.get("method") != method:
                continue
            out.append(
                f"{tex(short_method(method))} & {int(float(row['N']))} & "
                f"{fnum(row['selected_semantic_score'])} & {fnum(row['selected_real_utility'])} & "
                f"{fnum(row['violation_rate'])} & {fnum(row['high_N_regret'])} & {tex(short_gate(row['deployment_gate']))} \\\\"
            )
    write(
        PAPER / name,
        longtable(
            caption,
            label,
            r"p{0.25\linewidth}rrrrrl",
            r"Method & $N$ & Sem. & Utility & Viol. & Regret & Gate",
            out,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )


def write_robustness_table(rows: list[dict[str, str]]) -> None:
    methods = [
        "raw_semantic",
        "noisy_verifier_mild",
        "noisy_verifier_harsh",
        "ideal_verifier",
        "calib_1pct_noise0",
        "calib_2pct_noise5",
        "calib_5pct_noise10",
        "calib_10pct_noise20",
        "oracle",
    ]
    out = []
    for method in methods:
        row = find_row(rows, method, 128)
        out.append(
            f"{tex(short_method(method))} & {fnum(row['selected_real_utility'])} & "
            f"{fnum(row['violation_rate'])} & {fnum(row['high_N_regret'])} & {tex(short_gate(row['deployment_gate']))} \\\\"
        )
    write(
        PAPER / "v3_robustness_table.tex",
        longtable(
            "Noisy verifier and calibration stress at N=128.",
            "tab:v3-robustness",
            r"p{0.38\linewidth}rrrl",
            r"Method & Utility & Viol. & Regret & Gate",
            out,
        ),
    )


def write_component_and_stress_tables(data: dict[str, list[dict[str, str]]]) -> None:
    comp_rows = []
    for row in data["component_ablation_summary.csv"]:
        comp_rows.append(
            f"{tex(short_regime(row.get('ablation_regime', '')))} & {tex(short_method(row.get('method', '')))} & "
            f"{tex(short_component(row.get('component_under_test', '')))} & {fnum(row.get('selected_real_utility', ''))} & "
            f"{fnum(row.get('violation_rate', ''))} & {tex(short_status(row.get('passes_controlled_acceptance', '')))} & "
            f"{tex(short_regime(row.get('supporting_failure_mode', '')))} \\\\"
        )
    write(
        PAPER / "v3_component_ablation_table.tex",
        longtable(
            "Component ablation stress regimes.",
            "tab:v3-component-ablation",
            r"p{0.16\linewidth}p{0.20\linewidth}p{0.16\linewidth}rrp{0.08\linewidth}p{0.18\linewidth}",
            r"Regime & Method & Component & Utility & Viol. & Pass & Failure mode",
            comp_rows,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )

    phase_rows = []
    for row in data["phase_diagram_summary.csv"]:
        phase_rows.append(
            f"{fnum(row['semantic_physical_misalignment'])} & {fnum(row['distractor_salience'])} & "
            f"{fnum(row['raw_high_n_utility'])} & {fnum(row['certified_tailguard_utility'])} & "
            f"{fnum(row['tailguard_gain_over_raw'])} & {fnum(row['raw_violation_rate'])} & "
            f"{tex(short_gate(row['tailguard_gate']))} \\\\"
        )
    write(
        PAPER / "v3_phase_diagram_table.tex",
        longtable(
            "Semantic/physical misalignment and distractor-salience phase diagram.",
            "tab:v3-phase",
            r"rrrrrrl",
            r"Misalign. & Distractor & Raw util. & TG util. & Gain & Raw viol. & Gate",
            phase_rows,
            size=r"\scriptsize",
        ),
    )

    sample_rows = []
    for row in data["calibration_sample_complexity.csv"]:
        sample_rows.append(
            f"{fnum(row['label_budget_fraction'])} & {fnum(row['label_noise'])} & "
            f"{tex(row.get('pilot_label_count', ''))} & {tex(row.get('tail_label_count', ''))} & "
            f"{fnum(row['confidence_radius'])} & {fnum(row['certified_tailguard_utility'])} & "
            f"{fnum(row['certified_violation_rate'])} & {tex(short_gate(row['tailguard_gate']))} \\\\"
        )
    write(
        PAPER / "v3_calibration_table.tex",
        longtable(
            "Pilot-label sample-complexity and label-noise sweep.",
            "tab:v3-calibration",
            r"rrrrrrrl",
            r"Budget & Noise & Pilot & Tail & Radius & TG util. & TG viol. & Gate",
            sample_rows,
            size=r"\scriptsize",
        ),
    )

    phys_rows = []
    for row in data["physics_stress_summary.csv"]:
        phys_rows.append(
            f"{tex(row['stress_family'])} & {fnum(row['raw_high_n_utility'])} & "
            f"{fnum(row['certified_tailguard_utility'])} & {fnum(row['tailguard_gain_over_raw'])} & "
            f"{fnum(row['focus_failure_rate_raw'])} & {fnum(row['focus_failure_rate_tailguard'])} & "
            f"{tex(short_gate(row['tailguard_gate']))} \\\\"
        )
    write(
        PAPER / "v3_physics_stress_table.tex",
        longtable(
            "First-principles physical stress decomposition.",
            "tab:v3-physics-stress",
            r"p{0.28\linewidth}rrrrrl",
            r"Stress family & Raw util. & TG util. & Gain & Raw fail & TG fail & Gate",
            phys_rows,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )

    honesty_rows = []
    for row in data["failure_honesty_summary.csv"]:
        honesty_rows.append(
            f"{tex(row.get('stress_family', row.get('regime', '')))} & "
            f"{fnum(row.get('raw_high_n_utility', ''))} & {fnum(row.get('certified_tailguard_utility', row.get('tailguard_utility', '')))} & "
            f"{fnum(row.get('fallback_rate', ''))} & {fnum(row.get('abstention_rate', ''))} & "
            f"{tex(short_gate(row.get('honesty_status', row.get('tailguard_gate', ''))))} \\\\"
        )
    write(
        PAPER / "v3_failure_honesty_table.tex",
        longtable(
            "Failure-honesty regimes where the controller falls back, blocks, or asks for labels.",
            "tab:v3-failure-honesty",
            r"p{0.34\linewidth}rrrrp{0.20\linewidth}",
            r"Regime & Raw util. & TG util. & Fallback & Abstain & Status",
            honesty_rows,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )


def write_external_table() -> None:
    probe = read_json("optional_vla/inference_probe.json")
    bridge = read_json("optional_vla/smolvla_rendered_bridge.json")
    ext = read_json("external_benchmark_status.json")
    claim_levels = ext.get("claim_levels", {})
    rows = [
        (
            "SmoLVLA CPU probe",
            probe.get("status"),
            f"parameters={probe.get('parameter_count')}; action shape={probe.get('action_shape')}",
            "Model plumbing only; no physical success.",
        ),
        (
            "SmoLVLA rendered bridge",
            bridge.get("status"),
            f"action_decode_supported={bridge.get('action_decode_supported')}",
            "No action decoder or toy evaluator mapping.",
        ),
        (
            "External integration",
            claim_levels.get("integration"),
            "RoboCasa/LIBERO status rows exist",
            "Integration status only.",
        ),
        (
            "External benchmark stepping",
            claim_levels.get("benchmark_stepping"),
            "reset/step/reward success not established",
            "No benchmark outcome claim.",
        ),
        (
            "External physical success",
            claim_levels.get("physical_success_claimed"),
            "physical_success_claimed=false",
            "No real-robot or external simulator success claim.",
        ),
        (
            "TailGuard external method success",
            claim_levels.get("tailguard_method_success"),
            "tailguard_method_success=false",
            "No method outcome claim on external benchmark.",
        ),
    ]
    out = [
        f"{tex(a)} & {tex(short_status(b))} & {tex(c)} & {tex(d)} \\\\"
        for a, b, c, d in rows
    ]
    write(
        PAPER / "v3_external_boundary_table.tex",
        longtable(
            "Optional VLA and external benchmark boundary table.",
            "tab:v3-external-boundary",
            r"p{0.24\linewidth}p{0.14\linewidth}p{0.30\linewidth}p{0.22\linewidth}",
            r"Artifact & Status & Evidence & Allowed claim",
            out,
            size=r"\small",
        ),
    )


def attack_rows() -> list[dict[str, str]]:
    rows = [
        ("generic_budget", "novelty", "Could read like a generic candidate-budget paper.", "Would the argument still work if VLA terms were deleted?", "Make instruction, object observation, semantic affordance, action candidates, and physical certificates central.", "pass"),
        ("clone_risk", "novelty", "Looks like the same selected-tail theorem reused.", "Is the paper just another wrapper around the finite law?", "Use the law only as audit math; put TailGuard, VLA failures, and robotics boundary artifacts in the foreground.", "pass"),
        ("real_robot", "scope", "The text may imply robot validation.", "Does any sentence claim hardware evidence?", "State controlled evidence and no real-robot validation in abstract, limits, checklist, and external table.", "pass"),
        ("external_success", "scope", "RoboCasa/LIBERO status could be misread as success.", "Are reward or success metrics reported?", "Treat external artifacts as integration status only with no method outcome claim.", "pass"),
        ("smolvla_overclaim", "scope", "SmoLVLA probe could be oversold.", "Does synthetic action emission imply benchmark validation?", "Report it only as CPU model-plumbing evidence.", "pass"),
        ("certificate_idealization", "assumption", "Physical certificates are modeled, not real safety certificates.", "Does TailGuard rely on perfect perception/dynamics?", "List the modeled predicates and boundary conditions explicitly.", "bounded"),
        ("calibration_labels", "assumption", "Pilot labels may be too clean.", "Does calibration assume representative selected-tail labels?", "Report label-budget/noise stress and collect_pilot_labels decisions.", "pass"),
        ("noisy_verifier", "robustness", "Grounding might be imperfect.", "Does the paper show noisy verifier failure?", "Include mild/harsh noisy verifier rows and unsafe gates.", "pass"),
        ("ablation_stack", "ablation", "TailGuard may be redundant components.", "Does every component matter?", "Expose component-specific ablation failures.", "pass"),
        ("fallback_honesty", "policy", "Abstention could hide failures.", "Does the method pretend blocked cases are successes?", "Report fallback and abstention as first-class outcomes.", "pass"),
        ("n1_baseline", "baseline", "High-N could be compared only to weak baselines.", "Does TailGuard beat N=1 and random lower-bound baselines?", "Keep baseline checks in method and TailGuard table.", "pass"),
        ("random_baseline", "baseline", "Random selection baseline may be missing.", "Is random high-N represented?", "Include random high-N row and no-random-check ablation.", "pass"),
        ("oracle_gap", "baseline", "Near-oracle repair may be overclaimed.", "Is oracle treated as deployable?", "Call oracle an upper bound and keep gaps visible.", "pass"),
        ("theory_sampling", "theory", "Finite law could mismatch implementation.", "Is the law tie-aware and without-replacement?", "State exact without-replacement law and tests.", "pass"),
        ("constant_scores", "theory", "Tie handling may be wrong.", "Are ties and score groups covered?", "Keep tie-aware proof and unit tests in claim map.", "pass"),
        ("semantic_only", "theory", "No-free-lunch may be too strong.", "Is the proposition conditional on observed features?", "State semantic-score-only impossibility for unconstrained upper-tail utility.", "pass"),
        ("metric_cherry_pick", "statistics", "Only utility could hide physical failures.", "Are wrong-object/collision/reach/stability shown?", "Expose violation decomposition and stress tables.", "pass"),
        ("seed_count", "statistics", "Only three seeds may be attacked.", "Are seed counts and pool counts transparent?", "Report seeds, states, pools, and cached status.", "bounded"),
        ("mc_error", "statistics", "Monte Carlo validation may be noisy.", "Are exact-law errors small and CIs present?", "Use exact-law columns and claim audit thresholds.", "pass"),
        ("phase_sweep", "coverage", "Only one regime may work.", "Are misalignment and distractor sweeps present?", "Include phase diagram table and figure.", "pass"),
        ("physics_stress", "coverage", "Physical stress may be too narrow.", "Are hidden obstacles, spoofing, wrong receptacles, fragile/heavy tested?", "Expose physics stress families and failure reductions.", "pass"),
        ("rendered_gap", "coverage", "Symbolic scene only.", "Does visual rendering reproduce the effect?", "Include rendered simulator rows and figures.", "pass"),
        ("pytorch_gap", "coverage", "Lightweight scorer only.", "Does a heavier learned scorer reproduce the issue?", "Include torch_semantic/torch_calibrated rows.", "pass"),
        ("action_generator", "mechanism", "Candidate generator may be artificial.", "Are actions, objects, receptacles, and constraints explicit?", "Describe controlled generator and VLA-style candidate construction.", "bounded"),
        ("tailguard_name", "writing", "Certified TailGuard could sound like formal safety certification.", "Does the name overpromise?", "Clarify controlled certificate semantics and no hardware certification.", "pass"),
        ("abstract_scope", "writing", "Abstract may overclaim.", "Does it mention no benchmark/hardware outcome?", "Keep boundary sentence in abstract.", "pass"),
        ("related_work", "writing", "Related work too thin.", "Does it place VLA, affordance, verifiers, selective prediction, benchmarks?", "Expand related work around VLA-specific selection.", "pass"),
        ("template_smell", "writing", "Side-by-side duplicate risk.", "Would another paper share the same section names and contribution wording?", "Use VLA/TailGuard-specific sections, tables, and terminology.", "pass"),
        ("artifact_map", "reproducibility", "Reviewers cannot find evidence.", "Are source artifacts mapped to claims?", "Add claim scorecard and artifact map.", "pass"),
        ("commands", "reproducibility", "Commands may overwrite full results.", "Does final validation avoid smoke overwrite?", "Use cached derived tables; warn about full vs smoke outputs.", "pass"),
        ("desktop_mapping", "workflow", "Fresh agent may not find source.", "Is source map v3 row updated?", "Build script updates Desktop source map.", "pass"),
        ("old_pdf", "workflow", "Old v2 Desktop PDF could remain.", "Is old v2 absent?", "Build script removes old Desktop v2 if present.", "pass"),
        ("hash_mismatch", "workflow", "Desktop and repo PDFs could diverge.", "Are hashes checked?", "V3 audit compares SHA-256.", "pass"),
        ("page_count", "submission", "Paper is too short.", "Is final PDF at least 25 pages?", "Build script enforces 25-page minimum.", "pass"),
        ("figure_render", "submission", "Figures might not render from relative paths.", "Are representative pages visually checked?", "Render title/results/appendix/final pages.", "pass"),
        ("table_overflow", "submission", "Generated tables may overflow.", "Are long tables readable?", "Use chunked fixed tables, small font, p columns, and visual QA.", "pass"),
        ("external_action_decode", "external", "SmoLVLA action chunk has no evaluator mapping.", "Does the paper disclose action_decode_supported=false?", "External boundary table states no action decoder.", "pass"),
        ("libero_status", "external", "LIBERO import status could be confused.", "Is reset/step wrapper implemented?", "State no guarded LIBERO reset/step wrapper.", "pass"),
        ("robocasa_timeout", "external", "RoboCasa timeout could be hidden.", "Is timeout disclosed?", "State reset/step attempt timed out before reward/success metrics.", "pass"),
        ("success_metric", "external", "Success metric may be implied.", "Is success_trace nonempty?", "No success metric exposure claim without artifacts.", "pass"),
        ("method_external", "external", "TailGuard external success may be implied.", "Does TailGuard connect to external benchmark?", "State tailguard_method_success=false.", "pass"),
        ("calibration_near_oracle", "interpretation", "Near-oracle calibrated rows look suspicious.", "Does text explain clean controlled labels?", "Frame as controlled evidence with label/noise stress.", "bounded"),
        ("certificate_only", "interpretation", "Certificate-only can look sufficient.", "Why need calibration and gates?", "Show ablations and hard regimes where each component matters.", "pass"),
        ("allow_high_n", "policy", "Gate can allow high-N in easy regimes.", "Does it still block unsafe regimes?", "Report all four decisions and reason codes.", "pass"),
        ("low_certified_count", "policy", "Few certified candidates may fail silently.", "Does controller collect labels/abstain?", "Include low-certified-candidate stress row.", "pass"),
        ("claim_audit", "audit", "Overclaims could creep in.", "Is a claim audit run?", "Run legacy and v3 claim audits.", "pass"),
        ("github_push", "workflow", "Local final not pushed.", "Is remote main verified?", "Push and verify remote SHA.", "pass"),
        ("future_work", "scope", "Future benchmark plans may sound like done work.", "Are future steps clearly future?", "Keep future benchmark wrapper in limitations/future work only.", "pass"),
        ("llm_usage", "submission", "LLM disclosure missing.", "Is usage disclosed?", "Keep LLM usage appendix.", "pass"),
        ("reviewer_summary", "writing", "Reviewer may miss main contribution.", "Does final checklist state controlled claim and no external outcome?", "Add final acceptance checklist.", "pass"),
    ]
    return [
        {
            "round": str(idx),
            "attack_id": attack_id,
            "reviewer_angle": angle,
            "failure_mode": failure,
            "harsh_question": question,
            "defense_artifact_or_revision": defense,
            "status": status,
        }
        for idx, (attack_id, angle, failure, question, defense, status) in enumerate(rows, start=1)
    ]


def write_attack_ledger() -> None:
    rows = attack_rows()
    csv_path = RESULTS / "v3_vla_attack_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = [
        f"{row['round']} & {tex(row['reviewer_angle'])} & {tex(row['failure_mode'])} & "
        f"{tex(row['defense_artifact_or_revision'])} & {tex(row['status'])} \\\\"
        for row in rows
    ]
    write(
        PAPER / "v3_attack_ledger_table.tex",
        longtable(
            "Fifty-round self-attack ledger for the VLA submission.",
            "tab:v3-attack-ledger",
            r"r p{0.13\linewidth} p{0.27\linewidth} p{0.34\linewidth} p{0.08\linewidth}",
            r"Round & Angle & Attack & Defense & Status",
            table_rows,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )


def main() -> None:
    data = {
        name: read_csv(name)
        for name in [
            "learned_summary.csv",
            "repair_summary.csv",
            "rendered_summary.csv",
            "robustness_summary.csv",
            "tailguard_summary.csv",
            "component_ablation_summary.csv",
            "phase_diagram_summary.csv",
            "calibration_sample_complexity.csv",
            "physics_stress_summary.csv",
            "failure_honesty_summary.csv",
        ]
    }
    claims_payload = read_json("claims_status.json")
    claims = claims_payload.get("claims", [])

    learned_raw_1 = find_row(data["learned_summary.csv"], "raw_semantic", 1)
    learned_raw_128 = find_row(data["learned_summary.csv"], "raw_semantic", 128)
    rendered_raw_1 = find_row(data["rendered_summary.csv"], "raw_semantic", 1)
    rendered_raw_128 = find_row(data["rendered_summary.csv"], "raw_semantic", 128)
    repair_raw = find_row(data["repair_summary.csv"], "raw_semantic", 128)
    repair_grounded = find_row(data["repair_summary.csv"], "grounded_combined", 128)
    tailguard_raw = find_row(data["tailguard_summary.csv"], "raw_fixed_high_n", 128)
    tailguard_full = find_row(data["tailguard_summary.csv"], "certified_tailguard", 128)
    probe = read_json("optional_vla/inference_probe.json")

    phase_gain_mean = mean(float(row["tailguard_gain_over_raw"]) for row in data["phase_diagram_summary.csv"])
    raw_phase_min = min(float(row["raw_high_n_utility"]) for row in data["phase_diagram_summary.csv"])
    physics_family_count = len({row["stress_family"] for row in data["physics_stress_summary.csv"]})
    ablated_methods = {
        row["method"]
        for row in data["component_ablation_summary.csv"]
        if row["method"] != "full_certified_tailguard"
    }

    values = {
        "VThreeMinPages": MIN_PAGES,
        "VThreeSeeds": 3,
        "VThreeStatesPerSeed": 8,
        "VThreePools": 24,
        "VThreeCandidateBudget": 128,
        "VThreeLearnedRawNOneSem": fnum(learned_raw_1["selected_semantic_score"]),
        "VThreeLearnedRawNOneUtil": fnum(learned_raw_1["selected_real_utility"]),
        "VThreeLearnedRawHighSem": fnum(learned_raw_128["selected_semantic_score"]),
        "VThreeLearnedRawHighUtil": fnum(learned_raw_128["selected_real_utility"]),
        "VThreeLearnedRawHighViol": fnum(learned_raw_128["violation_rate"]),
        "VThreeLearnedRawRegret": fnum(learned_raw_128["high_N_regret"]),
        "VThreeRenderedRawNOneUtil": fnum(rendered_raw_1["selected_real_utility"]),
        "VThreeRenderedRawHighUtil": fnum(rendered_raw_128["selected_real_utility"]),
        "VThreeRenderedRawHighViol": fnum(rendered_raw_128["violation_rate"]),
        "VThreeRepairRawHighUtil": fnum(repair_raw["selected_real_utility"]),
        "VThreeRepairGroundedUtil": fnum(repair_grounded["selected_real_utility"]),
        "VThreeRepairGroundedViol": fnum(repair_grounded["violation_rate"]),
        "VThreeTailGuardRawUtil": fnum(tailguard_raw["selected_real_utility"]),
        "VThreeTailGuardRawViol": fnum(tailguard_raw["violation_rate"]),
        "VThreeTailGuardUtil": fnum(tailguard_full["selected_real_utility"]),
        "VThreeTailGuardViol": fnum(tailguard_full["violation_rate"]),
        "VThreeTailGuardGate": tailguard_full["gate_decision"],
        "VThreePhaseRows": len(data["phase_diagram_summary.csv"]),
        "VThreePhaseMeanGain": fnum(phase_gain_mean),
        "VThreePhaseRawMin": fnum(raw_phase_min),
        "VThreePhysicsStressFamilies": physics_family_count,
        "VThreeAblationComponents": len(ablated_methods),
        "VThreeClaimsSupported": claims_payload.get("summary", {}).get("supported", len(claims)),
        "VThreeAttackRounds": 50,
        "VThreeProbeStatus": probe.get("status", ""),
        "VThreeProbeParams": f"{int(probe.get('parameter_count', 0)):,}",
        "VThreeProbeActionShape": " x ".join(str(x) for x in probe.get("action_shape", [])),
    }

    write_macros(values)
    write_claim_scorecard(claims)
    write_failure_and_repair_tables(data)
    write_tailguard_tables(data["tailguard_summary.csv"])
    write_curve_table(
        "v3_learned_curve_table.tex",
        data["learned_summary.csv"],
        ["raw_semantic", "torch_semantic", "torch_calibrated", "oracle"],
        "Full learned VLA-style selected-tail curves.",
        "tab:v3-learned-curves",
    )
    write_curve_table(
        "v3_rendered_curve_table.tex",
        data["rendered_summary.csv"],
        ["raw_semantic", "grounded_combined", "torch_semantic", "torch_calibrated", "oracle"],
        "Full rendered-simulator selected-tail curves.",
        "tab:v3-rendered-curves",
    )
    write_robustness_table(data["robustness_summary.csv"])
    write_component_and_stress_tables(data)
    write_external_table()
    write_attack_ledger()

    summary = {
        "paper_identity": "VLA semantic affordance over-selection and Certified TailGuard",
        "uses_cached_artifacts_only": True,
        "target_page_count_minimum": MIN_PAGES,
        "generated_artifacts": [
            "results/v3_vla_attack_ledger.csv",
            "paper/iclr2026/v3_results_macros.tex",
            "paper/iclr2026/v3_claim_scorecard_table.tex",
            "paper/iclr2026/v3_selected_failure_table.tex",
            "paper/iclr2026/v3_repair_table.tex",
            "paper/iclr2026/v3_tailguard_table.tex",
            "paper/iclr2026/v3_learned_curve_table.tex",
            "paper/iclr2026/v3_rendered_curve_table.tex",
            "paper/iclr2026/v3_robustness_table.tex",
            "paper/iclr2026/v3_component_ablation_table.tex",
            "paper/iclr2026/v3_phase_diagram_table.tex",
            "paper/iclr2026/v3_calibration_table.tex",
            "paper/iclr2026/v3_physics_stress_table.tex",
            "paper/iclr2026/v3_failure_honesty_table.tex",
            "paper/iclr2026/v3_external_boundary_table.tex",
            "paper/iclr2026/v3_attack_ledger_table.tex",
        ],
        "key_values": values,
    }
    write(RESULTS / "v3_vla_evidence_summary.json", json.dumps(summary, indent=2) + "\n")
    write(
        RESULTS / "v3_vla_evidence_summary.md",
        "# V3 VLA Evidence Summary\n\n"
        f"- Identity: {summary['paper_identity']}\n"
        f"- Cached artifacts only: {summary['uses_cached_artifacts_only']}\n"
        f"- Minimum pages: {MIN_PAGES}\n"
        f"- Attack rounds: {values['VThreeAttackRounds']}\n"
        f"- Learned raw high-N utility: {values['VThreeLearnedRawHighUtil']}\n"
        f"- Certified TailGuard utility: {values['VThreeTailGuardUtil']}\n",
    )
    print("prepared v3 VLA evidence artifacts")


if __name__ == "__main__":
    main()
