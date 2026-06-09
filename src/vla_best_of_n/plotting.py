"""Paper-critical figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .io import ensure_dir


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def create_figures(results_dir: str | Path) -> list[str]:
    root = Path(results_dir)
    fig_dir = ensure_dir(root / "figures")
    paths: list[str] = []

    learned = _read(root / "learned_summary.csv")
    repair = _read(root / "repair_summary.csv")
    distractor = _read(root / "distractor_summary.csv")
    rendered_path = root / "rendered_summary.csv"
    robustness_path = root / "robustness_summary.csv"
    tailguard_path = root / "tailguard_summary.csv"
    phase_path = root / "phase_diagram_summary.csv"
    sample_complexity_path = root / "calibration_sample_complexity.csv"
    physics_stress_path = root / "physics_stress_summary.csv"
    component_ablation_path = root / "component_ablation_summary.csv"
    failure_honesty_path = root / "failure_honesty_summary.csv"
    smolvla_bridge_path = root / "optional_vla" / "smolvla_rendered_bridge.json"

    raw = learned[learned["method"] == "raw_semantic"].sort_values("N")
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(raw["N"], raw["selected_semantic_score"], marker="o", color="#2563eb", label="selected semantic score")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("N")
    ax1.set_ylabel("selected semantic score", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(raw["N"], raw["selected_real_utility"], marker="s", color="#dc2626", label="selected real utility")
    ax2.set_ylabel("selected real utility", color="#dc2626")
    ax1.set_title("Figure 1: Semantic affordance over-selection")
    fig.tight_layout()
    path = fig_dir / "figure1_semantic_overselection.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in ["raw_semantic", "semantic_plus_physical", "calibrated", "grounded_combined", "random", "oracle"]:
        curve = repair[repair["method"] == method].sort_values("N")
        if not curve.empty:
            ax.plot(curve["N"], curve["selected_real_utility"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("selected real utility")
    ax.set_title("Figure 2: Repair comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "figure2_repair_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(raw["N"], raw["tail_rank_correlation"], marker="o", color="#0f766e", label="tail rank correlation")
    ax1.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("N")
    ax1.set_ylabel("tail rank correlation", color="#0f766e")
    ax2 = ax1.twinx()
    ax2.plot(raw["N"], raw["semantic_real_tail_gap"], marker="s", color="#9333ea", label="tail gap")
    ax2.set_ylabel("semantic-real tail gap", color="#9333ea")
    ax1.set_title("Figure 3: Selected-tail diagnostics")
    fig.tight_layout()
    path = fig_dir / "figure3_tail_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    raw_d = distractor[distractor["method"] == "raw_semantic"].sort_values("N")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for column, label in [
        ("wrong_object_rate", "wrong object"),
        ("wrong_target_rate", "wrong target"),
        ("unreachable_rate", "unreachable"),
        ("collision_rate", "collision"),
        ("fragile_rate", "fragile/stability"),
    ]:
        ax.plot(raw_d["N"], raw_d[column], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("selected failure rate")
    ax.set_title("Figure 4: Failure type decomposition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "figure4_failure_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    combined = pd.concat([learned, repair, distractor], ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(combined["exact_selected_utility"], combined["mc_selected_utility_mean"], s=18, alpha=0.65)
    lo = min(combined["exact_selected_utility"].min(), combined["mc_selected_utility_mean"].min())
    hi = max(combined["exact_selected_utility"].max(), combined["mc_selected_utility_mean"].max())
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.0)
    ax.set_xlabel("exact finite-law selected utility")
    ax.set_ylabel("Monte Carlo selected utility")
    ax.set_title("Figure 5: Exact-law validation")
    fig.tight_layout()
    path = fig_dir / "figure5_exact_law_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    if rendered_path.exists():
        rendered = _read(rendered_path)
        raw_rendered = rendered[rendered["method"] == "raw_semantic"].sort_values("N")
        fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
        ax1.plot(raw_rendered["N"], raw_rendered["selected_semantic_score"], marker="o", color="#1d4ed8")
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("N")
        ax1.set_ylabel("selected semantic score", color="#1d4ed8")
        ax2 = ax1.twinx()
        ax2.plot(raw_rendered["N"], raw_rendered["selected_real_utility"], marker="s", color="#b91c1c")
        ax2.set_ylabel("simulator-selected real utility", color="#b91c1c")
        ax1.set_title("Figure 6: Rendered visual simulator over-selection")
        fig.tight_layout()
        path = fig_dir / "figure6_rendered_visual_simulator.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    if robustness_path.exists():
        robustness = _read(robustness_path)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for method in [
            "raw_semantic",
            "noisy_verifier_mild",
            "noisy_verifier_harsh",
            "ideal_verifier",
            "calib_1pct_noise0",
            "calib_5pct_noise10",
            "calib_10pct_noise20",
            "oracle",
        ]:
            curve = robustness[robustness["method"] == method].sort_values("N")
            if not curve.empty:
                ax.plot(curve["N"], curve["selected_real_utility"], marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_ylabel("simulator-selected real utility")
        ax.set_title("Figure 8: Noisy verifier and calibration robustness")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = fig_dir / "figure8_robustness_repairs.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

        high_n = robustness["N"].max()
        bars = robustness[
            (robustness["N"] == high_n)
            & robustness["method"].isin(["calib_1pct_noise0", "calib_2pct_noise5", "calib_5pct_noise10", "calib_10pct_noise20"])
        ].sort_values("method")
        if not bars.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.bar(bars["method"], bars["selected_real_utility"], color=["#60a5fa", "#34d399", "#f59e0b", "#f87171"])
            ax.set_ylabel("selected real utility at max N")
            ax.set_title("Figure 9: Calibration budget/noise stress test")
            ax.tick_params(axis="x", labelrotation=25)
            fig.tight_layout()
            path = fig_dir / "figure9_calibration_budget_noise.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if "torch_semantic" in set(learned["method"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in ["raw_semantic", "torch_semantic", "calibrated", "torch_calibrated", "oracle"]:
            curve = learned[learned["method"] == method].sort_values("N")
            if not curve.empty:
                ax.plot(curve["N"], curve["selected_real_utility"], marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_ylabel("selected real utility")
        ax.set_title("Figure 10: Learned scorer comparison")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = fig_dir / "figure10_learned_scorer_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    if tailguard_path.exists():
        tailguard = _read(tailguard_path)
        order = [
            "raw_fixed_high_n",
            "n1_baseline",
            "random_high_n",
            "verifier_filtered_high_n",
            "calibrated_high_n_no_certificates",
            "certificate_only_filtering_without_tail_calibration",
            "tailguard_without_lower_confidence_bound",
            "certified_tailguard_bon",
            "tailguard_bon",
            "oracle_high_n",
        ]
        bars = tailguard.set_index("method").reindex([m for m in order if m in set(tailguard["method"])])
        if not bars.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.bar(bars.index, bars["selected_real_utility"], color="#2563eb")
            if "violation_rate" in bars:
                ax.plot(bars.index, bars["violation_rate"], color="#dc2626", marker="o", label="violation rate")
                ax.legend(fontsize=8)
            ax.set_ylabel("selected real utility")
            ax.set_title("Figure 11: Certified TailGuard vs BoN baselines")
            ax.tick_params(axis="x", labelrotation=25)
            fig.tight_layout()
            path = fig_dir / "figure11_tailguard_adaptive_n.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if phase_path.exists():
        phase = _read(phase_path)
        if not phase.empty:
            pivot = phase.pivot_table(
                index="semantic_physical_misalignment",
                columns="distractor_salience",
                values="tailguard_gain_over_raw",
                aggfunc="mean",
            ).sort_index(ascending=True)
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{float(x):.2f}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{float(x):.2f}" for x in pivot.index])
            ax.set_xlabel("distractor salience")
            ax.set_ylabel("semantic/physical misalignment")
            ax.set_title("Figure 12: Misalignment phase diagram")
            fig.colorbar(im, ax=ax, label="TailGuard gain over raw")
            fig.tight_layout()
            path = fig_dir / "figure12_phase_diagram.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            scatter = ax.scatter(
                phase["verifier_false_positive_rate"],
                phase["verifier_false_negative_rate"],
                c=phase["tailguard_gain_over_raw"],
                s=70,
                cmap="RdYlGn",
                edgecolor="#111827",
                linewidth=0.3,
            )
            ax.set_xlabel("verifier false-positive rate")
            ax.set_ylabel("verifier false-negative rate")
            ax.set_title("Figure 14: Imperfect verifier failure map")
            fig.colorbar(scatter, ax=ax, label="TailGuard gain over raw")
            fig.tight_layout()
            path = fig_dir / "figure14_imperfect_verifier_map.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if sample_complexity_path.exists():
        sample = _read(sample_complexity_path)
        if not sample.empty:
            fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
            for noise, group in sample.groupby("label_noise"):
                curve = group.sort_values("label_budget_fraction")
                ax1.plot(
                    curve["label_budget_fraction"],
                    curve["tailguard_utility"],
                    marker="o",
                    label=f"noise={float(noise):.2f}",
                )
            ax1.set_xscale("log")
            ax1.set_xlabel("pilot label budget fraction")
            ax1.set_ylabel("TailGuard selected utility")
            ax2 = ax1.twinx()
            radius = sample.groupby("label_budget_fraction", as_index=False)["confidence_radius"].mean()
            ax2.plot(radius["label_budget_fraction"], radius["confidence_radius"], color="#111827", marker="s", label="LCB radius")
            ax2.set_ylabel("mean confidence radius")
            ax1.set_title("Figure 13: Calibration sample complexity")
            ax1.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            path = fig_dir / "figure13_calibration_sample_complexity.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if physics_stress_path.exists():
        physics = _read(physics_stress_path)
        if not physics.empty:
            fig, ax = plt.subplots(figsize=(8.6, 4.8))
            x = range(len(physics))
            width = 0.38
            ax.bar([i - width / 2 for i in x], physics["focus_failure_rate_raw"], width=width, label="raw high-N")
            ax.bar([i + width / 2 for i in x], physics["focus_failure_rate_tailguard"], width=width, label="TailGuard")
            ax.set_xticks(list(x))
            ax.set_xticklabels(physics["stress_family"], rotation=30, ha="right")
            ax.set_ylabel("selected focus failure rate")
            ax.set_title("Figure 15: First-principles physics failure decomposition")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = fig_dir / "figure15_physics_failure_decomposition.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if component_ablation_path.exists():
        ablation = _read(component_ablation_path)
        if not ablation.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.6))
            order = [
                "full_certified_tailguard",
                "no_physical_certificate",
                "no_verifier_score",
                "no_pilot_labels",
                "no_empirical_lower_bound",
                "no_adaptive_n",
                "no_abstention_fallback",
            ]
            worst_by_method = (
                ablation.groupby("method", as_index=True)
                .agg(selected_real_utility=("selected_real_utility", "min"), violation_rate=("violation_rate", "max"))
            )
            bars = worst_by_method.reindex([m for m in order if m in set(ablation["method"])])
            x = range(len(bars))
            ax.bar(x, bars["selected_real_utility"], color="#0f766e", label="utility")
            ax.plot(list(x), bars["violation_rate"], color="#dc2626", marker="o", label="violation")
            ax.axhline(0.98, color="#166534", linewidth=0.8, linestyle="--")
            ax.axhline(0.01, color="#991b1b", linewidth=0.8, linestyle=":")
            ax.set_xticks(list(x))
            ax.set_xticklabels(bars.index, rotation=30, ha="right")
            ax.set_ylabel("rate / utility")
            ax.set_title("Figure 17: Certified TailGuard component ablation")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = fig_dir / "figure17_component_ablation.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if failure_honesty_path.exists():
        honesty = _read(failure_honesty_path)
        if not honesty.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.4))
            x = range(len(honesty))
            width = 0.36
            ax.bar([i - width / 2 for i in x], honesty["abstention_rate"], width=width, label="abstention")
            ax.bar([i + width / 2 for i in x], honesty["fallback_rate"], width=width, label="fallback")
            ax.set_xticks(list(x))
            ax.set_xticklabels(honesty["stress_family"], rotation=30, ha="right")
            ax.set_ylabel("rate")
            ax.set_title("Figure 18: Failure honesty by stress regime")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = fig_dir / "figure18_failure_honesty.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))

    if smolvla_bridge_path.exists():
        import json

        bridge = json.loads(smolvla_bridge_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.axis("off")
        lines = [
            "Figure 16: Optional SmoLVLA rendered-input bridge",
            f"status: {bridge.get('status', 'missing')}",
            f"benchmark_validation: {bridge.get('benchmark_validation', False)}",
            f"decoded_physical_success: {bridge.get('decoded_physical_success', False)}",
            f"action_count: {bridge.get('action_count', 0)}",
        ]
        ax.text(0.02, 0.86, "\n".join(lines), va="top", ha="left", fontsize=10)
        fig.tight_layout()
        path = fig_dir / "figure16_smolvla_rendered_bridge_status.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    return paths
