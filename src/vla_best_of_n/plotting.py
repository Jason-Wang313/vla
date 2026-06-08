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

    return paths
