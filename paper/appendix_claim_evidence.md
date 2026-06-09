# Appendix: Claim-Evidence And Boundary Table

| Statement | Status | Artifact | Acceptance condition |
|---|---|---|---|
| Exact finite tie-aware Best-of-N law applies to fixed finite candidate pools. | Supported | `src/vla_best_of_n/theory.py`, `tests/test_theory.py` | The theorem tests pass and exact-law prediction error remains small in summary CSVs. |
| Semantic affordance over-selection appears in controlled VLA-style scenes. | Supported | `results/controlled_summary.csv`, `results/learned_summary.csv` | Selected semantic score rises with `N` while utility drops or violations rise. |
| Certified TailGuard-BoN repairs controlled high-N failure when certificates cover the modeled failure modes. | Supported | `results/tailguard_summary.csv`, `results/tailguard_artifact.json` | Utility is at least `0.98` and violation is at most `0.01` in the controlled TailGuard row. |
| Each named TailGuard component has ablation coverage. | Supported | `results/component_ablation_summary.csv` | Ablations cover certificates, verifier score, pilot labels, lower bounds, adaptive `N`, and fallback/abstention, with each named removal failing the controlled threshold in at least one stress regime. |
| Calibration sample complexity and confidence-radius behavior are measured. | Supported | `results/calibration_sample_complexity.csv` | Multiple budget and noise levels are present, and scarce selected-tail evidence can trigger label collection. |
| Failure-honesty regimes are recorded. | Supported | `results/failure_honesty_summary.csv` | At least one stress family falls back or abstains. |
| Optional SmoLVLA plumbing can emit synthetic actions. | Supported | `results/optional_vla/inference_probe.json` | Status is `INFERENCE_PROBE_PASS`; benchmark validation remains false. |
| RoboCasa/LIBERO external integration status is recorded. | Supported | `results/external_benchmark_status.json` | Package versions, assets, target registration, reset/step status, reward trace, success trace, policy kind, and skip reason fields are present. |
| External simulator reward/success evidence is not claimed. | Boundary recorded | `results/external_benchmark_status.json` | Upgrade only after reset ok, step ok, nonempty reward trace, and exposed success metric. |
| TailGuard external-simulator method success is not claimed. | Boundary recorded | `results/external_benchmark_status.json` | Upgrade only after an action-schema mapping, TailGuard-connected policy rows, and saved reward/success metrics. |
| Real-robot validation is not claimed. | Boundary recorded | none | Upgrade only after hardware runs and seed/episode-level outcome artifacts. |
