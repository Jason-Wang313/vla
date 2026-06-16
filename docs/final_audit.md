# Final Audit

## V4 Submission Addendum

Current repository path: `C:\Users\wangz\vla`.

Final v4 artifact:

- repo PDF: `paper/final/vla-v4.pdf`
- Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\vla-v4.pdf`
- pages: `29`
- SHA-256: `786742db18ea2c2b4c97d1ac4cd94241a38690a82fc5fd090b36339a3dc8556f`
- source-map row:

```text
| `vla-v4.pdf` | `C:\Users\wangz\vla` | `Jason-Wang313/vla` |
```

V4 hardening additions:

- cached v4 evidence generator: `experiments/v4_cached_evidence.py`
- v4 finalizer: `scripts/build_v4_paper.py`
- v4 final-state audit: `scripts/run_v4_claim_audit.py`
- scorecard, protocol-freeze gates, rubric map, and 60-round reviewer attack ledger under `results/`
- manuscript tables and macros under `paper/iclr2026/v4_*.tex`

Final checks run in the v4 pass:

| Command | Status | Notes |
|---|---:|---|
| `python scripts\build_v4_paper.py` | PASS | Built repo and Desktop `vla-v4.pdf`; removed old Desktop VLA PDFs; updated source map. |
| `python scripts\run_v4_claim_audit.py` | PASS | Verified 29 pages, SHA, source map, 60 attack rounds, boundaries, and v4 PDF markers. |
| `bash scripts/run_claim_audit.sh` | PASS | Legacy claim audit passed in the final v4 state. |
| `python -m compileall src experiments scripts tests -q` | PASS | Compile check passed. |
| `python -m pytest -q` | PASS | 39 tests passed. |
| Visual PDF render QA | PASS | Checked title, v4 scorecard/protocol page, attack-ledger page, and final checklist page. |

Final scope remains controlled and explicit: the paper is a VLA selected-tail audit with optional SmoLVLA model-plumbing and RoboCasa/LIBERO integration-status artifacts. It does not claim hardware results, external simulator outcomes, or universal deployment.

Repository path at latest audit: `C:\Users\wangz\score-tail vla`.

Git checkpoint: `7a8cc45` (`Initial paper-quality VLA score-tail checkpoint`). The repository was initialized during the resumability pass.

## V2 TailGuard Addendum

Latest v2 pass upgrades the method to **Certified TailGuard** while retaining `TailGuard` as the implementation short name. The headline method is implemented in `src/vla_tailguard_audit/tailguard.py`; stress sweeps are implemented in `src/vla_tailguard_audit/stress.py`.

Certified additions:

- hard certificate predicates for reach envelope, swept collision, receptacle compatibility, stability margin, fragile/heavy handling, tool/object match, blocked-path constraints, and modeled hidden obstacles
- public decision fields for certificate pass/failure types, certified candidate count, fallback, abstention, certified selected utility, and certified violation
- expanded baselines and ablations in `results/tailguard_summary.csv` and `results/component_ablation_summary.csv`
- adversarial stress families and certificate failure decomposition in `results/physics_stress_summary.csv`
- failure-honesty regimes in `results/failure_honesty_summary.csv`
- optional SmoLVLA rendered bridge fields `action_decode_supported: false` and `physical_success_claimed: false`

Latest commands run after the Certified TailGuard implementation:

| Command | Status | Runtime | Notes |
|---|---:|---:|---|
| `pytest -q` | PASS | 47.85 s test time | 36 tests passed |
| `bash scripts/run_smoke.sh` | PASS | 418.8 s wall time | 36 tests passed inside script; wrote `results/smoke/`; emitted a non-fatal WSL/systemd warning after successful completion |
| `bash scripts/run_all.sh` | PASS | 1009.1 s wall time | 36 tests passed inside script; wrote `results/` |
| `bash scripts/run_claim_audit.sh` | PASS | 24.5 s wall time | 23 supported, 0 partial, 0 unsupported; benchmark/robot validation remain explicitly not claimed |

Optional SmoLVLA CPU inference was not rerun in the Certified TailGuard pass summarized below. In the later evidence-strengthening pass, `bash scripts/run_optional_vla.sh --attempt-inference` completed and refreshed `results/optional_vla/inference_probe.json` with `INFERENCE_PROBE_PASS`; it remains optional model-plumbing evidence, not benchmark or physical-success evidence.

New v2 artifacts:

- `results/tailguard_summary.csv`
- `results/tailguard_artifact.json`
- `results/tailguard_gate_examples.csv`
- `results/phase_diagram_summary.csv`
- `results/calibration_sample_complexity.csv`
- `results/physics_stress_summary.csv`
- `results/component_ablation_summary.csv`
- `results/failure_honesty_summary.csv`
- `results/optional_vla/smolvla_rendered_bridge.json`
- `results/optional_vla/libero_benchmark_status.json`

New v2 figures:

- `results/figures/figure11_tailguard_adaptive_n.png`
- `results/figures/figure12_phase_diagram.png`
- `results/figures/figure13_calibration_sample_complexity.png`
- `results/figures/figure14_imperfect_verifier_map.png`
- `results/figures/figure15_physics_failure_decomposition.png`
- `results/figures/figure16_smolvla_rendered_bridge_status.png`
- `results/figures/figure17_component_ablation.png`
- `results/figures/figure18_failure_honesty.png`

Latest claim-audit additions are designed to support Certified TailGuard method claims, component ablations, failure-honesty claims, phase diagram claims, calibration sample-complexity claims, first-principles physics claims, optional SmoLVLA rendered-bridge status claims, and explicit benchmark/robot boundary claims. `results/tailguard_gate_examples.csv` covers `allow_high_n`, `stop_early`, `collect_pilot_labels`, and `block_high_n`; real VLA benchmark validation and real-robot validation remain not implemented and not claimed.

## External Benchmark Addendum

Latest guarded external runner additions:

- implementation: `src/vla_tailguard_audit/external_benchmark.py`
- CLI: `experiments/run_external_benchmark.py`
- shell wrapper: `scripts/run_external_benchmark.sh`
- tests: `tests/test_external_benchmark.py`
- artifacts: `results/external_benchmark_status.json`, `results/external_benchmark_summary.csv`, `results/external_benchmark_seed_level/`, and `results/figures/figure19_external_benchmark_status.png`

Commands run in this pass:

| Command | Status | Notes |
|---|---:|---|
| `bash scripts/run_external_benchmark.sh --status-only` | PASS | Wrote RoboCasa integration-only status. |
| `bash scripts/run_external_benchmark.sh --attempt-robocasa --attempt-libero --timeout-seconds 10` | PASS | RoboCasa imports and target task registration are recorded; bounded reset/step attempt timed out before reward or success metrics; LIBERO import status is recorded without a reset/step wrapper. |

Current external claim level: integration only. RoboCasa reset/step, reward traces, success metrics, physical outcomes, and TailGuard method outcomes remain unsupported.

Optional SmoLVLA note for this pass: `bash scripts/run_optional_vla.sh --attempt-inference` completed and refreshed `results/optional_vla/inference_probe.json` at `INFERENCE_PROBE_PASS`. The refreshed probe loaded cached `models--lerobot--smolvla_base` on CPU, reported `450,046,176` parameters, emitted action shape `[1, 50, 6]`, and kept `benchmark_validation: false`.

Component-ablation strengthening: `results/component_ablation_summary.csv` now includes the original joint adversarial stress plus six component-specific stress regimes. Each named removal (`no_physical_certificate`, `no_verifier_score`, `no_pilot_labels`, `no_empirical_lower_bound`, `no_adaptive_n`, and `no_abstention_fallback`) has at least one row failing the controlled acceptance threshold, while the corresponding full Certified TailGuard row passes.

## 1. Exact Commands Run

| Command | Status | Runtime | Failing tests |
|---|---:|---:|---|
| `pytest` | PASS | 53.50 s | None, 23 passed |
| `bash scripts/run_optional_vla.sh` | PASS | 6.67 s | Wrote `READY_TO_ATTEMPT` status |
| `bash scripts/run_optional_vla.sh --attempt-inference` | PASS | 186.18 s | Wrote `INFERENCE_PROBE_PASS` synthetic SmoLVLA action probe |
| `bash scripts/run_smoke.sh` | PASS | 132.68 s | None, 23 tests passed inside script |
| `bash scripts/run_all.sh` | PASS | 570.88 s | None, 23 tests passed inside script |
| `bash scripts/run_claim_audit.sh` | PASS | 10.32 s | None |

Intermediate note: plain `pytest` initially failed because `scripts` was not importable from the standalone pytest path. This was fixed by adding `scripts/__init__.py` and setting `pythonpath = ["src", "."]` in `pyproject.toml`; the final standalone `pytest` pass above is from the fixed state.

Rename note: the repository was later renamed from `C:\Users\wangz\vla-score-tail` to `C:\Users\wangz\score-tail vla`. The command table above reflects checks run from the renamed path.

## 2. Artifact Inventory

Key JSON files:

- `results/manifest.json`
- `results/controlled_semantic_overselection_artifact.json`
- `results/learned_vla_artifact.json`
- `results/repair_artifact.json`
- `results/rendered_visual_simulator_artifact.json`
- `results/robustness_artifact.json`
- `results/optional_vla/adapter_status.json`
- `results/optional_vla/inference_probe.json`
- `results/claims_status.json`

Key CSV files:

- `results/controlled_summary.csv`
- `results/learned_summary.csv`
- `results/distractor_summary.csv`
- `results/repair_summary.csv`
- `results/rendered_summary.csv`
- `results/rendered_visual_metadata.csv`
- `results/robustness_summary.csv`
- `results/robustness_seed_metrics.csv`
- `results/robustness_seed_variance.csv`
- `results/optional_vla/adapter_status.md`
- `results/optional_vla/inference_probe.md`
- `results/*_seed_metrics.csv`
- `results/*_seed_variance.csv`

Key figures:

- `results/figures/figure1_semantic_overselection.png`
- `results/figures/figure2_repair_comparison.png`
- `results/figures/figure3_tail_diagnostics.png`
- `results/figures/figure4_failure_decomposition.png`
- `results/figures/figure5_exact_law_validation.png`
- `results/figures/figure6_rendered_visual_simulator.png`
- `results/figures/figure7_rendered_scene_examples.png`
- `results/figures/figure8_robustness_repairs.png`
- `results/figures/figure9_calibration_budget_noise.png`
- `results/figures/figure10_learned_scorer_comparison.png`

Seed-level files:

- `results/seed_level/controlled_seed_001.json`
- `results/seed_level/controlled_seed_002.json`
- `results/seed_level/controlled_seed_003.json`
- `results/seed_level/learned_seed_001.json`
- `results/seed_level/learned_seed_002.json`
- `results/seed_level/learned_seed_003.json`
- `results/seed_level/distractor_seed_001.json`
- `results/seed_level/distractor_seed_002.json`
- `results/seed_level/distractor_seed_003.json`
- `results/seed_level/repair_seed_001.json`
- `results/seed_level/repair_seed_002.json`
- `results/seed_level/repair_seed_003.json`
- `results/seed_level/rendered_seed_001.json`
- `results/seed_level/rendered_seed_002.json`
- `results/seed_level/rendered_seed_003.json`
- `results/seed_level/robustness_seed_001.json`
- `results/seed_level/robustness_seed_002.json`
- `results/seed_level/robustness_seed_003.json`

## 3. Strongest Semantic Affordance Over-Selection Artifact

Path: `results/learned_summary.csv` and `results/learned_vla_artifact.json`.

N values: `1, 2, 4, 8, 16, 32, 64, 128`.

Selected semantic score vs N:

- `1: 0.876`, `2: 0.927`, `4: 0.952`, `8: 0.967`, `16: 0.974`, `32: 0.978`, `64: 0.982`, `128: 0.985`.

Selected real utility vs N:

- `1: 0.473`, `2: 0.411`, `4: 0.295`, `8: 0.205`, `16: 0.181`, `32: 0.175`, `64: 0.166`, `128: 0.152`.

At `N=128`:

- semantic-real tail gap: `0.796`
- tail rank correlation: `-0.039`
- high-N regret: `0.848`
- violation rate: `0.958`
- deployment gate: `block_high_n`

This is the cleanest v1 artifact: the learned semantic scorer becomes increasingly confident in language/vision plausibility while selected physical utility collapses.

## 4. Strongest Repair Artifact

Path: `results/repair_summary.csv` and `results/repair_artifact.json`.

At `N=128`:

| Method | Real utility | 95% MC CI | Oracle gap | Violation rate | Gate |
|---|---:|---:|---:|---:|---|
| raw semantic | 0.152 | [0.143, 0.160] | 0.848 | 0.958 | `block_high_n` |
| semantic plus physical | 0.965 | [0.958, 0.971] | 0.035 | 0.000 | `allow_high_n` |
| calibrated | 0.999 | [0.998, 0.999] | 0.001 | 0.000 | `stop_early` |
| grounded combined | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | `stop_early` |
| oracle | 1.000 | [1.000, 1.000] | 0.000 | 0.000 | `stop_early` |

Real utility improvement over raw:

- calibrated: `+0.847`
- grounded combined: `+0.848`

Violation reduction over raw:

- calibrated: `0.958 -> 0.000`
- grounded combined: `0.958 -> 0.000`

Deployment gate change:

- raw semantic: `block_high_n`
- calibrated: `stop_early`
- grounded combined: `stop_early`

## 5. Learned VLA Validity Checklist

- language instruction input exists? Yes, `language_vector` and instruction text are part of each candidate pool.
- visual/object observation input exists? Yes, `observation_vector` and scene/object state are used.
- action candidate/generator exists? Yes, `generate_candidate_pool` creates finite candidate action pools.
- learned semantic/affordance scorer exists? Yes, `LearnedVLAScorer` fits a semantic scorer from data.
- action selection is language-conditioned? Yes, learned features concatenate language, observation, and action features.
- real utility is measured separately? Yes, `real_utility_from_features` is separate from the semantic scorer.
- semantic score is not real utility? Yes, tests check they are not identical.
- raw reconstruction is not the main objective? Yes, the learned scorer predicts semantic affordance labels and is evaluated by physical utility.

## 5b. Rendered Visual Simulator Artifact

Path: `results/rendered_summary.csv`, `results/rendered_visual_metadata.csv`, `results/rendered/`, and `results/rendered_visual_simulator_artifact.json`.

N values: `1, 2, 4, 8, 16, 32, 64, 128`.

Selected semantic score vs N:

- `1: 0.876`, `2: 0.927`, `4: 0.952`, `8: 0.967`, `16: 0.974`, `32: 0.978`, `64: 0.982`, `128: 0.985`.

Simulator-computed selected real utility vs N:

- `1: 0.584`, `2: 0.572`, `4: 0.517`, `8: 0.465`, `16: 0.453`, `32: 0.452`, `64: 0.451`, `128: 0.446`.

At `N=128`:

- simulator violation rate: `1.000`
- high-N regret: `0.554`
- semantic-real tail gap: `0.526`
- tail rank correlation: `-0.051`
- raw semantic deployment gate: `block_high_n`
- grounded combined repair utility: `1.000`
- calibrated repair utility: `0.998`

## 6. Not-Clone Checklist

- reused math: finite tie-aware score-tail law only.
- new scientific object: VLA-style instruction, visual/object observation, action candidate, semantic score, physical feasibility, calibrated score, real utility.
- new VLA-specific failure: semantic affordance over-selection.
- new VLA-specific experiments: object/receptacle scenes, distractors, reachability, collision, wrong-object and wrong-target failures.
- forbidden clone patterns absent: no imagined rollout mismatch, no latent predictor, no energy model, no diffusion diversity framing.

## 6b. Noisy Verifier And Calibration Robustness

Path: `results/robustness_summary.csv` and `results/robustness_artifact.json`.

At `N=128`:

| Method | Real utility | Violation rate | High-N regret | Gate |
|---|---:|---:|---:|---|
| raw semantic | 0.446 | 1.000 | 0.554 | `block_high_n` |
| noisy verifier mild | 0.581 | 0.750 | 0.419 | `block_high_n` |
| noisy verifier harsh | 0.466 | 1.000 | 0.534 | `block_high_n` |
| ideal verifier | 1.000 | 0.000 | 0.000 | `stop_early` |
| calib 1pct/noise0 | 0.892 | 0.000 | 0.108 | `allow_high_n` |
| calib 2pct/noise5 | 1.000 | 0.000 | 0.000 | `stop_early` |
| calib 5pct/noise10 | 0.684 | 0.857 | 0.316 | `block_high_n` |
| calib 10pct/noise20 | 0.858 | 0.566 | 0.142 | `collect_pilot_labels` |

Interpretation: v2 no longer treats grounding/calibration as automatically clean repair. Mild noisy verification helps but remains unsafe at high N; harsh noisy verification barely repairs the tail; calibration can succeed or fail depending on label budget/noise.

## 6c. Optional Real-VLA / SmoLVLA Probe

Paths: `results/optional_vla/adapter_status.json`, `results/optional_vla/adapter_status.md`, `results/optional_vla/inference_probe.json`, and `results/optional_vla/inference_probe.md`.

Adapter status: `READY_TO_ATTEMPT`.

Reason: cached SmoLVLA files and the LeRobot/core runtime are available. The status records:

- `torch`: True
- `transformers`: True
- `huggingface_hub`: True
- `safetensors`: True
- `accelerate`: True
- `num2words`: True
- `lerobot`: True
- `libero`: False

Cached model folders detected:

- `models--lerobot--smolvla_base`: config and weights present.
- `models--lerobot--smolvla_libero`: config present, weights absent.
- `models--HuggingFaceVLA--smolvla_libero`: config and weights present.

Inference probe status: `INFERENCE_PROBE_PASS`.

Probe details:

- model cache: `models--lerobot--smolvla_base`
- model parameters: `450,046,176`
- task: `put the red mug in the cabinet`
- input type: synthetic visual tensors for three cameras, zero robot state, natural-language task
- action chunk shape: `[1, 50, 6]`
- action dtype: `torch.float32`
- max absolute action value: `0.9026641845703125`
- first action: `[0.05234574154019356, -0.0497642457485199, -0.03662483021616936, -0.0669940635561943, -0.04889203608036041, -0.3559389114379883]`
- config load: `1.5228 s`
- policy load: `70.31 s`
- preprocessing: `0.0239 s`
- inference: `56.7759 s`

This closes the “LeRobot/SmoLVLA cannot run at all” gap. It does not support real VLA benchmark validation because no LIBERO/OpenVLA task environment, physical utility evaluator, or real robot is used for the SmoLVLA output.

## 6d. PyTorch Learned VLA-Style Scorer

Implementation: `src/vla_tailguard_audit/torch_vla.py`.

At `N=128` in `results/learned_summary.csv`:

- `torch_semantic`: selected semantic score `0.971`, selected real utility `0.329`, violation rate `0.958`, gate `collect_pilot_labels`.
- `torch_calibrated`: selected real utility `0.999`, violation rate `0.000`, gate `stop_early`.

At `N=128` in `results/rendered_summary.csv`:

- `torch_semantic`: selected semantic score `0.969`, simulator real utility `0.455`, violation rate `1.000`, gate `block_high_n`.
- `torch_calibrated`: simulator real utility `1.000`, violation rate `0.000`, gate `stop_early`.

Interpretation: a heavier learned semantic scorer reproduces the selected-tail semantic/physical failure. Calibration repairs the tail in the controlled and rendered-simulator settings.

## 7. Paper Readiness Judgment

Judgment: paper-worthy v1 as a controlled diagnostic repository with optional real-SmoLVLA plumbing evidence, but not yet a robotics benchmark paper.

- The exact law, diagnostics, learned VLA-style artifact, distractor artifact, and repair artifact are all present.
- The evidence supports controlled and learned toy claims.
- The optional SmoLVLA probe shows real model loading/action emission, but the paper still needs benchmark validation for a venue expecting real VLA or robot foundation model evidence.
- No redesign is required for the v1 thesis; v2 should add real benchmark adapters.

## 8. Top Remaining Weaknesses

- The learned model is intentionally small and CPU-friendly.
- The environment is synthetic and constructed.
- No real robot or large VLA benchmark is evaluated.
- The optional SmoLVLA probe uses synthetic inputs and does not measure physical success.
- The calibrated scorer has access to clean pilot real-utility labels.
- The physical verifier is idealized relative to real robotics.
- Confidence intervals quantify Monte Carlo selection in the toy setting, not broad robotics uncertainty.
- Failure modes are convincing as controlled evidence but not representative coverage of all manipulation tasks.
- The grounded combined score can be near-oracle because the toy feasibility features are explicit.

## 9. Exact Next Steps After This Pass

1. Add a guarded LIBERO or LeRobot task wrapper so SmoLVLA actions are evaluated by a real benchmark/simulator success signal rather than only synthetic action emission.
2. Install or isolate `libero` in a compatible environment; the current global Python still lacks `libero` and has unrelated package-version conflicts outside this repo.
3. Add broader shortcut ablations across color, category, receptacle, distractor salience, language prior, and hidden physical constraints.
4. Convert the markdown paper skeleton into a conference-style draft using the now-supported rendered, simulator, robustness, PyTorch, and optional SmoLVLA plumbing evidence.
5. Add real-robot validation only with actual hardware evidence; keep current real-robot claims unsupported otherwise.

## 10. Current Git State

- Repository initialized: yes.
- Initial checkpoint commit: `7a8cc45`.
- Commit message: `Initial paper-quality VLA score-tail checkpoint`.
