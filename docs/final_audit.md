# Final Audit

Repository path at latest audit: `C:\Users\wangz\best of n vla`.

Git checkpoint: `7a8cc45` (`Initial paper-quality VLA Best-of-N checkpoint`). The repository was initialized during the resumability pass.

## 1. Exact Commands Run

| Command | Status | Runtime | Failing tests |
|---|---:|---:|---|
| `pytest` | PASS | 39.50 s | None, 22 passed |
| `bash scripts/run_optional_vla.sh` | PASS | 12.28 s | Wrote `SKIPPED_WITH_REASON` status |
| `bash scripts/run_smoke.sh` | PASS | 218.60 s | None |
| `bash scripts/run_all.sh` | PASS | 561.28 s | None |
| `bash scripts/run_claim_audit.sh` | PASS | 7.06 s | None |

Intermediate note: plain `pytest` initially failed because `scripts` was not importable from the standalone pytest path. This was fixed by adding `scripts/__init__.py` and setting `pythonpath = ["src", "."]` in `pyproject.toml`; the final standalone `pytest` pass above is from the fixed state.

Rename note: the repository was later renamed from `C:\Users\wangz\vla-best-of-n` to `C:\Users\wangz\best of n vla`. The command table above reflects checks run from the renamed path.

## 2. Artifact Inventory

Key JSON files:

- `results/manifest.json`
- `results/controlled_semantic_overselection_artifact.json`
- `results/learned_vla_artifact.json`
- `results/repair_artifact.json`
- `results/rendered_visual_simulator_artifact.json`
- `results/robustness_artifact.json`
- `results/optional_vla/adapter_status.json`
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

- reused math: finite tie-aware Best-of-N law only.
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

## 6c. Optional Real-VLA Adapter Status

Path: `results/optional_vla/adapter_status.json` and `results/optional_vla/adapter_status.md`.

Status: `SKIPPED_WITH_REASON`.

Reason: cached SmoLVLA files and core ML runtime are present, but `lerobot` is not importable. The adapter records:

- `torch`: True
- `transformers`: True
- `huggingface_hub`: True
- `safetensors`: True
- `accelerate`: True
- `lerobot`: False
- `libero`: False

Cached model folders detected:

- `models--lerobot--smolvla_base`: config and weights present.
- `models--lerobot--smolvla_libero`: config present, weights absent.
- `models--HuggingFaceVLA--smolvla_libero`: config and weights present.

This closes the “unknown optional VLA status” gap but does not support real VLA benchmark validation.

## 6d. PyTorch Learned VLA-Style Scorer

Implementation: `src/vla_best_of_n/torch_vla.py`.

At `N=128` in `results/learned_summary.csv`:

- `torch_semantic`: selected semantic score `0.971`, selected real utility `0.329`, violation rate `0.958`, gate `collect_pilot_labels`.
- `torch_calibrated`: selected real utility `0.999`, violation rate `0.000`, gate `stop_early`.

At `N=128` in `results/rendered_summary.csv`:

- `torch_semantic`: selected semantic score `0.969`, simulator real utility `0.455`, violation rate `1.000`, gate `block_high_n`.
- `torch_calibrated`: simulator real utility `1.000`, violation rate `0.000`, gate `stop_early`.

Interpretation: a heavier learned semantic scorer reproduces the selected-tail semantic/physical failure. Calibration repairs the tail in the controlled and rendered-simulator settings.

## 7. Paper Readiness Judgment

Judgment: paper-worthy v1 as a controlled diagnostic repository, but not yet a robotics benchmark paper.

- The exact law, diagnostics, learned VLA-style artifact, distractor artifact, and repair artifact are all present.
- The evidence supports controlled and learned toy claims.
- The paper still needs stronger experiments and real benchmark validation for a venue expecting real VLA or robot foundation model evidence.
- No redesign is required for the v1 thesis; v2 should add real benchmark adapters.

## 8. Top Remaining Weaknesses

- The learned model is intentionally small and CPU-friendly.
- The environment is synthetic and constructed.
- No real robot or large VLA benchmark is evaluated.
- The calibrated scorer has access to clean pilot real-utility labels.
- The physical verifier is idealized relative to real robotics.
- Confidence intervals quantify Monte Carlo selection in the toy setting, not broad robotics uncertainty.
- Failure modes are convincing as controlled evidence but not representative coverage of all manipulation tasks.
- The grounded combined score can be near-oracle because the toy feasibility features are explicit.

## 9. Exact Next Steps After This Pass

1. Install or vendor the LeRobot runtime if allowed, then move optional SmoLVLA status from `SKIPPED_WITH_REASON` toward actual local inference.
2. If LeRobot inference succeeds, add a guarded optional benchmark artifact under `results/optional_vla/` and update claim audit only to `PARTIAL` unless a real benchmark with physical evaluation runs.
3. Add broader shortcut ablations across color, category, receptacle, distractor salience, language prior, and hidden physical constraints.
4. Convert the markdown paper skeleton into a conference-style draft using the now-supported rendered, simulator, robustness, and PyTorch evidence.
5. Add external benchmark validation or real-robot validation only with actual runtime/hardware evidence; keep current claims unsupported otherwise.

## 10. Current Git State

- Repository initialized: yes.
- Initial checkpoint commit: `7a8cc45`.
- Commit message: `Initial paper-quality VLA Best-of-N checkpoint`.
