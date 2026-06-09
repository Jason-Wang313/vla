# V2 Gap Closure Roadmap

## Goal

Push the repository from a strong controlled v1 toward the best feasible paper-quality v2. RAM is not the limiting concern; evidence quality is.

## Current Verified Environment

Available Python packages from inspection:

- `torch`
- `transformers`
- `huggingface_hub`
- `accelerate`
- `safetensors`
- `gymnasium`
- `cv2`
- `PIL`
- `sapien`
- `robosuite`
- `lerobot`
- `num2words`

Not importable at inspection time:

- `libero`

Cached local model folders exist:

- `C:\Users\wangz\.cache\huggingface\hub\models--lerobot--smolvla_base`
- `C:\Users\wangz\.cache\huggingface\hub\models--lerobot--smolvla_libero`
- `C:\Users\wangz\.cache\huggingface\hub\models--HuggingFaceVLA--smolvla_libero`

## Best-ROI Order

1. Rendered visual toy benchmark.
2. Simulator-style physical evaluator.
3. Noisy and imperfect physical verifier.
4. Calibration budget/noise ablations.
5. Heavier PyTorch visual-language-action scorer.
6. Optional guarded SmoLVLA/real-VLA adapter.

Items 1-5 now have implemented artifacts. The repository now also includes Certified TailGuard-BoN, phase diagrams, calibration sample-complexity sweeps, component ablations, failure-honesty artifacts, and first-principles physics stress decompositions. Item 6 has guarded readiness/status artifacts and a heavyweight synthetic CPU inference probe path: cached SmoLVLA can load through LeRobot and emit a finite action chunk from synthetic visual/state/language input when the optional probe is run. Actual benchmark evaluation remains future work.

The latest external benchmark pass adds `experiments/run_external_benchmark.py` and `scripts/run_external_benchmark.sh`. It records RoboCasa package importability, assets/macros, task registration, bounded reset/step attempts, reward/success traces when available, and a separate LIBERO status row. The current artifact is integration-only: RoboCasa imports and registers the target task, but the bounded reset/step attempt timed out before reward or success metrics were produced.

## Certified TailGuard-BoN Additions

- `src/vla_best_of_n/tailguard.py` implements certificate predicates, `TailGuardConfig`, `TailGuardResult`, `fit_tail_calibrator`, `predict_selected_tail_curve`, and `tailguard_select`.
- `results/tailguard_summary.csv` compares Certified TailGuard with raw fixed-N BoN, `N=1`, random high-N, verifier filtering, calibrated high-N without certificates, certificate-only filtering, TailGuard ablations, and oracle.
- `results/tailguard_artifact.json` records controller decisions, selected N, curves, lower bounds, gate decisions, reason codes, certificate pass/failure types, certified candidate count, fallback, abstention, certified selected utility, and certified violation.
- `results/tailguard_gate_examples.csv` gives constructed examples for `allow_high_n`, `stop_early`, `collect_pilot_labels`, and `block_high_n`.
- `results/component_ablation_summary.csv` and `results/failure_honesty_summary.csv` record component necessity and abstain/fallback regimes.

## First-Principles Stress Additions

- `results/phase_diagram_summary.csv` sweeps semantic/physical misalignment, distractor salience, obstacle density, hidden constraints, verifier errors, and label noise.
- `results/calibration_sample_complexity.csv` crosses pilot label budgets `{0.5%, 1%, 2%, 5%, 10%, 20%}` with label noise `{0, 0.05, 0.10, 0.20}`.
- `results/physics_stress_summary.csv` decomposes reach, swept collision, receptacle, stability, fragile/heavy, blocked-path, hidden obstacle, semantic-tail verifier false positives, partial observation, object identity spoofing, high-plausibility wrong receptacles, correlated pilot-label noise, train/pilot/test shift, and low certified-candidate-count scenes.

## Acceptance Criteria For V2

- At least one rendered visual artifact reproduces semantic affordance over-selection.
- At least one simulator-computed utility artifact reproduces selected semantic score rising while physical utility drops or saturates.
- Noisy verifier experiments show repair is not assumed perfect.
- Calibration stress tests show label-budget and label-noise tradeoffs.
- Optional SmoLVLA adapter either runs a synthetic action probe or writes an exact skip reason.
- Optional SmoLVLA rendered bridge and LIBERO benchmark status artifacts write exact run-or-skip reasons, including `action_decode_supported: false` and `physical_success_claimed: false` when no action decoder/success evaluator is run.
- External RoboCasa/LIBERO artifacts write exact integration, reset/step, reward, success-metric, and claim-level status; external simulator outcome claims require reset/step/reward/success artifacts.
- Claim audit keeps real-robot validation unsupported unless actual hardware evidence exists.

## Why SmoLVLA Is Troublesome, Not Impossible

The first blocker was software integration, not hardware impossibility. Installing `lerobot`, pinning `transformers` to the compatible 4.x line, and adding `num2words` allowed cached SmoLVLA to load on CPU and produce an action chunk. The probe uses the LeRobot policy class, tokenizer/processor path, camera/state schema, and cached model weights.

This still does not close the benchmark gap. `libero` is not importable in the core optional path; the isolated external `libero310` environment imports `libero`, but no reset/step wrapper or physical utility evaluator is connected to SmoLVLA actions. The adapter should remain optional and should not be required for the core `run_all.sh` until a reliable benchmark wrapper exists.
