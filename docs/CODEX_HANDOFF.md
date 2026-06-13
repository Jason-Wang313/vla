# CODEX HANDOFF

## Current Goal

Make the worktree resumable in a new thread and push the VLA score-tail repository toward the best feasible paper-quality state. User requested the folder name `score-tail vla` and said RAM is not a concern; prioritize evidence quality.

## Repo Facts Verified From Files

- Current repository path: `C:\Users\wangz\score-tail vla`.
- The old path `C:\Users\wangz\vla-score-tail` was renamed during this continuation.
- No `AGENTS.md` existed before this update; one now exists.
- The repo is now a git repository.
- Initial checkpoint commit: `7a8cc45` (`Initial paper-quality VLA score-tail checkpoint`).
- Main Python package remains `src/vla_tailguard_audit`.
- `pyproject.toml` project name is now `vla-tailguard-audit`.
- v1 generated artifacts exist under `results/`, including summary CSVs, seed-level files, claim status files, and five figures.
- v2 rendered visual/simulator artifacts now exist under `results/rendered/`, `results/rendered_summary.csv`, and `results/rendered_visual_metadata.csv`.
- v2 noisy verifier and calibration robustness artifacts now exist under `results/robustness_summary.csv` and `results/robustness_artifact.json`.
- Optional SmoLVLA adapter status exists under `results/optional_vla/adapter_status.json` and `.md`; the heavyweight synthetic inference probe exists under `results/optional_vla/inference_probe.json` and `.md`.
- PyTorch learned scorer artifacts are included as `torch_semantic` and `torch_calibrated` methods in `results/learned_summary.csv` and `results/rendered_summary.csv`.
- Required v1 scripts exist:
  - `scripts/run_smoke.sh`
  - `scripts/run_all.sh`
  - `scripts/run_claim_audit.sh`
  - `scripts/run_with_src.py`

## Verified Scientific State

The repo currently supports controlled and learned VLA-style claims, not real-robot claims.

Strongest learned VLA-style artifact from `docs/final_audit.md`:

- path: `results/learned_summary.csv`
- `N=1`: selected semantic score `0.876`, selected real utility `0.473`
- `N=128`: selected semantic score `0.985`, selected real utility `0.152`
- `N=128` violation rate: `0.958`
- `N=128` high-N regret: `0.848`
- deployment gate: `block_high_n`

Strongest repair artifact from `docs/final_audit.md`:

- path: `results/repair_summary.csv`
- raw semantic utility at `N=128`: `0.152`
- calibrated utility at `N=128`: `0.999`
- grounded combined utility at `N=128`: `1.000`
- violations drop from `0.958` to `0.000` for calibrated/grounded repair.

Rendered visual simulator artifact:

- path: `results/rendered_summary.csv`
- `N=1`: selected semantic score `0.876`, simulator real utility `0.584`, violation rate `0.644`
- `N=128`: selected semantic score `0.985`, simulator real utility `0.446`, violation rate `1.000`
- `N=128` high-N regret: `0.554`
- grounded combined repair utility at `N=128`: `1.000`

Robustness stress artifact:

- path: `results/robustness_summary.csv`
- raw semantic at `N=128`: utility `0.446`, violation `1.000`, gate `block_high_n`
- mild noisy verifier at `N=128`: utility `0.581`, violation `0.750`, gate `block_high_n`
- harsh noisy verifier at `N=128`: utility `0.466`, violation `1.000`, gate `block_high_n`
- calibration stressed methods at `N=128`: utility range `0.684` to `1.000`

Optional VLA adapter status:

- path: `results/optional_vla/adapter_status.json`
- status: `READY_TO_ATTEMPT`
- `lerobot` and `num2words` are importable in the core optional path; `libero` is not importable there. A later isolated external benchmark pass found `libero` importable in `external_benchmarks/.venvs/libero310`, but no LIBERO reset/step wrapper is implemented.
- cached `models--lerobot--smolvla_base` has config and weights.
- optional synthetic CPU inference probe status: `INFERENCE_PROBE_PASS`
- inference probe action shape: `[1, 50, 6]`
- inference probe parameter count: `450,046,176`
- this supports real SmoLVLA plumbing only, not real VLA benchmark validation.

PyTorch learned scorer artifact:

- implementation: `src/vla_tailguard_audit/torch_vla.py`
- symbolic `N=128`: `torch_semantic` utility `0.329`, violation `0.958`; `torch_calibrated` utility `0.999`, violation `0.000`
- rendered `N=128`: `torch_semantic` utility `0.455`, violation `1.000`; `torch_calibrated` utility `1.000`, violation `0.000`

## Files Changed In This Continuation

- Renamed repo folder to `C:\Users\wangz\score-tail vla`.
- Updated `pyproject.toml` project name and added optional `v2` dependencies.
- Updated `src/vla_tailguard_audit/learned_vla.py` docstring to remove RAM-light framing.
- Updated `docs/current_state_before_v1.md` to record the rename.
- Added `AGENTS.md` for future agents.
- Added `docs/v2_gap_closure.md`.
- Added this handoff: `docs/CODEX_HANDOFF.md`.
- Added rendered visual benchmark implementation in `src/vla_tailguard_audit/rendering.py`.
- Added simulator-style evaluator in `src/vla_tailguard_audit/simulator.py`.
- Updated `experiments/run_experiments.py` and `src/vla_tailguard_audit/plotting.py` to produce rendered/simulator artifacts and figures.
- Added `tests/test_rendering_simulator.py`.
- Added robustness implementation in `src/vla_tailguard_audit/robustness.py`.
- Extended `RealUtilityCalibrator` with sample fraction and label-noise options.
- Added `tests/test_robustness.py`.
- Added `docs/robustness_stress_tests.md`.
- Added `docs/visual_benchmark.md` and `docs/simulator_evaluator.md`.
- Added optional VLA status module `src/vla_tailguard_audit/optional_vla.py`.
- Added optional VLA runner `experiments/run_optional_vla.py` and `scripts/run_optional_vla.sh`.
- Added `tests/test_optional_vla.py`.
- Added `docs/optional_real_vla_adapter.md`.
- Extended optional VLA runner with `--attempt-inference`, which loads cached SmoLVLA on CPU and writes `results/optional_vla/inference_probe.json`.
- Added PyTorch scorer `src/vla_tailguard_audit/torch_vla.py`.
- Added `tests/test_torch_vla.py`.
- Added `docs/torch_learned_scorer.md`.
- Updated experiments and plotting to include `torch_semantic`, `torch_calibrated`, and `figure10_learned_scorer_comparison.png`.
- Added `.gitattributes`.
- Initialized git and created initial checkpoint commit `7a8cc45`.

## Commands Run In This Continuation

- Inspected context-handoff skill: success.
- Verified folder state before rename:
  - `C:\Users\wangz\vla-score-tail`: existed.
  - `C:\Users\wangz\score-tail vla`: did not exist.
- Renamed folder with `Move-Item`: success.
- Checked git status before rename: failed with `fatal: not a git repository`.
- Scanned references with `rg`; command timed out after reporting relevant matches in `pyproject.toml`, `README.md`, `docs/current_state_before_v1.md`, and `src/vla_tailguard_audit/learned_vla.py`.
- Ran `pytest` from renamed path: PASS, 13 passed, runtime 25.06 s.
- Ran `bash scripts/run_claim_audit.sh` from renamed path: PASS, runtime 24.22 s.
- Ran `bash scripts/run_smoke.sh` from renamed path: PASS, runtime 109.10 s.
- Ran `bash scripts/run_all.sh` from renamed path: PASS, runtime 319.79 s.
- Ran `bash scripts/run_claim_audit.sh` again after final audit/handoff updates: PASS, runtime 10.56 s.
- After rendered/simulator implementation, ran `pytest`: PASS, 16 passed, runtime 17.52 s.
- After rendered/simulator implementation, ran `bash scripts/run_smoke.sh`: PASS, runtime 144.51 s.
- After rendered/simulator implementation, ran `bash scripts/run_all.sh`: PASS, runtime 296.72 s.
- After rendered/simulator docs update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 8.82 s.
- After robustness implementation, ran `pytest`: PASS, 18 passed, runtime 23.71 s.
- After robustness implementation, ran `bash scripts/run_smoke.sh`: PASS, runtime 122.86 s.
- After robustness implementation, ran `bash scripts/run_all.sh`: PASS, runtime 487.71 s.
- After robustness docs update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 5.41 s.
- After optional VLA status implementation, ran `bash scripts/run_optional_vla.sh`: PASS, runtime 12.28 s, status `SKIPPED_WITH_REASON`.
- After optional VLA status implementation, ran `pytest`: PASS, 21 passed, runtime 16.28 s.
- After optional VLA status implementation, ran `bash scripts/run_smoke.sh`: PASS, runtime 125.27 s.
- After optional VLA status implementation, ran `bash scripts/run_all.sh`: PASS, runtime 436.29 s.
- After optional VLA status implementation and docs update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 29.80 s.
- After PyTorch scorer implementation, ran `pytest`: PASS, 22 passed, runtime 39.50 s.
- After PyTorch scorer implementation, ran `bash scripts/run_smoke.sh`: PASS, runtime 218.60 s.
- After PyTorch scorer implementation, ran `bash scripts/run_all.sh`: PASS, runtime 561.28 s.
- After PyTorch docs update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 6.62 s.
- Initialized git repository and committed current verified state: `7a8cc45`.
- Verified `git status --short`: clean after initial checkpoint.
- After git handoff/final-audit doc update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 7.06 s.
- Installed `lerobot==0.4.4`, pinned `transformers` back to 4.x compatibility, and installed `num2words`.
- Ran `pytest tests/test_optional_vla.py -q`: PASS, 4 passed, runtime 2.54 s.
- Ran `bash scripts/run_optional_vla.sh --attempt-inference`: PASS, runtime 186.18 s, status `INFERENCE_PROBE_PASS`.
- Ran `bash scripts/run_optional_vla.sh`: PASS, runtime 6.67 s, status `READY_TO_ATTEMPT`.
- Final verification after optional inference implementation, ran `pytest`: PASS, 23 passed, runtime 56.38 s.
- Final verification after optional inference implementation, ran `bash scripts/run_smoke.sh`: PASS, runtime 235.04 s.
- Final verification after optional inference implementation, ran `bash scripts/run_all.sh`: PASS, runtime 491.46 s.
- Final verification after optional inference implementation, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 20.19 s.
- Post-doc-update claim audit, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 24.65 s.
- Final verification after manifest update, ran `pytest`: PASS, 23 passed, runtime 53.50 s.
- Final verification after manifest update, ran `bash scripts/run_smoke.sh`: PASS, runtime 132.68 s.
- Final verification after manifest update, ran `bash scripts/run_all.sh`: PASS, runtime 570.88 s.
- Final verification after manifest update, ran `bash scripts/run_claim_audit.sh`: PASS, runtime 10.32 s.

## Known Failures Or Bugs

- `libero` is not importable in the core optional path. The isolated external `libero310` environment imports `libero`, but no LIBERO benchmark reset/step or success artifact exists, so no LIBERO benchmark outcome can be claimed.
- `pip check` reports unrelated global-environment dependency conflicts around NumPy/protobuf; use an isolated environment for real benchmark work.
- Optional SmoLVLA/real-VLA benchmark evaluation is not implemented.
- Real-robot validation is unsupported.

## Open Questions

- RESOLVED: `lerobot` installed and cached SmoLVLA weights load on CPU.
- RESOLVED: cached SmoLVLA emits a finite action chunk from synthetic image/state/language input.
- UNKNOWN: Whether `robosuite`/`sapien` should be used for v2 or whether a pure Gymnasium 2D simulator is preferable.

## Next Recommended Steps

1. Build a guarded LIBERO/LeRobot benchmark wrapper if dependency conflicts can be isolated.
2. Re-run `pytest`, `bash scripts/run_claim_audit.sh`, `bash scripts/run_smoke.sh`, and `bash scripts/run_all.sh` after additional v2 code changes.

Safe to clear after handoff is updated.
