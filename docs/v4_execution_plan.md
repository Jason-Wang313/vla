# V4 Submission Hardening Plan

The v4 pass keeps the paper scoped as a controlled VLA selected-tail audit. It does not add a hardware or external simulator success claim without artifacts. The goal is to make the submission harder to reject by making the mechanism, baselines, stress failures, claim gates, and final source mapping explicit.

## Claim Thesis

Existing semantic action selection can fail for a specific reason: the selected semantic tail can contain physically invalid actions. Certified TailGuard fixes that reason in controlled finite candidate pools by combining modeled physical certificates, selected-tail calibration, lower confidence bounds, baseline comparisons, fallback, and abstention.

## Development Standard

- Use baselines and stress tests as adversarial teachers.
- Preserve raw semantic, random, N=1, verifier, certificate-only, no-bound, no-adaptive-N, no-fallback, and oracle comparisons.
- Freeze code, seeds, tasks, metrics, baselines, stress conditions, thresholds, and claim gates before final reporting.
- Treat external RoboCasa/LIBERO and optional SmoLVLA artifacts as boundary/status evidence only until reset/step reward, success metrics, action decoding, and TailGuard-connected outcome files exist.

## V4 Artifacts

- `experiments/v4_cached_evidence.py`
- `results/v4_vla_submission_scorecard.csv`
- `results/v4_protocol_freeze_gates.csv`
- `results/v4_iclr_style_rubric_map.csv`
- `results/v4_reviewer_attack_ledger.csv`
- `paper/iclr2026/v4_scorecard_table.tex`
- `paper/iclr2026/v4_protocol_gate_table.tex`
- `paper/iclr2026/v4_rubric_table.tex`
- `paper/iclr2026/v4_attack_ledger_table.tex`
- `scripts/build_v4_paper.py`
- `scripts/run_v4_claim_audit.py`

## Final Gates

- Build `vla-v4.pdf`.
- Remove old Desktop `vla-v3.pdf` and `vla-v2.pdf`.
- Update `PAPER_SOURCE_MAP.md` to `vla-v4.pdf`, `C:\Users\wangz\vla`, and `Jason-Wang313/vla`.
- Run legacy and v4 claim audits, tests, compile checks, and visual PDF QA in the final v4 source-map state.
- Commit, push, and verify remote `main` equals local `HEAD`.
