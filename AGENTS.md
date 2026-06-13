# Agent Handoff Notes

Repository: `C:\Users\wangz\score-tail vla`

Primary package: `src/vla_tailguard_audit`

Project title: **Certified TailGuard for VLA Action Selection: Auditing Semantic Affordance Over-Selection**

## Current Thesis

score-tail VLA action selection can amplify semantic/affordance ranking errors. As `N` grows, selected language/vision plausibility can improve while real physical utility saturates or drops. Physical grounding, simulator-style verification, or calibrated real-utility scoring should be evaluated by whether they repair the selected high-score tail.

## Current State

- v1 controlled and learned VLA-style experiments exist and pass.
- The repo is a git repository.
- Initial checkpoint commit: `7a8cc45`.
- The folder name intentionally contains spaces; keep shell scripts quoted/path-safe.
- Real-robot validation is unsupported.
- Optional real-VLA/SmoLVLA status is implemented as a guarded artifact. Cached SmoLVLA now loads on CPU and emits a synthetic action chunk via `bash scripts/run_optional_vla.sh --attempt-inference`; LIBERO/real benchmark validation is still unsupported.

## Required Commands

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

## Before Making Claims

Read:

- `docs/CODEX_HANDOFF.md`
- `docs/final_audit.md`
- `results/claims_status.md`
- `docs/v2_gap_closure.md`

Treat source files and generated artifacts as authoritative. If the handoff conflicts with the repo, trust the repo.

## Important Boundaries

- Do not claim real-robot validation.
- Do not claim real VLA benchmark validation unless `results/optional_vla/` contains a successful run artifact and claim audit is updated.
- Do not collapse this into WAM/JEPA/EBM/diffusion framing; the object is VLA semantic affordance over-selection.
