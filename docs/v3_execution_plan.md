# V3 Execution Plan: Certified TailGuard for VLA Action Selection

## Current Claim

The paper is a VLA-specific selected-tail audit. Its central claim is conditional and controlled: when a VLA-style action selector ranks language-conditioned action candidates by semantic affordance, increasing the candidate budget can raise selected semantic plausibility while lowering selected physical utility and raising violation rates. Certified TailGuard is an abstaining inference-time controller that uses modeled physical certificates, selected-tail calibration, lower confidence bounds, baseline comparisons, and fallback/abstention decisions to prevent unsupported high-budget action selection in controlled candidate pools.

The claim must remain distinct from the other Desktop papers. This is not a generic candidate-budget paper, not a WAM/JEP/EBM/diffusion wrapper, and not a token-codebook paper. The scientific object is instruction-conditioned vision-language-action selection: instruction, visual/object observation, action candidate, semantic affordance score, modeled physical certificate, calibrated selected-tail utility, gate decision, and real physical utility.

## Current Gaps

- The current ICLR PDF is only 6 pages, far below the 25-page submission-readiness target.
- The repo already has strong results, but the current manuscript exposes only a small fraction of the available evidence.
- The main text needs clearer distinction between semantic affordance over-selection, Certified TailGuard, rendered simulator evidence, robustness stress, external integration status, and unsupported robotics claims.
- Many CSV/JSON artifacts are not converted into paper tables, so a reviewer would need to inspect the repo manually.
- External RoboCasa/LIBERO and SmoLVLA artifacts must be framed as integration/plumbing status only, not benchmark or robot success.
- The exact-law section currently uses sampling without replacement; the manuscript must keep this consistent with the implementation and tests.
- The current appendix is too short and does not contain enough claim-evidence, ablation, failure-honesty, or reproducibility detail.

## Target Evidence And Experiments

Use existing full cached artifacts as the core evidence, avoiding heavy reruns unless a check fails:

- `results/learned_summary.csv`: learned VLA-style semantic scorer selected-tail failure.
- `results/rendered_summary.csv`: rendered visual simulator selected-tail failure.
- `results/repair_summary.csv`: physical, calibrated, grounded, PyTorch repair comparison.
- `results/robustness_summary.csv`: noisy verifier and calibration budget/noise failure modes.
- `results/tailguard_summary.csv`: Certified TailGuard high-budget gate behavior.
- `results/tailguard_gate_examples.csv`: all four public decisions.
- `results/component_ablation_summary.csv`: component-specific ablation failures.
- `results/failure_honesty_summary.csv`: fallback/abstention regimes.
- `results/phase_diagram_summary.csv`: semantic/physical misalignment and distractor-salience sweeps.
- `results/calibration_sample_complexity.csv`: pilot-label budget and label-noise curves.
- `results/physics_stress_summary.csv`: first-principles physical failure decomposition.
- `results/optional_vla/inference_probe.json`: SmoLVLA synthetic CPU action probe status.
- `results/optional_vla/smolvla_rendered_bridge.json`: no action decoder / no physical success claim.
- `results/external_benchmark_status.json` and summary CSV: RoboCasa/LIBERO integration status, no success metric.
- `results/claims_status.md`: repository claim boundary and not-clone audit.

If new v3 artifacts are needed, generate only derived evidence tables/figures from these cached full artifacts, not smoke-mode overwrites.

## Baselines

The final paper must explicitly compare Certified TailGuard against:

- raw semantic high-budget selection;
- `N=1` baseline;
- random high-budget selection;
- verifier-filtered high-budget selection;
- calibrated high-budget selection without certificates;
- certificate-only filtering without selected-tail calibration;
- TailGuard without lower confidence bound;
- TailGuard without random-baseline check;
- TailGuard without `N=1` baseline check;
- oracle high-budget selection.

The main narrative should not make the repair look free: calibration spends pilot labels; physical certificates are modeled; verifier quality can be noisy; abstention/fallback are part of the method.

## Ablations

Expose the component ablation artifact as a table and narrative:

- no physical certificate;
- no verifier score;
- no pilot labels;
- no empirical lower bound;
- no adaptive `N`;
- no abstention/fallback.

For each removal, state the stress regime where it fails controlled acceptance while the full controller passes. This is important because it defends TailGuard against the attack that it is only a stack of redundant heuristics.

## Stress Tests

The v3 manuscript should add substantive sections for:

- rendered visual simulator stress;
- noisy verifier stress;
- calibration budget and label-noise stress;
- semantic/physical misalignment phase diagram;
- distractor salience and object-binding failures;
- hidden obstacles and partial observability;
- object identity spoofing;
- wrong high-plausibility receptacles;
- fragile/heavy ambiguity;
- correlated pilot-label noise;
- train/pilot/test shift;
- low certified-candidate-count scenes.

The paper should state both positive and negative outcomes: TailGuard succeeds under modeled coverage, but can block, stop early, collect labels, or abstain when the evidence is insufficient.

## Figures And Tables

Reuse the existing 19 paper-critical figures where useful, but do not overcrowd the main text. Add v3 generated LaTeX tables from cached artifacts:

- compact claim-to-artifact scorecard;
- selected-tail failure table for learned, PyTorch, and rendered settings;
- repair and TailGuard comparison table;
- full `N` curves for learned/rendered/repair/robustness summaries;
- TailGuard baselines/ablations table;
- phase diagram summary table;
- calibration sample-complexity table;
- physics stress/failure-decomposition table;
- component ablation table;
- failure-honesty table;
- external and optional VLA boundary table;
- 50-round self-attack ledger.

Figures should be copied or referenced from `results/figures/`. The paper can use a curated set in the main text and place the rest in the appendix.

## Writing Expansion

Expand the manuscript into a full 25+ page submission-style paper:

- Abstract: emphasize semantic affordance over-selection, Certified TailGuard, controlled evidence, and external boundary.
- Introduction: motivate VLA candidate selection and why semantic plausibility is not a high-budget safety certificate.
- Theory: tie-aware finite selected-tail law and semantic-score no-free-lunch proposition.
- Method: physical certificates, calibrator, empirical lower bounds, baseline comparisons, public gate decisions.
- Experiments: learned scorer, PyTorch scorer, rendered simulator, repair, TailGuard stress, robustness, ablations, phase diagrams, external boundary.
- Related work: VLA models, affordance grounding, verifier/reranking, selective prediction, conformal or empirical Bernstein bounds, robotics benchmarks.
- Limitations: controlled evidence, idealized certificates, no benchmark success, no hardware validation, pilot-label assumptions.
- Reproducibility: commands, artifacts, source map, final build script, RAM-light workflow.
- Appendix: complete evidence tables, attack ledger, artifact map, claim checklist, external benchmark gate, LLM usage.

Avoid duplicate-paper language. The words and sections should read unmistakably like VLA robotics/action-selection work.

## Page-Count Strategy

The final PDF must be at least 25 pages. The length must come from real content:

- full tables from cached results;
- detailed method specification;
- ablation/failure-honesty analysis;
- external boundary documentation;
- reproducibility and artifact map;
- 50-round self-attack ledger.

Do not pad with empty prose. If the rebuilt PDF is still short, add more artifact-backed appendix tables or reviewer-facing analysis of actual VLA stress artifacts.

## RAM-Light Execution Strategy

- Prefer cached full artifacts already in `results/`.
- Avoid `bash scripts/run_all.sh` unless evidence is missing, because it is slow and can overwrite full artifacts.
- Avoid optional SmoLVLA inference reruns unless the existing JSON is missing or inconsistent; it is heavyweight CPU work.
- Generate derived v3 evidence tables with streaming CSV reads and small pandas frames.
- Keep visual PDF QA to selected representative pages.
- Run final checks sequentially or in parallel only for low-memory commands: claim audit, pytest, compileall, diff check, PDF hash/page checks.

## Final Acceptance Checklist

- `docs/v3_execution_plan.md` exists before manuscript/artifact edits.
- Final PDF is at least 25 pages.
- Final repo PDF is `paper/final/vla-v3.pdf`.
- Desktop PDF is `C:\Users\wangz\OneDrive\Desktop\vla-v3.pdf`.
- Old Desktop `vla-v2.pdf` is absent.
- `PAPER_SOURCE_MAP.md` maps `vla-v3.pdf` to `C:\Users\wangz\vla` and `Jason-Wang313/vla`.
- Repo and Desktop final PDFs have identical SHA-256 hashes.
- The v3 self-attack ledger contains exactly 50 rounds with pass or bounded outcomes.
- The manuscript contains no real-robot validation, real benchmark success, universal VLA recipe, or external physical-success claim.
- Existing `bash scripts/run_claim_audit.sh` passes.
- `python scripts/run_v3_claim_audit.py` passes.
- `pytest` passes.
- `python -m compileall src experiments scripts tests -q` passes.
- `git diff --check` passes, allowing only CRLF warnings if Git reports them.
- Visual QA checks title, main evidence table, TailGuard section, appendix tables, attack ledger, and final checklist pages.
- Final commit is pushed to GitHub `main`, and the remote SHA is verified.
