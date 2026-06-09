# Best-of-N Robot Foundation Models

**When Sampling More VLA Actions Helps or Hallucinates**

This repository studies a VLA-style failure mode: sampling more language-conditioned robot action candidates can improve selected semantic plausibility while hurting real physical utility. We call this **semantic affordance over-selection**.

The core claim is conditional and controlled: for a fixed VLA-style generator/scorer stack, Best-of-N selection stresses the high semantic-score tail. If that tail is misaligned with physical feasibility, selected actions can look more instruction-relevant while becoming less reachable, more collision-prone, or bound to the wrong object. Physical verification and pilot-label calibration can repair the selected tail in these toy settings.

The v2 headline repair is **Certified TailGuard-BoN** (`TailGuard-BoN` in code), an abstaining inference-time controller. It hard-filters candidates through modeled physical certificates, calibrates selected-tail utility from pilot labels, chooses `N`, returns one of `allow_high_n`, `stop_early`, `collect_pilot_labels`, or `block_high_n`, and only allows high-N when a lower confidence bound beats the `N=1` and random-selection baselines by a margin.

Near-100% repair claims are limited to controlled/simulator regimes where the modeled certificate covers the relevant failure modes. Otherwise the method is expected to fall back or abstain.

No real-robot validation or large robot foundation model benchmark is claimed.

## Why This Is VLA-Specific

The scientific object is:

```text
instruction l
visual/object observation o
candidate action or trajectory a
VLA generator G(a | o, l)
semantic affordance score S_sem(o, l, a)
physical feasibility score S_phys(o, a)
calibrated score S_calib
real physical utility R(a)
```

The experiments include object names, colors, categories, receptacles, obstacles, reachability, collisions, fragile/stability constraints, wrong-object errors, and wrong-target errors.

## Not WAM, JEPA, EBM, Or Diffusion

This repository reuses only the finite tie-aware Best-of-N selection law.

- WAM: imagined rollout mismatch and model-error amplification.
- JEPA: latent encoder/predictor compatibility and latent-real rank distortion.
- EBM: low-energy tail miscalibration.
- Diffusion: stochastic trajectory generator diversity and critic-selected outliers.
- This project: semantic affordance over-selection in language-conditioned vision-action selection.

See `docs/differentiation_from_wam_jepa_ebm_diffusion.md`.

## Quickstart

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

The full run writes results to `results/`. The smoke run writes to `results/smoke/`.

Optional, heavyweight local SmoLVLA plumbing probe:

```bash
bash scripts/run_optional_vla.sh --attempt-inference
```

This probe loads cached SmoLVLA on CPU and emits an action chunk for synthetic visual/state/language input. It is not a benchmark or robot validation.

Optional, guarded external benchmark status:

```bash
bash scripts/run_external_benchmark.sh --status-only
bash scripts/run_external_benchmark.sh --attempt-robocasa --attempt-libero --timeout-seconds 10
```

The external runner uses isolated environments under `C:\Users\wangz\external_benchmarks\.venvs`. The latest artifact records RoboCasa integration and target task registration, plus a bounded reset/step attempt that timed out before producing reward or success metrics. It does not support simulator-performance, method-performance, or robot claims.

Anonymous ICLR-style PDF build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_iclr_submission.ps1
```

The script builds `paper/iclr2026/main.tex` and copies the PDF to `C:\Users\wangz\Downloads\certified-tailguard-bon-iclr-submission.pdf`.

## Key Outputs

- `results/controlled_summary.csv`
- `results/learned_summary.csv`
- `results/distractor_summary.csv`
- `results/repair_summary.csv`
- `results/rendered_summary.csv`
- `results/rendered_visual_metadata.csv`
- `results/robustness_summary.csv`
- `results/tailguard_summary.csv`
- `results/tailguard_gate_examples.csv`
- `results/phase_diagram_summary.csv`
- `results/calibration_sample_complexity.csv`
- `results/physics_stress_summary.csv`
- `results/component_ablation_summary.csv`
- `results/failure_honesty_summary.csv`
- `results/claims_status.md`
- `results/optional_vla/adapter_status.md`
- `results/optional_vla/inference_probe.md` when the optional SmoLVLA probe is run
- `results/optional_vla/smolvla_rendered_bridge.json`
- `results/optional_vla/libero_benchmark_status.json`
- `results/external_benchmark_status.json`
- `results/external_benchmark_summary.csv`
- `results/external_benchmark_seed_level/`
- `docs/final_audit.md`

Paper-critical figures:

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
- `results/figures/figure11_tailguard_adaptive_n.png`
- `results/figures/figure12_phase_diagram.png`
- `results/figures/figure13_calibration_sample_complexity.png`
- `results/figures/figure14_imperfect_verifier_map.png`
- `results/figures/figure15_physics_failure_decomposition.png`
- `results/figures/figure16_smolvla_rendered_bridge_status.png`
- `results/figures/figure17_component_ablation.png`
- `results/figures/figure18_failure_honesty.png`
- `results/figures/figure19_external_benchmark_status.png` when the external runner is run

## Claim Boundaries

Supported claims are controlled VLA-style, learned VLA-style, Certified TailGuard-BoN, phase-diagram, calibration sample-complexity, component-ablation, failure-honesty, first-principles physics stress-test, optional model-plumbing, and guarded external-integration claims. The learned scorer consumes instruction, visual/object observation, and action candidate features, but it is still a CPU-friendly controlled model.

Unsupported in v1:

- real-robot validation
- large real VLA benchmark validation
- universal VLA deployment recipe
- proof that more samples always help
- proof that grounding always repairs errors
- real-world safety certification beyond the modeled certificate predicates

Inspect `docs/final_audit.md` and `results/claims_status.md` before citing any claim.

## Key V1 Results

Full run, learned VLA-style raw semantic scorer:

- `N=1`: selected semantic score `0.876`, selected real utility `0.473`, violation rate `0.441`.
- `N=128`: selected semantic score `0.985`, selected real utility `0.152`, violation rate `0.958`.
- High-N regret at `N=128`: `0.848`.
- Tail rank correlation at `N=128`: `-0.039`.
- Deployment gate for raw high-N selection: `block_high_n`.

Repair at `N=128`:

- raw semantic real utility: `0.152`, violation rate `0.958`.
- semantic plus physical real utility: `0.965`, violation rate `0.000`.
- calibrated real utility: `0.999`, violation rate `0.000`.
- grounded combined real utility: `1.000`, violation rate `0.000`.
- oracle real utility: `1.000`.

Rendered visual simulator artifact:

- `N=1`: selected semantic score `0.876`, simulator real utility `0.584`, violation rate `0.644`.
- `N=128`: selected semantic score `0.985`, simulator real utility `0.446`, violation rate `1.000`.
- high-N regret at `N=128`: `0.554`.
- grounded combined repair at `N=128`: real utility `1.000`, violation rate `0.000`.

Robustness stress test at `N=128`:

- raw semantic: utility `0.446`, violation `1.000`, gate `block_high_n`.
- mild noisy verifier: utility `0.581`, violation `0.750`, gate `block_high_n`.
- harsh noisy verifier: utility `0.466`, violation `1.000`, gate `block_high_n`.
- ideal verifier: utility `1.000`, violation `0.000`, gate `stop_early`.
- calibration under stress ranges from utility `0.684` to `1.000`, showing budget/noise sensitivity rather than assuming calibration always works.

Certified TailGuard controlled stress:

- raw fixed high-N remains high-violation in the controlled stress setting.
- Certified TailGuard selects only certified candidates, records certificate failure decomposition, and reaches the controlled target of utility at least `0.98` with violation at most `0.01` in supported runs.
- `results/component_ablation_summary.csv` shows a component-specific failing regime for every named core removal.
- `results/failure_honesty_summary.csv` lists regimes where the method falls back or abstains instead of claiming repair.

Optional real-VLA/SmoLVLA status:

- cached SmoLVLA config/weights are present locally.
- core ML runtime modules and `lerobot` are available.
- optional CPU SmoLVLA inference probe passes on synthetic visual/state/language input.
- rendered bridge status records `action_decode_supported: false` and `physical_success_claimed: false`.
- probe artifact: `results/optional_vla/inference_probe.json`.
- probe model size: `450,046,176` parameters; action chunk shape: `1 x 50 x 6`.
- the guarded external runner records RoboCasa package importability, asset/macros presence, target task registration, and a bounded reset/step timeout.
- an isolated LIBERO environment reports `libero` importability, but no guarded LIBERO reset/step wrapper is implemented.
- this does not support real VLA benchmark or real-robot claims.

PyTorch learned scorer at `N=128`:

- learned symbolic setting: `torch_semantic` utility `0.329`, violation `0.958`; `torch_calibrated` utility `0.999`, violation `0.000`.
- rendered simulator setting: `torch_semantic` utility `0.455`, violation `1.000`; `torch_calibrated` utility `1.000`, violation `0.000`.
- This strengthens the learned-model evidence while preserving the claim boundary: the scorer is still trained on semantic labels and evaluated separately by physical utility.
