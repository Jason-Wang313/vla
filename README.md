# Best-of-N Robot Foundation Models

**When Sampling More VLA Actions Helps or Hallucinates**

This repository studies a VLA-style failure mode: sampling more language-conditioned robot action candidates can improve selected semantic plausibility while hurting real physical utility. We call this **semantic affordance over-selection**.

The core claim is conditional and controlled: for a fixed VLA-style generator/scorer stack, Best-of-N selection stresses the high semantic-score tail. If that tail is misaligned with physical feasibility, selected actions can look more instruction-relevant while becoming less reachable, more collision-prone, or bound to the wrong object. Physical verification and pilot-label calibration can repair the selected tail in these toy settings.

No real-robot validation or large robot foundation model benchmark is claimed in v1.

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

## Key Outputs

- `results/controlled_summary.csv`
- `results/learned_summary.csv`
- `results/distractor_summary.csv`
- `results/repair_summary.csv`
- `results/rendered_summary.csv`
- `results/rendered_visual_metadata.csv`
- `results/robustness_summary.csv`
- `results/claims_status.md`
- `results/optional_vla/adapter_status.md`
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

## Claim Boundaries

Supported v1 claims are controlled VLA-style and learned VLA-style claims. The learned scorer consumes instruction, visual/object observation, and action candidate features, but it is still a small CPU-friendly model.

Unsupported in v1:

- real-robot validation
- large real VLA benchmark validation
- universal VLA deployment recipe
- proof that more samples always help
- proof that grounding always repairs errors

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

Optional real-VLA/SmoLVLA status:

- cached SmoLVLA config/weights are present locally.
- core ML runtime modules are available.
- `lerobot` is not importable, so optional real-VLA inference is skipped with a recorded reason.
- this does not support real VLA benchmark or real-robot claims.

PyTorch learned scorer at `N=128`:

- learned symbolic setting: `torch_semantic` utility `0.329`, violation `0.958`; `torch_calibrated` utility `0.999`, violation `0.000`.
- rendered simulator setting: `torch_semantic` utility `0.455`, violation `1.000`; `torch_calibrated` utility `1.000`, violation `0.000`.
- This strengthens the learned-model evidence while preserving the claim boundary: the scorer is still trained on semantic labels and evaluated separately by physical utility.
