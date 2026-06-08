# Current State Before V1

## What Exists

This repository was initially created at `C:\Users\wangz\vla-best-of-n` and later renamed to `C:\Users\wangz\best of n vla`. At the start of v1, only the requested directory scaffold existed: `docs/`, `paper/`, `src/vla_best_of_n/`, `experiments/`, `scripts/`, `tests/`, `results/`, and `results/figures/`.

Nearby Best-of-N repositories exist for WAM, JEPA, and diffusion-policy projects. They can inform lightweight command and packaging conventions only. They must not determine this repository's scientific object or claims.

## What Is Missing

- Exact finite tie-aware Best-of-N implementation and tests.
- VLA-specific scene/action generator with language, object observations, candidate actions, semantic scorer, physical scorer, calibrated scorer, and real utility.
- Controlled semantic affordance over-selection experiment.
- Learned language-conditioned VLA-style scorer.
- Distractor/object-binding experiment.
- Grounding/calibration repair experiment.
- Claim audit, final audit, paper skeleton, figures, and README.

## Strongest Possible VLA-Specific Contribution

The strongest contribution is a selected-tail diagnostic for language-conditioned robot action selection: Best-of-N can amplify semantic affordance ranking errors when the upper semantic-score tail contains actions that are language/vision plausible but physically infeasible. The paper should show that selected semantic plausibility rises with N while true physical utility saturates or falls, and that physical verification or calibrated real-utility scoring repairs that selected tail in controlled VLA-style settings.

## Risk Of Looking Like A WAM/JEPA/EBM/Diffusion Clone

The main risk is reusing generic Best-of-N misranking language without making the VLA stack concrete. This repository must keep the object distinct: language instruction `l`, visual/object observation `o`, action/trajectory `a`, VLA generator `G(a | o, l)`, semantic affordance score `S_sem(o, l, a)`, optional physical feasibility score `S_phys(o, a)`, calibrated score `S_calib`, and real physical utility `R(a)`.

The failure cannot be model-error amplification, latent-tail hallucination, low-energy miscalibration, or diffusion diversity-selection tradeoff. It must be semantic affordance over-selection: high-N selection chooses candidates that look more semantically appropriate while becoming physically worse.

## Top 10 Improvements Ranked By Paper Impact

1. Implement the exact finite tie-aware Best-of-N law and validate it empirically.
2. Build a controlled VLA-style manipulation/navigation environment with object, receptacle, reachability, collision, and stability failures.
3. Show the core artifact: semantic score improves with N while real utility drops and violation rate rises.
4. Add a learned language-conditioned VLA-style scorer whose semantic score is separate from real utility.
5. Add object-binding/distractor experiments with wrong-object, wrong-target, unreachable, and collision decomposition.
6. Add physical feasibility, calibrated scoring, combined scoring, random, and oracle baselines.
7. Implement selected-tail diagnostics and a deterministic deployment gate.
8. Generate paper-critical figures and seed-level CSV/JSON evidence.
9. Write honest claim audit and final audit documents that mark real-robot claims unsupported.
10. Write the paper skeleton and differentiation document so the repository is legible to reviewers.
