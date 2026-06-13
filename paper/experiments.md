# Experiments

## Controlled Semantic Affordance Over-Selection

Synthetic manipulation/navigation scenes include objects, colors, categories, receptacles, obstacles, reachability, collision, fragile, heavy, and tool constraints. The generator produces candidate actions such as grasping an object, moving to a receptacle, placing an object, or using a tool. The semantic scorer rewards language plausibility, object/category/receptacle matches, and affordance priors. Real utility separately evaluates physical success.

## Learned VLA-Style Model

A learned scorer consumes instruction features, visual/object observation features, and action candidate features. V2 includes both a NumPy scorer and a heavier PyTorch MLP scorer. Both are trained on semantic/affordance labels, not real physical utility. Real utility is evaluated separately.

## Distractors And Object Binding

Scenes include similar objects, target/distractor conflicts, salient wrong objects, unreachable plausible objects, correct object with wrong receptacle, and blocked trajectories.

## Rendered Visual Simulator

The v2 benchmark renders RGB scene observations with target objects, distractors, receptacles, robot pose, reachability regions, and obstacles. A simulator-style evaluator recomputes real physical utility from geometry and constraints rather than relying only on symbolic success labels. This closes part of the purely symbolic toy gap while remaining controlled evidence.

## Repair

Raw semantic scoring is compared with semantic plus physical verification, calibrated scoring from pilot real-utility labels, grounded combined scoring, random selection, and oracle real-utility selection.

## Robustness Stress Tests

V2 adds noisy physical verifiers with false positives, false negatives, hidden constraints, and score noise. It also evaluates calibration under pilot-label budget and label-noise stress. These experiments test whether repair actually fixes the selected tail rather than assuming a perfect feasibility oracle.

## Certified TailGuard Adaptive N

Certified TailGuard is compared with raw fixed high-N selection, `N=1`, random high-N, verifier-filtered high-N, calibrated high-N without certificates, certificate-only filtering without tail calibration, TailGuard without lower-confidence bounds, TailGuard without random-baseline checks, TailGuard without `N=1` checks, and oracle selection. The key outputs are adaptive selected `N`, selected real utility, violation rate, certified candidate count, certificate pass, certified selected utility, certified violation rate, fallback rate, abstention rate, confidence radius, gate decision, and gate rates across scenes.

Constructed gate examples separately exercise all four controller outputs: allowing high-N when the calibrated selected tail dominates baselines, stopping early when the tail adds no lower-bound value, requesting pilot labels when selected-tail evidence is too scarce, and blocking high-N when the lower bound falls below baselines.

## Phase Diagrams

V2 sweeps semantic/physical misalignment, distractor salience, obstacle density, hidden constraints, verifier false positives, verifier false negatives, and label noise. The phase diagram records raw high-N utility, TailGuard utility, oracle utility, random utility, violation rates, TailGuard gain over raw, and gate decisions.

## Calibration Sample Complexity

Pilot-label budgets `{0.5%, 1%, 2%, 5%, 10%, 20%}` are crossed with label noise `{0, 0.05, 0.10, 0.20}`. These runs test whether TailGuard asks for more labels when the selected-tail lower bound is too wide and whether bounds tighten as labels increase.

## Component Ablations

The component ablation table removes one element at a time: physical certificate, verifier score, pilot labels, empirical lower bound, adaptive `N`, or abstention/fallback. The controlled acceptance target is Certified TailGuard utility at least `0.98` and violation at most `0.01`. Each named removal must fail that threshold in at least one component-specific stress regime while the full controller passes.

## First-Principles Physics And Adversarial Stress

The physics stress tests decompose failures into reach envelopes, swept-volume collision, receptacle compatibility, stability margin, fragile/heavy handling, blocked paths, hidden obstacles, verifier false positives concentrated in the semantic tail, partial observation, object identity spoofing, wrong receptacles with high semantic plausibility, fragile/heavy ambiguity, correlated pilot-label noise, train/pilot/test shift, and low certified-candidate-count scenes. Each family reports raw high-N utility, Certified TailGuard utility, violation rate, abstention rate, fallback rate, certificate failure decomposition, and gate distribution.

The failure-honesty table lists regimes where the method abstains or falls back instead of pretending to repair high-N selection.

## Exact-Law Validation

For every main experiment, exact finite-pool selected utility is compared with Monte Carlo estimates and confidence intervals.

## Guarded External Benchmark Status

The external bridge is optional and artifact-gated. The runner uses isolated environments under `external_benchmarks/.venvs`, records package versions, asset/macros status, task names, action space, observation keys, reset status, step status, reward traces, success traces, policy kind, TailGuard connection status, and explicit skip reasons.

The current RoboCasa artifact uses `robocasa/PickPlaceCounterToCabinet` with split `pretrain`. It confirms that RoboCasa imports, assets/macros are present, and the target task is registered. The bounded reset/step attempt timed out before reward or success metrics were produced. The current LIBERO row records import status only; no guarded reset/step wrapper is implemented. Therefore the paper should use the controlled-only evidence variant, not an upgraded external-simulator variant.
