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

## Exact-Law Validation

For every main experiment, exact finite-pool selected utility is compared with Monte Carlo estimates and confidence intervals.
