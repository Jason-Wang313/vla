# Differentiation From Other score-tail Projects

## Core Distinction

This repository studies language-conditioned vision-action selection. The scientific object is a VLA-style generator and scorer stack:

```text
l, o, optional x, G(a | o, l), S_sem(o, l, a), S_phys(o, a), S_calib, R(a)
```

The failure mode is semantic affordance over-selection: high-N selection finds actions that look increasingly language/vision plausible while becoming physically worse.

## Not WAM

WAM studies imagined rollouts and imagined-vs-real dynamics mismatch. Its failure is model-error amplification. This project does not use imagined rollout quality as the object; it studies semantic affordance ranking over language-conditioned robot action candidates.

## Not JEPA

JEPA studies latent encoders, latent predictors, and latent compatibility scores. Its failure is latent-real rank distortion. This project does not audit latent compatibility; it audits semantic and physical ranking in object-conditioned VLA-style action selection.

## Not EBM

EBM studies energy functions over actions or trajectories. Its failure is low-energy tail miscalibration. This project does not define success as low energy; it separates semantic affordance plausibility from real physical utility.

## Not Diffusion

Diffusion-policy work studies stochastic trajectory generators and critic-selected outliers. Its failure emphasizes diversity-selection tradeoffs. This project uses score-tail selection pressure only as a finite selection law and focuses on language/object binding, semantic distractors, and physical feasibility failures.

## Not-Clone Checklist

- Reused math: only the finite tie-aware score-tail law.
- New scientific object: VLA-style language, observation, action, semantic score, physical utility stack.
- New failure: semantic affordance over-selection.
- New experiments: language-conditioned object/receptacle scenes, distractors, reachability, collision, wrong-object and wrong-target failures.
- Forbidden clone patterns absent: no imagined rollout mismatch, no latent predictor, no energy model, no diffusion diversity framing.
