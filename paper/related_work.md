# Related Work

Vision-language-action models and robot foundation models connect instruction following, perception, and control. Language-conditioned imitation learning trains policies to map instructions and observations to actions. Affordance learning and grounding study whether actions are possible and useful in a scene. Model-based verification and world models can check physical feasibility before acting.

Best-of-N and inference-time scaling improve outputs when the scorer aligns with real utility, but can amplify scorer errors in the selected tail. Calibration and ranker misalignment work studies related failures in learned evaluators.

This project differs from WAM, JEPA, EBM, and diffusion Best-of-N projects. It does not study imagined rollout mismatch, latent compatibility distortion, low-energy tail miscalibration, or diffusion diversity-selection tradeoffs. It studies semantic affordance over-selection in VLA-style action selection.
