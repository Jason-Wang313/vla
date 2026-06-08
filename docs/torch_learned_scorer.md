# PyTorch Learned VLA-Style Scorer

The v2 PyTorch scorer is a heavier learned semantic/affordance model than the NumPy ridge scorer.

Implementation:

- `src/vla_best_of_n/torch_vla.py`

Inputs:

- language instruction vector,
- visual/object observation vector,
- rendered visual vector when available,
- action candidate features,
- cross terms between language/visual/action features.

Training target:

- semantic/affordance score, not real physical utility.

Evaluation target:

- real physical utility and simulator utility are measured separately.

Full-run `N=128` result:

- symbolic learned setting: `torch_semantic` utility `0.329`, violation `0.958`.
- symbolic learned setting after calibration: `torch_calibrated` utility `0.999`, violation `0.000`.
- rendered simulator setting: `torch_semantic` utility `0.455`, violation `1.000`.
- rendered simulator setting after calibration: `torch_calibrated` utility `1.000`, violation `0.000`.

This strengthens the learned-model evidence while keeping the paper boundary honest: it is still controlled VLA-style evidence, not real robot validation.
