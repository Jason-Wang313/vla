# V2 Gap Closure Roadmap

## Goal

Push the repository from a strong controlled v1 toward the best feasible paper-quality v2. RAM is not the limiting concern; evidence quality is.

## Current Verified Environment

Available Python packages from inspection:

- `torch`
- `transformers`
- `huggingface_hub`
- `accelerate`
- `safetensors`
- `gymnasium`
- `cv2`
- `PIL`
- `sapien`
- `robosuite`

Not importable at inspection time:

- `lerobot`
- `libero`

Cached local model folders exist:

- `C:\Users\wangz\.cache\huggingface\hub\models--lerobot--smolvla_base`
- `C:\Users\wangz\.cache\huggingface\hub\models--lerobot--smolvla_libero`
- `C:\Users\wangz\.cache\huggingface\hub\models--HuggingFaceVLA--smolvla_libero`

## Best-ROI Order

1. Rendered visual toy benchmark.
2. Simulator-style physical evaluator.
3. Noisy and imperfect physical verifier.
4. Calibration budget/noise ablations.
5. Heavier PyTorch visual-language-action scorer.
6. Optional guarded SmoLVLA/real-VLA adapter.

Items 1-5 now have implemented artifacts. Item 6 has a guarded status adapter and skip artifact. Actual real-VLA inference remains future work.

## Acceptance Criteria For V2

- At least one rendered visual artifact reproduces semantic affordance over-selection.
- At least one simulator-computed utility artifact reproduces selected semantic score rising while physical utility drops or saturates.
- Noisy verifier experiments show repair is not assumed perfect.
- Calibration stress tests show label-budget and label-noise tradeoffs.
- Optional SmoLVLA adapter either runs or writes an exact skip reason.
- Claim audit keeps real-robot validation unsupported unless actual hardware evidence exists.

## Why SmoLVLA Is Troublesome, Not Impossible

Weights and configs appear to be cached, but `lerobot` is not currently importable. Loading `model.safetensors` directly would require reconstructing the LeRobot/SmoLVLA architecture, preprocessing, action normalization, camera/state schema, and postprocessing. That is software integration risk, not a hardware impossibility.

The adapter should be attempted aggressively, but it should not be required for the core `run_all.sh` until it is proven reliable.
