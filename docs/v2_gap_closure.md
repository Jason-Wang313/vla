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
- `lerobot`
- `num2words`

Not importable at inspection time:

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

Items 1-5 now have implemented artifacts. Item 6 now has both a guarded readiness artifact and a heavyweight synthetic CPU inference probe: cached SmoLVLA loads through LeRobot and emits a finite action chunk from synthetic visual/state/language input. Actual benchmark evaluation remains future work.

## Acceptance Criteria For V2

- At least one rendered visual artifact reproduces semantic affordance over-selection.
- At least one simulator-computed utility artifact reproduces selected semantic score rising while physical utility drops or saturates.
- Noisy verifier experiments show repair is not assumed perfect.
- Calibration stress tests show label-budget and label-noise tradeoffs.
- Optional SmoLVLA adapter either runs a synthetic action probe or writes an exact skip reason.
- Claim audit keeps real-robot validation unsupported unless actual hardware evidence exists.

## Why SmoLVLA Is Troublesome, Not Impossible

The first blocker was software integration, not hardware impossibility. Installing `lerobot`, pinning `transformers` to the compatible 4.x line, and adding `num2words` allowed cached SmoLVLA to load on CPU and produce an action chunk. The probe uses the LeRobot policy class, tokenizer/processor path, camera/state schema, and cached model weights.

This still does not close the benchmark gap. `libero` is not importable, and no real task environment or physical utility evaluator is connected to SmoLVLA actions. The adapter should remain optional and should not be required for the core `run_all.sh` until a reliable benchmark wrapper exists.
