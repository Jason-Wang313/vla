# Optional VLA Adapter Status

- status: `READY_TO_ATTEMPT`
- can_attempt_inference: `True`
- reason: Cached SmoLVLA files and LeRobot/core runtime modules are available; use --attempt-inference for a local synthetic action probe, but benchmark evidence still requires a real task wrapper and physical evaluation.

## Runtime Modules
- torch: True
- transformers: True
- huggingface_hub: True
- safetensors: True
- accelerate: True
- num2words: True
- lerobot: True
- libero: False

## Cached Models
- models--lerobot--smolvla_base: exists=True, snapshots=1
- models--lerobot--smolvla_libero: exists=True, snapshots=1
- models--HuggingFaceVLA--smolvla_libero: exists=True, snapshots=1
