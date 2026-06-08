# Optional Real-VLA / SmoLVLA Adapter

The optional adapter records whether this machine can attempt local SmoLVLA/real-VLA inference.

Command:

```bash
bash scripts/run_optional_vla.sh
```

Heavyweight local inference probe:

```bash
bash scripts/run_optional_vla.sh --attempt-inference
```

Outputs:

- `results/optional_vla/adapter_status.json`
- `results/optional_vla/adapter_status.md`
- `results/optional_vla/inference_probe.json` when `--attempt-inference` is used
- `results/optional_vla/inference_probe.md` when `--attempt-inference` is used

The adapter is guarded and non-blocking. It may return:

- `READY_TO_ATTEMPT`: cached SmoLVLA files and the LeRobot/core runtime are available.
- `SKIPPED_WITH_REASON`: the status artifact explains which runtime or cache component is missing.
- `INFERENCE_PROBE_PASS`: the optional heavyweight probe loaded cached SmoLVLA on CPU and emitted an action chunk from synthetic image/state/language input.
- `INFERENCE_PROBE_FAIL`: the optional heavyweight probe attempted real model inference but failed before producing a finite action chunk.

Current status after the v2 gap-closure pass: cached SmoLVLA config/weights exist, `lerobot` is importable, and the CPU inference probe can run if optional dependencies are installed. `libero` is still not importable, so LIBERO benchmark validation remains unsupported.

No claim audit should mark real VLA benchmark validation as supported unless the optional adapter runs model inference inside an actual benchmark or simulator task with physical success evaluation. The synthetic SmoLVLA action probe is useful plumbing evidence, not robotics validation.
