# Optional Real-VLA / SmoLVLA Adapter

The optional adapter records whether this machine can attempt local SmoLVLA/real-VLA inference.

Command:

```bash
bash scripts/run_optional_vla.sh
```

Outputs:

- `results/optional_vla/adapter_status.json`
- `results/optional_vla/adapter_status.md`

The adapter is guarded and non-blocking. It may return:

- `READY_TO_ATTEMPT`: cached SmoLVLA files and the LeRobot/core runtime are available, but a real inference wrapper still needs to be implemented before benchmark claims are made.
- `SKIPPED_WITH_REASON`: the status artifact explains which runtime or cache component is missing.

Current expected status from prior inspection: cached SmoLVLA config/weights exist, but `lerobot` and `libero` are not importable. This is a software integration gap, not a real-robot hardware gap.

No claim audit should mark real VLA benchmark validation as supported unless the optional adapter actually runs model inference and stores evaluation artifacts.
