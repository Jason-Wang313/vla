# Paper Variants

## Controlled-Only Variant

Use this as the current submission stance. It claims the selected-tail theorem, semantic affordance over-selection in controlled and learned VLA-style settings, Certified TailGuard, stress tests, ablations, and optional integration-status artifacts. It does not claim external simulator reward/success evidence.

## External-Simulator-Upgraded Variant

Use only if `results/external_benchmark_status.json` records reset ok, step ok, reward traces, exposed success metrics, and a documented action/control mapping. If TailGuard is claimed on the external simulator, the artifact must also contain TailGuard-connected policy rows and seed-level outcome files.

## Current Choice

Current artifacts justify the controlled-only variant. RoboCasa integration is visible, but reset/step did not complete within the bounded probe.
