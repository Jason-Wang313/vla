# Benchmark Plan

No real VLA or real-robot benchmark claim is made in v1.

A v2 benchmark adapter should be guarded behind optional dependencies and should never block the controlled CPU runs. Candidate paths:

- Wrap a local OpenVLA/RT-style policy if weights and dependencies are already installed.
- Add a LIBERO-style or Gymnasium manipulation adapter if it can run locally.
- Render the controlled object scenes as image observations and compare text/object binding failures.
- Evaluate selected actions with a simulator-side feasibility checker.

Current guarded status artifacts:

- `results/optional_vla/adapter_status.json` records dependency/cache readiness.
- `results/optional_vla/inference_probe.json` records the optional synthetic SmoLVLA action probe when it is explicitly run.
- `results/optional_vla/smolvla_rendered_bridge.json` records whether rendered toy observations have been bridged into SmoLVLA and whether actions were decoded into the toy evaluator. In the core status, `action_decode_supported` is `false` and `physical_success_claimed` is `false`; decoded physical success is not claimed.
- `results/optional_vla/libero_benchmark_status.json` records whether LIBERO is importable and whether a benchmark wrapper was skipped.
- `results/external_benchmark_status.json` records isolated RoboCasa/LIBERO importability, task registration, reset/step status, reward traces, success traces, action space, observation keys, policy kind, TailGuard connection status, and explicit claim levels.
- `results/external_benchmark_summary.csv` and `results/external_benchmark_seed_level/` record row-level and seed-level external status.

Latest guarded external status:

- RoboCasa environment: `robocasa/PickPlaceCounterToCabinet`, split `pretrain`.
- Isolated Python: `C:\Users\wangz\external_benchmarks\.venvs\robocasa\Scripts\python.exe`.
- Importability: `robocasa`, `gymnasium`, `robosuite`, `lerobot`, `torch`, and `transformers` import in the RoboCasa venv.
- Assets/macros: RoboCasa source root, model assets, and `macros_private.py` are present.
- Task registration: the target environment is registered and 396 RoboCasa Gymnasium tasks are visible.
- Reset/step: a bounded attempt timed out before producing reward or success traces, so benchmark stepping is not claimed.
- LIBERO: the isolated `libero310` environment imports `libero`, `robosuite`, `lerobot`, `torch`, and `transformers`, but lacks `gymnasium` and has no guarded reset/step wrapper here.
- TailGuard connection: no action-schema mapping from controlled TailGuard candidates to RoboCasa controls is implemented.

Minimum evidence before upgrading claims:

- Real model or benchmark version recorded.
- Prompt/instruction set recorded.
- Candidate generation and reranking interface documented.
- Real utility or simulator success metric measured separately from semantic score.
- Seed-level or episode-level outputs saved.
- Claim audit updated only for claim levels that have actually run benchmark artifacts.
- `results/external_benchmark_status.json` must show reset, step, reward trace, and exposed success metric before any external simulator outcome is discussed as evidence.
- `physical_success_claimed` must remain `false` unless a real policy/control mapping and benchmark success signal are run and saved.
