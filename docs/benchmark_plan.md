# Benchmark Plan

No real VLA or real-robot benchmark claim is made in v1.

A v2 benchmark adapter should be guarded behind optional dependencies and should never block the controlled CPU runs. Candidate paths:

- Wrap a local OpenVLA/RT-style policy if weights and dependencies are already installed.
- Add a LIBERO-style or Gymnasium manipulation adapter if it can run locally.
- Render the controlled object scenes as image observations and compare text/object binding failures.
- Evaluate selected actions with a simulator-side feasibility checker.

Minimum evidence before upgrading claims:

- Real model or benchmark version recorded.
- Prompt/instruction set recorded.
- Candidate generation and reranking interface documented.
- Real utility or simulator success metric measured separately from semantic score.
- Seed-level or episode-level outputs saved.
- Claim audit updated from `UNSUPPORTED` to `PARTIAL` or `SUPPORTED` only for actually run benchmarks.
