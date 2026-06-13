# Limitations

- Evidence is controlled CPU toy evidence.
- Rendered visual scenes and simulator-style geometry improve v2 realism but remain controlled evidence.
- The learned VLA-style models are controlled semantic scorers, including a PyTorch MLP, but still not a deployed robot foundation model.
- No real robot is used.
- No real benchmark success signal or real-robot success signal is produced.
- Failure modes are constructed to isolate semantic affordance over-selection.
- The theorem is conditional on a fixed generator/scorer stack.
- Calibration requires pilot real-utility labels.
- Certified TailGuard inherits any blind spots in the modeled certificate predicates, pilot labels, verifier scores, and selected-tail coverage.
- Physical verifiers can be wrong.
- A certificate only covers modeled physics; unmodeled perception, dynamics, contact, calibration, or hardware failures are outside the guarantee.
- Near-100% repair claims apply only to controlled/simulator regimes where the certificate covers the relevant failure modes.
- V2 noisy-verifier stress tests are still synthetic approximations of real perception and dynamics errors.
- Phase diagrams and first-principles physics sweeps are constructed stress tests, not representative coverage of real manipulation.
- Future benchmark validation is needed.
- Optional SmoLVLA artifacts record rendered-bridge status and synthetic action emission only; `action_decode_supported` and `physical_success_claimed` remain false unless an actual action-to-evaluator mapping is implemented and run.
- The guarded RoboCasa/LIBERO runner currently supports external integration status only. RoboCasa reset/step timed out in the bounded probe, and no reward trace, exposed success metric, physical outcome, or TailGuard method outcome is available from an external simulator.
