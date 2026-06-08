# Limitations

- Evidence is controlled CPU toy evidence.
- Rendered visual scenes and simulator-style geometry improve v2 realism but remain controlled evidence.
- The learned VLA-style models are controlled semantic scorers, including a PyTorch MLP, but still not a deployed robot foundation model.
- No real robot is used.
- No large robot foundation model is evaluated in v1.
- Failure modes are constructed to isolate semantic affordance over-selection.
- The theorem is conditional on a fixed generator/scorer stack.
- Calibration requires pilot real-utility labels.
- Physical verifiers can be wrong.
- V2 noisy-verifier stress tests are still synthetic approximations of real perception and dynamics errors.
- Future benchmark validation is needed.
