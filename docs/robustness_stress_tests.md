# Robustness Stress Tests

The v2 robustness experiment tests whether repair survives imperfect physical grounding and limited/noisy pilot labels.

Implementation:

- robustness module: `src/vla_best_of_n/robustness.py`
- calibrator: `src/vla_best_of_n/learned_vla.py`
- experiment output: `results/robustness_summary.csv`
- artifact: `results/robustness_artifact.json`
- figures:
  - `results/figures/figure8_robustness_repairs.png`
  - `results/figures/figure9_calibration_budget_noise.png`

## Stress Conditions

Noisy verifier conditions:

- false positives: infeasible actions sometimes look feasible,
- false negatives: feasible actions can be rejected,
- score noise,
- hidden constraints that are partially invisible to the verifier.

Calibration conditions:

- `1%` pilot labels, no label noise,
- `2%` pilot labels, `5%` label noise,
- `5%` pilot labels, `10%` label noise,
- `10%` pilot labels, `20%` label noise.

## Full-Run High-N Result

At `N=128`, raw semantic selection has utility `0.446` and violation rate `1.000`.

- mild noisy verifier: utility `0.581`, violation `0.750`, still `block_high_n`.
- harsh noisy verifier: utility `0.466`, violation `1.000`, still `block_high_n`.
- ideal verifier: utility `1.000`, violation `0.000`.
- calibration ranges from utility `0.684` to `1.000` depending on budget/noise.

This closes the “repair is too clean” gap partially: grounding and calibration are now stress-tested, and the audit records cases where repair is insufficient.
