# Claims Status

- **SUPPORTED** `theorem claims`: Exact finite tie-aware Best-of-N law is implemented and documented.
  Evidence: src/vla_best_of_n/theory.py; docs/theory.md; tests/test_theory.py
- **SUPPORTED** `controlled VLA-style toy claims`: Controlled VLA-style experiment shows semantic score improves while real utility saturates or drops and violations rise.
  Evidence: results\controlled_summary.csv
- **SUPPORTED** `learned VLA-style claims`: Learned language-conditioned VLA-style scorer exists and exhibits selected-tail regret growth.
  Evidence: results\learned_summary.csv
- **SUPPORTED** `PyTorch learned VLA-style claims`: A heavier PyTorch language/visual/action scorer is trained on semantic labels and evaluated separately from real utility.
  Evidence: results\learned_summary.csv
- **SUPPORTED** `semantic distractor/object-binding claims`: Distractor/object-binding setup measures wrong-object, wrong-target, unreachable, and collision failures.
  Evidence: results\distractor_summary.csv
- **SUPPORTED** `grounding/calibration repair claims`: Grounded or calibrated scorer improves high-N real utility and lowers violations over raw semantic selection.
  Evidence: results\repair_summary.csv
- **SUPPORTED** `rendered visual simulator claims`: Rendered visual scenes with simulator-style physical utility reproduce selected-tail semantic over-selection.
  Evidence: results\rendered_summary.csv
- **SUPPORTED** `noisy verifier robustness claims`: Noisy physical verifiers improve selected-tail utility without being treated as perfect repair.
  Evidence: results\robustness_summary.csv
- **SUPPORTED** `calibration budget/noise claims`: Pilot-label calibration is stress-tested across label budgets and label noise.
  Evidence: results\robustness_summary.csv
- **SUPPORTED** `paper-critical figures`: At least eight paper-critical or v2 figures exist.
  Evidence: results\figures\figure1_semantic_overselection.png, results\figures\figure2_repair_comparison.png, results\figures\figure3_tail_diagnostics.png, results\figures\figure4_failure_decomposition.png, results\figures\figure5_exact_law_validation.png, results\figures\figure6_rendered_visual_simulator.png, results\figures\figure8_robustness_repairs.png, results\figures\figure9_calibration_budget_noise.png, results\figures\figure10_learned_scorer_comparison.png, results\figures\figure7_rendered_scene_examples.png
- **UNSUPPORTED** `optional benchmark claims`: Real VLA benchmark validation is implemented.
  Evidence: docs/benchmark_plan.md marks benchmark validation as future work.
- **SUPPORTED** `optional VLA adapter status claims`: Optional SmoLVLA/real-VLA adapter status is recorded with run-or-skip reason.
  Evidence: results\optional_vla\adapter_status.json
- **UNSUPPORTED** `unsupported future robotics claims`: Real-robot validation is established.
  Evidence: No real-robot adapter or hardware results are included.
- **SUPPORTED** `forbidden/overclaim claims`: Forbidden universal or real-robot claims are absent from README, docs, and paper skeleton.
  Evidence: Text scan over README.md, docs/*.md, paper/*.md.

## Not-Clone Audit
- this_is_not_WAM_clone: True
- this_is_not_JEPA_clone: True
- this_is_not_EBM_clone: True
- this_is_not_diffusion_clone: True
- learned_language_conditioned_model_exists: True
- visual_object_observation_input_exists: True
- semantic_score_separate_from_real_utility: True
- physical_utility_evaluated_separately: True
- object_distractor_reachability_collision_failures_represented: True
- learned_vla_artifact_exists: True
- torch_vla_methods_exist: True
- repair_artifact_exists: True
- rendered_visual_simulator_artifact_exists: True
- robustness_artifact_exists: True
- optional_vla_status_exists: True
- no_real_robot_claim_unless_implemented: True
- no_universal_training_recipe_claim: True
