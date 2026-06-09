# Claims Status

- **SUPPORTED** `theorem claims`: Exact finite tie-aware Best-of-N law, selected-tail principle, and semantic-score no-free-lunch examples are implemented and documented.
  Evidence: src/vla_best_of_n/theory.py; docs/theory.md; tests/test_theory.py
- **SUPPORTED** `exact-law validation claims`: Every main experiment compares exact finite-law selected utility with Monte Carlo estimates and confidence intervals.
  Evidence: controlled/learned/distractor/repair/rendered/robustness summary CSVs.
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
- **SUPPORTED** `Certified TailGuard method claims`: Certified TailGuard-BoN uses hard physical certificates, tail lower bounds, fallback/abstention metadata, and reaches >=0.98 utility with <=0.01 violation in controlled stress.
  Evidence: results\tailguard_summary.csv; results\tailguard_artifact.json; results\tailguard_gate_examples.csv
- **SUPPORTED** `Certified TailGuard ablation claims`: Component ablations cover certificates, verifier score, pilot labels, empirical lower bound, adaptive N, and abstention/fallback, with each named removal failing controlled acceptance in at least one stress regime.
  Evidence: results\component_ablation_summary.csv
- **SUPPORTED** `phase diagram claims`: Semantic/physical misalignment and distractor salience phase diagrams are generated with TailGuard outcomes.
  Evidence: results\phase_diagram_summary.csv
- **SUPPORTED** `calibration sample complexity claims`: Tail calibration sample-complexity curves cover label budgets and label noise, lower bounds tighten as labels increase, and scarce selected-tail evidence triggers pilot-label collection.
  Evidence: results\calibration_sample_complexity.csv
- **SUPPORTED** `first-principles physics claims`: Geometry-based certificates and adversarial full-scene stress families are evaluated with utility, violation, fallback/abstention, and certificate failure decomposition.
  Evidence: results\physics_stress_summary.csv
- **SUPPORTED** `failure honesty claims`: Stress artifacts include regimes where Certified TailGuard abstains or falls back instead of pretending to repair unsafe high-N selection.
  Evidence: results\failure_honesty_summary.csv
- **SUPPORTED** `paper-critical figures`: At least eight paper-critical or v2 figures exist.
  Evidence: results\figures\figure1_semantic_overselection.png, results\figures\figure2_repair_comparison.png, results\figures\figure3_tail_diagnostics.png, results\figures\figure4_failure_decomposition.png, results\figures\figure5_exact_law_validation.png, results\figures\figure6_rendered_visual_simulator.png, results\figures\figure8_robustness_repairs.png, results\figures\figure9_calibration_budget_noise.png, results\figures\figure10_learned_scorer_comparison.png, results\figures\figure11_tailguard_adaptive_n.png, results\figures\figure12_phase_diagram.png, results\figures\figure14_imperfect_verifier_map.png, results\figures\figure13_calibration_sample_complexity.png, results\figures\figure15_physics_failure_decomposition.png, results\figures\figure17_component_ablation.png, results\figures\figure18_failure_honesty.png, results\figures\figure16_smolvla_rendered_bridge_status.png, results\figures\figure7_rendered_scene_examples.png
- **SUPPORTED** `optional benchmark boundary claims`: Real VLA benchmark validation is not implemented and is not claimed.
  Evidence: docs/benchmark_plan.md marks benchmark validation as future work; results\optional_vla\libero_benchmark_status.json status=SKIPPED_WITH_REASON.
- **SUPPORTED** `external benchmark artifact gate claims`: External simulator claims are artifact-gated: integration, reset/step/reward, exposed success metric, physical success, and TailGuard method success are separate claim levels.
  Evidence: results\external_benchmark_status.json exists=True, reset_step_reward=False, success_metric=False, physical_success_claimed=False.
- **SUPPORTED** `optional VLA adapter status claims`: Optional SmoLVLA/real-VLA adapter status is recorded with run-or-skip reason.
  Evidence: results\optional_vla\adapter_status.json
- **SUPPORTED** `optional VLA inference probe claims`: Cached SmoLVLA can be loaded locally and emit an action chunk from synthetic visual/state/language input.
  Evidence: results\optional_vla\inference_probe.json
- **SUPPORTED** `optional SmoLVLA rendered bridge claims`: Optional SmoLVLA rendered-input bridge status is recorded, without claiming decoded physical success.
  Evidence: results\optional_vla\smolvla_rendered_bridge.json
- **SUPPORTED** `unsupported future robotics boundary claims`: Real-robot validation is not established and is not claimed.
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
- tailguard_artifact_exists: True
- phase_diagram_exists: True
- calibration_sample_complexity_exists: True
- physics_stress_exists: True
- component_ablation_exists: True
- failure_honesty_exists: True
- optional_vla_status_exists: True
- optional_vla_inference_probe_passes: True
- optional_smolvla_rendered_bridge_status_exists: True
- optional_libero_benchmark_status_exists: True
- external_benchmark_status_exists: True
- external_benchmark_step_reward_artifact_exists: False
- external_success_metric_artifact_exists: False
- external_physical_success_claimed: False
- no_real_robot_claim_unless_implemented: True
- no_universal_training_recipe_claim: True
