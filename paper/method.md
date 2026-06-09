# Method

We model a VLA-style stack with instruction `l`, visual/object observation `o`, action or trajectory `a`, generator `G(a | o, l)`, semantic score `S_sem(o, l, a)`, physical feasibility score `S_phys(o, a)`, calibrated score `S_calib`, and real utility `R(a)`.

For a fixed finite candidate pool, Best-of-N samples `N` candidates without replacement and selects the highest-scoring candidate with uniform tie breaking. The exact law computes selected real utility by summing candidate selection probabilities times `R(a_i)`.

Diagnostics include selected semantic score, selected real utility, physical feasibility, exact-law prediction error, semantic-real tail gap, top-score-tail utility, overall and tail rank correlation, high-N regret, oracle gap, physical violation rate, wrong-object and wrong-target rates, collision and reachability failures, marginal value per added candidate, seed variance, and a deterministic deployment gate.

Baselines include raw fixed-N BoN, `N=1`, random high-N, verifier-filtered high-N, calibrated high-N without certificates, certificate-only filtering without tail calibration, TailGuard ablations without lower confidence bounds or baseline checks, and oracle selection.

## Certified TailGuard-BoN

Certified TailGuard-BoN is the v2 inference-time controller. The implementation-compatible short name remains `TailGuard-BoN`. Inputs are a finite candidate pool, semantic scores, optional verifier scores, pilot real-utility labels, simulator flags or modeled-physics checks, and a candidate `N` grid.

The controller has four layers.

1. Certificate layer: candidates must pass modeled checks for reach envelope, swept-volume collision, receptacle compatibility, stability margin, fragile/heavy handling, tool/object match, blocked-path constraints, and hidden modeled obstacles when present. Failing candidates are hard-rejected from certified selection.
2. Tail calibration: a bounded real-utility calibrator is fit from pilot labels using semantic and verifier features.
3. Lower-bound gating: selected-tail utility curves are predicted with the exact finite law and converted to empirical lower confidence bounds.
4. Fallback/abstention: high-N is allowed only when the lower bound beats both `N=1` and random baselines by a margin. Otherwise the controller stops early, falls back to the best certified low-N candidate, blocks high-N, or returns `collect_pilot_labels`.

Public result fields include `certificate_pass`, `certificate_failure_types`, `certified_candidate_count`, `fallback_used`, `abstention_reason`, `certified_selected_utility`, and `certified_violation_rate`, in addition to selected `N`, selected index, calibrated score, predicted utility curve, lower-confidence curve, gate decision, reason code, pilot-label count, and tail-label count.

The method is deliberately conditional. A certificate is evidence only for the modeled physics predicates it covers. When too few certified candidates or too few selected-tail labels remain, the correct behavior is fallback or abstention, not an unsafe high-N repair claim.
