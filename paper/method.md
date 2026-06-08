# Method

We model a VLA-style stack with instruction `l`, visual/object observation `o`, action or trajectory `a`, generator `G(a | o, l)`, semantic score `S_sem(o, l, a)`, physical feasibility score `S_phys(o, a)`, calibrated score `S_calib`, and real utility `R(a)`.

For a fixed finite candidate pool, Best-of-N samples `N` candidates without replacement and selects the highest-scoring candidate with uniform tie breaking. The exact law computes selected real utility by summing candidate selection probabilities times `R(a_i)`.

Diagnostics include selected semantic score, selected real utility, physical feasibility, exact-law prediction error, semantic-real tail gap, top-score-tail utility, overall and tail rank correlation, high-N regret, oracle gap, physical violation rate, wrong-object and wrong-target rates, collision and reachability failures, marginal value per added candidate, seed variance, and a deterministic deployment gate.

Repair methods include physical feasibility filtering, semantic plus physical scoring, learned calibration from pilot real-utility labels, combined grounded scoring, random selection, and oracle selection.
