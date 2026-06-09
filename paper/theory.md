# Theory

The theorem audits a fixed finite VLA generator/scorer stack. It does not say whether VLAs are good or bad universally.

## Exact Selected-Tail Law

The exact finite tie-aware law assigns each candidate a selection probability under sampling without replacement and uniform tie breaking among top-score candidates. Expected selected real utility is the dot product of these probabilities with real utility.

## Selected-Tail Principle

High-N utility is governed by `E[R | S in the selected upper tail]`, not by average score-utility correlation. In the finite law, probability mass moves toward high-score candidates as `N` grows. Therefore a scorer can have reasonable full-pool correlation while its selected semantic tail is physically bad. In that regime, increasing `N` improves selected semantic plausibility while decreasing or saturating selected real utility.

## Conditional Certificate-Abstention Guarantee

For a finite candidate pool, suppose the certificate predicates are sound for the modeled physics failure modes and the calibrated selected-tail utility error is bounded by the reported confidence radius. Then Certified TailGuard-BoN either selects a certified action whose lower-bound utility beats the configured `N=1` and random baselines by the margin, falls back to certified low-N selection, or abstains/request labels. The guarantee is conditional on the candidate pool, modeled physics predicates, and calibration assumptions; it is not a real-world safety theorem.

## No-Free-Lunch Proposition

No high-N controller can guarantee physical utility from semantic scores alone when upper-tail physical utility is unconstrained. Two finite candidate pools can have identical semantic scores and average semantic statistics while assigning opposite physical utility to the semantic tail. Any controller that only observes semantic scores must behave identically on both pools, so it fails on at least one. Physical certificates, verifier evidence, pilot labels, or abstention are necessary for controlled high-N guarantees.

Physical grounding helps only if it changes the ranking or admissible set in the selected tail. A verifier that improves average feasibility estimates but leaves the top semantic tail misranked will not fix high-N selection.
