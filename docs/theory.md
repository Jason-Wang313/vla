# Theory

## VLA Setup

A VLA-style policy stack receives an instruction `l`, a visual/object observation `o`, and samples candidate actions or trajectories `a_i` from a fixed generator `G(a | o, l)`. Candidates are ranked by a semantic affordance score `S_sem(o, l, a_i)`, optionally by a physical feasibility score `S_phys(o, a_i)`, a calibrated score `S_calib`, or an oracle score equal to real utility `R(a_i)`.

score-tail selects

```text
a* = argmax_i S(a_i)
```

with uniform tie breaking among candidates tied for maximal score. The theorem audits a fixed finite generator/scorer stack. It does not prove that all VLAs succeed or fail.

## Exact Finite Tie-Aware Law

For a finite candidate pool of size `M`, sample `N` candidates uniformly without replacement. Candidate `i` with score `s_i` is selected only if no higher-score candidate is sampled. If `k` other candidates tied with `s_i` are sampled, candidate `i` receives probability `1 / (k + 1)` among the tied maxima.

Let `H_i` be the number of candidates with score greater than `s_i`, `E_i` the number of other candidates tied with `s_i`, and `L_i` the number with lower score. The exact probability of selecting candidate `i` is

```text
p_i(N) = sum_k C(E_i, k) C(L_i, N - 1 - k) / C(M, N) / (k + 1)
```

over valid `k` such that `0 <= k <= E_i` and `0 <= N - 1 - k <= L_i`. Higher-score candidates are excluded by construction.

Expected selected real utility is

```text
E[R(a*)] = sum_i p_i(N) R(a_i).
```

The binary utility law is the same expression with `R(a_i)` in `{0,1}`, yielding selected success probability.

## Edge Cases

- Tied scores: ties are handled exactly by the `1 / (k + 1)` term.
- Constant real utility: selected expected utility remains constant for every scorer and every `N`.
- Oracle score: if `S = R`, expected selected real utility is nondecreasing in `N` for a fixed finite pool.
- Anti-aligned score: if high scores indicate low real utility, increasing `N` can decrease selected real utility.

## VLA Diagnostics

The experiments report selected real utility, selected semantic score, selected physical feasibility, exact finite-pool expected selected utility, Monte Carlo confidence intervals, exact-law prediction error, semantic-real tail gap, top-score-tail real utility, overall rank correlation, tail rank correlation, high-N regret, oracle-minus-VLA gap, violation rate, wrong-object rate, wrong-target rate, collision rate, reachability failure rate, instruction satisfaction proxy, physical success, calibration improvement, marginal value per added candidate, deployment gate, and seed-level variance.

## Corollaries

- If the upper semantic-score tail aligns with real utility, increasing `N` helps or saturates.
- If the upper semantic-score tail is noisy or anti-aligned, increasing `N` can improve semantic plausibility while hurting real utility.
- Average score-utility correlation can miss high-N failure because score-tail stresses the upper score tail.
- Physical grounding helps only when it repairs tail ranking, not merely when it improves average feasibility accuracy.

## Selected-Tail Utility Principle

For large `N`, score-tail is governed by the conditional real utility in the scorer's upper tail, not by average score-utility correlation over the whole candidate pool. In finite pools this is visible directly from the exact law: as `N` increases, probability mass shifts toward candidates whose score is near the maximum. A scorer with good average correlation but bad upper-tail ranking can therefore fail at high `N`.

This motivates reporting:

```text
E[R | S in upper tail]
tail rank correlation between S and R
semantic-real tail gap
```

rather than only full-pool rank correlation.

## Certified TailGuard Guarantee

Certified TailGuard keeps `TailGuard` as the implementation short name but adds a hard certificate layer before calibrated scoring. Candidate actions must pass modeled predicates for reach envelope, swept-volume collision, receptacle compatibility, stability margin, fragile/heavy handling, tool/object match, blocked-path constraints, and modeled hidden obstacles when such flags are available.

After certificate filtering, the controller fits a bounded real-utility calibrator from pilot labels and predicts the exact selected utility curve for a candidate `N` grid. Let `L_N` be a one-sided lower confidence bound on predicted selected utility at `N`, and let `B` be the larger of the `N=1` and random-selection baselines. The default controller allows high-N only when

```text
L_N >= B + margin.
```

If pilot labels, selected-tail labels, or certified candidates are insufficient, it returns `collect_pilot_labels` or `block_high_n`. If high-N adds no lower-bound value, it returns `stop_early`. Public outputs include certificate pass/failure types, certified candidate count, fallback use, abstention reason, certified selected utility, and certified violation rate.

The guarantee is conditional: if the modeled certificate predicates are sound for the relevant physics failures and the bounded-error tail utility estimates are calibrated at the requested confidence level, then any `allow_high_n` decision has a lower-bound selected utility advantage over both baselines. Otherwise the controller falls back or abstains. This does not prove that every real verifier, calibrator, perception system, or robot is accurate.

## No-Free-Lunch For Semantic Scores Alone

Semantic scores alone cannot certify physical high-N utility when upper-tail utility is unconstrained. Two pools can share the same semantic scores and average semantic statistics while swapping the physical utilities of the semantic tail. A semantic-only controller must make the same high-N decision in both pools and therefore fails on one. The repository's affirmative repair claims require certificates, verifier evidence, pilot labels, or abstention.

## Pilot-Label Sample Complexity

The implementation uses an empirical-Bernstein-style lower-bound radius for bounded utilities in `[0, 1]`:

```text
radius(n, delta) =
sqrt(2 * empirical_variance * log(2/delta) / n)
+ 7 * log(2/delta) / (3 * (n - 1)).
```

Here `n` is the number of selected-tail pilot labels used for the bound. The bound tightens as tail labels increase and widens under noisy labels through the empirical residual variance. The sample-complexity experiment crosses pilot label budgets `{0.5%, 1%, 2%, 5%, 10%, 20%}` with label noise `{0, 0.05, 0.10, 0.20}`.

## Boundary

The theorem is conditional on a fixed finite generator/scorer stack. The controlled experiments are VLA-style toy experiments unless real VLA benchmarks are added. This repository claims no real-robot validation, no universal VLA training recipe, no proof that score-tail always helps, and no proof that grounding always fixes the issue.
