# Theory

## VLA Setup

A VLA-style policy stack receives an instruction `l`, a visual/object observation `o`, and samples candidate actions or trajectories `a_i` from a fixed generator `G(a | o, l)`. Candidates are ranked by a semantic affordance score `S_sem(o, l, a_i)`, optionally by a physical feasibility score `S_phys(o, a_i)`, a calibrated score `S_calib`, or an oracle score equal to real utility `R(a_i)`.

Best-of-N selects

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
- Average score-utility correlation can miss high-N failure because Best-of-N stresses the upper score tail.
- Physical grounding helps only when it repairs tail ranking, not merely when it improves average feasibility accuracy.

## Boundary

The theorem is conditional on a fixed finite generator/scorer stack. The controlled experiments are VLA-style toy experiments unless real VLA benchmarks are added. This repository claims no real-robot validation, no universal VLA training recipe, no proof that Best-of-N always helps, and no proof that grounding always fixes the issue.
