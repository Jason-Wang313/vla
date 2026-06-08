# Reviewer Attacks

1. "This is not a real VLA." Response: v1 is controlled and learned VLA-style evidence, not validation of a deployed robot foundation model.
2. "This is only a toy." Response: yes, the toy isolates the selected-tail failure; benchmark validation remains future work.
3. "The theorem is reused." Response: the finite law is intentionally reused; the contribution is the VLA-specific object, diagnostics, and evidence.
4. "This is just WAM with language." Response: WAM audits imagined dynamics mismatch; this audits semantic affordance ranking over language-conditioned action candidates.
5. "This is just reward model misalignment." Response: related, but the paper identifies a robot-action version involving object binding, affordance plausibility, and physical feasibility under high-N selection.
6. "This is just distribution shift." Response: distribution shift can contribute, but the diagnostic target is the upper semantic-score tail under fixed generator/scorer sampling.
7. "The failure is constructed." Response: yes, deliberately; controlled construction is used to test whether the diagnostic can expose the failure cleanly.
8. "Why should semantic score preserve real utility?" Response: it need not; that is the deployment risk when semantic plausibility is used to choose physical actions.
9. "Where is robotics?" Response: robotics appears through object, receptacle, reachability, collision, stability, fragile, and tool constraints, but real-robot validation is not claimed.
10. "Does grounding solve everything?" Response: no; grounding helps only if it repairs selected-tail ranking.
11. "Does high N always hurt?" Response: no; if the upper score tail aligns with utility, high N helps or saturates.
12. "Why not train a better VLA?" Response: better training is complementary; this repository audits inference-time selection risk for any fixed scorer.
13. "Why not use a real robot foundation model?" Response: v1 keeps runs lightweight and reproducible; the benchmark plan states what is needed for v2.
14. "What is scientifically new?" Response: the VLA-specific selected-tail phenomenon and diagnostics for semantic affordance over-selection.
15. "Why is this different from diffusion reranking?" Response: the object is not diffusion trajectory diversity; it is semantic/physical ranking in language-conditioned robot actions.
16. "Why is this different from EBM low-energy selection?" Response: there is no energy function; the failure is high semantic-affordance score selecting infeasible actions.
17. "Why is this different from JEPA latent rank distortion?" Response: no latent predictor or compatibility score is being audited.
18. "Are the metrics cherry-picked?" Response: the audit reports semantic, real utility, exact law, tail rank, violations, object binding, oracle gaps, and seed variance.
19. "Are confidence intervals meaningful?" Response: they quantify Monte Carlo selection estimates in the controlled setup; they are not population-level robotics guarantees.
20. "What would make this publishable?" Response: stronger learned artifacts, real VLA benchmark adapters, and ablations showing when grounding succeeds or fails.
21. "Could the physical verifier be wrong?" Response: yes, and the limitations state that verifier error can preserve or worsen tail failures.
22. "Is calibration just training on the answer?" Response: calibration uses small pilot real-utility labels and is evaluated on separate held-out scenes.
23. "Does the exact law assume independent candidates?" Response: no, it audits a fixed finite candidate pool sampled without replacement.
24. "Do average correlations suffice?" Response: no, because Best-of-N stresses the upper score tail, which can behave differently from the average.
