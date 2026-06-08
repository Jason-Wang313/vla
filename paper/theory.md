# Theory

The theorem audits a fixed finite VLA generator/scorer stack. It does not say whether VLAs are good or bad universally.

The exact finite tie-aware law assigns each candidate a selection probability under sampling without replacement and uniform tie breaking among top-score candidates. Expected selected real utility is the dot product of these probabilities with real utility.

The important implication for VLA action selection is selected-tail dependence. Average semantic-real correlation can be acceptable while the high semantic-score tail is physically poor. In that case increasing `N` can improve selected semantic plausibility while decreasing or saturating selected real utility.

Physical grounding helps only if it changes the ranking in the selected tail. A verifier that improves average feasibility estimates but leaves the top semantic tail misranked will not fix high-N selection.
