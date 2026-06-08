# Simulator-Style Physical Evaluator

The v2 simulator-style evaluator recomputes real utility from rendered scene geometry and action-candidate constraints.

Implementation: `src/vla_best_of_n/simulator.py`

The evaluator checks:

- whether the selected object is the target or distractor,
- whether the selected receptacle is the requested target,
- whether the object is reachable from the robot pose,
- whether the object-to-receptacle path intersects obstacles,
- whether blocked-path flags defeat otherwise plausible semantic actions,
- whether placement is stable,
- whether fragile/heavy/tool constraints are satisfied.

The simulator-style utility is still controlled and simplified, but it is less idealized than the v1 symbolic utility because geometry and path obstruction affect outcomes.

Main artifact:

- `results/rendered_summary.csv`

At full-run `N=128`, raw semantic selection has simulator utility `0.446` and violation rate `1.000`; grounded combined repair reaches utility `1.000` and violation rate `0.000`.
