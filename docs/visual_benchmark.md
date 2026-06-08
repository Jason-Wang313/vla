# Rendered Visual Benchmark

The v2 rendered benchmark adds RGB scene observations to the controlled VLA-style setting. Each scene image contains:

- target object and distractor object,
- target color,
- target receptacle,
- robot start pose,
- reachability region,
- obstacles,
- instruction text overlay.

Implementation:

- rendering module: `src/vla_best_of_n/rendering.py`
- experiment output: `results/rendered/`
- metadata: `results/rendered_visual_metadata.csv`
- summary: `results/rendered_summary.csv`
- figures:
  - `results/figures/figure6_rendered_visual_simulator.png`
  - `results/figures/figure7_rendered_scene_examples.png`

The rendered benchmark still remains controlled evidence. It closes part of the “purely symbolic toy” gap, but it is not a real robot benchmark.
