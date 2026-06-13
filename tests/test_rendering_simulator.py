from pathlib import Path

import numpy as np
from PIL import Image

from vla_tailguard_audit.rendering import attach_visual_observation, render_pool_scene, visual_observation_vector
from vla_tailguard_audit.simulator import simulate_pool
from vla_tailguard_audit.vla_env import generate_pools


def test_rendered_scene_is_nonblank(tmp_path: Path):
    pool = generate_pools([1], 1, 16, "rendered_visual")[0]
    path = render_pool_scene(pool, tmp_path / "scene.png")
    img = Image.open(path).convert("RGB")
    pixels = np.asarray(img)
    assert pixels.shape == (256, 256, 3)
    assert float(pixels.std()) > 5.0


def test_visual_vector_is_fixed_width_and_finite():
    pool = generate_pools([1], 1, 16, "rendered_visual")[0]
    vec = visual_observation_vector(pool)
    assert vec.shape == (8,)
    assert np.all(np.isfinite(vec))
    with_visual = attach_visual_observation(pool)
    assert with_visual.visual_vector.shape == (8,)


def test_simulator_recomputes_physical_utility_and_flags():
    pool = attach_visual_observation(generate_pools([2], 1, 64, "rendered_visual")[0])
    sim = simulate_pool(pool)
    assert sim.real_utility.shape == pool.real_utility.shape
    assert sim.physical_feasibility.shape == pool.physical_feasibility.shape
    assert set(["violation", "wrong_object", "wrong_target", "unreachable", "collision"]).issubset(sim.flags)
    assert not np.allclose(sim.real_utility, pool.real_utility)
    assert np.any(sim.flags["violation"] > 0.0)
