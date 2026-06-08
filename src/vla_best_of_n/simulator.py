"""Simulator-style physical utility for rendered toy VLA scenes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .rendering import scene_layout
from .vla_env import CandidatePool, FEATURE_NAMES, flags_from_features


def _point_to_rect_distance(point: np.ndarray, rect: tuple[float, float, float, float]) -> float:
    cx, cy, w, h = rect
    dx = max(abs(float(point[0]) - cx) - w / 2, 0.0)
    dy = max(abs(float(point[1]) - cy) - h / 2, 0.0)
    return float(np.hypot(dx, dy))


def _segment_hits_rect(a: np.ndarray, b: np.ndarray, rect: tuple[float, float, float, float], samples: int = 25) -> bool:
    for t in np.linspace(0.0, 1.0, samples):
        p = (1.0 - t) * a + t * b
        if _point_to_rect_distance(p, rect) <= 0.015:
            return True
    return False


def simulate_pool(pool: CandidatePool) -> CandidatePool:
    """Return a copy whose real utility is computed by geometry plus constraints."""

    layout = scene_layout(pool)
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    utilities = []
    sim_flags = {name: [] for name in flags_from_features(pool.candidate_features)}
    for feature in pool.candidate_features:
        correct_object = feature[idx["correct_object"]] >= 0.5
        correct_receptacle = feature[idx["correct_receptacle"]] >= 0.5
        object_pos = layout["target"] if correct_object else layout["distractor"]
        if correct_receptacle:
            receptacle_pos = layout["receptacle"]
        else:
            receptacle_pos = np.asarray([0.22, 0.68], dtype=float)

        robot_dist = float(np.linalg.norm(object_pos - layout["robot"]))
        reachable_geom = robot_dist <= float(layout["reach_radius"])
        reachable = reachable_geom and feature[idx["reachable"]] >= 0.5
        path_hits = any(_segment_hits_rect(object_pos, receptacle_pos, rect) for rect in layout["obstacles"])
        blocked = feature[idx["blocked_path"]] >= 0.5
        planned_route = correct_object and correct_receptacle and feature[idx["affordance_prior"]] < 0.85
        collision_free = (not blocked) and feature[idx["collision_free"]] >= 0.5 and ((not path_hits) or planned_route)
        stable = feature[idx["stable_placement"]] >= 0.5 and receptacle_pos[1] > 0.55
        fragile_safe = feature[idx["fragile_safe"]] >= 0.5
        heavy_safe = feature[idx["heavy_safe"]] >= 0.5
        tool_match = feature[idx["tool_match"]] >= 0.5

        hard_success = all(
            [
                correct_object,
                correct_receptacle,
                reachable,
                collision_free,
                stable,
                fragile_safe,
                heavy_safe,
                tool_match,
            ]
        )
        partial = (
            0.22 * float(correct_object)
            + 0.16 * float(correct_receptacle)
            + 0.16 * float(reachable)
            + 0.16 * float(collision_free)
            + 0.12 * float(stable)
            + 0.08 * float(fragile_safe)
            + 0.05 * float(heavy_safe)
            + 0.05 * float(tool_match)
        )
        utility = 1.0 if hard_success else max(0.0, partial - 0.20 * float(not correct_object) - 0.18 * float(not reachable))
        utilities.append(float(np.clip(utility, 0.0, 1.0)))

        sim_flags["wrong_object"].append(float(not correct_object))
        sim_flags["wrong_target"].append(float(not correct_receptacle))
        sim_flags["unreachable"].append(float(not reachable))
        sim_flags["collision"].append(float(not collision_free))
        sim_flags["fragile"].append(float(not fragile_safe))
        sim_flags["stability_failure"].append(float(not stable))
        sim_flags["violation"].append(float((not reachable) or (not collision_free) or (not stable) or (not fragile_safe)))

    flags = {name: np.asarray(values, dtype=float) for name, values in sim_flags.items()}
    physical = 1.0 - np.maximum.reduce(
        [flags["unreachable"], flags["collision"], flags["stability_failure"], flags["fragile"]]
    )
    return replace(pool, real_utility=np.asarray(utilities, dtype=float), physical_feasibility=physical, flags=flags)
