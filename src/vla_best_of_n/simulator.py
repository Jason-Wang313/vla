"""Simulator-style physical utility for rendered toy VLA scenes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .rendering import scene_layout
from .vla_env import CandidatePool, FEATURE_NAMES, flags_from_features


@dataclass(frozen=True)
class GeometryEvaluation:
    """Named physical checks for one candidate in a rendered toy scene."""

    correct_object: bool
    correct_receptacle: bool
    reachable: bool
    swept_collision_free: bool
    receptacle_compatible: bool
    stability_margin: float
    stable: bool
    fragile_safe: bool
    heavy_safe: bool
    tool_match: bool
    blocked_path: bool

    @property
    def violation(self) -> bool:
        return not all(
            [
                self.correct_object,
                self.correct_receptacle,
                self.reachable,
                self.swept_collision_free,
                self.receptacle_compatible,
                self.stable,
                self.fragile_safe,
                self.heavy_safe,
                self.tool_match,
            ]
        )


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


def evaluate_candidate_geometry(pool: CandidatePool, feature: np.ndarray) -> GeometryEvaluation:
    """Evaluate reach, swept path, receptacle, stability, and handling checks."""

    layout = scene_layout(pool)
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
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
    blocked_path = feature[idx["blocked_path"]] >= 0.5
    planned_route = correct_object and correct_receptacle and feature[idx["affordance_prior"]] < 0.85
    swept_collision_free = (not blocked_path) and feature[idx["collision_free"]] >= 0.5 and ((not path_hits) or planned_route)

    receptacle_height_ok = bool(receptacle_pos[1] > 0.55)
    receptacle_compatible = correct_receptacle and receptacle_height_ok
    stability_margin = float(receptacle_pos[1] - 0.55 + 0.20 * feature[idx["stable_placement"]])
    stable = feature[idx["stable_placement"]] >= 0.5 and stability_margin >= 0.10
    fragile_safe = feature[idx["fragile_safe"]] >= 0.5
    heavy_safe = feature[idx["heavy_safe"]] >= 0.5
    tool_match = feature[idx["tool_match"]] >= 0.5
    return GeometryEvaluation(
        correct_object=bool(correct_object),
        correct_receptacle=bool(correct_receptacle),
        reachable=bool(reachable),
        swept_collision_free=bool(swept_collision_free),
        receptacle_compatible=bool(receptacle_compatible),
        stability_margin=stability_margin,
        stable=bool(stable),
        fragile_safe=bool(fragile_safe),
        heavy_safe=bool(heavy_safe),
        tool_match=bool(tool_match),
        blocked_path=bool(blocked_path),
    )


def simulate_pool(pool: CandidatePool) -> CandidatePool:
    """Return a copy whose real utility is computed by geometry plus constraints."""

    utilities = []
    sim_flags = {name: [] for name in flags_from_features(pool.candidate_features)}
    for extra in ["receptacle_incompatible", "heavy_failure", "blocked_path", "tool_mismatch"]:
        sim_flags[extra] = []
    for feature in pool.candidate_features:
        geom = evaluate_candidate_geometry(pool, feature)

        hard_success = all(
            [
                geom.correct_object,
                geom.correct_receptacle,
                geom.reachable,
                geom.swept_collision_free,
                geom.receptacle_compatible,
                geom.stable,
                geom.fragile_safe,
                geom.heavy_safe,
                geom.tool_match,
            ]
        )
        partial = (
            0.20 * float(geom.correct_object)
            + 0.14 * float(geom.correct_receptacle)
            + 0.14 * float(geom.reachable)
            + 0.14 * float(geom.swept_collision_free)
            + 0.10 * float(geom.receptacle_compatible)
            + 0.10 * float(geom.stable)
            + 0.07 * float(geom.fragile_safe)
            + 0.04 * float(geom.heavy_safe)
            + 0.03 * float(geom.tool_match)
        )
        utility = 1.0 if hard_success else max(
            0.0,
            partial
            - 0.20 * float(not geom.correct_object)
            - 0.18 * float(not geom.reachable)
            - 0.12 * float(not geom.receptacle_compatible),
        )
        utilities.append(float(np.clip(utility, 0.0, 1.0)))

        sim_flags["wrong_object"].append(float(not geom.correct_object))
        sim_flags["wrong_target"].append(float(not geom.correct_receptacle))
        sim_flags["unreachable"].append(float(not geom.reachable))
        sim_flags["collision"].append(float(not geom.swept_collision_free))
        sim_flags["fragile"].append(float(not geom.fragile_safe))
        sim_flags["stability_failure"].append(float(not geom.stable))
        sim_flags["receptacle_incompatible"].append(float(not geom.receptacle_compatible))
        sim_flags["heavy_failure"].append(float(not geom.heavy_safe))
        sim_flags["tool_mismatch"].append(float(not geom.tool_match))
        sim_flags["blocked_path"].append(float(geom.blocked_path))
        sim_flags["violation"].append(float(geom.violation))

    flags = {name: np.asarray(values, dtype=float) for name, values in sim_flags.items()}
    physical = 1.0 - np.maximum.reduce(
        [
            flags["unreachable"],
            flags["collision"],
            flags["stability_failure"],
            flags["fragile"],
            flags["receptacle_incompatible"],
            flags["heavy_failure"],
            flags["tool_mismatch"],
            flags["blocked_path"],
        ]
    )
    return replace(pool, real_utility=np.asarray(utilities, dtype=float), physical_feasibility=physical, flags=flags)
