"""Controlled VLA-style scenes and action candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURE_NAMES = [
    "object_name_match",
    "color_match",
    "category_match",
    "receptacle_match",
    "affordance_prior",
    "distractor_salience",
    "target_binding",
    "reachable",
    "collision_free",
    "stable_placement",
    "fragile_safe",
    "tool_match",
    "correct_object",
    "correct_receptacle",
    "blocked_path",
    "heavy_safe",
]


@dataclass
class CandidatePool:
    """A finite set of language-conditioned VLA action candidates."""

    seed: int
    state_id: int
    family: str
    instruction: str
    observation: dict
    language_vector: np.ndarray
    observation_vector: np.ndarray
    candidate_features: np.ndarray
    action_texts: list[str]
    semantic_proxy: np.ndarray
    physical_feasibility: np.ndarray
    real_utility: np.ndarray
    random_score: np.ndarray
    flags: dict[str, np.ndarray]
    visual_vector: np.ndarray | None = None
    render_path: str | None = None

    @property
    def size(self) -> int:
        return int(len(self.real_utility))


OBJECTS = ["mug", "bowl", "block", "tool", "cube", "bottle"]
COLORS = ["red", "blue", "green", "yellow", "white", "black"]
CATEGORIES = ["container", "fragile", "heavy", "tool", "toy", "kitchen"]
RECEPTACLES = ["cabinet", "shelf", "tray", "bin", "drawer", "table"]


def _scene(seed: int, state_id: int, rng: np.random.Generator) -> dict:
    obj = OBJECTS[(seed + state_id) % len(OBJECTS)]
    color = COLORS[(2 * seed + state_id) % len(COLORS)]
    category = CATEGORIES[(seed + 3 * state_id) % len(CATEGORIES)]
    receptacle = RECEPTACLES[(3 * seed + state_id) % len(RECEPTACLES)]
    distractor = OBJECTS[(OBJECTS.index(obj) + 2) % len(OBJECTS)]
    if distractor == obj:
        distractor = OBJECTS[(OBJECTS.index(obj) + 1) % len(OBJECTS)]
    return {
        "target_object": obj,
        "target_color": color,
        "target_category": category,
        "target_receptacle": receptacle,
        "distractor_object": distractor,
        "fragile_instruction": category == "fragile" or obj == "bowl",
        "heavy_instruction": category == "heavy" or obj == "block",
        "requires_tool": obj == "tool" or "push" in category,
        "obstacle_density": float(rng.uniform(0.25, 0.8)),
    }


def _instruction(scene: dict, state_id: int) -> str:
    templates = [
        "put the {color} {obj} in the {rec}",
        "move the {cat} {obj} to the {rec}",
        "place the {color} object in the reachable {rec}",
        "pick the tool that can push the {color} object",
        "put the object in the reachable container",
    ]
    text = templates[state_id % len(templates)]
    return text.format(
        color=scene["target_color"],
        obj=scene["target_object"],
        cat=scene["target_category"],
        rec=scene["target_receptacle"],
    )


def _language_vector(scene: dict, state_id: int) -> np.ndarray:
    return np.asarray(
        [
            (OBJECTS.index(scene["target_object"]) + 1) / len(OBJECTS),
            (COLORS.index(scene["target_color"]) + 1) / len(COLORS),
            (CATEGORIES.index(scene["target_category"]) + 1) / len(CATEGORIES),
            (RECEPTACLES.index(scene["target_receptacle"]) + 1) / len(RECEPTACLES),
            1.0 if state_id % 2 == 0 else 0.0,
        ],
        dtype=float,
    )


def _observation_vector(scene: dict, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        [
            scene["obstacle_density"],
            1.0 if scene["fragile_instruction"] else 0.0,
            1.0 if scene["heavy_instruction"] else 0.0,
            1.0 if scene["requires_tool"] else 0.0,
            float(rng.uniform(0.35, 0.95)),
        ],
        dtype=float,
    )


def _feature(**kwargs: float) -> np.ndarray:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(kwargs)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)


def _mode_probs(family: str) -> list[tuple[str, float]]:
    if family == "distractor":
        return [
            ("success", 0.19),
            ("semantic_trap", 0.22),
            ("distractor", 0.38),
            ("wrong_target", 0.11),
            ("fragile_violation", 0.05),
            ("random", 0.05),
        ]
    if family == "learned_train":
        return [
            ("success", 0.25),
            ("semantic_trap", 0.30),
            ("distractor", 0.25),
            ("wrong_target", 0.10),
            ("fragile_violation", 0.05),
            ("random", 0.05),
        ]
    return [
        ("success", 0.28),
        ("semantic_trap", 0.34),
        ("distractor", 0.20),
        ("wrong_target", 0.10),
        ("fragile_violation", 0.04),
        ("random", 0.04),
    ]


def _choose_mode(family: str, rng: np.random.Generator) -> str:
    modes, probs = zip(*_mode_probs(family))
    return str(rng.choice(modes, p=np.asarray(probs, dtype=float)))


def _candidate(mode: str, scene: dict, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    obj = scene["target_object"]
    rec = scene["target_receptacle"]
    if mode == "success":
        feature = _feature(
            object_name_match=1.0,
            color_match=1.0,
            category_match=1.0,
            receptacle_match=1.0,
            affordance_prior=float(rng.uniform(0.62, 0.78)),
            distractor_salience=float(rng.uniform(0.0, 0.16)),
            target_binding=1.0,
            reachable=1.0,
            collision_free=1.0,
            stable_placement=1.0,
            fragile_safe=1.0,
            tool_match=1.0,
            correct_object=1.0,
            correct_receptacle=1.0,
            blocked_path=0.0,
            heavy_safe=1.0,
        )
        return feature, f"grasp {obj}, move collision-free, place in {rec}"

    if mode == "semantic_trap":
        reachable = float(rng.random() < 0.20)
        collision_free = float(rng.random() < 0.30)
        stable = float(rng.random() < 0.45)
        feature = _feature(
            object_name_match=1.0,
            color_match=1.0,
            category_match=1.0,
            receptacle_match=1.0,
            affordance_prior=float(rng.uniform(0.90, 1.0)),
            distractor_salience=float(rng.uniform(0.10, 0.35)),
            target_binding=1.0,
            reachable=reachable,
            collision_free=collision_free,
            stable_placement=stable,
            fragile_safe=float(rng.random() < 0.65),
            tool_match=1.0,
            correct_object=1.0,
            correct_receptacle=1.0,
            blocked_path=1.0 - reachable,
            heavy_safe=float(rng.random() < 0.70),
        )
        return feature, f"directly place the {scene['target_color']} {obj} in {rec}"

    if mode == "distractor":
        feature = _feature(
            object_name_match=float(rng.uniform(0.35, 0.65)),
            color_match=1.0,
            category_match=1.0,
            receptacle_match=1.0,
            affordance_prior=float(rng.uniform(0.82, 0.97)),
            distractor_salience=float(rng.uniform(0.75, 1.0)),
            target_binding=float(rng.uniform(0.0, 0.25)),
            reachable=1.0,
            collision_free=float(rng.random() < 0.85),
            stable_placement=1.0,
            fragile_safe=1.0,
            tool_match=1.0,
            correct_object=0.0,
            correct_receptacle=1.0,
            blocked_path=0.0,
            heavy_safe=1.0,
        )
        return feature, f"grasp salient {scene['distractor_object']} and place in {rec}"

    if mode == "wrong_target":
        wrong_rec = RECEPTACLES[(RECEPTACLES.index(rec) + 2) % len(RECEPTACLES)]
        feature = _feature(
            object_name_match=1.0,
            color_match=1.0,
            category_match=1.0,
            receptacle_match=float(rng.uniform(0.25, 0.55)),
            affordance_prior=float(rng.uniform(0.62, 0.80)),
            distractor_salience=float(rng.uniform(0.0, 0.25)),
            target_binding=1.0,
            reachable=1.0,
            collision_free=1.0,
            stable_placement=1.0,
            fragile_safe=1.0,
            tool_match=1.0,
            correct_object=1.0,
            correct_receptacle=0.0,
            blocked_path=0.0,
            heavy_safe=1.0,
        )
        return feature, f"move {obj} to nearby {wrong_rec}"

    if mode == "fragile_violation":
        feature = _feature(
            object_name_match=1.0,
            color_match=1.0,
            category_match=1.0,
            receptacle_match=1.0,
            affordance_prior=float(rng.uniform(0.82, 0.98)),
            distractor_salience=float(rng.uniform(0.0, 0.25)),
            target_binding=1.0,
            reachable=1.0,
            collision_free=1.0,
            stable_placement=0.0,
            fragile_safe=0.0,
            tool_match=1.0,
            correct_object=1.0,
            correct_receptacle=1.0,
            blocked_path=0.0,
            heavy_safe=float(rng.random() < 0.35),
        )
        return feature, f"quickly drop fragile {obj} into {rec}"

    feature = _feature(
        object_name_match=float(rng.random() < 0.45),
        color_match=float(rng.random() < 0.45),
        category_match=float(rng.random() < 0.45),
        receptacle_match=float(rng.random() < 0.45),
        affordance_prior=float(rng.uniform(0.05, 0.55)),
        distractor_salience=float(rng.uniform(0.0, 0.8)),
        target_binding=float(rng.uniform(0.0, 0.7)),
        reachable=float(rng.random() < 0.65),
        collision_free=float(rng.random() < 0.70),
        stable_placement=float(rng.random() < 0.65),
        fragile_safe=float(rng.random() < 0.75),
        tool_match=float(rng.random() < 0.65),
        correct_object=float(rng.random() < 0.45),
        correct_receptacle=float(rng.random() < 0.45),
        blocked_path=float(rng.random() < 0.35),
        heavy_safe=float(rng.random() < 0.75),
    )
    return feature, "free-form low-confidence manipulation action"


def semantic_score_from_features(features: np.ndarray) -> np.ndarray:
    """Language/vision plausibility proxy, intentionally not real utility."""

    w = np.asarray([0.20, 0.13, 0.12, 0.18, 0.22, 0.06, 0.09], dtype=float)
    base = features[:, :7] @ w
    # Slight semantic boost for salient distractors mimics object-binding priors.
    boost = 0.08 * features[:, FEATURE_NAMES.index("distractor_salience")]
    return np.clip(base + boost, 0.0, 1.0)


def physical_feasibility_from_features(features: np.ndarray) -> np.ndarray:
    cols = [
        "reachable",
        "collision_free",
        "stable_placement",
        "fragile_safe",
        "tool_match",
        "heavy_safe",
    ]
    values = features[:, [FEATURE_NAMES.index(c) for c in cols]]
    return np.clip(values.mean(axis=1), 0.0, 1.0)


def real_utility_from_features(features: np.ndarray) -> np.ndarray:
    correct = 0.5 * features[:, FEATURE_NAMES.index("correct_object")] + 0.25 * features[:, FEATURE_NAMES.index("correct_receptacle")]
    phys = 0.25 * physical_feasibility_from_features(features)
    severe_penalty = 0.45 * (1.0 - features[:, FEATURE_NAMES.index("reachable")])
    severe_penalty += 0.35 * (1.0 - features[:, FEATURE_NAMES.index("collision_free")])
    severe_penalty += 0.30 * (1.0 - features[:, FEATURE_NAMES.index("stable_placement")])
    severe_penalty += 0.35 * (1.0 - features[:, FEATURE_NAMES.index("correct_object")])
    return np.clip(correct + phys - severe_penalty, 0.0, 1.0)


def flags_from_features(features: np.ndarray) -> dict[str, np.ndarray]:
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    unreachable = 1.0 - features[:, idx["reachable"]]
    collision = 1.0 - features[:, idx["collision_free"]]
    unstable = 1.0 - features[:, idx["stable_placement"]]
    fragile = 1.0 - features[:, idx["fragile_safe"]]
    wrong_object = 1.0 - features[:, idx["correct_object"]]
    wrong_target = 1.0 - features[:, idx["correct_receptacle"]]
    violation = np.maximum.reduce([unreachable, collision, unstable, fragile])
    return {
        "violation": violation,
        "wrong_object": wrong_object,
        "wrong_target": wrong_target,
        "unreachable": unreachable,
        "collision": collision,
        "fragile": fragile,
        "stability_failure": unstable,
    }


def generate_candidate_pool(
    seed: int,
    state_id: int,
    family: str,
    candidates: int,
) -> CandidatePool:
    rng = np.random.default_rng(seed * 100_003 + state_id * 997)
    scene = _scene(seed, state_id, rng)
    instruction = _instruction(scene, state_id)
    language = _language_vector(scene, state_id)
    observation = _observation_vector(scene, rng)

    mandatory = ["success", "success", "semantic_trap", "semantic_trap", "distractor", "wrong_target"]
    modes = mandatory[: min(len(mandatory), candidates)]
    while len(modes) < candidates:
        modes.append(_choose_mode(family, rng))
    rng.shuffle(modes)

    features = []
    actions = []
    for mode in modes:
        feature, action = _candidate(mode, scene, rng)
        features.append(feature)
        actions.append(action)
    x = np.vstack(features)
    semantic = semantic_score_from_features(x)
    # Add small deterministic observation-dependent semantic jitter without
    # making semantic score aware of physical success.
    semantic = np.clip(semantic + rng.normal(0.0, 0.015, size=candidates), 0.0, 1.0)
    physical = physical_feasibility_from_features(x)
    utility = real_utility_from_features(x)
    random_score = rng.random(candidates)
    return CandidatePool(
        seed=seed,
        state_id=state_id,
        family=family,
        instruction=instruction,
        observation=scene,
        language_vector=language,
        observation_vector=observation,
        candidate_features=x,
        action_texts=actions,
        semantic_proxy=semantic,
        physical_feasibility=physical,
        real_utility=utility,
        random_score=random_score,
        flags=flags_from_features(x),
    )


def generate_pools(
    seeds: Iterable[int],
    states_per_seed: int,
    candidates: int,
    family: str,
) -> list[CandidatePool]:
    pools: list[CandidatePool] = []
    for seed in seeds:
        for state_id in range(states_per_seed):
            pools.append(generate_candidate_pool(int(seed), state_id, family, candidates))
    return pools
