"""Rendered visual observations for the VLA toy benchmark."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .io import ensure_dir
from .vla_env import CandidatePool, COLORS


RGB = {
    "red": (220, 38, 38),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "yellow": (234, 179, 8),
    "white": (245, 245, 245),
    "black": (24, 24, 27),
    "cabinet": (120, 113, 108),
    "shelf": (168, 85, 247),
    "tray": (20, 184, 166),
    "bin": (75, 85, 99),
    "drawer": (202, 138, 4),
    "table": (132, 204, 22),
}


def scene_layout(pool: CandidatePool) -> dict[str, object]:
    """Deterministic visual layout derived from seed/state."""

    rng = np.random.default_rng(pool.seed * 4099 + pool.state_id * 131)
    target = np.asarray([0.25 + 0.08 * (pool.state_id % 3), 0.28 + 0.08 * (pool.seed % 3)])
    distractor = np.asarray([0.70 + rng.uniform(-0.06, 0.06), 0.28 + rng.uniform(-0.05, 0.09)])
    receptacle = np.asarray([0.68 + rng.uniform(-0.04, 0.04), 0.72 + rng.uniform(-0.04, 0.04)])
    robot = np.asarray([0.14, 0.82])
    obstacle_count = 2 + int(pool.observation["obstacle_density"] > 0.55)
    obstacles = []
    for k in range(obstacle_count):
        cx = 0.42 + 0.12 * k + rng.uniform(-0.03, 0.03)
        cy = 0.48 + rng.uniform(-0.12, 0.10)
        w = 0.09 + rng.uniform(0.0, 0.04)
        h = 0.18 + rng.uniform(0.0, 0.08)
        obstacles.append((cx, cy, w, h))
    return {
        "target": target,
        "distractor": distractor,
        "receptacle": receptacle,
        "robot": robot,
        "obstacles": obstacles,
        "reach_radius": 0.72 - 0.08 * float(pool.observation["obstacle_density"]),
    }


def visual_observation_vector(pool: CandidatePool) -> np.ndarray:
    layout = scene_layout(pool)
    color_id = (COLORS.index(pool.observation["target_color"]) + 1) / len(COLORS)
    obstacle_count = len(layout["obstacles"]) / 4.0
    return np.asarray(
        [
            float(layout["target"][0]),
            float(layout["target"][1]),
            float(layout["distractor"][0]),
            float(layout["distractor"][1]),
            float(layout["receptacle"][0]),
            float(layout["receptacle"][1]),
            float(layout["reach_radius"]),
            float(0.5 * obstacle_count + 0.5 * color_id),
        ],
        dtype=float,
    )


def _pt(xy: np.ndarray, size: int) -> tuple[int, int]:
    return int(round(float(xy[0]) * size)), int(round(float(xy[1]) * size))


def render_pool_scene(pool: CandidatePool, path: str | Path, size: int = 256) -> Path:
    """Render one scene-level visual observation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = scene_layout(pool)
    img = Image.new("RGB", (size, size), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Workspace and reachability.
    margin = int(size * 0.06)
    draw.rectangle([margin, margin, size - margin, size - margin], outline=(71, 85, 105), width=2)
    robot = _pt(layout["robot"], size)
    rr = int(float(layout["reach_radius"]) * size)
    draw.ellipse([robot[0] - rr, robot[1] - rr, robot[0] + rr, robot[1] + rr], outline=(125, 211, 252), width=2)
    draw.ellipse([robot[0] - 7, robot[1] - 7, robot[0] + 7, robot[1] + 7], fill=(15, 23, 42))

    for cx, cy, w, h in layout["obstacles"]:
        x0 = int((cx - w / 2) * size)
        y0 = int((cy - h / 2) * size)
        x1 = int((cx + w / 2) * size)
        y1 = int((cy + h / 2) * size)
        draw.rectangle([x0, y0, x1, y1], fill=(100, 116, 139), outline=(51, 65, 85), width=2)

    rec = _pt(layout["receptacle"], size)
    draw.rounded_rectangle(
        [rec[0] - 26, rec[1] - 18, rec[0] + 26, rec[1] + 18],
        radius=4,
        fill=RGB.get(pool.observation["target_receptacle"], (148, 163, 184)),
        outline=(31, 41, 55),
        width=2,
    )

    target = _pt(layout["target"], size)
    color = RGB.get(pool.observation["target_color"], (220, 38, 38))
    draw.ellipse([target[0] - 15, target[1] - 15, target[0] + 15, target[1] + 15], fill=color, outline=(15, 23, 42), width=2)
    distractor = _pt(layout["distractor"], size)
    draw.rectangle(
        [distractor[0] - 15, distractor[1] - 15, distractor[0] + 15, distractor[1] + 15],
        fill=(251, 146, 60),
        outline=(15, 23, 42),
        width=2,
    )

    try:
        font = ImageFont.load_default()
        draw.text((10, 8), pool.instruction[:42], fill=(15, 23, 42), font=font)
    except Exception:
        pass
    img.save(path)
    return path


def attach_visual_observation(pool: CandidatePool, render_path: str | None = None) -> CandidatePool:
    return replace(pool, visual_vector=visual_observation_vector(pool), render_path=render_path)


def render_pools(pools: list[CandidatePool], output_dir: str | Path, max_images: int | None = None) -> tuple[list[CandidatePool], pd.DataFrame]:
    """Render scene observations and return pools carrying visual vectors."""

    root = ensure_dir(output_dir)
    rows = []
    out = []
    for idx, pool in enumerate(pools):
        should_render = max_images is None or idx < max_images
        path = root / f"seed_{pool.seed:03d}_state_{pool.state_id:03d}.png"
        render_path = str(path) if should_render else None
        if should_render:
            render_pool_scene(pool, path)
        visual = attach_visual_observation(pool, render_path=render_path)
        out.append(visual)
        layout = scene_layout(pool)
        rows.append(
            {
                "seed": pool.seed,
                "state_id": pool.state_id,
                "instruction": pool.instruction,
                "render_path": render_path or "",
                "target_x": float(layout["target"][0]),
                "target_y": float(layout["target"][1]),
                "distractor_x": float(layout["distractor"][0]),
                "distractor_y": float(layout["distractor"][1]),
                "receptacle_x": float(layout["receptacle"][0]),
                "receptacle_y": float(layout["receptacle"][1]),
                "reach_radius": float(layout["reach_radius"]),
                "obstacle_count": len(layout["obstacles"]),
            }
        )
    return out, pd.DataFrame(rows)


def create_render_montage(image_paths: list[str], output_path: str | Path, columns: int = 4) -> Path:
    """Create a compact montage of rendered scene examples."""

    paths = [Path(p) for p in image_paths if p]
    if not paths:
        raise ValueError("no image paths for montage")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = imgs[0].size
    rows = int(np.ceil(len(imgs) / columns))
    canvas = Image.new("RGB", (columns * w, rows * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        canvas.paste(img, ((i % columns) * w, (i // columns) * h))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
