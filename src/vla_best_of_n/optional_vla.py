"""Guarded optional real-VLA/SmoLVLA adapter status.

This module does not claim real-VLA validation by itself. It records whether
local runtime support and cached model files are sufficient to attempt such an
adapter.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from .io import ensure_dir, write_json


RUNTIME_MODULES = ["torch", "transformers", "huggingface_hub", "safetensors", "accelerate", "lerobot", "libero"]
MODEL_CACHE_NAMES = [
    "models--lerobot--smolvla_base",
    "models--lerobot--smolvla_libero",
    "models--HuggingFaceVLA--smolvla_libero",
]


@dataclass(frozen=True)
class OptionalVLAStatus:
    status: str
    reason: str
    runtime_modules: dict[str, bool]
    cached_models: list[dict[str, object]]
    can_attempt_inference: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "runtime_modules": self.runtime_modules,
            "cached_models": self.cached_models,
            "can_attempt_inference": self.can_attempt_inference,
        }


def module_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in RUNTIME_MODULES}


def _read_config(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"config_read_error": str(exc)}


def discover_cached_models(cache_root: str | Path | None = None) -> list[dict[str, object]]:
    root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    out: list[dict[str, object]] = []
    for name in MODEL_CACHE_NAMES:
        model_root = root / name
        snapshots = model_root / "snapshots"
        entries: list[dict[str, object]] = []
        if snapshots.exists():
            for snap in sorted(p for p in snapshots.iterdir() if p.is_dir()):
                config = snap / "config.json"
                model_files = sorted(list(snap.glob("*.safetensors")) + list(snap.glob("*.bin")))
                entries.append(
                    {
                        "snapshot": str(snap),
                        "has_config": config.exists(),
                        "has_weights": bool(model_files),
                        "weight_files": [str(p) for p in model_files],
                        "config": _read_config(config) if config.exists() else {},
                    }
                )
        out.append({"cache_name": name, "path": str(model_root), "exists": model_root.exists(), "snapshots": entries})
    return out


def assess_optional_vla(cache_root: str | Path | None = None) -> OptionalVLAStatus:
    runtimes = module_status()
    cached = discover_cached_models(cache_root)
    has_complete_cache = any(
        model["exists"] and any(snapshot["has_config"] and snapshot["has_weights"] for snapshot in model["snapshots"])
        for model in cached
    )
    has_lerobot = runtimes.get("lerobot", False)
    has_core_runtime = all(runtimes.get(name, False) for name in ["torch", "transformers", "huggingface_hub", "safetensors"])
    can_attempt = bool(has_complete_cache and has_lerobot and has_core_runtime)
    if can_attempt:
        return OptionalVLAStatus(
            status="READY_TO_ATTEMPT",
            reason="Cached SmoLVLA files and LeRobot/core runtime modules are available; implement a real inference wrapper before claiming benchmark evidence.",
            runtime_modules=runtimes,
            cached_models=cached,
            can_attempt_inference=True,
        )
    missing = []
    if not has_complete_cache:
        missing.append("complete cached SmoLVLA config+weights")
    if not has_lerobot:
        missing.append("importable lerobot runtime")
    if not has_core_runtime:
        missing.append("core torch/transformers/huggingface_hub/safetensors runtime")
    return OptionalVLAStatus(
        status="SKIPPED_WITH_REASON",
        reason="Cannot run optional real-VLA adapter yet: missing " + ", ".join(missing) + ".",
        runtime_modules=runtimes,
        cached_models=cached,
        can_attempt_inference=False,
    )


def write_optional_vla_status(results_dir: str | Path, cache_root: str | Path | None = None) -> dict[str, object]:
    root = ensure_dir(Path(results_dir) / "optional_vla")
    status = assess_optional_vla(cache_root)
    data = status.to_dict()
    write_json(root / "adapter_status.json", data)
    lines = [
        "# Optional VLA Adapter Status",
        "",
        f"- status: `{status.status}`",
        f"- can_attempt_inference: `{status.can_attempt_inference}`",
        f"- reason: {status.reason}",
        "",
        "## Runtime Modules",
    ]
    for name, present in status.runtime_modules.items():
        lines.append(f"- {name}: {present}")
    lines.append("")
    lines.append("## Cached Models")
    for model in status.cached_models:
        snapshot_count = len(model["snapshots"])
        lines.append(f"- {model['cache_name']}: exists={model['exists']}, snapshots={snapshot_count}")
    (root / "adapter_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data
