"""Guarded optional real-VLA/SmoLVLA adapter status.

This module does not claim real-VLA validation by itself. It records whether
local runtime support and cached model files are sufficient to attempt such an
adapter.
"""

from __future__ import annotations

import importlib.util
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .io import ensure_dir, write_json


RUNTIME_MODULES = [
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "accelerate",
    "num2words",
    "lerobot",
    "libero",
]
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


@dataclass(frozen=True)
class OptionalVLAInferenceProbe:
    status: str
    reason: str
    model_cache_name: str | None
    model_snapshot: str | None
    task: str
    config_load_seconds: float | None
    policy_load_seconds: float | None
    preprocess_seconds: float | None
    inference_seconds: float | None
    action_shape: list[int]
    action_dtype: str | None
    action_abs_max: float | None
    first_action: list[float]
    parameter_count: int | None
    benchmark_validation: bool
    traceback_tail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "model_cache_name": self.model_cache_name,
            "model_snapshot": self.model_snapshot,
            "task": self.task,
            "config_load_seconds": self.config_load_seconds,
            "policy_load_seconds": self.policy_load_seconds,
            "preprocess_seconds": self.preprocess_seconds,
            "inference_seconds": self.inference_seconds,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "action_abs_max": self.action_abs_max,
            "first_action": self.first_action,
            "parameter_count": self.parameter_count,
            "benchmark_validation": self.benchmark_validation,
            "traceback_tail": self.traceback_tail,
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
    has_core_runtime = all(
        runtimes.get(name, False)
        for name in ["torch", "transformers", "huggingface_hub", "safetensors", "num2words"]
    )
    can_attempt = bool(has_complete_cache and has_lerobot and has_core_runtime)
    if can_attempt:
        return OptionalVLAStatus(
            status="READY_TO_ATTEMPT",
            reason="Cached SmoLVLA files and LeRobot/core runtime modules are available; use --attempt-inference for a local synthetic action probe, but benchmark evidence still requires a real task wrapper and physical evaluation.",
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
        missing.append("core torch/transformers/huggingface_hub/safetensors/num2words runtime")
    return OptionalVLAStatus(
        status="SKIPPED_WITH_REASON",
        reason="Cannot run optional real-VLA adapter yet: missing " + ", ".join(missing) + ".",
        runtime_modules=runtimes,
        cached_models=cached,
        can_attempt_inference=False,
    )


def _select_complete_smolvla_snapshot(
    cached_models: list[dict[str, object]], preferred_cache_name: str | None = None
) -> tuple[str, Path] | None:
    names = [preferred_cache_name] if preferred_cache_name else []
    names.extend(name for name in MODEL_CACHE_NAMES if name not in names)
    by_name = {str(model["cache_name"]): model for model in cached_models}
    for name in names:
        model = by_name.get(name)
        if not model or not model.get("exists"):
            continue
        for snapshot in model.get("snapshots", []):
            if snapshot.get("has_config") and snapshot.get("has_weights"):
                return name, Path(str(snapshot["snapshot"]))
    return None


def _write_inference_probe(root: Path, probe: OptionalVLAInferenceProbe) -> dict[str, object]:
    data = probe.to_dict()
    write_json(root / "inference_probe.json", data)
    lines = [
        "# Optional SmoLVLA Inference Probe",
        "",
        f"- status: `{probe.status}`",
        f"- reason: {probe.reason}",
        f"- benchmark_validation: `{probe.benchmark_validation}`",
        f"- model_cache_name: `{probe.model_cache_name}`",
        f"- model_snapshot: `{probe.model_snapshot}`",
        f"- task: {probe.task}",
        f"- action_shape: `{probe.action_shape}`",
        f"- action_dtype: `{probe.action_dtype}`",
        f"- action_abs_max: `{probe.action_abs_max}`",
        f"- first_action: `{probe.first_action}`",
        f"- parameter_count: `{probe.parameter_count}`",
        f"- config_load_seconds: `{probe.config_load_seconds}`",
        f"- policy_load_seconds: `{probe.policy_load_seconds}`",
        f"- preprocess_seconds: `{probe.preprocess_seconds}`",
        f"- inference_seconds: `{probe.inference_seconds}`",
    ]
    if probe.traceback_tail:
        lines.extend(["", "## Traceback Tail", "", "```text", probe.traceback_tail, "```"])
    (root / "inference_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def write_smolvla_rendered_bridge_status(results_dir: str | Path) -> dict[str, object]:
    """Write a guarded rendered-to-SmoLVLA bridge status artifact.

    The core run does not decode SmoLVLA actions back into the toy evaluator, so
    this artifact is deliberately a run-or-skip status rather than success
    evidence.
    """

    root = ensure_dir(Path(results_dir) / "optional_vla")
    data: dict[str, object] = {
        "status": "SKIPPED_WITH_REASON",
        "reason": "Rendered toy observations were not fed through SmoLVLA in the core run. Use the optional probe path only as model-plumbing evidence unless an action-to-toy-evaluator mapping is implemented and documented.",
        "bridge_name": "smolvla_rendered_bridge",
        "benchmark_validation": False,
        "decoded_physical_success": False,
        "action_decode_supported": False,
        "physical_success_claimed": False,
        "action_count": 0,
        "action_shape": [],
        "action_abs_max_mean": None,
        "rendered_input_count": 0,
        "mapping_documented": False,
    }
    write_json(root / "smolvla_rendered_bridge.json", data)
    lines = [
        "# Optional SmoLVLA Rendered Bridge",
        "",
        f"- status: `{data['status']}`",
        f"- benchmark_validation: `{data['benchmark_validation']}`",
        f"- decoded_physical_success: `{data['decoded_physical_success']}`",
        f"- action_decode_supported: `{data['action_decode_supported']}`",
        f"- physical_success_claimed: `{data['physical_success_claimed']}`",
        f"- reason: {data['reason']}",
    ]
    (root / "smolvla_rendered_bridge.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def write_libero_benchmark_status(results_dir: str | Path) -> dict[str, object]:
    """Write a guarded LIBERO/LeRobot benchmark wrapper status artifact."""

    root = ensure_dir(Path(results_dir) / "optional_vla")
    has_libero = importlib.util.find_spec("libero") is not None
    data: dict[str, object] = {
        "status": "READY_TO_IMPLEMENT_WRAPPER" if has_libero else "SKIPPED_WITH_REASON",
        "reason": (
            "libero is importable, but no benchmark task wrapper has been run in this core pipeline."
            if has_libero
            else "libero is not importable, so the guarded LIBERO/OpenVLA-style benchmark wrapper is skipped."
        ),
        "benchmark_validation": False,
        "real_robot_validation": False,
        "libero_importable": bool(has_libero),
        "wrapper_skeleton": "guarded_optional_status_only",
    }
    write_json(root / "libero_benchmark_status.json", data)
    lines = [
        "# Optional LIBERO Benchmark Status",
        "",
        f"- status: `{data['status']}`",
        f"- libero_importable: `{data['libero_importable']}`",
        f"- benchmark_validation: `{data['benchmark_validation']}`",
        f"- reason: {data['reason']}",
    ]
    (root / "libero_benchmark_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def attempt_smolvla_cpu_inference_probe(
    results_dir: str | Path,
    cache_root: str | Path | None = None,
    *,
    task: str = "put the red mug in the cabinet",
    preferred_cache_name: str | None = "models--lerobot--smolvla_base",
) -> dict[str, object]:
    """Load cached SmoLVLA on CPU and ask it for an action on synthetic inputs.

    This is a guarded integration probe, not a robotics benchmark. It verifies
    that a real cached VLA policy can consume language, visual tensors, and robot
    state and emit an action chunk through the LeRobot API.
    """

    root = ensure_dir(Path(results_dir) / "optional_vla")
    status = assess_optional_vla(cache_root)
    selected = _select_complete_smolvla_snapshot(status.cached_models, preferred_cache_name)
    if not status.can_attempt_inference or selected is None:
        probe = OptionalVLAInferenceProbe(
            status="SKIPPED_WITH_REASON",
            reason=status.reason if not status.can_attempt_inference else "No complete cached SmoLVLA snapshot.",
            model_cache_name=None,
            model_snapshot=None,
            task=task,
            config_load_seconds=None,
            policy_load_seconds=None,
            preprocess_seconds=None,
            inference_seconds=None,
            action_shape=[],
            action_dtype=None,
            action_abs_max=None,
            first_action=[],
            parameter_count=None,
            benchmark_validation=False,
        )
        return _write_inference_probe(root, probe)

    model_cache_name, snapshot_path = selected
    config_load_seconds = policy_load_seconds = preprocess_seconds = inference_seconds = None
    try:
        import torch

        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        start = time.perf_counter()
        config = PreTrainedConfig.from_pretrained(snapshot_path, cli_overrides=["--device=cpu"])
        config_load_seconds = time.perf_counter() - start

        start = time.perf_counter()
        policy = SmolVLAPolicy.from_pretrained(
            snapshot_path,
            config=config,
            local_files_only=True,
            strict=False,
        )
        policy_load_seconds = time.perf_counter() - start

        preprocessor, _ = make_smolvla_pre_post_processors(config, dataset_stats=None)
        observation = {}
        for key, feature in config.image_features.items():
            channels, height, width = feature.shape
            image = torch.zeros((channels, height, width), dtype=torch.float32)
            image[0, height // 4 : height // 2, width // 4 : width // 2] = 1.0
            if channels > 1:
                image[1, height // 2 : 3 * height // 4, width // 2 : 3 * width // 4] = 0.5
            if channels > 2:
                image[2] = torch.linspace(0.0, 0.25, width).repeat(height, 1)
            observation[key] = image
        if config.robot_state_feature is not None:
            observation["observation.state"] = torch.zeros(config.robot_state_feature.shape, dtype=torch.float32)

        start = time.perf_counter()
        batch = preprocessor({**observation, "task": task})
        preprocess_seconds = time.perf_counter() - start

        start = time.perf_counter()
        actions = policy.predict_action_chunk(batch)
        inference_seconds = time.perf_counter() - start

        finite = bool(torch.isfinite(actions).all().item())
        first_action = [float(x) for x in actions[0, 0].detach().cpu().tolist()]
        probe = OptionalVLAInferenceProbe(
            status="INFERENCE_PROBE_PASS" if finite else "INFERENCE_PROBE_FAIL",
            reason=(
                "Cached SmoLVLA loaded on CPU and emitted a finite action chunk from synthetic visual/state/language input."
                if finite
                else "SmoLVLA emitted non-finite action values."
            ),
            model_cache_name=model_cache_name,
            model_snapshot=str(snapshot_path),
            task=task,
            config_load_seconds=round(config_load_seconds, 4),
            policy_load_seconds=round(policy_load_seconds, 4),
            preprocess_seconds=round(preprocess_seconds, 4),
            inference_seconds=round(inference_seconds, 4),
            action_shape=list(actions.shape),
            action_dtype=str(actions.dtype),
            action_abs_max=float(actions.detach().abs().max().cpu().item()),
            first_action=first_action,
            parameter_count=sum(parameter.numel() for parameter in policy.parameters()),
            benchmark_validation=False,
        )
        return _write_inference_probe(root, probe)
    except Exception:
        tail = traceback.format_exc(limit=8)
        probe = OptionalVLAInferenceProbe(
            status="INFERENCE_PROBE_FAIL",
            reason="SmoLVLA CPU inference probe failed before producing an action chunk.",
            model_cache_name=model_cache_name,
            model_snapshot=str(snapshot_path),
            task=task,
            config_load_seconds=round(config_load_seconds, 4) if config_load_seconds is not None else None,
            policy_load_seconds=round(policy_load_seconds, 4) if policy_load_seconds is not None else None,
            preprocess_seconds=round(preprocess_seconds, 4) if preprocess_seconds is not None else None,
            inference_seconds=round(inference_seconds, 4) if inference_seconds is not None else None,
            action_shape=[],
            action_dtype=None,
            action_abs_max=None,
            first_action=[],
            parameter_count=None,
            benchmark_validation=False,
            traceback_tail=tail,
        )
        return _write_inference_probe(root, probe)


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
    write_smolvla_rendered_bridge_status(results_dir)
    write_libero_benchmark_status(results_dir)
    return data
