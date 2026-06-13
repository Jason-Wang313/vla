"""Guarded external benchmark status and smoke-step runner.

The external benchmark path is deliberately artifact-first. It records import,
environment, reset, step, reward, and success-metric status without upgrading
paper claims unless those artifacts actually exist.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

from .io import ensure_dir, write_json


DEFAULT_EXTERNAL_ROOT = Path.home() / "external_benchmarks"
DEFAULT_ROBOCASA_ENV_ID = "robocasa/PickPlaceCounterToCabinet"
REQUIRED_RUN_FIELDS = (
    "benchmark",
    "env_id",
    "python_executable",
    "package_versions",
    "asset_status",
    "seed",
    "reset_ok",
    "step_ok",
    "reward_trace",
    "success_trace",
    "action_space",
    "observation_keys",
    "policy_kind",
    "tailguard_connected",
    "physical_success_claimed",
    "skip_reason",
)


def _venv_python(external_root: Path, name: str) -> Path:
    return external_root / ".venvs" / name / "Scripts" / "python.exe"


def _base_run(
    *,
    benchmark: str,
    env_id: str,
    python_executable: str,
    seed: int | None,
    policy_kind: str,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "env_id": env_id,
        "python_executable": python_executable,
        "package_versions": {},
        "asset_status": {},
        "seed": seed,
        "reset_ok": False,
        "step_ok": False,
        "reward_trace": [],
        "success_trace": [],
        "termination_trace": [],
        "truncation_trace": [],
        "action_space": {},
        "observation_keys": [],
        "policy_kind": policy_kind,
        "tailguard_connected": False,
        "physical_success_claimed": False,
        "skip_reason": skip_reason,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _probe_code() -> str:
    return textwrap.dedent(
        r"""
        import importlib.metadata as md
        import importlib.util
        import json
        import os
        import sys
        import traceback

        REQUIRED_MODULES = ["robocasa", "gymnasium", "robosuite", "lerobot", "torch", "transformers", "libero"]
        SENTINEL = "__VLA_TAILGUARD_EXTERNAL_JSON__="

        def version(name):
            try:
                return md.version(name)
            except Exception as exc:
                return f"UNAVAILABLE:{type(exc).__name__}"

        def to_jsonable(value):
            try:
                import numpy as np
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
            except Exception:
                pass
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {str(k): to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_jsonable(v) for v in value]
            return str(value)

        def space_summary(space):
            try:
                from gymnasium import spaces
            except Exception:
                return {"type": type(space).__name__}
            if isinstance(space, spaces.Dict):
                return {"type": "Dict", "spaces": {str(k): space_summary(v) for k, v in space.spaces.items()}}
            if isinstance(space, spaces.Box):
                return {
                    "type": "Box",
                    "shape": list(space.shape),
                    "dtype": str(space.dtype),
                    "low_min": float(space.low.min()),
                    "high_max": float(space.high.max()),
                }
            if isinstance(space, spaces.Discrete):
                return {"type": "Discrete", "n": int(space.n)}
            if isinstance(space, spaces.Tuple):
                return {"type": "Tuple", "spaces": [space_summary(v) for v in space.spaces]}
            return {"type": type(space).__name__}

        def zero_action(space):
            import numpy as np
            from gymnasium import spaces
            if isinstance(space, spaces.Dict):
                return {k: zero_action(v) for k, v in space.spaces.items()}
            if isinstance(space, spaces.Tuple):
                return tuple(zero_action(v) for v in space.spaces)
            if isinstance(space, spaces.Box):
                return np.zeros(space.shape, dtype=space.dtype)
            if isinstance(space, spaces.Discrete):
                return 0
            return space.sample()

        def obs_keys(obs):
            if isinstance(obs, dict):
                return sorted(str(k) for k in obs.keys())
            return [type(obs).__name__]

        def success_value(info, env):
            if isinstance(info, dict):
                for key in ["success", "is_success", "task_success", "episode_success"]:
                    if key in info:
                        return bool(info[key])
            raw_env = getattr(env, "unwrapped", env)
            for attr in ["_check_success", "check_success", "is_success"]:
                fn = getattr(raw_env, attr, None)
                if callable(fn):
                    try:
                        return bool(fn())
                    except Exception:
                        pass
            return None

        def base_run(benchmark, env_id, python_executable, seed, policy_kind, skip_reason):
            return {
                "benchmark": benchmark,
                "env_id": env_id,
                "python_executable": python_executable,
                "package_versions": {},
                "asset_status": {},
                "seed": seed,
                "reset_ok": False,
                "step_ok": False,
                "reward_trace": [],
                "success_trace": [],
                "termination_trace": [],
                "truncation_trace": [],
                "action_space": {},
                "observation_keys": [],
                "policy_kind": policy_kind,
                "tailguard_connected": False,
                "physical_success_claimed": False,
                "skip_reason": skip_reason,
            }

        def emit(payload):
            print(SENTINEL + json.dumps(payload, sort_keys=True, default=to_jsonable))

        benchmark = os.environ.get("VLA_TAILGUARD_EXTERNAL_BENCHMARK", "robocasa")
        env_id = os.environ.get("VLA_TAILGUARD_EXTERNAL_ENV_ID", "robocasa/PickPlaceCounterToCabinet")
        split = os.environ.get("VLA_TAILGUARD_EXTERNAL_SPLIT", "pretrain")
        attempt_reset = os.environ.get("VLA_TAILGUARD_EXTERNAL_ATTEMPT_RESET", "0") == "1"
        max_steps = int(os.environ.get("VLA_TAILGUARD_EXTERNAL_MAX_STEPS", "1"))
        seeds = [int(item) for item in os.environ.get("VLA_TAILGUARD_EXTERNAL_SEEDS", "0").split(",") if item]
        policy_kinds = [item for item in os.environ.get("VLA_TAILGUARD_EXTERNAL_POLICIES", "zero,random").split(",") if item]
        repo_root = os.environ.get("VLA_TAILGUARD_EXTERNAL_REPO_ROOT", "")

        module_importable = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
        package_versions = {name: version(name) for name in REQUIRED_MODULES}
        asset_status = {
            "repo_root": repo_root,
            "repo_root_exists": bool(repo_root and os.path.isdir(repo_root)),
            "robocasa_assets_dir_exists": bool(repo_root and os.path.isdir(os.path.join(repo_root, "robocasa", "models", "assets"))),
            "robocasa_macros_private_exists": bool(repo_root and os.path.exists(os.path.join(repo_root, "robocasa", "macros_private.py"))),
            "libero_package_dir_exists": bool(repo_root and os.path.isdir(os.path.join(repo_root, "libero"))),
        }
        payload = {
            "benchmark": benchmark,
            "env_id": env_id,
            "split": split,
            "python_executable": sys.executable,
            "module_importable": module_importable,
            "package_versions": package_versions,
            "asset_status": asset_status,
            "registered_env_count": 0,
            "task_names": [],
            "target_env_registered": False,
            "runs": [],
            "traceback_tail": None,
        }

        try:
            if benchmark != "robocasa":
                row = base_run(benchmark, env_id, sys.executable, None, "status_only", "no guarded runtime wrapper is implemented for this benchmark")
                row["package_versions"] = package_versions
                row["asset_status"] = asset_status
                payload["runs"].append(row)
                emit(payload)
                raise SystemExit(0)

            import gymnasium as gym
            import robocasa  # noqa: F401
            from gymnasium.envs.registration import registry

            task_names = sorted(str(key) for key in registry.keys() if str(key).startswith("robocasa/"))
            payload["registered_env_count"] = len(task_names)
            payload["task_names"] = task_names[:50]
            payload["target_env_registered"] = env_id in task_names
            if not payload["target_env_registered"]:
                row = base_run(benchmark, env_id, sys.executable, None, "status_only", "target environment is not registered")
                row["package_versions"] = package_versions
                row["asset_status"] = asset_status
                payload["runs"].append(row)
                emit(payload)
                raise SystemExit(0)

            if not attempt_reset:
                row = base_run(benchmark, env_id, sys.executable, None, "status_only", "status-only run; reset and step not attempted")
                row["package_versions"] = package_versions
                row["asset_status"] = asset_status
                payload["runs"].append(row)
                emit(payload)
                raise SystemExit(0)

            for seed in seeds:
                for policy_kind in policy_kinds:
                    row = base_run(benchmark, env_id, sys.executable, seed, policy_kind, "")
                    row["package_versions"] = package_versions
                    row["asset_status"] = asset_status
                    env = None
                    try:
                        env = gym.make(env_id, split=split, seed=seed)
                        row["action_space"] = space_summary(env.action_space)
                        reset_out = env.reset(seed=seed)
                        if isinstance(reset_out, tuple) and len(reset_out) == 2:
                            obs, info = reset_out
                        else:
                            obs, info = reset_out, {}
                        row["reset_ok"] = True
                        row["observation_keys"] = obs_keys(obs)
                        for _ in range(max_steps):
                            action = zero_action(env.action_space) if policy_kind == "zero" else env.action_space.sample()
                            step_out = env.step(action)
                            if len(step_out) == 5:
                                obs, reward, terminated, truncated, info = step_out
                            elif len(step_out) == 4:
                                obs, reward, done, info = step_out
                                terminated, truncated = bool(done), False
                            else:
                                raise RuntimeError(f"unexpected step tuple length {len(step_out)}")
                            row["step_ok"] = True
                            row["reward_trace"].append(float(reward))
                            row["success_trace"].append(success_value(info, env))
                            row["termination_trace"].append(bool(terminated))
                            row["truncation_trace"].append(bool(truncated))
                            if terminated or truncated:
                                break
                    except Exception:
                        row["skip_reason"] = traceback.format_exc(limit=8)
                    finally:
                        try:
                            if env is not None:
                                env.close()
                        except Exception:
                            pass
                    payload["runs"].append(row)

            decoded = base_run(
                benchmark,
                env_id,
                sys.executable,
                None,
                "decoded_tailguard_candidate",
                "action-schema mapping from Certified TailGuard toy candidates to RoboCasa controls is not implemented; no method success is claimed",
            )
            decoded["package_versions"] = package_versions
            decoded["asset_status"] = asset_status
            payload["runs"].append(decoded)
            emit(payload)
        except SystemExit:
            raise
        except Exception:
            payload["traceback_tail"] = traceback.format_exc(limit=8)
            row = base_run(benchmark, env_id, sys.executable, None, "status_only", payload["traceback_tail"])
            row["package_versions"] = package_versions
            row["asset_status"] = asset_status
            payload["runs"].append(row)
            emit(payload)
        """
    )


def _run_probe_subprocess(
    *,
    python_executable: Path,
    benchmark: str,
    env_id: str,
    split: str,
    seeds: Iterable[int],
    policy_kinds: Iterable[str],
    attempt_reset: bool,
    max_steps: int,
    timeout_seconds: int,
    external_repo_root: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "VLA_TAILGUARD_EXTERNAL_BENCHMARK": benchmark,
            "VLA_TAILGUARD_EXTERNAL_ENV_ID": env_id,
            "VLA_TAILGUARD_EXTERNAL_SPLIT": split,
            "VLA_TAILGUARD_EXTERNAL_ATTEMPT_RESET": "1" if attempt_reset else "0",
            "VLA_TAILGUARD_EXTERNAL_MAX_STEPS": str(max_steps),
            "VLA_TAILGUARD_EXTERNAL_SEEDS": ",".join(str(int(seed)) for seed in seeds),
            "VLA_TAILGUARD_EXTERNAL_POLICIES": ",".join(policy_kinds),
            "VLA_TAILGUARD_EXTERNAL_REPO_ROOT": str(external_repo_root),
        }
    )
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", _probe_code()],
            cwd=str(external_repo_root) if external_repo_root.exists() else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        asset_status = {"repo_root": str(external_repo_root), "repo_root_exists": external_repo_root.exists()}
        row = _base_run(
            benchmark=benchmark,
            env_id=env_id,
            python_executable=str(python_executable),
            seed=None,
            policy_kind="reset_step_attempt",
            skip_reason=f"external probe timed out after {timeout_seconds} seconds",
        )
        row["asset_status"] = asset_status
        return {
            "benchmark": benchmark,
            "env_id": env_id,
            "split": split,
            "python_executable": str(python_executable),
            "module_importable": {},
            "package_versions": {},
            "asset_status": asset_status,
            "registered_env_count": 0,
            "task_names": [],
            "target_env_registered": False,
            "runs": [row],
            "traceback_tail": (exc.stderr or exc.stdout or "")[-4000:] if hasattr(exc, "stderr") else None,
            "subprocess_returncode": "TIMEOUT",
        }

    payload = None
    sentinel = "__VLA_TAILGUARD_EXTERNAL_JSON__="
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith(sentinel):
            payload = json.loads(line[len(sentinel) :])
            break
    if payload is None:
        row = _base_run(
            benchmark=benchmark,
            env_id=env_id,
            python_executable=str(python_executable),
            seed=None,
            policy_kind="status_only",
            skip_reason="external probe did not emit a JSON status sentinel",
        )
        payload = {
            "benchmark": benchmark,
            "env_id": env_id,
            "split": split,
            "python_executable": str(python_executable),
            "module_importable": {},
            "package_versions": {},
            "asset_status": {"repo_root": str(external_repo_root), "repo_root_exists": external_repo_root.exists()},
            "registered_env_count": 0,
            "task_names": [],
            "target_env_registered": False,
            "runs": [row],
            "traceback_tail": (proc.stderr + "\n" + proc.stdout)[-4000:],
        }
    payload["subprocess_returncode"] = proc.returncode
    payload["subprocess_stderr_tail"] = proc.stderr[-4000:]
    return payload


def _missing_python_payload(benchmark: str, env_id: str, split: str, python_executable: Path, repo_root: Path) -> dict[str, Any]:
    row = _base_run(
        benchmark=benchmark,
        env_id=env_id,
        python_executable=str(python_executable),
        seed=None,
        policy_kind="status_only",
        skip_reason="isolated external benchmark Python executable is missing",
    )
    row["asset_status"] = {"repo_root": str(repo_root), "repo_root_exists": repo_root.exists()}
    return {
        "benchmark": benchmark,
        "env_id": env_id,
        "split": split,
        "python_executable": str(python_executable),
        "module_importable": {},
        "package_versions": {},
        "asset_status": row["asset_status"],
        "registered_env_count": 0,
        "task_names": [],
        "target_env_registered": False,
        "runs": [row],
        "traceback_tail": None,
        "subprocess_returncode": "MISSING_PYTHON",
    }


def _has_reward_and_success(row: dict[str, Any]) -> bool:
    success_trace = row.get("success_trace") or []
    return bool(row.get("reset_ok")) and bool(row.get("step_ok")) and bool(row.get("reward_trace")) and any(
        item is not None for item in success_trace
    )


def _claim_levels(runs: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "integration": any(run.get("package_versions") and not run.get("skip_reason", "").startswith("isolated") for run in runs),
        "benchmark_stepping": any(
            bool(run.get("reset_ok")) and bool(run.get("step_ok")) and bool(run.get("reward_trace")) for run in runs
        ),
        "success_metric_exposed": any(_has_reward_and_success(run) for run in runs),
        "physical_success_claimed": any(bool(run.get("physical_success_claimed")) for run in runs),
        "tailguard_method_success": any(
            bool(run.get("tailguard_connected")) and bool(run.get("physical_success_claimed")) for run in runs
        ),
    }


def _flatten_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        item = dict(row)
        for key in ["package_versions", "asset_status", "action_space", "observation_keys", "reward_trace", "success_trace", "termination_trace", "truncation_trace"]:
            item[key] = json.dumps(item.get(key), sort_keys=True, default=_json_default)
        flat.append(item)
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    flat = _flatten_for_csv(rows)
    fieldnames = list(REQUIRED_RUN_FIELDS)
    for row in flat:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)


def _write_seed_level(results_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    seed_dir = ensure_dir(results_dir / "external_benchmark_seed_level")
    for old_file in seed_dir.glob("*.json"):
        old_file.unlink()
    paths = []
    for idx, row in enumerate(rows):
        seed = row.get("seed")
        seed_label = "status" if seed is None else f"seed_{int(seed):03d}"
        policy = str(row.get("policy_kind", "unknown")).replace("/", "_")
        benchmark = str(row.get("benchmark", "unknown")).replace("/", "_")
        path = seed_dir / f"{benchmark}_{policy}_{seed_label}_{idx:03d}.json"
        write_json(path, row)
        paths.append(str(path))
    return paths


def _write_status_figure(results_dir: Path, claim_levels: dict[str, bool]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig_dir = ensure_dir(results_dir / "figures")
    labels = [
        "integration",
        "reset+step",
        "reward",
        "success metric",
        "physical claim",
        "TailGuard claim",
    ]
    values = [
        claim_levels["integration"],
        claim_levels["benchmark_stepping"],
        claim_levels["benchmark_stepping"],
        claim_levels["success_metric_exposed"],
        claim_levels["physical_success_claimed"],
        claim_levels["tailguard_method_success"],
    ]
    colors = ["#2f8f5b" if value else "#b8bec6" for value in values]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    y = list(range(len(labels)))
    ax.barh(y, [1] * len(labels), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("External Benchmark Artifact Gate")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for pos, value in zip(y, values):
        ax.text(0.5, pos, "present" if value else "not claimed", ha="center", va="center", color="white" if value else "#2f3437")
    fig.tight_layout()
    out = fig_dir / "figure19_external_benchmark_status.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out)


def run_external_benchmark(
    *,
    results_dir: str | Path = "results",
    external_root: str | Path = DEFAULT_EXTERNAL_ROOT,
    env_id: str = DEFAULT_ROBOCASA_ENV_ID,
    split: str = "pretrain",
    seeds: Iterable[int] = (0, 1, 2),
    max_steps: int = 1,
    status_only: bool = True,
    attempt_robocasa: bool = False,
    attempt_libero: bool = False,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Run guarded external benchmark status or short reset/step probes."""

    results_path = ensure_dir(Path(results_dir))
    external_path = Path(external_root)
    all_payloads: list[dict[str, Any]] = []

    robocasa_python = _venv_python(external_path, "robocasa")
    robocasa_root = external_path / "robocasa"
    if robocasa_python.exists():
        all_payloads.append(
            _run_probe_subprocess(
                python_executable=robocasa_python,
                benchmark="robocasa",
                env_id=env_id,
                split=split,
                seeds=seeds,
                policy_kinds=("zero", "random"),
                attempt_reset=False,
                max_steps=max_steps,
                timeout_seconds=max(timeout_seconds, 90),
                external_repo_root=robocasa_root,
            )
        )
        if attempt_robocasa and not status_only:
            all_payloads.append(
                _run_probe_subprocess(
                    python_executable=robocasa_python,
                    benchmark="robocasa",
                    env_id=env_id,
                    split=split,
                    seeds=seeds,
                    policy_kinds=("zero", "random"),
                    attempt_reset=True,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    external_repo_root=robocasa_root,
                )
            )
    else:
        all_payloads.append(_missing_python_payload("robocasa", env_id, split, robocasa_python, robocasa_root))

    if attempt_libero:
        libero_python = _venv_python(external_path, "libero310")
        libero_root = external_path / "LIBERO"
        if libero_python.exists():
            all_payloads.append(
                _run_probe_subprocess(
                    python_executable=libero_python,
                    benchmark="libero",
                    env_id="libero/status_only",
                    split="n/a",
                    seeds=seeds,
                    policy_kinds=("status_only",),
                    attempt_reset=False,
                    max_steps=max_steps,
                    timeout_seconds=min(timeout_seconds, 60),
                    external_repo_root=libero_root,
                )
            )
        else:
            all_payloads.append(_missing_python_payload("libero", "libero/status_only", "n/a", libero_python, libero_root))

    metadata_by_benchmark: dict[str, dict[str, Any]] = {}
    for payload in all_payloads:
        benchmark = str(payload.get("benchmark", ""))
        current = metadata_by_benchmark.setdefault(
            benchmark,
            {"package_versions": {}, "asset_status": {}, "module_importable": {}},
        )
        if payload.get("package_versions"):
            current["package_versions"] = payload.get("package_versions", {})
        if payload.get("module_importable"):
            current["module_importable"] = payload.get("module_importable", {})
        if payload.get("asset_status"):
            merged_asset_status = dict(payload.get("asset_status", {}))
            merged_asset_status.update(current.get("asset_status", {}))
            current["asset_status"] = merged_asset_status
    runs = [run for payload in all_payloads for run in payload.get("runs", [])]
    for run in runs:
        inherited = metadata_by_benchmark.get(str(run.get("benchmark", "")), {})
        inherited_any = False
        if inherited and not run.get("package_versions"):
            run["package_versions"] = inherited.get("package_versions", {})
            run["module_importable"] = inherited.get("module_importable", {})
            inherited_any = True
        if inherited:
            original_asset_status = run.get("asset_status") or {}
            inherited_asset_status = dict(inherited.get("asset_status", {}))
            inherited_asset_status.update(original_asset_status)
            inherited_any = inherited_any or set(inherited_asset_status) != set(original_asset_status)
            run["asset_status"] = inherited_asset_status
        if inherited_any:
            run["metadata_inherited_from_status_probe"] = True
    for run in runs:
        for field in REQUIRED_RUN_FIELDS:
            run.setdefault(field, _base_run(benchmark=str(run.get("benchmark", "unknown")), env_id=str(run.get("env_id", "")), python_executable=str(run.get("python_executable", "")), seed=run.get("seed"), policy_kind=str(run.get("policy_kind", "unknown")), skip_reason=str(run.get("skip_reason", "")))[field])
    claim_levels = _claim_levels(runs)
    status = (
        "BENCHMARK_STEP_PASS"
        if claim_levels["benchmark_stepping"]
        else "INTEGRATION_ONLY"
        if claim_levels["integration"]
        else "SKIPPED_WITH_REASON"
    )
    figure_path = _write_status_figure(results_path, claim_levels)
    seed_paths = _write_seed_level(results_path, runs)
    summary_path = results_path / "external_benchmark_summary.csv"
    _write_csv(summary_path, runs)

    artifact = {
        "artifact_version": 1,
        "status": status,
        "external_root": str(external_path),
        "status_only": bool(status_only),
        "attempt_robocasa": bool(attempt_robocasa),
        "attempt_libero": bool(attempt_libero),
        "claim_levels": claim_levels,
        "required_fields": list(REQUIRED_RUN_FIELDS),
        "payloads": all_payloads,
        "runs": runs,
        "summary_csv": str(summary_path),
        "seed_level_json": seed_paths,
        "figure": figure_path,
        "physical_success_claimed": claim_levels["physical_success_claimed"],
        "method_success_claimed": claim_levels["tailguard_method_success"],
        "claim_boundary": "External integration, benchmark stepping, success-metric exposure, physical success, and TailGuard method success are separate claim levels.",
    }
    write_json(results_path / "external_benchmark_status.json", artifact)
    return artifact
