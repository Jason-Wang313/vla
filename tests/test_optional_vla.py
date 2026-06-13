from pathlib import Path

from vla_tailguard_audit.optional_vla import (
    assess_optional_vla,
    attempt_smolvla_cpu_inference_probe,
    discover_cached_models,
    write_libero_benchmark_status,
    write_optional_vla_status,
    write_smolvla_rendered_bridge_status,
)


def test_optional_vla_cleanly_skips_with_empty_cache(tmp_path: Path):
    status = assess_optional_vla(tmp_path)
    assert status.status in {"SKIPPED_WITH_REASON", "READY_TO_ATTEMPT"}
    assert isinstance(status.reason, str)


def test_optional_vla_writes_status_artifacts(tmp_path: Path):
    data = write_optional_vla_status(tmp_path, cache_root=tmp_path / "empty_cache")
    assert (tmp_path / "optional_vla" / "adapter_status.json").exists()
    assert (tmp_path / "optional_vla" / "adapter_status.md").exists()
    assert (tmp_path / "optional_vla" / "smolvla_rendered_bridge.json").exists()
    assert (tmp_path / "optional_vla" / "libero_benchmark_status.json").exists()
    assert "status" in data


def test_cached_model_discovery_shape(tmp_path: Path):
    models = discover_cached_models(tmp_path)
    assert models
    assert {"cache_name", "path", "exists", "snapshots"}.issubset(models[0])


def test_optional_vla_inference_probe_skips_without_cache(tmp_path: Path):
    data = attempt_smolvla_cpu_inference_probe(tmp_path, cache_root=tmp_path / "empty_cache")
    assert data["status"] == "SKIPPED_WITH_REASON"
    assert data["benchmark_validation"] is False
    assert (tmp_path / "optional_vla" / "inference_probe.json").exists()
    assert (tmp_path / "optional_vla" / "inference_probe.md").exists()


def test_optional_bridge_and_benchmark_statuses_do_not_claim_success(tmp_path: Path):
    bridge = write_smolvla_rendered_bridge_status(tmp_path)
    benchmark = write_libero_benchmark_status(tmp_path)
    assert bridge["benchmark_validation"] is False
    assert bridge["decoded_physical_success"] is False
    assert benchmark["benchmark_validation"] is False
    assert benchmark["real_robot_validation"] is False
