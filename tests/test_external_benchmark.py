from pathlib import Path

from vla_best_of_n.external_benchmark import REQUIRED_RUN_FIELDS, run_external_benchmark


def test_external_benchmark_missing_venv_writes_guarded_artifacts(tmp_path: Path):
    artifact = run_external_benchmark(
        results_dir=tmp_path / "results",
        external_root=tmp_path / "missing_external",
        status_only=True,
        attempt_robocasa=False,
        timeout_seconds=1,
    )
    assert artifact["status"] == "SKIPPED_WITH_REASON"
    assert artifact["physical_success_claimed"] is False
    assert (tmp_path / "results" / "external_benchmark_status.json").exists()
    assert (tmp_path / "results" / "external_benchmark_summary.csv").exists()
    assert artifact["seed_level_json"]
    for field in REQUIRED_RUN_FIELDS:
        assert field in artifact["runs"][0]


def test_external_benchmark_keeps_claim_levels_separate(tmp_path: Path):
    artifact = run_external_benchmark(
        results_dir=tmp_path / "results",
        external_root=tmp_path / "missing_external",
        status_only=True,
        attempt_robocasa=True,
        timeout_seconds=1,
    )
    levels = artifact["claim_levels"]
    assert levels["benchmark_stepping"] is False
    assert levels["success_metric_exposed"] is False
    assert levels["physical_success_claimed"] is False
    assert levels["tailguard_method_success"] is False
