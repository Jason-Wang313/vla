import numpy as np
import pandas as pd

from vla_tailguard_audit.deployment import VALID_GATES
from vla_tailguard_audit.rendering import attach_visual_observation
from vla_tailguard_audit.simulator import evaluate_candidate_geometry, simulate_pool
from vla_tailguard_audit.stress import component_ablation_summary, phase_diagram_summary, tailguard_gate_examples_summary
from vla_tailguard_audit.tailguard import (
    CERTIFICATE_FAILURE_TYPES,
    TailGuardConfig,
    certify_candidates,
    fit_tail_calibrator,
    tailguard_select,
)
from vla_tailguard_audit.vla_env import FEATURE_NAMES, generate_pools


def _aligned_pilot(count: int = 256):
    sem = np.linspace(0.0, 1.0, count)
    phys = sem.copy()
    util = sem.copy()
    return sem, phys, util


def test_tailguard_returns_exactly_one_valid_gate():
    sem, phys, util = _aligned_pilot()
    calibrator = fit_tail_calibrator(sem, util, phys)
    result = tailguard_select(sem, calibrator=calibrator, physical_scores=phys, n_grid=[1, 2, 4, 8, 16])
    gates = {result.gate_decision}
    assert gates.issubset(VALID_GATES)
    assert len(gates) == 1


def test_tailguard_gate_examples_cover_all_decisions():
    summary = tailguard_gate_examples_summary()
    assert set(summary["gate_decision"]) == {"allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"}
    assert bool(summary["expected_behavior_verified"].all())


def test_tailguard_blocks_high_n_when_lcb_falls_below_baseline():
    sem = np.linspace(0.0, 1.0, 512)
    phys = np.ones_like(sem)
    util = np.full_like(sem, 0.40)
    config = TailGuardConfig(margin=0.03, uncertainty_collect_threshold=0.50)
    calibrator = fit_tail_calibrator(sem, util, phys, config=config)
    result = tailguard_select(sem, calibrator=calibrator, physical_scores=phys, n_grid=[1, 2, 4, 8, 16], config=config)
    assert result.gate_decision == "block_high_n"
    assert result.reason_code == "tail_lcb_below_baselines"


def test_tailguard_allows_when_calibrated_tail_dominates_baseline():
    sem, phys, util = _aligned_pilot(512)
    config = TailGuardConfig(margin=0.02, uncertainty_collect_threshold=0.50)
    calibrator = fit_tail_calibrator(sem, util, phys, config=config)
    result = tailguard_select(sem, calibrator=calibrator, physical_scores=phys, n_grid=[1, 2, 4, 8, 16], config=config)
    assert result.gate_decision in {"allow_high_n", "stop_early"}
    assert result.lower_confidence_bound_curve[max(result.lower_confidence_bound_curve)] >= result.random_baseline - 0.05


def test_certificate_layer_catches_each_named_failure():
    zeros = np.zeros(len(CERTIFICATE_FAILURE_TYPES) + 1)
    flags = {
        "unreachable": zeros.copy(),
        "collision": zeros.copy(),
        "receptacle_incompatible": zeros.copy(),
        "stability_failure": zeros.copy(),
        "fragile": zeros.copy(),
        "tool_mismatch": zeros.copy(),
        "blocked_path": zeros.copy(),
        "hidden_constraint": zeros.copy(),
    }
    flag_by_failure = {
        "reach_envelope": "unreachable",
        "swept_volume_collision": "collision",
        "receptacle_compatibility": "receptacle_incompatible",
        "stability_margin": "stability_failure",
        "fragile_heavy_handling": "fragile",
        "tool_object_match": "tool_mismatch",
        "blocked_path_constraints": "blocked_path",
        "hidden_obstacle": "hidden_constraint",
    }
    for idx, failure_type in enumerate(CERTIFICATE_FAILURE_TYPES):
        flags[flag_by_failure[failure_type]][idx] = 1.0

    certificate = certify_candidates(flags)
    for idx, failure_type in enumerate(CERTIFICATE_FAILURE_TYPES):
        assert failure_type in certificate.failure_types[idx]
        assert not bool(certificate.pass_mask[idx])
    assert bool(certificate.pass_mask[-1])


def test_certified_tailguard_rejects_unsafe_semantic_tail_candidate():
    sem = np.asarray([0.30, 0.99, 0.88, 0.40, 0.50])
    util = np.asarray([0.45, 0.0, 1.0, 0.50, 0.55])
    phys = util.copy()
    flags = {name: np.zeros_like(sem) for name in ["violation", "collision"]}
    flags["collision"][1] = 1.0
    flags["violation"][1] = 1.0
    certificate = certify_candidates(flags, candidate_count=len(sem))
    config = TailGuardConfig(
        confidence_delta=0.5,
        min_pilot_labels=2,
        min_tail_labels=1,
        min_certified_for_high_n=2,
        uncertainty_collect_threshold=1.1,
    )
    calibrator = fit_tail_calibrator(sem, util, phys, config=config)

    result = tailguard_select(
        sem,
        calibrator=calibrator,
        physical_scores=phys,
        certificate=certificate,
        real_utilities=util,
        violation_flags=flags["violation"],
        n_grid=[1, 2, 4],
        config=config,
    )
    assert result.selected_index != 1
    assert result.certificate_pass
    assert result.certified_selected_utility == 1.0
    assert result.certified_violation_rate == 0.0


def test_certified_tailguard_collects_when_too_few_certified_candidates_remain():
    sem = np.linspace(0.0, 1.0, 16)
    phys = np.ones_like(sem)
    util = sem.copy()
    flags = {"hidden_constraint": np.ones_like(sem), "violation": np.ones_like(sem)}
    flags["hidden_constraint"][3] = 0.0
    flags["violation"][3] = 0.0
    certificate = certify_candidates(flags, candidate_count=len(sem))
    config = TailGuardConfig(confidence_delta=0.5, min_certified_candidates=2, min_certified_for_high_n=4)
    calibrator = fit_tail_calibrator(sem, util, phys, config=config)

    result = tailguard_select(
        sem,
        calibrator=calibrator,
        physical_scores=phys,
        certificate=certificate,
        real_utilities=util,
        violation_flags=flags["violation"],
        n_grid=[1, 2, 4, 8],
        config=config,
    )
    assert result.gate_decision == "collect_pilot_labels"
    assert result.reason_code == "insufficient_certified_candidates"
    assert result.fallback_used
    assert result.abstention_reason == "insufficient_certified_candidates"


def test_certified_tailguard_blocks_high_n_when_certificate_count_is_low():
    sem = np.linspace(0.0, 1.0, 32)
    phys = np.ones_like(sem)
    util = sem.copy()
    flags = {"hidden_constraint": np.ones_like(sem), "violation": np.ones_like(sem)}
    flags["hidden_constraint"][[5, 20, 21]] = 0.0
    flags["violation"][[5, 20, 21]] = 0.0
    certificate = certify_candidates(flags, candidate_count=len(sem))
    config = TailGuardConfig(
        confidence_delta=0.5,
        min_pilot_labels=2,
        min_tail_labels=1,
        min_certified_candidates=2,
        min_certified_for_high_n=8,
        uncertainty_collect_threshold=1.1,
    )
    calibrator = fit_tail_calibrator(sem, util, phys, config=config)

    result = tailguard_select(
        sem,
        calibrator=calibrator,
        physical_scores=phys,
        certificate=certificate,
        real_utilities=util,
        violation_flags=flags["violation"],
        n_grid=[1, 2, 4, 8, 16],
        config=config,
    )
    assert result.gate_decision == "block_high_n"
    assert result.reason_code == "certified_candidate_count_too_low"
    assert result.selected_n == 1


def test_tailguard_confidence_bounds_tighten_with_more_pilot_labels():
    rng = np.random.default_rng(7)
    sem = rng.random(2000)
    phys = rng.random(2000)
    util = np.clip(0.35 * sem + 0.55 * phys + rng.normal(0.0, 0.03, 2000), 0.0, 1.0)
    low = fit_tail_calibrator(sem, util, phys, sample_fraction=0.02, seed=1)
    high = fit_tail_calibrator(sem, util, phys, sample_fraction=0.20, seed=1)
    assert high.tail_label_count > low.tail_label_count
    assert high.confidence_radius < low.confidence_radius


def test_phase_sweep_generation_is_deterministic_by_seed():
    a = phase_diagram_summary([1], 1, 16, [1, 2, 4, 8, 16])
    b = phase_diagram_summary([1], 1, 16, [1, 2, 4, 8, 16])
    pd.testing.assert_frame_equal(a, b)


def test_component_ablation_has_failure_for_each_named_removal():
    summary = component_ablation_summary([1], 1, 16, [1, 2, 4, 8, 16])
    ablations = {
        "no_physical_certificate",
        "no_verifier_score",
        "no_pilot_labels",
        "no_empirical_lower_bound",
        "no_adaptive_n",
        "no_abstention_fallback",
    }
    full = summary[summary["method"] == "full_certified_tailguard"]
    assert not full.empty
    assert ((full["selected_real_utility"] >= 0.98) & (full["violation_rate"] <= 0.01)).all()
    for method in ablations:
        rows = summary[summary["method"] == method]
        assert not rows.empty, method
        assert ((rows["selected_real_utility"] < 0.98) | (rows["violation_rate"] > 0.01)).any(), method
    assert ablations.issubset(set(summary["component_under_test"].astype(str)))


def test_geometry_evaluator_catches_named_physical_failures():
    pool = simulate_pool(attach_visual_observation(generate_pools([1], 1, 32, "rendered_visual")[0]))
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    feature = pool.candidate_features[0].copy()

    wrong_object = feature.copy()
    wrong_object[idx["correct_object"]] = 0.0
    assert evaluate_candidate_geometry(pool, wrong_object).correct_object is False

    wrong_receptacle = feature.copy()
    wrong_receptacle[idx["correct_receptacle"]] = 0.0
    assert evaluate_candidate_geometry(pool, wrong_receptacle).receptacle_compatible is False

    unreachable = feature.copy()
    unreachable[idx["reachable"]] = 0.0
    assert evaluate_candidate_geometry(pool, unreachable).reachable is False

    collision = feature.copy()
    collision[idx["collision_free"]] = 0.0
    collision[idx["blocked_path"]] = 1.0
    assert evaluate_candidate_geometry(pool, collision).swept_collision_free is False

    unstable = feature.copy()
    unstable[idx["stable_placement"]] = 0.0
    assert evaluate_candidate_geometry(pool, unstable).stable is False
