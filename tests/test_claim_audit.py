from pathlib import Path

from scripts.claim_audit import FORBIDDEN_CLAIMS


def test_forbidden_claim_list_contains_user_forbidden_claims():
    assert "Best-of-N always helps." in FORBIDDEN_CLAIMS
    assert "We validate on real robots." in FORBIDDEN_CLAIMS


def test_required_docs_paths_are_declared():
    expected = [
        Path("docs/current_state_before_v1.md"),
        Path("docs/theory.md"),
        Path("docs/claims.md"),
        Path("paper/abstract.md"),
    ]
    for path in expected:
        assert path.exists(), path
