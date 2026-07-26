from __future__ import annotations

from botcolosseo.cli.audit_extraction_v3_release import _release_hash


def test_release_hash_excludes_only_its_signature() -> None:
    unsigned = {
        "schema_version": 1,
        "policies": {"strong": {"checkpoint_sha256": "a" * 64}},
        "test_cases_accessed": False,
    }
    signature = _release_hash(unsigned)

    assert _release_hash({**unsigned, "release_sha256": signature}) == signature
    assert _release_hash({**unsigned, "schema_version": 2}) != signature
