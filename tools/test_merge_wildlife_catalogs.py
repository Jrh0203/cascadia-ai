from tools.merge_wildlife_catalogs import merge_catalogs


def test_merge_prefers_proof_then_score_and_strips_audit_fields(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        '{"schema":"catalog","candidate_sha256":"x","results":['
        '{"counts":[1,2,3,4,5],"optimum":10,"proof_complete":false},'
        '{"counts":[2,2,2,6,8],"optimum":20,"proof_complete":true,'
        '"proof_provenance":{"receipt":"x"}}]}'
    )
    second.write_text(
        '{"schema":"catalog","results":['
        '{"counts":[1,2,3,4,5],"optimum":9,"proof_complete":true},'
        '{"counts":[2,2,2,6,8],"optimum":25,"proof_complete":false}]}'
    )
    merged = merge_catalogs([first, second])
    assert merged["completed_count"] == 2
    assert merged["allocation_count"] == 2
    assert merged["proof_complete"] is True
    assert [row["optimum"] for row in merged["results"]] == [9, 20]
    assert "candidate_sha256" not in merged
    assert "proof_provenance" not in merged["results"][1]


def test_merge_does_not_call_partial_catalog_complete(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        '{"allocation_count":3,"results":['
        '{"counts":[1,2,3,4,5],"optimum":10,"proof_complete":true}]}'
    )
    second.write_text(
        '{"allocation_count":3,"results":['
        '{"counts":[2,2,2,6,8],"optimum":20,"proof_complete":true}]}'
    )
    merged = merge_catalogs([first, second])
    assert merged["allocation_count"] == 3
    assert merged["completed_count"] == 2
    assert merged["proof_complete"] is False
