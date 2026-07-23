from pathlib import Path

from tools.aaaaa_wildlife_union_catalog import union

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_base_and_fleet_union_has_724_exact_rows() -> None:
    paths = [
        ROOT
        / "cascadiav3"
        / "fleet"
        / "inputs_aaaaa_exact_tail_fleet3_20260723"
        / "import_ledger.json",
        *[
            ROOT
            / "cascadiav3"
            / "fleet"
            / "staging_aaaaa_exact_tail_fleet3_20260723"
            / f"shard_john{host}.json"
            for host in (2, 3, 4)
        ],
    ]
    payload = union(paths)
    assert payload["allocation_count"] == 826
    assert payload["completed_count"] == 724
    assert len(payload["results"]) == 826
