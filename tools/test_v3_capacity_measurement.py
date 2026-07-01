from __future__ import annotations

import v3_capacity_measurement as measurement


def test_capacity_measurement_separates_density_from_collection_rate(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model").write_bytes(b"x" * 123)
    value = measurement.measure(
        {
            "schema_id": "direct",
            "scientific_eligible": False,
            "compact_shard": {"bytes": 1_001, "games": 100},
        },
        {
            "schema_id": "corpus",
            "scientific_eligible": False,
            "elapsed_seconds": 20.0,
            "games": 2_000,
        },
        checkpoint,
    )
    assert value["bytes_per_game"] == 11
    assert value["collection_seconds_per_game"] == 0.01
    assert value["checkpoint_bytes"] == 123
