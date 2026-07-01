from __future__ import annotations

import json
import shutil
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx import corrected_mid_tail_parity as parity

REPOSITORY = Path(__file__).resolve().parents[2]


def _activation_row(game_index: int, decision_index: int) -> dict[str, object]:
    focal_seat = decision_index % 4
    return {
        "game_index": game_index,
        "decision_index": decision_index,
        "features": [0, parity.HISTORICAL_OPPONENT_START],
        "raw_feature_count": 2,
        "focal_seat": focal_seat,
        "personal_turn": decision_index // 4 + 1,
        "phase": "opening",
        "policy": parity.POLICIES[(game_index + focal_seat) % 4],
        "free_overflow_applied": decision_index == 0,
    }


def _mini_corpus(tmp_path: Path) -> tuple[Path, parity.CorpusContract]:
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    declarations = []
    parsed_declarations = []
    for shard_index in range(2):
        path = root / f"part-{shard_index:05d}.jsonl"
        rows = [_activation_row(shard_index, decision_index) for decision_index in range(4)]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        declaration = {
            "file": path.name,
            "row_count": 4,
            "bytes": path.stat().st_size,
            "blake3": parity.checksum_file(path),
            "first_game_index": shard_index,
            "games": 1,
        }
        declarations.append(declaration)
        parsed_declarations.append(
            parity.ShardDeclaration(
                shard_index=shard_index,
                file=path.name,
                row_count=4,
                byte_count=path.stat().st_size,
                blake3=declaration["blake3"],
                first_game_index=shard_index,
                games=1,
            )
        )
    payload = parity.payload_blake3(parsed_declarations)
    scientific = "1" * 64
    manifest = {
        "schema_version": 1,
        "dataset_id": "mini-parity-v1",
        "feature_schema": parity.FEATURE_SCHEMA,
        "feature_count": parity.LEGACY_NNUE_FEATURES,
        "split": "train",
        "rows": 8,
        "shards": declarations,
        "payload_blake3": payload,
        "scientific_blake3": scientific,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract = parity.CorpusContract(
        dataset_id="mini-parity-v1",
        feature_schema=parity.FEATURE_SCHEMA,
        feature_count=parity.LEGACY_NNUE_FEATURES,
        split="train",
        manifest_file_blake3=parity.checksum_file(manifest_path),
        manifest_scientific_blake3=scientific,
        payload_blake3=payload,
        shard_count=2,
        rows_per_shard=4,
        games_per_shard=1,
        rows_per_game=4,
        first_game_index=0,
        expected_statistics={
            "games": 2,
            "rows": 8,
            "rows_by_phase": {
                "opening": 8,
                "early": 0,
                "middle": 0,
                "late": 0,
            },
            "rows_by_focal_seat": {"0": 2, "1": 2, "2": 2, "3": 2},
            "rows_by_policy": {
                "greedy": 2,
                "random_draft": 2,
                "scarcity_draft": 2,
                "preference_draft": 2,
            },
            "free_overflow_preludes": 2,
            "raw_feature_emissions": 16,
            "unique_feature_activations": 16,
            "duplicate_feature_emissions_removed": 0,
            "minimum_unique_features_per_row": 2,
            "maximum_unique_features_per_row": 2,
        },
    )
    return root, contract


def _fake_shard_report(
    path: Path,
    *,
    contract: parity.CorpusContract,
    shard_index: int,
    prediction_digest: str | None = None,
) -> Path:
    prediction_digest = (
        prediction_digest or blake3.blake3(f"predictions-{shard_index}".encode()).hexdigest()
    )
    implementation = {
        "identity_kind": "test-implementation",
        "bundle_blake3": "2" * 64,
        "files": [],
    }
    checkpoints = {
        "C0": {"blake3": parity.HISTORICAL_CHECKPOINT_BLAKE3},
        "T1": {"blake3": parity.CORRECTED_CHECKPOINT_BLAKE3},
    }
    mapping = {"mapping_id": "test", "gates": {"exact": True}}
    first_game_index = contract.first_game_index + shard_index * contract.games_per_shard
    scientific = {
        "schema_version": 1,
        "experiment_id": parity.EXPERIMENT_ID,
        "classification": "corrected_mid_tail_frozen_parity_shard_complete",
        "mode": "production",
        "implementation": implementation,
        "corpus": {
            "dataset_id": contract.dataset_id,
            "feature_schema": contract.feature_schema,
            "feature_count": contract.feature_count,
            "manifest_file_blake3": contract.manifest_file_blake3,
            "manifest_scientific_blake3": contract.manifest_scientific_blake3,
            "payload_blake3": contract.payload_blake3,
            "shard_index": shard_index,
            "shard_blake3": f"{shard_index + 3:064x}",
            "shard_bytes": 100 + shard_index,
            "declared_rows": contract.rows_per_shard,
            "first_game_index": first_game_index,
            "games": contract.games_per_shard,
        },
        "checkpoints": checkpoints,
        "mapping": mapping,
        "coverage": {
            "requested_rows": contract.rows_per_shard,
            "evaluated_rows": contract.rows_per_shard,
            "complete_shard": True,
            "first_row_identity": [first_game_index, 0],
            "last_row_identity": [
                first_game_index + contract.games_per_shard - 1,
                contract.rows_per_game - 1,
            ],
        },
        "statistics": {
            "rows_by_phase": {
                "opening": 4,
                "early": 0,
                "middle": 0,
                "late": 0,
            },
            "rows_by_focal_seat": {"0": 1, "1": 1, "2": 1, "3": 1},
            "rows_by_policy": {
                "greedy": 1,
                "random_draft": 1,
                "scarcity_draft": 1,
                "preference_draft": 1,
            },
            "free_overflow_preludes": 1,
            "raw_feature_emissions": 8,
            "unique_feature_activations": 8,
            "duplicate_feature_emissions_removed": 0,
            "minimum_unique_features_per_row": 2,
            "maximum_unique_features_per_row": 2,
        },
        "activations": {
            "historical_base": 4,
            "historical_discarded": 0,
            "historical_opponent": 4,
            "corrected_base": 4,
            "corrected_opponent": 4,
            "corrected_tail": 0,
        },
        "feature_streams": {
            "C0_blake3": blake3.blake3(f"c0-{shard_index}".encode()).hexdigest(),
            "T1_blake3": blake3.blake3(f"t1-{shard_index}".encode()).hexdigest(),
        },
        "predictions": {
            "dtype": "float32-little-endian",
            "C0_blake3": prediction_digest,
            "T1_blake3": prediction_digest,
            "bit_identical_rows": contract.rows_per_shard,
            "mismatched_rows": 0,
            "nonfinite_C0": 0,
            "nonfinite_T1": 0,
        },
        "gates": {"all_exact": True},
    }
    report = {
        "schema_version": 1,
        "scientific": scientific,
        "scientific_blake3": parity.scientific_blake3(scientific),
        "operational": {
            "host": f"john{shard_index + 1}",
            "timing": {
                "wall_seconds": 2.0,
                "C0_inference_seconds": 0.5,
                "T1_inference_seconds": 0.5,
                "paired_inference_seconds": 1.0,
            },
        },
        "passed": True,
        "aggregate_eligible": True,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def test_canonical_scientific_hash_is_key_order_independent_and_finite() -> None:
    assert parity.scientific_blake3({"a": 1, "b": 2}) == parity.scientific_blake3({"b": 2, "a": 1})
    with pytest.raises(parity.ParityCampaignError, match="canonical JSON"):
        parity.scientific_blake3({"invalid": float("nan")})


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(parity.ParityCampaignError, match="duplicate key"):
        parity.strict_json_loads('{"a": 1, "a": 2}', label="duplicate")
    with pytest.raises(parity.ParityCampaignError, match="non-finite"):
        parity.strict_json_loads('{"a": NaN}', label="nonfinite")


def test_activation_row_contract_rejects_discarded_and_malformed_rows() -> None:
    valid = _activation_row(0, 0)
    assert parity.validate_activation_row(
        valid,
        expected_game_index=0,
        expected_decision_index=0,
    )["features"] == [0, parity.HISTORICAL_OPPONENT_START]

    discarded = dict(valid)
    discarded["features"] = [0, parity.HISTORICAL_DEFECT_START]
    with pytest.raises(parity.ParityCampaignError, match="discarded"):
        parity.validate_activation_row(
            discarded,
            expected_game_index=0,
            expected_decision_index=0,
        )

    duplicate = dict(valid)
    duplicate["features"] = [0, 0]
    with pytest.raises(parity.ParityCampaignError, match="strictly increasing"):
        parity.validate_activation_row(
            duplicate,
            expected_game_index=0,
            expected_decision_index=0,
        )

    wrong_policy = dict(valid)
    wrong_policy["policy"] = "preference_draft"
    with pytest.raises(parity.ParityCampaignError, match="policy"):
        parity.validate_activation_row(
            wrong_policy,
            expected_game_index=0,
            expected_decision_index=0,
        )


def test_manifest_and_payload_identity_validation_fail_closed(tmp_path: Path) -> None:
    root, contract = _mini_corpus(tmp_path)
    corpus = parity.validate_all_corpus_payload_identities(root, contract)
    assert len(corpus.shards) == 2

    shard = root / "part-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b" ")
    with pytest.raises(parity.ParityCampaignError, match="byte count drifted"):
        parity.validate_all_corpus_payload_identities(root, contract)

    root, contract = _mini_corpus(tmp_path / "second")
    manifest = root / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(parity.ParityCampaignError, match="manifest identity drifted"):
        parity.validate_corpus_manifest(root, contract)


def test_checkpoint_hash_drift_fails_before_parsing(tmp_path: Path) -> None:
    source = REPOSITORY / parity.DEFAULT_HISTORICAL_CHECKPOINT
    drifted = tmp_path / source.name
    shutil.copyfile(source, drifted)
    with drifted.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))
    with pytest.raises(parity.ParityCampaignError, match="BLAKE3 drifted"):
        parity.validate_checkpoint_identity(
            drifted,
            parity.HISTORICAL_CHECKPOINT_CONTRACT,
        )


def test_prediction_comparison_rejects_nonfinite_and_reports_first_bit_mismatch() -> None:
    rows = [(1, 2), (1, 3)]
    with pytest.raises(parity.ParityCampaignError, match="non-finite"):
        parity.compare_prediction_bytes(
            np.asarray([1.0, np.nan], dtype=np.float32),
            np.asarray([1.0, np.nan], dtype=np.float32),
            rows,
        )
    with pytest.raises(
        parity.ParityCampaignError,
        match=r"game=1, decision=3.*C0=0x.*T1=0x",
    ):
        parity.compare_prediction_bytes(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0, 3.0], dtype=np.float32),
            rows,
        )


def test_scientific_section_rejects_operational_fields() -> None:
    parity.assert_scientific_section_is_portable({"prediction": {"rows": 1}})
    with pytest.raises(parity.ParityCampaignError, match="operational keys"):
        parity.assert_scientific_section_is_portable({"prediction": {"wall_seconds": 1.0}})


def test_order_independent_aggregate_requires_complete_exact_shards(
    tmp_path: Path,
) -> None:
    _root, contract = _mini_corpus(tmp_path / "corpus")
    reports = [
        _fake_shard_report(
            tmp_path / f"shard-{shard_index}.json",
            contract=contract,
            shard_index=shard_index,
        )
        for shard_index in range(2)
    ]
    forward_path = tmp_path / "forward.json"
    reverse_path = tmp_path / "reverse.json"
    forward = parity.aggregate_reports(
        reports,
        output=forward_path,
        contract=contract,
    )
    reverse = parity.aggregate_reports(
        list(reversed(reports)),
        output=reverse_path,
        contract=contract,
    )
    assert forward == reverse
    assert forward_path.read_bytes() == reverse_path.read_bytes()
    assert forward["scientific"]["coverage"]["rows"] == 8
    assert forward["scientific"]["predictions"]["bit_identical_rows"] == 8

    with pytest.raises(parity.ParityCampaignError, match="exactly 2"):
        parity.aggregate_reports(reports[:1], output=None, contract=contract)

    duplicate = json.loads(reports[1].read_text())
    duplicate["scientific"]["corpus"]["shard_index"] = 0
    duplicate["scientific_blake3"] = parity.scientific_blake3(duplicate["scientific"])
    reports[1].write_text(json.dumps(duplicate))
    with pytest.raises(parity.ParityCampaignError, match="duplicate"):
        parity.aggregate_reports(reports, output=None, contract=contract)


def test_aggregate_rejects_tampered_hash_and_prediction_receipt(tmp_path: Path) -> None:
    _root, contract = _mini_corpus(tmp_path / "corpus")
    reports = [
        _fake_shard_report(
            tmp_path / f"shard-{shard_index}.json",
            contract=contract,
            shard_index=shard_index,
        )
        for shard_index in range(2)
    ]
    tampered = json.loads(reports[0].read_text())
    tampered["scientific"]["predictions"]["T1_blake3"] = "f" * 64
    reports[0].write_text(json.dumps(tampered))
    with pytest.raises(parity.ParityCampaignError, match="scientific hash drifted"):
        parity.aggregate_reports(reports, output=None, contract=contract)

    tampered["scientific_blake3"] = parity.scientific_blake3(tampered["scientific"])
    reports[0].write_text(json.dumps(tampered))
    with pytest.raises(parity.ParityCampaignError, match="prediction parity drifted"):
        parity.aggregate_reports(reports, output=None, contract=contract)


def test_aggregate_rejects_overlapping_game_intervals(tmp_path: Path) -> None:
    _root, contract = _mini_corpus(tmp_path / "corpus")
    reports = [
        _fake_shard_report(
            tmp_path / f"shard-{shard_index}.json",
            contract=contract,
            shard_index=shard_index,
        )
        for shard_index in range(2)
    ]
    overlapping = json.loads(reports[1].read_text())
    overlapping["scientific"]["corpus"]["first_game_index"] = 0
    overlapping["scientific"]["coverage"]["first_row_identity"] = [0, 0]
    overlapping["scientific"]["coverage"]["last_row_identity"] = [0, 3]
    overlapping["scientific_blake3"] = parity.scientific_blake3(overlapping["scientific"])
    reports[1].write_text(json.dumps(overlapping))
    with pytest.raises(parity.ParityCampaignError, match="game interval drifted"):
        parity.aggregate_reports(reports, output=None, contract=contract)


def test_real_corpus_smoke_is_bit_identical_and_batch_independent(
    tmp_path: Path,
) -> None:
    historical = REPOSITORY / parity.DEFAULT_HISTORICAL_CHECKPOINT
    corrected = REPOSITORY / parity.DEFAULT_CORRECTED_CHECKPOINT
    models = parity.prepare_models(historical, corrected)
    first = parity.run_shard(
        corpus_root=REPOSITORY / parity.DEFAULT_CORPUS_ROOT,
        shard_index=0,
        historical_checkpoint=historical,
        corrected_checkpoint=corrected,
        output=tmp_path / "batch-7.json",
        batch_rows=7,
        row_limit=32,
        prepared_models=models,
    )
    second = parity.run_shard(
        corpus_root=REPOSITORY / parity.DEFAULT_CORPUS_ROOT,
        shard_index=0,
        historical_checkpoint=historical,
        corrected_checkpoint=corrected,
        output=tmp_path / "batch-13.json",
        batch_rows=13,
        row_limit=32,
        prepared_models=models,
    )
    assert first["scientific_blake3"] == second["scientific_blake3"]
    assert first["scientific"]["classification"].endswith("_smoke_complete")
    assert first["scientific"]["predictions"]["bit_identical_rows"] == 32
    assert first["scientific"]["predictions"]["mismatched_rows"] == 0
    assert first["scientific"]["activations"]["historical_discarded"] == 0
    assert first["scientific"]["mapping"]["gates"] == {
        "base_rows_byte_identical": True,
        "opponent_rows_byte_identical_after_remap": True,
        "corrected_tail_all_ieee754_signed_zero": True,
        "all_downstream_tensors_byte_identical": True,
    }
