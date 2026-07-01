from __future__ import annotations

import json
from pathlib import Path

import blake3
import pytest
import r2_sparse_mlx_campaign as campaign


def _synthetic_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    monkeypatch.setattr(campaign, "Dataset", lambda *args, **kwargs: object())
    cards = {
        wildlife: "A" for wildlife in ("bear", "elk", "salmon", "hawk", "fox")
    }
    roots: list[Path] = []
    parts: list[campaign.DatasetPart] = []
    specifications = [
        ("train", index, games, first)
        for index, (games, first) in enumerate(
            zip(
                campaign.TRAIN_GAMES,
                campaign.TRAIN_FIRST_GAME_INDEX,
                strict=True,
            )
        )
    ] + [
        ("validation", index, games, first)
        for index, (games, first) in enumerate(
            zip(
                campaign.VALIDATION_GAMES,
                campaign.VALIDATION_FIRST_GAME_INDEX,
                strict=True,
            )
        )
    ]
    for split, index, games, first in specifications:
        root = tmp_path / f"corpus-{split}-part-{index}"
        root.mkdir()
        manifest = {
            "schema_version": 1,
            "dataset_id": f"{campaign.STRATEGY_ID}-{split}-{first}",
            "feature_schema": campaign.FEATURE_SCHEMA,
            "target_schema": campaign.TARGET_SCHEMA,
            "record_size": 864,
            "game": {
                "player_count": 4,
                "mode": "Standard",
                "scoring_cards": cards,
                "habitat_bonuses": False,
            },
            "split": split,
            "strategy": campaign.STRATEGY_ID,
            "first_game_index": first,
            "requested_games": games,
            "completed_games": games,
            "total_records": games * 80,
            "provenance": {"v2_source_blake3": "7" * 64},
        }
        manifest_path = root / "dataset.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        digest = blake3.blake3(manifest_path.read_bytes()).hexdigest()
        parts.append(
            campaign.DatasetPart(
                split=split,
                part_index=index,
                games=games,
                first_game_index=first,
                root=root,
                manifest_blake3=digest,
            )
        )
        roots.append(root)
    monkeypatch.setattr(campaign, "dataset_parts", lambda *args, **kwargs: tuple(parts))
    return roots


def _r0_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scientific_identity = {"arm": campaign.R0_EXACT_CONTROL, "complete": True}
    report = {
        "arm": campaign.R0_EXACT_CONTROL,
        "report_id": campaign.canonical_blake3(scientific_identity),
        "scientific_identity": scientific_identity,
        "metrics": {
            "validation": {
                "samples": 10_000,
                "mean_component_mae": 2.7,
                "total_mae": 2.65,
                "total_rmse": 3.37,
            }
        },
        "integrity": {"all_metrics_finite": True},
        "claims": {"promotion_authorized": False},
    }
    report_path = tmp_path / "r0-exact.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    collection = tmp_path / "collection.json"
    collection.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "arm": campaign.R0_EXACT_CONTROL,
                        "file": str(report_path),
                        "blake3": campaign.file_blake3(report_path),
                        "report_id": report["report_id"],
                    }
                ]
            }
        )
    )
    classification = {
        "experiment_id": "r0-spatial-mlx-tournament-v1",
        "adr": "0142",
        "classification": campaign.R0_COMPLETE_CLASSIFICATION,
        "aggregate_id": "a" * 64,
        "selected_stage2_candidate": None,
    }
    payload = json.dumps(classification, indent=2, sort_keys=True).encode() + b"\n"
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    forward.write_bytes(payload)
    reverse.write_bytes(payload)
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "byte_identical": True,
                "classification_blake3": blake3.blake3(payload).hexdigest(),
            }
        )
    )
    return forward, reverse, proof, collection


def test_corpus_lock_is_deterministic_and_binds_foundation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _synthetic_corpus(tmp_path, monkeypatch)
    first = campaign.freeze_corpus(roots, dataset_root=tmp_path / "unused")
    second = campaign.freeze_corpus(roots, dataset_root=tmp_path / "unused")
    assert first == second
    assert first["identity"]["foundation_scientific_blake3"] == (
        campaign.FOUNDATION_SCIENTIFIC_BLAKE3
    )
    assert first["identity"]["layer_maxima"] == campaign.EXPECTED_LAYER_MAXIMA
    assert first["identity"]["type_token_totals"] == (
        campaign.EXPECTED_TYPE_TOKEN_TOTALS
    )
    assert first["identity"]["active_tokens"] == campaign.EXPECTED_ACTIVE_TOKENS
    assert first["identity"]["per_board_p99_active_tokens"] == 83
    assert first["identity"]["per_board_max_active_tokens"] == 92
    assert first["identity"]["train_records"] == 50_000
    assert first["identity"]["validation_records"] == 10_000


def test_r0_null_selection_binds_exact_control_and_fails_closed(
    tmp_path: Path,
) -> None:
    forward, reverse, proof, collection = _r0_files(tmp_path)
    binding = campaign.bind_r0_control(
        classification_forward=forward,
        classification_reverse=reverse,
        order_proof=proof,
        collection=collection,
    )
    assert binding["identity"]["r0_selected_stage2_candidate"] is None
    assert binding["identity"]["selected_control_arm"] == campaign.R0_EXACT_CONTROL
    reverse.write_text("{}\n")
    with pytest.raises(campaign.CampaignError, match="byte-identical"):
        campaign.bind_r0_control(
            classification_forward=forward,
            classification_reverse=reverse,
            order_proof=proof,
            collection=collection,
        )


def test_authorization_binds_bundle_corpus_r0_and_full_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    lock_path = tmp_path / "control/corpus-lock.json"
    r0_path = tmp_path / "control/r0-control-binding.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "lock_id": "1" * 64,
    }
    r0 = {"binding_id": "2" * 64}
    monkeypatch.setattr(campaign, "validate_corpus_lock", lambda path: lock)
    monkeypatch.setattr(campaign, "validate_r0_control_binding", lambda path: r0)
    monkeypatch.setattr(
        campaign,
        "validate_bundle_for_campaign",
        lambda path: {
            "bundle_id": "3" * 64,
            "identity": {
                "binaries": [
                    {
                        "name": campaign.EXPORTER_BINARY,
                        "blake3": "4" * 64,
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(
        campaign,
        "source_provenance",
        lambda path: {"v2_source_blake3": "5" * 64},
    )
    monkeypatch.setattr(campaign, "file_blake3", lambda path: "4" * 64)
    authorization = campaign.create_authorization(
        bundle=bundle,
        corpus_lock=lock_path,
        r0_control=r0_path,
        approved_by="parent",
        approved_unix_ms=99,
    )
    identity = authorization["identity"]
    assert identity["bundle_id"] == "3" * 64
    assert identity["corpus_lock_id"] == "1" * 64
    assert identity["r0_control_binding_id"] == "2" * 64
    assert identity["authorized_runs"] == list(campaign.AUTHORIZED_RUNS)
    assert identity["run_architectures"] == campaign.RUN_ARCHITECTURES


def test_inert_task_graph_uses_all_four_hosts_without_queue_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/experiment/bundles/bundle-id"
    control = repository / "artifacts/experiment/control"
    experiment = repository / "artifacts/experiment"
    bundle.mkdir(parents=True)
    control.mkdir(parents=True)
    lock = control / "corpus-lock.json"
    r0 = control / "r0-control-binding.json"
    authorization = control / "authorization.json"
    for path in (lock, r0, authorization):
        path.touch()
    monkeypatch.setattr(
        campaign,
        "validate_bundle_for_campaign",
        lambda path: {"bundle_id": "a" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "validate_authorization",
        lambda *args, **kwargs: {"authorization_id": "b" * 64},
    )
    specs = campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        corpus_lock=lock,
        r0_control=r0,
        authorization=authorization,
        experiment_root=experiment,
        dataset_root=Path("artifacts/datasets/corpus"),
    )
    by_id = {spec["id"]: spec for spec in specs}
    assert len(specs) == 16
    assert sum(spec["id"] == "r2smlx-export-cache" for spec in specs) == 1
    for role, host in campaign.RUN_HOSTS.items():
        task = by_id[f"r2smlx-run-{role}"]
        assert task["compatible_hosts"] == [host]
        assert task["resources"]["uses_mlx"] is True
    assert by_id["r2smlx-classification-order-proof"]["decision_terminal"] is True
    for spec in specs:
        command_text = "\0".join(spec["command"])
        assert "cluster_research_queue" not in command_text
        if "tools/r2_sparse_mlx_" in command_text:
            assert "-B" in spec["command"]


def test_classification_order_proof_rejects_byte_drift(tmp_path: Path) -> None:
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    forward.write_bytes(b'{"same":true}\n')
    reverse.write_bytes(b'{"same":true}\n')
    assert campaign.compare_classifications(forward, reverse)["byte_identical"] is True
    reverse.write_bytes(b'{"same":false}\n')
    with pytest.raises(campaign.CampaignError, match="differ"):
        campaign.compare_classifications(forward, reverse)
