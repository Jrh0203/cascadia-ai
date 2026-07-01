from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from cascadia_mlx.p1_relational_pointer_data import PointerParentBatch
from cascadia_mlx.p1_relational_pointer_model import PointerParentEncoding
from cascadia_mlx.p1_relational_pointer_train import (
    AUTHORIZED_SUCCESSOR,
    DEFAULT_WARM_START_CHECKPOINT,
    DEFAULT_WARM_START_REPORT,
    EXPECTED_PARENT_PARAMETER_BLAKE3,
    FOUNDATION_ADR_ID,
    FOUNDATION_EXPERIMENT_ID,
    FOUNDATION_PASS,
    FOUNDATION_PROTOCOL_ID,
    PointerParentEncodingMemo,
    PointerStageTrainingConfig,
    PointerStageTrainingProtocol,
    _append_metric_event_once,
    _canonical_blake3,
    _read_metric_events,
    calibrated_stage_selection_key,
    load_verified_c0_parent,
    publish_selected_checkpoint,
    require_foundation_classification,
)
from mlx.utils import tree_flatten


class _FakeParentModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode_parent(self, parent: PointerParentBatch) -> PointerParentEncoding:
        self.calls += 1
        groups = parent.global_features.shape[0]
        values = mx.arange(groups * 2, dtype=mx.float32).reshape(groups, 2)
        return PointerParentEncoding(
            summary=values,
            active_tokens=values[:, None, :],
            active_mask=mx.ones((groups, 1), dtype=mx.bool_),
            active_types=mx.ones((groups, 1), dtype=mx.int32),
        )


def _parent(groups: int) -> PointerParentBatch:
    return PointerParentBatch(
        r2_token_features=mx.zeros((groups, 1, 1, 1)),
        r2_token_types=mx.zeros((groups, 1, 1), dtype=mx.int32),
        r2_token_mask=mx.ones((groups, 1, 1), dtype=mx.bool_),
        relational_values=mx.zeros((groups, 1, 0, 1)),
        relational_classes=mx.zeros((groups, 1, 0), dtype=mx.int32),
        relational_mask=mx.zeros((groups, 1, 0), dtype=mx.bool_),
        market_features=mx.zeros((groups, 1, 1)),
        market_mask=mx.ones((groups, 1), dtype=mx.bool_),
        player_features=mx.zeros((groups, 1, 1)),
        player_mask=mx.ones((groups, 1), dtype=mx.bool_),
        global_features=mx.zeros((groups, 1)),
    )


def test_parent_encoding_memo_reuses_group_transform_pairs() -> None:
    model = _FakeParentModel()
    batch = SimpleNamespace(
        parent=_parent(2),
        parent_group_ids=[7, 9],
        parent_transform_ids=[1, 4],
    )
    memo = PointerParentEncodingMemo()
    first = memo.encoding(model, batch)
    second = memo.encoding(model, batch)
    mx.eval(first.summary, second.summary)
    assert model.calls == 1
    assert memo.stats.requested_parents == 4
    assert memo.stats.encoded_parents == 2
    assert memo.stats.cache_hits == 2


def test_protocol_and_smoke_launch_contracts_are_frozen(tmp_path: Path) -> None:
    PointerStageTrainingProtocol.frozen("tile").validate()
    with pytest.raises(ValueError, match="drifted"):
        PointerStageTrainingProtocol(
            stage="tile",
            seed=1,
            epochs=20,
            batch_size=32,
        ).validate()
    PointerStageTrainingConfig(
        stage="draft",
        run_dir=tmp_path / "run",
        output=tmp_path / "report.json",
        foundation_classification=None,
        smoke_batches=1,
    ).validate()
    with pytest.raises(ValueError, match="at most 10"):
        PointerStageTrainingConfig(
            stage="draft",
            run_dir=tmp_path / "run",
            output=tmp_path / "report.json",
            foundation_classification=None,
            smoke_batches=11,
        ).validate()


def test_foundation_classifier_gate_is_fail_closed(tmp_path: Path) -> None:
    identity = {
        "schema_version": 1,
        "experiment_id": FOUNDATION_EXPERIMENT_ID,
        "protocol_id": FOUNDATION_PROTOCOL_ID,
        "adr": FOUNDATION_ADR_ID,
        "kind": "terminal-classification",
        "splits": {
            "train": {"scientific_blake3": "a" * 64},
            "validation": {"scientific_blake3": "b" * 64},
        },
        "source": {"bundle_id": "c" * 64},
        "gates": {
            "structural": True,
            "cross_host_consistent": True,
            "all_split_gates_passed": True,
        },
        "passed": True,
        "classification": FOUNDATION_PASS,
        "authorized_successor": AUTHORIZED_SUCCESSOR,
        "claim_boundary": "test",
    }
    report = {
        "schema_version": 1,
        "scientific_identity": identity,
        "scientific_blake3": _canonical_blake3(identity),
        "runtime": {"host": "test"},
    }
    path = tmp_path / "classification.json"
    path.write_text(json.dumps(report))
    accepted = require_foundation_classification(path)
    assert accepted["classification"] == FOUNDATION_PASS
    identity["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="did not authorize"):
        require_foundation_classification(path)


def test_metric_jsonl_is_idempotent_and_detects_drift(tmp_path: Path) -> None:
    identity = {
        "schema_version": 1,
        "experiment_id": "test",
        "epoch": 1,
        "selection_key": [1.0, 0.5, -2.0],
    }
    event = {
        "schema_version": 1,
        "scientific_identity": identity,
        "scientific_blake3": _canonical_blake3(identity),
        "runtime": {},
        "checkpoint": "one",
    }
    path = tmp_path / "metrics.jsonl"
    _append_metric_event_once(path, event)
    _append_metric_event_once(path, event)
    assert len(path.read_text().splitlines()) == 1
    assert _read_metric_events(path)[0]["selection_key"] == [1.0, 0.5, -2.0]
    changed = dict(event)
    changed_identity = {**identity, "selection_key": [0.0, 0.0, 0.0]}
    changed["scientific_identity"] = changed_identity
    changed["scientific_blake3"] = _canonical_blake3(changed_identity)
    with pytest.raises(ValueError, match="changed scientific"):
        _append_metric_event_once(path, changed)


def test_selected_checkpoint_is_published_at_stable_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints/step-000000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text('{"schema_version": 1}\n')
    (checkpoint / "model.safetensors").write_bytes(b"model")
    identity = publish_selected_checkpoint(
        run_dir=tmp_path,
        checkpoint=checkpoint,
    )
    assert publish_selected_checkpoint(
        run_dir=tmp_path,
        checkpoint=checkpoint,
    ) == identity
    assert (tmp_path / "selected/model.safetensors").read_bytes() == b"model"


def test_calibrated_selection_key_matches_historical_ordering() -> None:
    assert calibrated_stage_selection_key(
        {
            "target_factor_recall": 0.9,
            "exact_query_fraction": 0.8,
            "rank_mean_absolute_error": 1.5,
        }
    ) == (0.9, 0.8, -1.5)


@pytest.mark.skipif(
    not DEFAULT_WARM_START_CHECKPOINT.is_dir()
    or not DEFAULT_WARM_START_REPORT.is_file(),
    reason="accepted C0 checkpoint is not installed",
)
def test_actual_c0_parent_warm_start_is_exact() -> None:
    model, identity = load_verified_c0_parent(
        checkpoint_dir=DEFAULT_WARM_START_CHECKPOINT,
        report_path=DEFAULT_WARM_START_REPORT,
        stage="draft",
        seed=PointerStageTrainingProtocol.frozen("draft").seed,
    )
    assert identity["parent_parameter_tensor_blake3"] == (
        EXPECTED_PARENT_PARAMETER_BLAKE3
    )
    assert all(
        not name.startswith("parent_encoder.")
        for name, _value in tree_flatten(model.trainable_parameters())
    )
