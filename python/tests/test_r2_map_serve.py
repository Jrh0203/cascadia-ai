from __future__ import annotations

import io
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.r2_map_model import (
    R2MapMarketDecisionPrediction,
    R2MapMarketSurvival,
    R2MapOpponentNextAction,
    R2MapPrediction,
)
from cascadia_mlx.r2_map_serve import (
    FRAME_HEADER,
    MARKET_REQUEST_SCHEMA,
    MARKET_REQUEST_SCHEMA_BLAKE3,
    MARKET_RESPONSE_SCHEMA,
    MARKET_RESPONSE_SCHEMA_BLAKE3,
    MARKET_RESPONSE_TENSOR_DTYPES,
    MAX_GROUPS,
    MESSAGE_ERROR,
    MESSAGE_SCORE_GROUPS,
    MESSAGE_SCORE_RESPONSE,
    MESSAGE_SHUTDOWN,
    PROTOCOL_MAGIC,
    PROTOCOL_MAX_CANDIDATES_PER_GROUP,
    PROTOCOL_VERSION,
    REFERENCE_MAX_CANDIDATES_PER_GROUP,
    REQUEST_SCHEMA,
    REQUEST_SCHEMA_BLAKE3,
    REQUEST_TENSOR_DTYPES,
    RESPONSE_SCHEMA,
    RESPONSE_TENSOR_DTYPES,
    SERVING_BUNDLE_SCHEMA,
    R2MapCheckpointRegistry,
    R2MapProtocolError,
    R2MapRegistryEntry,
    _evaluate_prediction,
    _materialize_wave_batch,
    decode_tensor_payload,
    encode_tensor_frame,
    ordered_action_ids_blake3,
    score_grouped_request,
    score_market_decision_request,
    serve_r2_map,
    validate_action_groups,
    validate_grouped_request,
    validate_market_decision_request,
)
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)


def _digest(label: str) -> str:
    return blake3.blake3(label.encode()).hexdigest()


class FakeR2MapModel:
    def __init__(self, bias: float):
        self.bias = bias
        self.calls = 0
        self.parent_groups = 0
        self.action_evaluations = 0

    def __call__(self, batch: Any) -> R2MapPrediction:
        groups, candidates = batch.validate()
        self.calls += 1
        self.parent_groups += groups
        self.action_evaluations += int(mx.sum(batch.candidate_mask).item())
        valid = batch.candidate_mask
        scores = mx.where(valid, batch.exact_afterstate_scores + self.bias, -mx.inf)
        to_go = valid.astype(mx.float32) * self.bias
        components = mx.ones((groups, candidates, 11)) * self.bias * valid[..., None]
        policy = mx.where(valid, mx.zeros((groups, candidates)), -mx.inf)
        opponent = R2MapOpponentNextAction(
            tile_slot_logits=mx.ones((groups, candidates, 3, 4)) * self.bias,
            wildlife_slot_logits=mx.ones((groups, candidates, 3, 4)) * self.bias,
            draft_kind_logits=mx.ones((groups, candidates, 3, 2)) * self.bias,
            drafted_wildlife_logits=mx.ones((groups, candidates, 3, 5)) * self.bias,
            replace_three_logits=mx.ones((groups, candidates, 3, 2)) * self.bias,
            paid_wipe_count_logits=mx.ones((groups, candidates, 3, 21)) * self.bias,
            paid_wipe_mask_logits=mx.ones((groups, candidates, 3, 20, 16))
            * self.bias,
        )
        survival = R2MapMarketSurvival(
            disposition_logits=mx.ones((groups, candidates, 4, 4)) * self.bias,
            pair_survival_logits=mx.ones((groups, candidates, 4, 2)) * self.bias,
            final_slot_logits=mx.ones((groups, candidates, 4, 4)) * self.bias,
        )
        return R2MapPrediction(
            action_scores=scores,
            predicted_score_to_go=to_go,
            predicted_score_components_to_go=components,
            bootstrap_policy_logits=policy,
            opponent_next_action=opponent,
            market_survival=survival,
            candidate_mask=valid,
        )

    def score_actions(self, batch: Any) -> R2MapPrediction:
        return self(batch)

    def score_market_decisions(self, batch: Any) -> R2MapMarketDecisionPrediction:
        groups, actions = batch.validate()
        self.calls += 1
        self.parent_groups += groups
        self.action_evaluations += int(mx.sum(batch.action_mask).item())
        valid = batch.action_mask
        to_go = valid.astype(mx.float32) * self.bias
        return R2MapMarketDecisionPrediction(
            action_scores=mx.where(
                valid,
                batch.exact_current_scores[:, None] + to_go,
                -mx.inf,
            ),
            predicted_score_to_go=to_go,
            bootstrap_policy_logits=mx.where(
                valid, mx.zeros((groups, actions), dtype=mx.float32), -mx.inf
            ),
            action_mask=valid,
        )


def _entry(label: str, bias: float) -> R2MapRegistryEntry:
    return R2MapRegistryEntry(
        checkpoint_id=f"checkpoint-{label}",
        checkpoint_manifest_blake3=_digest(f"manifest-{label}"),
        model_config_blake3=_digest(f"config-{label}"),
        model_weights_blake3=_digest(f"weights-{label}"),
        verification_id=_digest(f"verification-{label}"),
        model=FakeR2MapModel(bias),
    )


def _group(index: int, count: int, entry: R2MapRegistryEntry) -> dict[str, Any]:
    action_ids = [_digest(f"group-{index}-action-{action}") for action in range(count)]
    return {
        "group_id": _digest(f"group-{index}"),
        "decision_id": _digest(f"decision-{index}"),
        "model": entry.identity(),
        "expected_legal_action_count": count,
        "action_ids": action_ids,
        "enumeration_indices": list(range(count)),
        "ordered_action_ids_blake3": ordered_action_ids_blake3(action_ids),
    }


def _request(
    counts: list[int], entries: list[R2MapRegistryEntry]
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    assert len(counts) == len(entries)
    offsets = np.asarray([0, *np.cumsum(counts)], dtype="<i4")
    groups = [_group(index, count, entries[index]) for index, count in enumerate(counts)]
    group_count = len(groups)
    candidate_count = int(offsets[-1])
    tensors: dict[str, np.ndarray[Any, Any]] = {
        "candidate_offsets": offsets,
        "parent_token_features": np.zeros(
            (group_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES), dtype="<f4"
        ),
        "parent_token_types": np.zeros(
            (group_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="<i4"
        ),
        "parent_token_mask": np.zeros(
            (group_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="|u1"
        ),
        "parent_market_features": np.zeros((group_count, 4, MARKET_FEATURES), dtype="<f4"),
        "parent_market_mask": np.ones((group_count, 4), dtype="|u1"),
        "parent_player_features": np.zeros(
            (group_count, BOARD_SLOTS, PLAYER_FEATURES), dtype="<f4"
        ),
        "parent_player_mask": np.ones((group_count, BOARD_SLOTS), dtype="|u1"),
        "parent_global_features": np.zeros((group_count, GLOBAL_FEATURES), dtype="<f4"),
        "candidate_token_features": np.zeros(
            (candidate_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
            dtype="<f4",
        ),
        "candidate_token_types": np.zeros(
            (candidate_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="<i4"
        ),
        "candidate_token_mask": np.zeros(
            (candidate_count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="|u1"
        ),
        "candidate_market_features": np.zeros((candidate_count, 4, MARKET_FEATURES), dtype="<f4"),
        "candidate_market_mask": np.ones((candidate_count, 4), dtype="|u1"),
        "candidate_player_features": np.zeros(
            (candidate_count, BOARD_SLOTS, PLAYER_FEATURES), dtype="<f4"
        ),
        "candidate_player_mask": np.ones((candidate_count, BOARD_SLOTS), dtype="|u1"),
        "candidate_global_features": np.zeros((candidate_count, GLOBAL_FEATURES), dtype="<f4"),
        "action_bytes": np.zeros((candidate_count, 128), dtype="|u1"),
        "exact_afterstate_scores": np.arange(10, 10 + candidate_count, dtype="<f4"),
    }
    tensors["parent_token_mask"][..., :4] = 1
    tensors["candidate_token_mask"][..., :4] = 1
    metadata = {
        "schema_version": 1,
        "schema_id": REQUEST_SCHEMA,
        "request_schema_blake3": REQUEST_SCHEMA_BLAKE3,
        "group_count": group_count,
        "candidate_count": candidate_count,
        "groups": groups,
    }
    return metadata, tensors


def _market_request(
    entry: R2MapRegistryEntry, *, case_name: str = "paid-all-subsets"
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    fixture = json.loads(
        Path("tests/fixtures/r2_map/public-market-decision-protocol-v3.json").read_text()
    )
    case = next(value for value in fixture["cases"] if value["name"] == case_name)
    count = len(case["action_ids"])
    tensors: dict[str, np.ndarray[Any, Any]] = {
        "action_offsets": np.asarray([0, count], dtype="<i4"),
        "parent_token_features": np.zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES), dtype="<f4"
        ),
        "parent_token_types": np.zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="<i4"
        ),
        "parent_token_mask": np.zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype="|u1"
        ),
        "parent_market_features": np.zeros((1, 4, MARKET_FEATURES), dtype="<f4"),
        "parent_market_mask": np.ones((1, 4), dtype="|u1"),
        "parent_player_features": np.zeros(
            (1, BOARD_SLOTS, PLAYER_FEATURES), dtype="<f4"
        ),
        "parent_player_mask": np.ones((1, BOARD_SLOTS), dtype="|u1"),
        "parent_global_features": np.zeros((1, GLOBAL_FEATURES), dtype="<f4"),
        "action_bytes": np.asarray(
            [list(bytes.fromhex(value)) for value in case["action_bytes_hex"]],
            dtype="|u1",
        ),
        "exact_current_scores": np.asarray([10.0], dtype="<f4"),
    }
    tensors["parent_token_mask"][..., :4] = 1
    group = {
        "group_id": _digest(f"market-{case_name}-group"),
        "decision_id": case["decision_id"],
        "model": entry.identity(),
        "expected_legal_action_count": count,
        "action_ids": case["action_ids"],
        "enumeration_indices": list(range(count)),
        "ordered_action_ids_blake3": case["ordered_action_ids_blake3"],
        "public_nature_tokens": case["public_nature_tokens"],
        "public_wildlife_bag_total": case["public_wildlife_bag_total"],
        "public_wildlife_bag_counts": case["public_wildlife_bag_counts"],
        "public_market_wildlife": case["public_market_wildlife"],
        "decision_kind": case["decision_kind"],
    }
    return (
        {
            "schema_version": 1,
            "schema_id": MARKET_REQUEST_SCHEMA,
            "request_schema_blake3": MARKET_REQUEST_SCHEMA_BLAKE3,
            "group_count": 1,
            "action_count": count,
            "groups": [group],
        },
        tensors,
    )


def _registry(*entries: R2MapRegistryEntry) -> R2MapCheckpointRegistry:
    registry = R2MapCheckpointRegistry(capacity=max(len(entries), 1))
    for entry in entries:
        registry.register_model(entry)
    return registry


def _protocols(value: int) -> dict[str, list[int]]:
    return {
        "collector_hash": [value] * 32,
        "source_hash": [value + 1] * 32,
        "serving_protocol_hash": [value + 2] * 32,
    }


def test_serving_bundle_v2_rejects_stale_or_zero_protocol_identity(tmp_path: Path) -> None:
    entry = {
        "manifest_identity_blake3": _digest("compact"),
        "run_dir": str(tmp_path.resolve()),
        "checkpoint_path": str((tmp_path / "checkpoint").resolve()),
        "model": _entry("bundle", 0.0).identity(),
        "pinned": False,
    }
    path = tmp_path / "bundle.json"
    stale = {
        "schema_version": 2,
        "schema_id": SERVING_BUNDLE_SCHEMA,
        "protocols": _protocols(10),
        "entries": [entry],
    }
    path.write_text(json.dumps(stale))
    registry = R2MapCheckpointRegistry(expected_protocols=_protocols(20))
    with pytest.raises(R2MapProtocolError, match="protocol identity is stale"):
        registry.register_verified_bundle(path)

    stale["protocols"]["source_hash"] = [0] * 32
    path.write_text(json.dumps(stale))
    with pytest.raises(R2MapProtocolError, match="source_hash is invalid"):
        R2MapCheckpointRegistry().register_verified_bundle(path)


def _decode_single_response(data: bytes) -> tuple[tuple[Any, ...], dict, dict]:
    header = FRAME_HEADER.unpack(data[: FRAME_HEADER.size])
    metadata_size = header[4]
    tensor_size = header[5]
    start = FRAME_HEADER.size
    metadata, tensors = decode_tensor_payload(
        data[start : start + metadata_size],
        data[start + metadata_size : start + metadata_size + tensor_size],
        expected_dtypes=RESPONSE_TENSOR_DTYPES,
    )
    return header, metadata, tensors


def test_reference_service_scores_every_action_once_and_shuts_down_cleanly() -> None:
    entry = _entry("a", 1.5)
    metadata, tensors = _request([3], [entry])
    request = encode_tensor_frame(
        message_type=MESSAGE_SCORE_GROUPS,
        request_id=17,
        metadata=metadata,
        tensors=tensors,
        expected_dtypes=REQUEST_TENSOR_DTYPES,
    )
    shutdown = FRAME_HEADER.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, MESSAGE_SHUTDOWN, 18, 0, 0)
    registry = _registry(entry)
    output = io.BytesIO()
    serve_r2_map(registry, io.BytesIO(request + shutdown), output)
    header, response, values = _decode_single_response(output.getvalue())

    assert header[:4] == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SCORE_RESPONSE,
        17,
    )
    assert response["schema_id"] == RESPONSE_SCHEMA
    assert response["groups"][0]["action_ids"] == metadata["groups"][0]["action_ids"]
    assert response["groups"][0]["diagnostics"] == {
        "parent_groups_encoded": 1,
        "actions_enumerated": 3,
        "actions_scored": 3,
        "complete_cardinality": True,
    }
    np.testing.assert_array_equal(values["action_scores"], [11.5, 12.5, 13.5])
    assert entry.model.calls == 1
    assert entry.model.parent_groups == 1
    assert entry.model.action_evaluations == 3
    assert registry.closed is True


def test_public_market_service_scores_exact_full_screen_without_refill_inputs() -> None:
    entry = _entry("market", 2.5)
    metadata, tensors = _market_request(entry)
    registry = _registry(entry)
    response, values = score_market_decision_request(registry, metadata, tensors)
    assert response["schema_id"] == MARKET_RESPONSE_SCHEMA
    assert response["request_schema_blake3"] == MARKET_REQUEST_SCHEMA_BLAKE3
    assert response["response_schema_blake3"] == MARKET_RESPONSE_SCHEMA_BLAKE3
    assert set(values) == set(MARKET_RESPONSE_TENSOR_DTYPES) == {
        "market_action_scores",
        "market_predicted_score_to_go",
    }
    assert values["market_action_scores"].tolist() == [12.5] * 16
    assert values["market_predicted_score_to_go"].tolist() == [2.5] * 16
    group = response["groups"][0]
    assert group["decision_kind"] == 1
    assert group["public_nature_tokens"] == 2
    assert group["public_wildlife_bag_total"] == sum(group["public_wildlife_bag_counts"])
    assert group["public_market_wildlife"] == [0, 1, 2, 3]
    assert group["diagnostics"] == {
        "actions_enumerated": 16,
        "actions_scored": 16,
        "complete_cardinality": True,
        "hidden_refill_inputs": 0,
    }
    assert response["diagnostics"]["pruned_actions"] == 0
    assert response["diagnostics"]["future_refill_tensors"] == 0
    assert entry.model.parent_groups == 1
    assert entry.model.action_evaluations == 16


def test_public_market_service_rejects_partial_or_future_augmented_requests() -> None:
    entry = _entry("market-reject", 0.0)
    metadata, tensors = _market_request(entry)
    partial = {name: value.copy() for name, value in tensors.items()}
    partial["action_offsets"][-1] -= 1
    metadata_partial = {
        **metadata,
        "action_count": metadata["action_count"] - 1,
        "groups": [
            {
                **metadata["groups"][0],
                "expected_legal_action_count": metadata["action_count"] - 1,
                "action_ids": metadata["groups"][0]["action_ids"][:-1],
                "enumeration_indices": list(range(metadata["action_count"] - 1)),
                "ordered_action_ids_blake3": ordered_action_ids_blake3(
                    metadata["groups"][0]["action_ids"][:-1]
                ),
            }
        ],
    }
    with pytest.raises(R2MapProtocolError):
        validate_market_decision_request(metadata_partial, partial)

    future = {**tensors, "future_refill": np.zeros((1, 4), dtype="|u1")}
    with pytest.raises(R2MapProtocolError, match="tensor names"):
        validate_market_decision_request(metadata, future)


def test_multi_checkpoint_groups_are_cohorted_without_reordering() -> None:
    first = _entry("first", 1.0)
    second = _entry("second", 100.0)
    metadata, tensors = _request([2, 1, 2], [first, second, first])
    registry = _registry(first, second)
    response, values = score_grouped_request(registry, metadata, tensors)

    np.testing.assert_array_equal(values["action_scores"], [11.0, 12.0, 112.0, 14.0, 15.0])
    assert [group["group_id"] for group in response["groups"]] == [
        group["group_id"] for group in metadata["groups"]
    ]
    assert response["diagnostics"]["checkpoint_waves"] == 2
    assert values["predicted_score_components_to_go"].shape == (5, 11)
    assert set(values) == set(RESPONSE_TENSOR_DTYPES) == {
        "action_scores",
        "predicted_score_to_go",
        "predicted_score_components_to_go",
        "bootstrap_policy_logits",
    }
    assert first.model.calls == second.model.calls == 1
    assert first.model.parent_groups == 2
    assert first.model.action_evaluations == 4
    assert second.model.parent_groups == 1
    assert second.model.action_evaluations == 1


def test_maximum_width_action_contract_is_accepted_without_materializing_padding() -> None:
    entry = _entry("maximum", 0.0)
    group = _group(0, REFERENCE_MAX_CANDIDATES_PER_GROUP, entry)
    offsets = np.asarray([0, REFERENCE_MAX_CANDIDATES_PER_GROUP], dtype="<i4")
    validate_action_groups([group], offsets)
    group["expected_legal_action_count"] += 1
    with pytest.raises(R2MapProtocolError, match="partial"):
        validate_action_groups([group], offsets)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("partial", "partial"),
        ("duplicate-action", "duplicate action"),
        ("reordered", "reordered"),
        ("order-digest", "ordered action digest"),
        ("duplicate-group", "duplicate group"),
    ],
)
def test_group_contract_rejects_partial_duplicate_and_reordered_requests(
    mutation: str, message: str
) -> None:
    entry = _entry("malformed", 0.0)
    metadata, tensors = _request([2, 2], [entry, entry])
    groups = metadata["groups"]
    if mutation == "partial":
        groups[0]["expected_legal_action_count"] = 1
    elif mutation == "duplicate-action":
        groups[0]["action_ids"][1] = groups[0]["action_ids"][0]
    elif mutation == "reordered":
        groups[0]["enumeration_indices"] = [1, 0]
    elif mutation == "order-digest":
        groups[0]["ordered_action_ids_blake3"] = _digest("wrong-order")
    elif mutation == "duplicate-group":
        groups[1]["group_id"] = groups[0]["group_id"]
    with pytest.raises(R2MapProtocolError, match=message):
        validate_grouped_request(metadata, tensors)


def test_tensor_contract_rejects_missing_wrong_dtype_and_wrong_shape() -> None:
    entry = _entry("tensor", 0.0)
    metadata, tensors = _request([2], [entry])
    missing = dict(tensors)
    del missing["action_bytes"]
    with pytest.raises(R2MapProtocolError, match="tensor names"):
        validate_grouped_request(metadata, missing)
    wrong_dtype = dict(tensors)
    wrong_dtype["exact_afterstate_scores"] = tensors["exact_afterstate_scores"].astype("<f8")
    with pytest.raises(R2MapProtocolError, match="shape or dtype"):
        validate_grouped_request(metadata, wrong_dtype)
    wrong_shape = dict(tensors)
    wrong_shape["action_bytes"] = tensors["action_bytes"][:, :127]
    with pytest.raises(R2MapProtocolError, match="shape or dtype"):
        validate_grouped_request(metadata, wrong_shape)


def test_live_protocol_rejects_the_obsolete_4x92_padding_shape() -> None:
    entry = _entry("obsolete-shape", 0.0)
    metadata, tensors = _request([1], [entry])
    assert BOARD_TOKEN_CAPACITY == 139
    obsolete = dict(tensors)
    obsolete["parent_token_features"] = tensors["parent_token_features"][:, :, :92, :]
    with pytest.raises(R2MapProtocolError, match="shape or dtype"):
        validate_grouped_request(metadata, obsolete)

    market_metadata, market_tensors = _market_request(entry)
    obsolete_market = dict(market_tensors)
    obsolete_market["parent_token_mask"] = market_tensors["parent_token_mask"][:, :, :92]
    with pytest.raises(R2MapProtocolError, match="shape or dtype"):
        validate_market_decision_request(market_metadata, obsolete_market)


def test_live_protocol_rejects_v2_frame_and_market_fixture() -> None:
    entry = _entry("obsolete-v2", 0.0)
    obsolete_shutdown = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        2,
        MESSAGE_SHUTDOWN,
        73,
        0,
        0,
    )
    registry = _registry(entry)
    output = io.BytesIO()
    serve_r2_map(registry, io.BytesIO(obsolete_shutdown), output)
    header = FRAME_HEADER.unpack(output.getvalue()[: FRAME_HEADER.size])
    assert header[:4] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, MESSAGE_ERROR, 73)
    assert b"incompatible R2-MAP protocol header" in output.getvalue()[
        FRAME_HEADER.size :
    ]
    assert registry.closed is True

    obsolete_fixture = json.loads(
        Path("tests/fixtures/r2_map/public-market-decision-protocol-v2.json").read_text()
    )
    metadata, tensors = _market_request(entry)
    metadata["schema_id"] = obsolete_fixture["request_schema"]
    metadata["request_schema_blake3"] = obsolete_fixture["request_schema_blake3"]
    with pytest.raises(R2MapProtocolError, match="schema or hash"):
        validate_market_decision_request(metadata, tensors)


def test_frame_payload_tamper_is_fatal_and_registry_closes() -> None:
    entry = _entry("tamper", 0.0)
    metadata, tensors = _request([1], [entry])
    request = bytearray(
        encode_tensor_frame(
            message_type=MESSAGE_SCORE_GROUPS,
            request_id=4,
            metadata=metadata,
            tensors=tensors,
            expected_dtypes=REQUEST_TENSOR_DTYPES,
        )
    )
    request[-1] ^= 1
    registry = _registry(entry)
    output = io.BytesIO()
    serve_r2_map(registry, io.BytesIO(request), output)
    header = FRAME_HEADER.unpack(output.getvalue()[: FRAME_HEADER.size])
    assert header[2] == MESSAGE_ERROR
    assert b"payload hash differs" in output.getvalue()[FRAME_HEADER.size :]
    assert registry.closed is True


def test_partial_frame_is_rejected_and_service_exits_cleanly() -> None:
    entry = _entry("partial-frame", 0.0)
    metadata, tensors = _request([1], [entry])
    request = encode_tensor_frame(
        message_type=MESSAGE_SCORE_GROUPS,
        request_id=44,
        metadata=metadata,
        tensors=tensors,
        expected_dtypes=REQUEST_TENSOR_DTYPES,
    )
    registry = _registry(entry)
    output = io.BytesIO()
    serve_r2_map(registry, io.BytesIO(request[:-7]), output)
    header = FRAME_HEADER.unpack(output.getvalue()[: FRAME_HEADER.size])
    assert header[2] == MESSAGE_ERROR
    assert b"ended inside a frame" in output.getvalue()[FRAME_HEADER.size :]
    assert registry.closed is True


def test_registry_is_bounded_pinned_and_identity_strict() -> None:
    first = _entry("one", 1.0)
    second = _entry("two", 2.0)
    third = _entry("three", 3.0)
    registry = R2MapCheckpointRegistry(capacity=2)
    registry.register_model(first, pinned=True)
    registry.register_model(second)
    registry.register_model(third)
    assert registry.checkpoint_ids == (first.checkpoint_id, third.checkpoint_id)
    registry.pin(third.checkpoint_id)
    with pytest.raises(R2MapProtocolError, match="pinned"):
        registry.register_model(_entry("four", 4.0))
    drifted = dict(first.identity())
    drifted["model_weights_blake3"] = _digest("different")
    with pytest.raises(R2MapProtocolError, match="hashes differ"):
        registry.get(drifted)


def test_service_can_restart_with_identical_output() -> None:
    entry_one = _entry("restart", 7.0)
    metadata, tensors = _request([2], [entry_one])
    request = encode_tensor_frame(
        message_type=MESSAGE_SCORE_GROUPS,
        request_id=1,
        metadata=metadata,
        tensors=tensors,
        expected_dtypes=REQUEST_TENSOR_DTYPES,
    )
    shutdown = FRAME_HEADER.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, MESSAGE_SHUTDOWN, 2, 0, 0)
    first_output = io.BytesIO()
    serve_r2_map(_registry(entry_one), io.BytesIO(request + shutdown), first_output)

    entry_two = _entry("restart", 7.0)
    second_output = io.BytesIO()
    serve_r2_map(_registry(entry_two), io.BytesIO(request + shutdown), second_output)
    assert first_output.getvalue() == second_output.getvalue()


def test_protocol_header_and_limits_are_cross_language_fixed() -> None:
    assert FRAME_HEADER.size == struct.calcsize("<4sHHIII") == 20
    assert MAX_GROUPS == 16
    assert REFERENCE_MAX_CANDIDATES_PER_GROUP == 6_372
    assert PROTOCOL_MAX_CANDIDATES_PER_GROUP == 8_192


def test_prediction_validation_rejects_bad_shapes_and_nonfinite_legal_values() -> None:
    entry = _entry("prediction", 0.0)
    _, tensors = _request([2], [entry])
    batch = _materialize_wave_batch(tensors, tensors["candidate_offsets"], [0])

    prediction = replace(entry.model(batch), action_scores=mx.array([[1.0]]))
    with pytest.raises(R2MapProtocolError, match="action_scores shape"):
        _evaluate_prediction(prediction)

    prediction = replace(
        entry.model(batch), predicted_score_to_go=mx.array([[1.0, mx.nan]])
    )
    with pytest.raises(R2MapProtocolError, match="non-finite"):
        _evaluate_prediction(prediction)

    prediction = replace(
        entry.model(batch), candidate_mask=mx.ones((1, 2), dtype=mx.float32)
    )
    with pytest.raises(R2MapProtocolError, match="mask shape or dtype"):
        _evaluate_prediction(prediction)
