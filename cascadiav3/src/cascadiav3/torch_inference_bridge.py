"""JSONL stdio inference bridge for Rust-side CascadiaFormer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .replay import read_replay_jsonl

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)


TRAINING_LABEL_KEYS = {
    "per_action_Q",
    "per_action_Q_valid",
    "per_action_score_to_go",
    "per_action_Q_variance",
    "per_action_Q_count",
    "per_action_truncated_count",
    "visits",
    "selected_action",
    "final_score_vector",
    "rank_vector",
    "score_decomposition",
}


def derived_final_q_values(root: dict[str, Any], score_to_go: list[float]) -> list[float]:
    exact = root.get("exact_afterstate_score_active")
    if not isinstance(exact, list):
        raise ValueError("eval_request root missing exact_afterstate_score_active")
    if len(exact) != len(score_to_go):
        raise ValueError("exact_afterstate_score_active length must match score_to_go")
    return [float(afterstate) + float(remaining) for afterstate, remaining in zip(exact, score_to_go)]


def q_selection_index(root: dict[str, Any], score_to_go: list[float]) -> int:
    final_q = derived_final_q_values(root, score_to_go)
    if not final_q:
        raise ValueError("cannot select from an empty q vector")
    return max(range(len(final_q)), key=lambda index: (final_q[index], -index))


def _response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


PROTOCOL_FEATURES = ["eval_batch", "value_vector"]
EVAL_BATCH_CHUNK_SIZE = 32


def _uniform_eval(root: dict[str, Any], *, model_fallback: bool) -> dict[str, Any]:
    action_count = len(root["legal_actions"])
    if action_count == 0:
        raise ValueError("eval_request root has no legal actions")
    prior = 1.0 / action_count
    if root.get("per_action_score_to_go") is not None:
        score_to_go = [float(value) for value in root["per_action_score_to_go"]]
    elif root.get("per_action_Q") is not None and root.get("exact_afterstate_score_active") is not None:
        score_to_go = [
            float(q_value) - float(afterstate)
            for q_value, afterstate in zip(root["per_action_Q"], root["exact_afterstate_score_active"])
        ]
    else:
        score_to_go = [0.0] * action_count
    q_values = derived_final_q_values(root, score_to_go)
    return {
        "type": "eval_response",
        "schema_id": root.get("schema_id"),
        "state_hash": root.get("state_hash"),
        "action_ids": [action["action_id"] for action in root["legal_actions"]],
        "priors": [prior] * action_count,
        "q": q_values,
        "score_to_go": score_to_go,
        "uncertainty": [1.0] * action_count,
        "value": [0.0, 0.0, 0.0, 0.0],
        "model_fallback": model_fallback,
    }


def inference_request_view(root: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the public-only request fields used for inference."""
    required = ("state_hash", "active_seat", "legal_actions", "public_tokens", "exact_afterstate_score_active")
    missing = [key for key in required if key not in root]
    if missing:
        raise KeyError(f"eval_request root missing required public field(s): {missing}")
    legal_actions = root["legal_actions"]
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("eval_request root requires at least one legal action")
    action_ids = []
    for index, action in enumerate(legal_actions):
        action_id = action.get("action_id") if isinstance(action, dict) else None
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"eval_request legal action {index} is missing action_id")
        action_ids.append(action_id)
    public_tokens = root["public_tokens"]
    if not isinstance(public_tokens, dict) or "tokens" not in public_tokens:
        raise ValueError("eval_request root public_tokens must include tokens")
    exact_afterstate = root["exact_afterstate_score_active"]
    if not isinstance(exact_afterstate, list) or len(exact_afterstate) != len(legal_actions):
        raise ValueError("eval_request exact_afterstate_score_active must align with legal_actions")
    return {
        "schema_id": root.get("schema_id"),
        "state_hash": root["state_hash"],
        "active_seat": root["active_seat"],
        "legal_actions": legal_actions,
        "public_tokens": public_tokens,
        "exact_afterstate_score_active": [float(value) for value in exact_afterstate],
        "action_ids": action_ids,
        "training_labels_present": sorted(key for key in TRAINING_LABEL_KEYS if key in root),
    }


def collate_inference_roots(records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    from .torch_public_token_merit import PUBLIC_TOKEN_FEATURE_DIM, public_token_features
    from .torch_relation_bias_merit import combined_relation_ids_array, relation_counts
    from .torch_semantic_relation_bias_merit import (
        SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        semantic_public_token_action_features,
    )

    if not records:
        raise ValueError("collate_inference_roots requires at least one record")
    views = [inference_request_view(record) for record in records]
    batch_size = len(views)
    action_counts = [len(view["legal_actions"]) for view in views]
    token_counts = [int(view["public_tokens"].get("token_count", len(view["public_tokens"]["tokens"]))) for view in views]
    max_actions = max(action_counts)
    max_tokens = max(token_counts)
    tokens = torch.zeros((batch_size, max_tokens, PUBLIC_TOKEN_FEATURE_DIM), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    actions = torch.zeros((batch_size, max_actions, SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    seq_len = max_tokens + max_actions
    relation_ids = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long)
    exact_afterstate = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    relation_summaries = []
    for batch_index, record in enumerate(records):
        token_count = token_counts[batch_index]
        action_count = action_counts[batch_index]
        tokens[batch_index, :token_count] = torch.tensor(public_token_features(record), dtype=torch.float32)
        token_mask[batch_index, :token_count] = True
        actions[batch_index, :action_count] = torch.tensor(
            semantic_public_token_action_features(record),
            dtype=torch.float32,
        )
        action_mask[batch_index, :action_count] = True
        matrix = combined_relation_ids_array(record, action_offset=max_tokens, seq_len=seq_len)
        relation_ids[batch_index] = torch.from_numpy(matrix)
        exact_afterstate[batch_index, :action_count] = torch.tensor(
            views[batch_index]["exact_afterstate_score_active"],
            dtype=torch.float32,
        )
        relation_summaries.append(relation_counts(matrix))
    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "action_mask": action_mask,
        "relation_ids": relation_ids,
        "combined_seq_len": seq_len,
        "relation_id_counts": relation_summaries,
        "action_counts": action_counts,
        "token_counts": token_counts,
        "state_hashes": [view["state_hash"] for view in views],
        "action_ids": [view["action_ids"] for view in views],
        "exact_afterstate_score_active": exact_afterstate,
    }


def _inferred_project_roots(*paths: Path | None) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        if path is None:
            continue
        for parent in [path, *path.parents]:
            if parent.name == "cascadiav3":
                candidate = parent.parent
                if candidate not in roots:
                    roots.append(candidate)
    return roots


def resolve_checkpoint_path(
    raw_path: str,
    *,
    manifest_path: Path | None = None,
    checkpoint_path: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd = cwd or Path.cwd()
    candidates: list[Path] = [cwd / path]
    for root in _inferred_project_roots(manifest_path, checkpoint_path):
        candidate = root / path
        if candidate not in candidates:
            candidates.append(candidate)
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate not in candidates:
            candidates.append(candidate)
    if checkpoint_path is not None:
        candidate = checkpoint_path.parent / path
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _config_from_payload(payload: dict[str, Any]):  # type: ignore[no-untyped-def]
    from .torch_cascadiaformer import CascadiaFormerConfig

    config = payload.get("config", payload)
    allowed = set(CascadiaFormerConfig.__dataclass_fields__)
    values = {key: value for key, value in config.items() if key in allowed}
    if "score_categories" in values:
        values["score_categories"] = tuple(values["score_categories"])
    return CascadiaFormerConfig(**values)


def _load_model(
    checkpoint: Path,
    *,
    manifest_path: Path | None,
    manifest_payload: dict[str, Any] | None,
    device_name: str = "cpu",
):  # type: ignore[no-untyped-def]
    import torch
    from .torch_cascadiaformer import build_cascadiaformer

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    payload = manifest_payload
    if payload is None and checkpoint.suffix == ".json":
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        manifest_path = checkpoint
    if payload is None:
        raise RuntimeError("checkpoint serving requires a model manifest with config and weights")
    weights = resolve_checkpoint_path(
        payload.get("weights", str(checkpoint)),
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
    )
    model = build_cascadiaformer(_config_from_payload(payload))
    if payload.get("weights_format") == "safetensors" or weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _move_batch_to_device(batch: dict[str, Any], device):  # type: ignore[no-untyped-def]
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _model_eval_batch(
    model,
    roots: list[dict[str, Any]],
    *,
    device_name: str = "cpu",
    chunk_size: int = EVAL_BATCH_CHUNK_SIZE,
) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """One collated forward per chunk of roots. Chunking bounds the dense
    relation_ids tensor (batch x seq x seq int64) at full action menus."""
    import torch

    if not roots:
        return []
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    responses: list[dict[str, Any]] = []
    for start in range(0, len(roots), max(1, chunk_size)):
        chunk = roots[start : start + max(1, chunk_size)]
        batch = collate_inference_roots(chunk)
        eval_batch = _move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = model(
                eval_batch["tokens"],
                eval_batch["token_mask"],
                eval_batch["actions"],
                eval_batch["action_mask"],
                eval_batch.get("relation_ids"),
            )
            masked_logits = outputs["logits"].masked_fill(~eval_batch["action_mask"], -1.0e9)
            priors = torch.softmax(masked_logits, dim=1).cpu()
        score_to_go_all = outputs["q"].cpu()
        uncertainty_all = outputs["uncertainty"].cpu()
        value_all = outputs["value_vector"].cpu()
        exact_all = eval_batch["exact_afterstate_score_active"].cpu()
        for row_index, root in enumerate(chunk):
            action_count = batch["action_counts"][row_index]
            score_to_go = score_to_go_all[row_index, :action_count].tolist()
            final_q = (
                exact_all[row_index, :action_count] + score_to_go_all[row_index, :action_count]
            ).tolist()
            responses.append(
                {
                    "type": "eval_response",
                    "schema_id": root.get("schema_id"),
                    "state_hash": root.get("state_hash"),
                    "action_ids": batch["action_ids"][row_index],
                    "priors": priors[row_index, :action_count].tolist(),
                    "q": final_q,
                    "score_to_go": score_to_go,
                    "uncertainty": uncertainty_all[row_index, :action_count].tolist(),
                    "value": value_all[row_index].tolist(),
                    "model_fallback": False,
                }
            )
    return responses


def _model_eval(model, root: dict[str, Any], *, device_name: str = "cpu") -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _model_eval_batch(model, [root], device_name=device_name)[0]


def serve(
    *,
    checkpoint: Path | None,
    manifest: Path | None,
    allow_dry_run_fallback: bool,
    device_name: str = "cpu",
) -> int:
    loaded_model = None
    manifest_payload = None
    if manifest is not None:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    load_checkpoint = checkpoint or manifest
    if load_checkpoint is not None:
        try:
            loaded_model = _load_model(
                load_checkpoint,
                manifest_path=manifest,
                manifest_payload=manifest_payload,
                device_name=device_name,
            )
        except ModuleNotFoundError as exc:
            if not allow_dry_run_fallback:
                _response({"type": "error", "error": f"torch_dependency_unavailable: {exc}", "checkpoint": str(load_checkpoint)})
                return 2
        except Exception as exc:
            if not allow_dry_run_fallback:
                _response({"type": "error", "error": f"checkpoint_load_failed: {exc}", "checkpoint": str(load_checkpoint)})
                return 2

    _response(
        {
            "type": "hello",
            "protocol": "cascadiav3.torch_jsonl_stdio.v1",
            "checkpoint": str(checkpoint) if checkpoint else None,
            "manifest": manifest_payload,
            "model_loaded": loaded_model is not None,
            "allow_dry_run_fallback": allow_dry_run_fallback,
            "device": device_name,
            "protocol_features": PROTOCOL_FEATURES,
        }
    )
    for line in sys.stdin:
        try:
            message = json.loads(line)
            message_type = message.get("type")
            if message_type == "hello":
                _response({"type": "hello", "protocol": "cascadiav3.torch_jsonl_stdio.v1"})
            elif message_type == "shutdown":
                _response({"type": "shutdown", "status": "ok"})
                return 0
            elif message_type == "eval_request":
                root = message["root"]
                if loaded_model is None:
                    if not allow_dry_run_fallback and not message.get("allow_model_fallback", False):
                        raise RuntimeError("no model loaded and dry-run fallback is disabled")
                    _response(_uniform_eval(root, model_fallback=True))
                else:
                    _response(_model_eval(loaded_model, root, device_name=device_name))
            elif message_type == "eval_batch_request":
                roots = message["roots"]
                if not isinstance(roots, list) or not roots:
                    raise ValueError("eval_batch_request requires a non-empty roots list")
                if loaded_model is None:
                    if not allow_dry_run_fallback and not message.get("allow_model_fallback", False):
                        raise RuntimeError("no model loaded and dry-run fallback is disabled")
                    results = [_uniform_eval(root, model_fallback=True) for root in roots]
                else:
                    results = _model_eval_batch(loaded_model, roots, device_name=device_name)
                _response({"type": "eval_batch_response", "results": results})
            else:
                raise ValueError(f"unknown message type {message_type!r}")
        except Exception as exc:  # pragma: no cover - protocol errors are surfaced as JSON.
            _response({"type": "error", "error": str(exc)})
    return 0


def _self_test_manifest_resolution() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tempdir:
        project_root = Path(tempdir)
        manifest_dir = project_root / "cascadiav3" / "checkpoints" / "cascadiaformer"
        manifest_dir.mkdir(parents=True)
        project_weight = manifest_dir / "step_0000001.weights.pt"
        project_weight.write_text("weights", encoding="utf-8")
        manifest_path = manifest_dir / "step_0000001.manifest.json"
        manifest_path.write_text(
            json.dumps({"weights": "cascadiav3/checkpoints/cascadiaformer/step_0000001.weights.pt"}),
            encoding="utf-8",
        )
        resolved_project = resolve_checkpoint_path(
            "cascadiav3/checkpoints/cascadiaformer/step_0000001.weights.pt",
            manifest_path=manifest_path,
            cwd=project_root,
        )
        if resolved_project != project_weight:
            raise AssertionError(f"project-root-relative resolution failed: {resolved_project} != {project_weight}")
        relative_weight = manifest_dir / "relative.weights.pt"
        relative_weight.write_text("weights", encoding="utf-8")
        resolved_relative = resolve_checkpoint_path(
            "relative.weights.pt",
            manifest_path=manifest_path,
            cwd=project_root / "elsewhere",
        )
        if resolved_relative != relative_weight:
            raise AssertionError(f"manifest-relative resolution failed: {resolved_relative} != {relative_weight}")
        return {
            "status": "pass",
            "project_root_relative": str(resolved_project),
            "manifest_relative": str(resolved_relative),
        }


def _self_test_inference_request(root_path: Path) -> dict[str, Any]:
    root = read_replay_jsonl(root_path)[0]
    public_root = {key: value for key, value in root.items() if key not in TRAINING_LABEL_KEYS}
    view = inference_request_view(public_root)
    report = {
        "status": "pass",
        "action_count": len(view["action_ids"]),
        "token_count": int(view["public_tokens"].get("token_count", len(view["public_tokens"]["tokens"]))),
        "training_labels_present": view["training_labels_present"],
        "torch_collate_checked": False,
    }
    try:
        batch = collate_inference_roots([public_root])
        report.update(
            {
                "torch_collate_checked": True,
                "token_shape": list(batch["tokens"].shape),
                "action_shape": list(batch["actions"].shape),
                "relation_shape": list(batch["relation_ids"].shape),
            }
        )
    except ModuleNotFoundError as exc:
        report["torch_skip_reason"] = str(exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dry-run-fallback", action="store_true")
    parser.add_argument("--self-test-root")
    parser.add_argument("--self-test-manifest-resolution", action="store_true")
    parser.add_argument("--self-test-inference-request")
    args = parser.parse_args()

    if args.self_test_manifest_resolution:
        print(json.dumps(_self_test_manifest_resolution(), indent=2, sort_keys=True))
        return 0
    if args.self_test_inference_request:
        print(json.dumps(_self_test_inference_request(Path(args.self_test_inference_request)), indent=2, sort_keys=True))
        return 0
    if args.self_test_root:
        root = read_replay_jsonl(Path(args.self_test_root))[0]
        print(json.dumps(_uniform_eval(root, model_fallback=True), indent=2, sort_keys=True))
        return 0
    return serve(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        manifest=Path(args.manifest) if args.manifest else None,
        allow_dry_run_fallback=args.allow_dry_run_fallback,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
