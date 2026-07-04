from __future__ import annotations

import base64
import json
import struct
import sys


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def pack_f64_b64(values: list[float]) -> str:
    """Base64 little-endian f64 packing; bit-identical to the JSON float path."""
    return base64.b64encode(struct.pack("<%dd" % len(values), *values)).decode("ascii")


def eval_response(root: dict, *, model_fallback: bool, packed: bool = False) -> dict:
    raw_ids = root.get("action_ids")
    if isinstance(raw_ids, list) and raw_ids:
        action_ids = [str(action_id) for action_id in raw_ids]
    else:
        action_ids = [action["action_id"] for action in root["legal_actions"]]
    weights = [float(index + 1) for index in range(len(action_ids))]
    total = sum(weights)
    exact = root.get("exact_afterstate_score_active")
    if isinstance(exact, list) and len(exact) == len(action_ids):
        score_to_go = [1.0 for _ in action_ids]
        q_values = [float(afterstate) + 1.0 for afterstate in exact]
    else:
        score_to_go = [0.0 for _ in action_ids]
        q_values = [0.0 for _ in action_ids]
    priors = [weight / total for weight in weights]
    uncertainty = [1.0 for _ in action_ids]
    value = [80.0, 80.0, 80.0, 80.0]
    response = {
        "type": "eval_response",
        "state_hash": root["state_hash"],
        "action_ids": action_ids,
        "model_fallback": model_fallback,
    }
    if packed:
        response["packed"] = {
            "priors_f64_b64": pack_f64_b64(priors),
            "q_f64_b64": pack_f64_b64(q_values),
            "score_to_go_f64_b64": pack_f64_b64(score_to_go),
            "uncertainty_f64_b64": pack_f64_b64(uncertainty),
            "value_f64_b64": pack_f64_b64(value),
        }
    else:
        response.update(
            {
                "priors": priors,
                "q": q_values,
                "score_to_go": score_to_go,
                "uncertainty": uncertainty,
                "value": value,
            }
        )
    return response


def main() -> int:
    model_fallback = "--model-fallback" in sys.argv[1:]
    no_batch = "--no-batch" in sys.argv[1:]
    no_packed_response = "--no-packed-response" in sys.argv[1:]
    hello: dict = {"type": "hello", "protocol": "cascadiav3.mock_model_bridge.v1"}
    if not no_batch:
        features = ["eval_batch", "value_vector", "packed_features"]
        if not no_packed_response:
            features.append("packed_response")
        hello["protocol_features"] = features
    emit(hello)
    for line in sys.stdin:
        message = json.loads(line)
        message_type = message.get("type")
        packed = (
            bool(message.get("packed_response", False))
            and not no_packed_response
            and not no_batch
        )
        if message_type == "shutdown":
            emit({"type": "shutdown", "status": "ok"})
            return 0
        if message_type == "eval_request":
            emit(eval_response(message["root"], model_fallback=model_fallback, packed=packed))
            continue
        if message_type == "eval_batch_request" and not no_batch:
            results = [
                eval_response(root, model_fallback=model_fallback, packed=packed)
                for root in message["roots"]
            ]
            emit({"type": "eval_batch_response", "results": results})
            continue
        emit({"type": "error", "error": f"unknown message type {message_type!r}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
