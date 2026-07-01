"""Validate CascadiaFormer Q serving uses derived final Q.

The model Q head predicts score-to-go. Serving must rank by
exact_afterstate_score_active + predicted_score_to_go.
"""

from __future__ import annotations

import json

from .torch_inference_bridge import derived_final_q_values, inference_request_view, q_selection_index


def main() -> int:
    root = {
        "schema_id": "cascadiav3.pre_gpu.v0",
        "state_hash": "synthetic:q-serving-rank-flip",
        "active_seat": 0,
        "legal_actions": [{"action_id": "keep-high-current"}, {"action_id": "chase-remaining"}],
        "public_tokens": {"tokens": [], "token_count": 0},
        "exact_afterstate_score_active": [100.0, 0.0],
    }
    score_to_go = [-1.0, 10.0]
    raw_score_to_go_winner = max(range(len(score_to_go)), key=lambda index: score_to_go[index])
    derived_winner = q_selection_index(root, score_to_go)
    if raw_score_to_go_winner == derived_winner:
        raise AssertionError("synthetic q-serving fixture did not create a rank flip")
    if derived_winner != 0:
        raise AssertionError("derived final-Q serving chose the wrong action")
    view = inference_request_view(root)
    final_q = derived_final_q_values(root, score_to_go)
    report = {
        "status": "pass",
        "raw_score_to_go_winner": raw_score_to_go_winner,
        "derived_final_q_winner": derived_winner,
        "winning_action_id": view["action_ids"][derived_winner],
        "score_to_go": score_to_go,
        "exact_afterstate_score_active": view["exact_afterstate_score_active"],
        "derived_final_q": final_q,
        "invariant": "selection_head=q ranks exact_afterstate_score_active + predicted_score_to_go",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
