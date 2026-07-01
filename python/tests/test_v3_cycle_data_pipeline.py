from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _module() -> Any:
    path = Path(__file__).resolve().parents[2] / "tools/v3_cycle_data_pipeline.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v3_cycle_data_pipeline_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_stages_reclaim_before_their_downstream_consumers(
    tmp_path: Path,
) -> None:
    module = _module()
    module.ROOT = tmp_path
    module.CAMPAIGN_STATE = tmp_path / "control/campaign-state.json"
    state = {"phase": "cycle-03-collecting"}
    calls: list[str] = []

    accepted = tmp_path / "accepted"
    for index in range(100):
        shard = accepted / f"item-{index:03d}" / f"item-{index:03d}.v3g"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.touch()
    collection = tmp_path / "collection/completion.json"
    label_completion = tmp_path / "labeling/completion.json"

    module._read = lambda _: state
    module._collection = lambda *_: (calls.append("collection") or collection, accepted)
    module.reclaim_completed_increment = lambda completion, _: calls.append(
        "reclaim-collection" if completion == collection else "reclaim-label"
    )
    module.reclaim_remote_workers = lambda completion, _: calls.append(
        "remote-reclaim-collection" if completion == collection else "remote-reclaim-label"
    )
    module._verification = lambda *_: calls.append("verification") or tmp_path / "verify"
    module._corpus = lambda *_: calls.append("corpus") or tmp_path / "corpus.json"
    module._roots = lambda *_: calls.append("roots") or [tmp_path / "root.v3r"]
    module._label = lambda *_: (
        calls.append("label") or label_completion,
        tmp_path / "label-accepted",
    )
    module._label_evidence = lambda *_: calls.append("evidence") or tmp_path / "labels.json"

    def advance(destination: str, _: Path) -> None:
        calls.append(f"advance-{destination}")
        state["phase"] = destination

    module._advance = advance
    module.run(3, "image", tmp_path / "newest", [])

    assert calls == [
        "collection",
        "reclaim-collection",
        "remote-reclaim-collection",
        "verification",
        "corpus",
        "advance-cycle-03-labeling",
        "roots",
        "label",
        "reclaim-label",
        "remote-reclaim-label",
        "evidence",
        "advance-cycle-03-training",
    ]
