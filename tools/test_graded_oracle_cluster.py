from __future__ import annotations

import graded_oracle_cluster as cluster


def test_frozen_host_split_is_game_disjoint_and_complete() -> None:
    observed = []
    for splits in cluster.HOST_SPLITS.values():
        for seeds in splits.values():
            observed.extend(seeds)
    assert sorted(observed) == list(range(61000, 61013))
    assert len(observed) == len(set(observed))
    assert {
        host: splits["validation"]
        for host, splits in cluster.HOST_SPLITS.items()
    } == {
        "john1": (61003,),
        "john2": (61007,),
        "john3": (61011,),
    }
    assert {
        host: splits["test"]
        for host, splits in cluster.HOST_SPLITS.items()
    } == {
        "john1": (61004,),
        "john2": (61008,),
        "john3": (61012,),
    }
