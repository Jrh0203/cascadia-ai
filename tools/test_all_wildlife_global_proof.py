from tools.all_wildlife_global_proof import _proof_complete
from tools.all_wildlife_rules import count_upper, count_vectors


def test_proof_complete_accepts_exactly_the_required_count_exclusions() -> None:
    incumbent = 68
    required = {
        counts: incumbent + 1
        for counts in count_vectors()
        if count_upper(counts, "AAAAA") > incumbent
    }
    assert _proof_complete("AAAAA", incumbent, required)
    required.pop(next(iter(required)))
    assert not _proof_complete("AAAAA", incumbent, required)
