import unittest

from ortools.sat.python import cp_model

from tools.aaaaa_wildlife_exact import (
    KNOWN_INCUMBENT_TOKENS,
    SPECIES,
    build_model,
    count_relaxation,
    count_vectors,
    score_tokens,
    species_tokens,
)


class AaaaaWildlifeExactTests(unittest.TestCase):
    def test_count_relaxation_space_is_pinned(self) -> None:
        all_vectors = count_vectors()
        proof_vectors = count_vectors(69)
        self.assertEqual(826, len(all_vectors))
        self.assertEqual(((6, 1, 6, 1, 6), 73), all_vectors[0])
        self.assertEqual(128, len(proof_vectors))
        self.assertTrue(all(bound >= 69 for _, bound in proof_vectors))
        self.assertTrue(all(counts[4] > 0 for counts, _ in proof_vectors))

    def test_optimal_witness_scores_68(self) -> None:
        self.assertEqual((19, 13, 20, 0, 16), score_tokens(KNOWN_INCUMBENT_TOKENS))

    def test_token_symmetry_order_and_model_validation(self) -> None:
        counts = (6, 1, 6, 1, 6)
        self.assertEqual(20, len(species_tokens(counts)))
        self.assertEqual([4] * 6, species_tokens(counts)[:6])
        self.assertEqual(73, count_relaxation(counts))
        model, _ = build_model(counts, 69)
        self.assertEqual("", model.validate())

        fox_free = (6, 6, 6, 2, 0)
        self.assertEqual([0] * 6, species_tokens(fox_free)[:6])
        fox_free_model, _ = build_model(fox_free, 0)
        self.assertEqual("", fox_free_model.validate())

    def test_exact_model_accepts_the_production_witness(self) -> None:
        model, variables = build_model((6, 4, 6, 0, 4), 68)
        foxes = [
            (int(token["q"]), int(token["r"]))
            for token in KNOWN_INCUMBENT_TOKENS
            if token["wildlife"] == "fox"
        ]
        origin_q, origin_r = min(foxes)
        positions = {
            species: sorted(
                (int(token["q"]) - origin_q, int(token["r"]) - origin_r)
                for token in KNOWN_INCUMBENT_TOKENS
                if token["wildlife"] == species
            )
            for species in SPECIES
        }
        offsets = {species: 0 for species in SPECIES}
        for token, species_code in enumerate(variables.species_by_token):
            species = SPECIES[species_code]
            q, r = positions[species][offsets[species]]
            offsets[species] += 1
            model.add(variables.q[token] == q)
            model.add(variables.r[token] == r)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        status = solver.solve(model)
        self.assertEqual(cp_model.OPTIMAL, status)
        self.assertEqual(68, solver.value(variables.total_score))


if __name__ == "__main__":
    unittest.main()
