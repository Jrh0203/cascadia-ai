#!/usr/bin/env python3
"""AAAAA model variant that eagerly channels deterministic score features.

The base feasibility model permits a true geometric score feature to remain
unselected because only overstatement would be unsound. That is logically
complete, but it leaves many equivalent Boolean assignments. This wrapper
adds the missing reverse implications for deterministic Fox-A, Hawk-A, and
Bear-A features without changing the represented coordinates or scores.
"""

from __future__ import annotations

import itertools

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base


def _variables_by_name(model: cp_model.CpModel) -> dict[str, cp_model.IntVar]:
    return {
        variable.name: model.get_bool_var_from_proto_index(index)
        for index, variable in enumerate(model.proto.variables)
        if len(variable.domain) == 2 and variable.domain[0] == 0 and variable.domain[1] == 1
    }


def build_model(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, int | str]] | None = None,
    fix_initial_tokens: bool = False,
) -> tuple[cp_model.CpModel, base.ExactVariables]:
    model, variables = base.build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
        fix_initial_tokens=fix_initial_tokens,
    )
    named = _variables_by_name(model)
    by_species = {
        species: [
            token
            for token, token_species in enumerate(variables.species_by_token)
            if token_species == species
        ]
        for species in range(len(base.SPECIES))
    }

    # Fox-A distinct-species flags are deterministic ORs of their adjacency
    # edges. The base model already has distinct <= sum(edges); these reverse
    # implications complete the equivalence and remove false-valued slack.
    foxes = by_species[base.SPECIES_CODE["fox"]]
    for fox in foxes:
        for species in range(len(base.SPECIES)):
            distinct = named[f"fox_distinct_{fox}_{species}"]
            for target in by_species[species]:
                if target != fox:
                    model.add(distinct >= named[f"adj_{min(fox, target)}_{max(fox, target)}"])

    # Hawk-A isolated flags are true exactly when every hawk-hawk adjacency is
    # false, not merely allowed to be true in that situation.
    hawks = by_species[base.SPECIES_CODE["hawk"]]
    for hawk in hawks:
        incident = [
            named[f"adj_{min(hawk, other)}_{max(hawk, other)}"] for other in hawks if other != hawk
        ]
        isolated = named[f"hawk_isolated_{hawk}"]
        model.add(isolated + sum(incident) >= 1)

    # A Bear-A pair flag is true exactly for a two-token full component. The
    # base upper implications reject every invalid pair; this lower inequality
    # forces the one remaining valid case.
    bears = by_species[base.SPECIES_CODE["bear"]]
    for left, right in itertools.combinations(bears, 2):
        pair = named[f"bear_pair_{left}_{right}"]
        together = named[f"adj_{left}_{right}"]
        boundary = [
            named[f"adj_{min(member, other)}_{max(member, other)}"]
            for member in (left, right)
            for other in bears
            if other not in (left, right)
        ]
        model.add(pair >= together - sum(boundary))

    return model, variables
