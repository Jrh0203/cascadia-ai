//! End-to-end tests for the WI-2 selfish tomography optimizers and harness.
//!
//! Everything here is deterministic, CPU-only, and hermetic: fixture games
//! are seeded random legal play sealed into trajectory ledgers, and every
//! claim a witness makes is re-verified through the canonical engine.

use std::fs;
use std::path::PathBuf;

use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, score_board};
use cascadia_rival::{
    HarnessConfig, HarnessError, MenuComposer, PolicyMemoryBank, PrivateSimState, PublicRootId,
    RepackConfig, ReplayConfig, RootActionOccurrenceId, RootDecisionRecord, RulesDecision,
    RulesLegalMenu, SeatIndex, SelectedDecisionKind, TomographyEvidenceDomain, TomographyKind,
    TomographySummary, TrajectoryLedger, TrajectoryLedgerBuilder,
    WITNESS_SEMANTICS_LOWER_BOUND_ONLY, chronology_deviation_is_feasible, repack_seat, replay_seat,
    run_directory,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn small_config(seed: u64) -> HarnessConfig {
    HarnessConfig {
        seed,
        repack_iterations: 150,
        beam_width: 2,
        candidate_cap: 4,
    }
}

/// A sealed terminal ledger from seeded random legal play (places wildlife,
/// mints and spends nature tokens, exercises free refreshes).
fn random_play_ledger(seed: u64, source_game_id: &str) -> TrajectoryLedger {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(seed),
    )
    .unwrap();
    let mut builder = TrajectoryLedgerBuilder::new(source_game_id, game).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x746f_6d6f);
    while !builder.game().is_game_over() {
        let preludes = builder.game().free_three_of_a_kind_choices().unwrap();
        let prelude = &preludes[rng.gen_range(0..preludes.len())];
        let actions = builder.game().legal_turn_actions(prelude).unwrap();
        let action = actions[rng.gen_range(0..actions.len())].clone();
        builder.push_fixture_turn(action).unwrap();
    }
    builder.seal_terminal().unwrap()
}

/// A sealed terminal ledger whose every turn carries a complete
/// public-root policy decision trace (deterministic first-legal policy).
fn policy_trace_ledger(seed: u64, source_game_id: &str) -> TrajectoryLedger {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(seed),
    )
    .unwrap();
    let mut builder = TrajectoryLedgerBuilder::new(source_game_id, game).unwrap();
    let mut memories = PolicyMemoryBank::new(4).unwrap();
    let mut ordinal = 0u32;
    while !builder.game().is_game_over() {
        let (action, records) =
            first_legal_policy_turn(builder.game(), &mut memories, &mut ordinal);
        builder.push_policy_turn(action, records).unwrap();
    }
    builder.seal_terminal().unwrap()
}

fn first_legal_policy_turn(
    state: &GameState,
    memories: &mut PolicyMemoryBank,
    ordinal: &mut u32,
) -> (TurnAction, Vec<RootDecisionRecord>) {
    let actor = SeatIndex::new(state.current_player() as u8).unwrap();
    let mut records = Vec::new();
    let prelude_menu = MenuComposer::prelude_root(state).unwrap();
    let staged = if prelude_menu.len() == 1 {
        let RulesDecision::Prelude(prelude) = prelude_menu.decision(0).unwrap() else {
            panic!("singleton prelude root must be the default decline");
        };
        state.preview_market_prelude(prelude).unwrap()
    } else {
        records.push(decision_record(
            state,
            actor,
            &prelude_menu,
            0,
            ordinal,
            memories,
        ));
        let RulesDecision::Prelude(prelude) = prelude_menu.decision(0).unwrap() else {
            panic!("prelude root decision 0 must be a prelude");
        };
        state.preview_market_prelude(prelude).unwrap()
    };
    let draft_menu = MenuComposer::draft_root(&staged, &MarketPrelude::default()).unwrap();
    let index = draft_menu.first_draft_index().unwrap();
    records.push(decision_record(
        &staged,
        actor,
        &draft_menu,
        index,
        ordinal,
        memories,
    ));
    let RulesDecision::Draft(action) = draft_menu.decision(index).unwrap() else {
        panic!("first draft index must be a draft decision");
    };
    (action.clone(), records)
}

fn decision_record(
    state: &GameState,
    actor: SeatIndex,
    menu: &RulesLegalMenu,
    index: usize,
    ordinal: &mut u32,
    memories: &mut PolicyMemoryBank,
) -> RootDecisionRecord {
    let memory = memories.get(actor).unwrap().clone();
    let observation = PrivateSimState::new(state.clone())
        .unwrap()
        .public_observation(actor, memory.clone())
        .unwrap();
    let root = PublicRootId::new(&observation, menu.root_kind());
    let (selected_kind, draft_occurrence_id) = match menu.decision(index).unwrap() {
        RulesDecision::Prelude(_) => (SelectedDecisionKind::Prelude, None),
        RulesDecision::PaidWipe(_) => (SelectedDecisionKind::PaidWipe, None),
        RulesDecision::Draft(_) => (
            SelectedDecisionKind::Draft,
            Some(RootActionOccurrenceId::new(&root, menu, index).unwrap()),
        ),
    };
    let record = RootDecisionRecord {
        decision_ordinal: *ordinal,
        root_kind: menu.root_kind(),
        public_observation: observation,
        public_root_id: root,
        ordered_menu_hash: menu.hash(),
        menu_len: menu.len() as u32,
        selected_index: index as u32,
        selected_kind,
        draft_occurrence_id,
        next_memory: memory,
    };
    *ordinal += 1;
    record
}

fn fresh_directory(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "cascadia-rival-tomography-{label}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

fn write_ledger(directory: &std::path::Path, file_name: &str, ledger: &TrajectoryLedger) {
    ledger
        .write_json_immutable(&directory.join(file_name))
        .unwrap();
}

#[test]
fn witnesses_rebuild_through_the_canonical_engine_with_exact_scores() {
    let ledger = random_play_ledger(41, "rival-tomography-itest-041");
    let terminal = ledger.replay().unwrap();
    let population = cascadia_rival::TomographyPopulation {
        incumbent_policy_id: "rival-tomography-itest".to_owned(),
        opponent_population_id: "rival-tomography-itest:table".to_owned(),
        evidence_domain: TomographyEvidenceDomain::CpuProxy,
    };
    let repack_config = RepackConfig {
        seed: 5,
        iterations: 150,
    };
    let replay_config = ReplayConfig {
        seed: 5,
        beam_width: 1,
        candidate_cap: 3,
    };
    for seat in 0..4u8 {
        let seat_index = SeatIndex::new(seat).unwrap();
        let realized = ledger.terminal_scores().unwrap()[usize::from(seat)];

        let repack = repack_seat(&ledger, seat_index, &repack_config, &population).unwrap();
        assert_eq!(repack.witness.realized_score, realized);
        assert!(repack.witness.witness_score.total >= realized.total);
        assert_eq!(
            repack.witness.score_delta,
            i32::from(repack.witness.witness_score.total) - i32::from(realized.total)
        );
        // Witness legality and exact-score round trip through the canonical
        // engine, from the sealed starter cluster.
        let board = repack.witness.rebuild_board(&ledger).unwrap();
        assert_eq!(
            score_board(&board, ledger.config().scoring_cards),
            repack.witness.witness_score
        );
        // Frozen multiset: same tiles, same wildlife, same terminal tokens.
        let realized_board = &terminal.boards()[usize::from(seat)];
        let mut witness_tiles: Vec<u8> = board
            .placed_tiles()
            .map(|(_, placed)| placed.tile.id.0)
            .collect();
        let mut realized_tiles: Vec<u8> = realized_board
            .placed_tiles()
            .map(|(_, placed)| placed.tile.id.0)
            .collect();
        witness_tiles.sort_unstable();
        realized_tiles.sort_unstable();
        assert_eq!(witness_tiles, realized_tiles);
        let mut witness_wildlife: Vec<u8> = board
            .placed_tiles()
            .filter_map(|(_, placed)| placed.wildlife.map(|wildlife| wildlife as u8))
            .collect();
        let mut realized_wildlife: Vec<u8> = realized_board
            .placed_tiles()
            .filter_map(|(_, placed)| placed.wildlife.map(|wildlife| wildlife as u8))
            .collect();
        witness_wildlife.sort_unstable();
        realized_wildlife.sort_unstable();
        assert_eq!(witness_wildlife, realized_wildlife);
        assert_eq!(board.nature_tokens(), realized_board.nature_tokens());
        assert_eq!(
            repack.result.kind(),
            TomographyKind::T0OwnBoardRepack,
            "repacking is the acquired-resources-only kind"
        );

        let replay = replay_seat(&ledger, seat_index, &replay_config, &population).unwrap();
        assert_eq!(replay.witness.realized_score, realized);
        assert!(replay.witness.witness_score.total >= realized.total);
        replay.witness.certify(&ledger).unwrap();
        assert_eq!(
            replay.result.kind(),
            TomographyKind::T3KnownWorldOneSeatOracle,
            "hindsight replay is labeled with the known-chance-tape boundary"
        );
        assert_eq!(replay.result.evidence().upper_bound(), None);
        assert_eq!(repack.result.evidence().upper_bound(), None);
    }
}

#[test]
fn chronology_breaking_deviation_is_rejected_as_infeasible() {
    let ledger = random_play_ledger(42, "rival-tomography-itest-042");
    let seat = SeatIndex::new(0).unwrap();
    let mut flips = 0usize;
    let mut pruned = 0usize;
    for (index, record) in ledger.turns().iter().enumerate() {
        if record.actor != 0 || record.action.wildlife.is_none() {
            continue;
        }
        // Hold the chronology (prelude + draft) fixed; decline the wildlife
        // placement instead.  The returned token perturbs the bag, so the
        // recorded public stream must eventually diverge for at least one
        // such flip, and the branch must be pruned, never patched.
        let mut flipped = record.action.clone();
        flipped.wildlife = None;
        flips += 1;
        if !chronology_deviation_is_feasible(&ledger, seat, index, &flipped).unwrap() {
            pruned += 1;
        }
        // The realized action itself is always feasible.
        assert!(chronology_deviation_is_feasible(&ledger, seat, index, &record.action).unwrap());
    }
    assert!(
        flips > 0,
        "fixture must contain own-seat wildlife placements"
    );
    assert!(
        pruned > 0,
        "at least one place->decline flip must break a later recorded market action"
    );
}

#[test]
fn harness_summary_is_deterministic_labeled_and_fail_closed() {
    let directory = fresh_directory("proxy");
    write_ledger(
        &directory,
        "game-043.json",
        &random_play_ledger(43, "rival-tomography-itest-043"),
    );
    let config = small_config(9);

    let left = run_directory(&directory, &config).unwrap();
    let right = run_directory(&directory, &config).unwrap();
    let left_bytes = left.canonical_json_bytes().unwrap();
    let right_bytes = right.canonical_json_bytes().unwrap();
    assert_eq!(
        left_bytes, right_bytes,
        "same inputs and seed must produce a byte-identical summary"
    );

    // Kill-bar discipline and population labeling.
    assert_eq!(left.witness_semantics(), WITNESS_SEMANTICS_LOWER_BOUND_ONLY);
    assert_eq!(
        left.population().evidence_domain,
        TomographyEvidenceDomain::CpuProxy
    );
    assert_eq!(
        left.population().incumbent_policy_id,
        "rival-tomography-itest"
    );
    assert_eq!(left.games().len(), 1);
    assert_eq!(left.games()[0].seats.len(), 4);
    assert_eq!(left.results().len(), 8);
    for result in left.results() {
        assert_eq!(result.evidence_domain(), TomographyEvidenceDomain::CpuProxy);
        assert!(!result.eligible_for_high_fidelity_funding_claim());
        assert_eq!(
            result.evidence().upper_bound(),
            None,
            "never an upper bound"
        );
        assert!(result.evidence().lower_bound() >= 0);
    }
    assert_eq!(left.aggregates().len(), 2);
    assert_eq!(left.aggregates()[0].kind, TomographyKind::T0OwnBoardRepack);
    assert_eq!(
        left.aggregates()[1].kind,
        TomographyKind::T3KnownWorldOneSeatOracle
    );
    assert_eq!(left.inputs().len(), 1);
    assert_eq!(left.inputs()[0].file_name, "game-043.json");

    // The published document round-trips through full validation.
    let decoded = TomographySummary::from_json_slice(&left_bytes).unwrap();
    assert_eq!(decoded, left);

    // Tampering fails closed on read.
    let mut value: serde_json::Value = serde_json::from_slice(&left_bytes).unwrap();
    value["witness_semantics"] = serde_json::json!("upper_bound");
    assert!(TomographySummary::from_json_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(&left_bytes).unwrap();
    let delta = value["games"][0]["seats"][0]["repack_delta"]
        .as_i64()
        .unwrap();
    value["games"][0]["seats"][0]["repack_delta"] = serde_json::json!(delta + 1);
    assert!(TomographySummary::from_json_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(&left_bytes).unwrap();
    value["results"][0]["evidence_domain"] = serde_json::json!("incumbent_measured");
    assert!(TomographySummary::from_json_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(&left_bytes).unwrap();
    let max_delta = value["aggregates"][0]["max_delta"].as_i64().unwrap();
    value["aggregates"][0]["max_delta"] = serde_json::json!(max_delta + 1);
    assert!(TomographySummary::from_json_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn mixed_population_inputs_are_refused() {
    let directory = fresh_directory("mixed");
    write_ledger(
        &directory,
        "alpha.json",
        &random_play_ledger(44, "alpha-battery-001"),
    );
    write_ledger(
        &directory,
        "beta.json",
        &random_play_ledger(45, "beta-battery-001"),
    );
    assert!(matches!(
        run_directory(&directory, &small_config(1)),
        Err(HarnessError::MixedPopulations { .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn incumbent_domain_requires_policy_decision_traces() {
    // An incumbent-namespaced population whose ledgers do NOT carry complete
    // policy decision traces must be refused, not silently downgraded.
    let directory = fresh_directory("incumbent-mislabeled");
    write_ledger(
        &directory,
        "game.json",
        &random_play_ledger(46, "incumbent:mislabeled-fixture-046"),
    );
    assert!(matches!(
        run_directory(&directory, &small_config(1)),
        Err(HarnessError::IncumbentWithoutDecisionTraces(_))
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn incumbent_population_with_traces_is_labeled_incumbent_measured() {
    let directory = fresh_directory("incumbent");
    // Serialize without the extra sealed-write verification pass: the
    // harness re-verifies the full policy decision trace on read, and each
    // trace verification pass recomposes 80 canonical root menus (an
    // expensive deliberate proof), so this test avoids redundant passes.
    let ledger = policy_trace_ledger(47, "incumbent:first-legal-fixture-047");
    fs::write(
        directory.join("game-047.json"),
        serde_json::to_vec_pretty(&ledger).unwrap(),
    )
    .unwrap();
    let config = HarnessConfig {
        seed: 3,
        repack_iterations: 40,
        beam_width: 1,
        candidate_cap: 2,
    };
    let summary = run_directory(&directory, &config).unwrap();
    assert_eq!(
        summary.population().evidence_domain,
        TomographyEvidenceDomain::IncumbentMeasured
    );
    assert_eq!(
        summary.population().incumbent_policy_id,
        "incumbent:first-legal-fixture"
    );
    for result in summary.results() {
        assert_eq!(
            result.evidence_domain(),
            TomographyEvidenceDomain::IncumbentMeasured
        );
    }
    // The summary still validates and round-trips byte-identically.
    let bytes = summary.canonical_json_bytes().unwrap();
    assert_eq!(TomographySummary::from_json_slice(&bytes).unwrap(), summary);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn harness_refuses_foreign_partial_and_duplicate_inputs() {
    // Foreign files.
    let directory = fresh_directory("foreign");
    fs::write(directory.join("notes.txt"), b"not a ledger").unwrap();
    assert!(matches!(
        run_directory(&directory, &small_config(1)),
        Err(HarnessError::ForeignInput(_))
    ));
    fs::remove_dir_all(&directory).unwrap();

    // Empty directory.
    let directory = fresh_directory("empty");
    assert!(matches!(
        run_directory(&directory, &small_config(1)),
        Err(HarnessError::EmptyInputSet)
    ));
    fs::remove_dir_all(&directory).unwrap();

    // Duplicate source game ids.
    let directory = fresh_directory("duplicate");
    let ledger = random_play_ledger(48, "dup-battery-001");
    write_ledger(&directory, "a.json", &ledger);
    write_ledger(&directory, "b.json", &ledger);
    assert!(matches!(
        run_directory(&directory, &small_config(1)),
        Err(HarnessError::DuplicateSourceGameId(_))
    ));
    fs::remove_dir_all(directory).unwrap();
}
