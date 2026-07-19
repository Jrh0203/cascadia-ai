//! End-to-end tests for the incumbent exporter ingest bridge.
//!
//! The fixture is a REAL Gate 0 champion battery game
//! (`gumbel_game_seed_2027160000.jsonl`, 80 `gumbel_decision` rows plus one
//! `gumbel_game_done` row) produced by `cascadiav3/real-root-exporter`.
//! Resolving all 80 recorded `chosen_action_id` digests to exactly one legal
//! draft each, and reproducing the recorded terminal scores bit-for-bit, is
//! the equivalence proof for the reimplemented exporter action hash.

use std::fs;
use std::path::{Path, PathBuf};

use cascadia_rival::{
    HarnessConfig, IngestError, LedgerCompletion, TomographyEvidenceDomain, TurnEvidenceKind,
    ingest_exporter_game, ingest_exporter_game_file, run_directory,
};

const FIXTURE_SEED: u64 = 2027160000;
const FIXTURE_POLICY_ID: &str = "incumbent:cascadia-v3-cycle4-n1024-d16-gate0-20260716";
/// Recorded per-seat totals from the fixture's `gumbel_game_done` row.
const FIXTURE_TOTALS: [u16; 4] = [99, 99, 98, 99];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("gumbel_game_seed_{FIXTURE_SEED}.jsonl"))
}

fn fixture_contents() -> String {
    fs::read_to_string(fixture_path()).expect("read committed exporter fixture")
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cascadia-rival-ingest-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create isolated ingest test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn real_exporter_game_ingests_seals_and_feeds_incumbent_measured_tomography() {
    let ingested = ingest_exporter_game_file(&fixture_path(), FIXTURE_POLICY_ID)
        .expect("all 80 recorded decisions resolve to exactly one legal action each");
    assert_eq!(ingested.seed, FIXTURE_SEED);
    assert_eq!(ingested.decision_count, 80);
    assert_eq!(ingested.terminal_totals(), FIXTURE_TOTALS);

    let ledger = &ingested.ledger;
    assert_eq!(ledger.completion(), LedgerCompletion::Terminal);
    assert_eq!(
        ledger.source_game_id(),
        format!("{FIXTURE_POLICY_ID}-{FIXTURE_SEED}")
    );
    assert_eq!(ledger.turns().len(), 80);
    assert!(
        ledger
            .turns()
            .iter()
            .all(|turn| turn.evidence_kind == TurnEvidenceKind::PolicyDecisionTrace
                && !turn.root_decisions.is_empty()),
        "every ingested turn must carry a complete policy decision trace"
    );
    // One explicit full verification pass: `replay` recomposes every root
    // menu, re-verifies every decision record, and re-applies every action
    // (it is `verify()` returning the terminal state).
    let replayed = ledger.replay().expect("sealed ledger replays exactly");
    assert!(replayed.is_game_over());
    let scores = ledger.terminal_scores().expect("terminal scores are sealed");
    assert_eq!(
        scores.iter().map(|score| score.total).collect::<Vec<_>>(),
        FIXTURE_TOTALS
    );
    // The recorded free three-of-a-kind accepts must survive as compound
    // prelude actions (the fixture accepts at plies 4 and 24, among others).
    assert!(
        ledger
            .turns()
            .iter()
            .any(|turn| turn.action.replace_three_of_a_kind),
        "fixture contains accepted free three-of-a-kind decisions"
    );

    // Round-trip through the immutable publisher and the WI-2 tomography
    // harness: the population derives from the ledger's source_game_id and
    // must be labeled IncumbentMeasured.
    let directory = TemporaryDirectory::new("harness");
    ledger
        .write_json_immutable(&directory.path().join("game-2027160000.json"))
        .expect("publish sealed ledger");
    let summary = run_directory(
        directory.path(),
        &HarnessConfig {
            seed: 11,
            repack_iterations: 60,
            beam_width: 1,
            candidate_cap: 2,
        },
    )
    .expect("tomography harness accepts the ingested ledger directory");
    assert_eq!(
        summary.population().evidence_domain,
        TomographyEvidenceDomain::IncumbentMeasured
    );
    assert_eq!(summary.population().incumbent_policy_id, FIXTURE_POLICY_ID);
    assert_eq!(
        summary.population().opponent_population_id,
        format!("{FIXTURE_POLICY_ID}:table")
    );
    assert_eq!(summary.inputs().len(), 1);
    assert_eq!(summary.games().len(), 1);
    for result in summary.results() {
        assert_eq!(
            result.evidence_domain(),
            TomographyEvidenceDomain::IncumbentMeasured
        );
    }
}

#[test]
fn tampered_chosen_action_id_fails_closed() {
    let contents = fixture_contents();
    let tampered = tamper_first(&contents, "\"chosen_action_id\":\"sha256:a", |line| {
        line.replacen("\"chosen_action_id\":\"sha256:a", "\"chosen_action_id\":\"sha256:b", 1)
    });
    let error = ingest_exporter_game(&tampered, Some(FIXTURE_SEED), FIXTURE_POLICY_ID)
        .expect_err("a tampered chosen action digest must abort ingest");
    assert!(
        matches!(error, IngestError::ChosenActionNotFound { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn tampered_final_score_fails_closed() {
    let contents = fixture_contents();
    let tampered = tamper_first(&contents, "\"gumbel_game_done\"", |line| {
        line.replacen("\"base_total\":99,", "\"base_total\":98,", 1)
            .replacen("\"total\":99,", "\"total\":98,", 1)
    });
    assert_ne!(tampered, contents, "tamper must hit the game_done row");
    let error = ingest_exporter_game(&tampered, Some(FIXTURE_SEED), FIXTURE_POLICY_ID)
        .expect_err("a tampered recorded final score must abort ingest");
    assert!(
        matches!(error, IngestError::ScoreMismatch { seat: 0, .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn wrong_ruleset_id_fails_closed() {
    let contents = fixture_contents().replace(
        "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16",
        "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_17",
    );
    let error = ingest_exporter_game(&contents, Some(FIXTURE_SEED), FIXTURE_POLICY_ID)
        .expect_err("an unknown ruleset must abort ingest");
    assert!(
        matches!(error, IngestError::WrongRulesetId { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn duplicate_and_missing_plies_fail_closed() {
    let contents = fixture_contents();
    let lines: Vec<&str> = contents.lines().collect();

    let mut duplicated = lines.clone();
    duplicated.insert(6, lines[5]);
    let error = ingest_exporter_game(
        &duplicated.join("\n"),
        Some(FIXTURE_SEED),
        FIXTURE_POLICY_ID,
    )
    .expect_err("a duplicated decision row must abort ingest");
    assert!(
        matches!(
            error,
            IngestError::PlyMismatch {
                expected: 6,
                observed: 5
            }
        ),
        "unexpected error: {error}"
    );

    let mut missing = lines.clone();
    missing.remove(5);
    let error = ingest_exporter_game(&missing.join("\n"), Some(FIXTURE_SEED), FIXTURE_POLICY_ID)
        .expect_err("a missing decision row must abort ingest");
    assert!(
        matches!(
            error,
            IngestError::PlyMismatch {
                expected: 5,
                observed: 6
            }
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn declared_decision_count_and_seed_are_enforced() {
    let contents = fixture_contents();
    let tampered = tamper_first(&contents, "\"decision_count\":80", |line| {
        line.replacen("\"decision_count\":80", "\"decision_count\":79", 1)
    });
    assert!(matches!(
        ingest_exporter_game(&tampered, Some(FIXTURE_SEED), FIXTURE_POLICY_ID),
        Err(IngestError::DecisionCountMismatch {
            declared: 79,
            observed: 80
        })
    ));
    assert!(matches!(
        ingest_exporter_game(&contents, Some(FIXTURE_SEED + 1), FIXTURE_POLICY_ID),
        Err(IngestError::SeedMismatch { .. })
    ));
}

#[test]
fn non_incumbent_policy_ids_are_refused_before_any_replay() {
    let error = ingest_exporter_game(&fixture_contents(), Some(FIXTURE_SEED), "gate0-proxy")
        .expect_err("evidence-domain-downgrading policy ids must be refused");
    assert!(matches!(
        error,
        IngestError::PolicyIdOutsideIncumbentNamespace(_)
    ));
}

/// Applies `tamper` to the first line containing `needle`; panics if the
/// needle is absent so a stale fixture cannot silently weaken a test.
fn tamper_first(contents: &str, needle: &str, tamper: impl Fn(&str) -> String) -> String {
    let mut hit = false;
    let lines: Vec<String> = contents
        .lines()
        .map(|line| {
            if !hit && line.contains(needle) {
                hit = true;
                tamper(line)
            } else {
                line.to_owned()
            }
        })
        .collect();
    assert!(hit, "fixture no longer contains {needle:?}");
    lines.join("\n")
}
