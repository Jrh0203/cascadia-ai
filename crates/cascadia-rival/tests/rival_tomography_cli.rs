//! End-to-end test of the real `rival-tomography` binary: a sealed fixture
//! ledger directory in, one validated deterministic summary out, and an
//! immutable `--out` artifact that is never replaced.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use cascadia_game::{GameConfig, GameSeed, GameState};
use cascadia_rival::{
    TomographySummary, TrajectoryLedgerBuilder, WITNESS_SEMANTICS_LOWER_BOUND_ONLY,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cascadia-rival-tomography-cli-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create isolated CLI test directory");
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

fn write_fixture_ledger(directory: &Path, seed: u64) {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(seed),
    )
    .unwrap();
    let mut builder =
        TrajectoryLedgerBuilder::new(format!("rival-tomography-cli-{seed:03}"), game).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x636c_6921);
    while !builder.game().is_game_over() {
        let preludes = builder.game().free_three_of_a_kind_choices().unwrap();
        let prelude = &preludes[rng.gen_range(0..preludes.len())];
        let actions = builder.game().legal_turn_actions(prelude).unwrap();
        let action = actions[rng.gen_range(0..actions.len())].clone();
        builder.push_fixture_turn(action).unwrap();
    }
    builder
        .seal_terminal()
        .unwrap()
        .write_json_immutable(&directory.join(format!("game-{seed:03}.json")))
        .unwrap();
}

fn run_tomography(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rival-tomography"))
        .args(arguments)
        .output()
        .expect("run rival-tomography")
}

#[test]
fn binary_emits_a_validated_summary_and_never_replaces_an_artifact() {
    let inputs = TemporaryDirectory::new("inputs");
    write_fixture_ledger(inputs.path(), 71);
    let outputs = TemporaryDirectory::new("outputs");
    let out = outputs.path().join("summary.json");
    let directory = inputs.path().to_str().unwrap().to_owned();
    let out_argument = out.to_str().unwrap().to_owned();
    let arguments = [
        directory.as_str(),
        "--seed",
        "13",
        "--repack-iterations",
        "60",
        "--beam-width",
        "1",
        "--candidate-cap",
        "2",
        "--out",
        out_argument.as_str(),
    ];

    let first = run_tomography(&arguments);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let stdout_summary = TomographySummary::from_json_slice(&first.stdout)
        .expect("stdout must be one validated summary document");
    assert_eq!(
        stdout_summary.witness_semantics(),
        WITNESS_SEMANTICS_LOWER_BOUND_ONLY
    );
    let published = fs::read(&out).expect("--out must publish the summary");
    let published_summary = TomographySummary::from_json_slice(&published).unwrap();
    assert_eq!(published_summary, stdout_summary);

    // Deterministic: a second run reproduces the summary byte-for-byte on
    // stdout, and the immutable publish refuses to replace the artifact.
    let second = run_tomography(&arguments);
    assert!(
        !second.status.success(),
        "immutable --out must refuse replacement"
    );
    assert_eq!(
        second.stdout, first.stdout,
        "summary must be byte-identical"
    );
    assert!(
        String::from_utf8_lossy(&second.stderr).contains("already exists"),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(fs::read(&out).unwrap(), published);
}

#[test]
fn binary_fails_closed_on_bad_arguments_and_bad_inputs() {
    let missing_seed = run_tomography(&["somewhere"]);
    assert!(!missing_seed.status.success());
    assert!(missing_seed.stdout.is_empty());

    let inputs = TemporaryDirectory::new("bad-inputs");
    fs::write(inputs.path().join("garbage.json"), b"{not a ledger}").unwrap();
    let bad = run_tomography(&[inputs.path().to_str().unwrap(), "--seed", "1"]);
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty(), "failure must not emit a summary");
}
