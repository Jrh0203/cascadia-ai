//! Ingest bridge CLI: raw incumbent exporter games in, sealed trajectory
//! ledgers out.
//!
//! Consumes every `gumbel_game_seed_<seed>.jsonl` in a directory of raw
//! `cascadiav3/real-root-exporter` game files (non-matching names such as
//! stderr logs are skipped with a note on stderr), deterministically replays
//! each game through the canonical engine, and publishes one sealed
//! `cascadiav3.rival_trajectory_ledger.v1` JSON per game through the
//! immutable no-replace publisher.  The resulting directory is the M1
//! selfish-ceiling input for `rival-tomography`.
//!
//! Fail-closed: any per-file failure (unknown ruleset, unresolvable chosen
//! action, score mismatch, malformed row, existing output artifact) aborts
//! the whole run with a non-zero exit; there is no partial-success exit.
//! stdout carries one deterministic summary line per ingested game plus a
//! final manifest (count and per-file SHA-256 of the published bytes).

use std::{fs, path::PathBuf, process::ExitCode};

use cascadia_rival::{
    Sha256Digest, ingest_exporter_game_file, parse_exporter_game_file_name,
    validate_incumbent_policy_id,
};

/// The Gate 0 champion battery policy identity (100 games, cycle-4 model,
/// n=1024, d=16, recorded 2026-07-16).
pub const DEFAULT_INCUMBENT_POLICY_ID: &str =
    "incumbent:cascadia-v3-cycle4-n1024-d16-gate0-20260716";

fn usage() -> &'static str {
    "usage: rival-ingest-exporter <raw-games-dir> --out-dir <ledger-dir> \
     [--policy-id <incumbent:...>]"
}

struct Arguments {
    raw_directory: PathBuf,
    out_directory: PathBuf,
    policy_id: String,
}

fn parse_arguments(mut arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let raw_directory = arguments.next().ok_or_else(|| usage().to_owned())?;
    if raw_directory.starts_with("--") {
        return Err(usage().to_owned());
    }
    let mut out_directory: Option<PathBuf> = None;
    let mut policy_id: Option<String> = None;
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} requires a value; {}", usage()))
        };
        match flag.as_str() {
            "--out-dir" => {
                if out_directory.is_some() {
                    return Err(format!("--out-dir was supplied twice; {}", usage()));
                }
                out_directory = Some(PathBuf::from(value()?));
            }
            "--policy-id" => {
                if policy_id.is_some() {
                    return Err(format!("--policy-id was supplied twice; {}", usage()));
                }
                policy_id = Some(value()?);
            }
            _ => return Err(format!("unknown argument {flag:?}; {}", usage())),
        }
    }
    let out_directory = out_directory.ok_or_else(|| format!("--out-dir is required; {}", usage()))?;
    let policy_id = policy_id.unwrap_or_else(|| DEFAULT_INCUMBENT_POLICY_ID.to_owned());
    validate_incumbent_policy_id(&policy_id)
        .map_err(|error| format!("invalid --policy-id: {error}"))?;
    Ok(Arguments {
        raw_directory: PathBuf::from(raw_directory),
        out_directory,
        policy_id,
    })
}

fn ledger_file_name(seed: u64) -> String {
    format!("rival_incumbent_ledger_seed_{seed}.json")
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments(std::env::args().skip(1))?;

    let mut game_files: Vec<(u64, PathBuf)> = Vec::new();
    let entries = fs::read_dir(&arguments.raw_directory)
        .map_err(|error| format!("cannot read {:?}: {error}", arguments.raw_directory))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot enumerate raw games: {error}"))?;
        let file_name = entry.file_name().to_string_lossy().into_owned();
        match parse_exporter_game_file_name(&file_name) {
            Some(seed) => game_files.push((seed, entry.path())),
            None => eprintln!("skipping non-game file {file_name:?}"),
        }
    }
    game_files.sort_unstable_by_key(|(seed, _)| *seed);
    if game_files.is_empty() {
        return Err(format!(
            "no gumbel_game_seed_<seed>.jsonl files in {:?}",
            arguments.raw_directory
        ));
    }

    let mut manifest: Vec<(String, Sha256Digest)> = Vec::with_capacity(game_files.len());
    for (seed, path) in &game_files {
        let ingested = ingest_exporter_game_file(path, &arguments.policy_id)
            .map_err(|error| format!("ingest failed for {:?}: {error}", path.display()))?;
        let destination = arguments.out_directory.join(ledger_file_name(*seed));
        ingested
            .ledger
            .write_json_immutable(&destination)
            .map_err(|error| format!("cannot publish {:?}: {error}", destination.display()))?;
        let published = fs::read(&destination)
            .map_err(|error| format!("cannot read back {:?}: {error}", destination.display()))?;
        let file_sha256 = Sha256Digest::of_bytes(&published);
        let totals = ingested
            .terminal_totals()
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join("/");
        println!(
            "ingested seed={seed} decisions={} scores={totals} ledger={} {file_sha256}",
            ingested.decision_count,
            ledger_file_name(*seed),
        );
        manifest.push((ledger_file_name(*seed), file_sha256));
    }

    manifest.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    println!("manifest policy_id={} count={}", arguments.policy_id, manifest.len());
    for (file_name, file_sha256) in &manifest {
        println!("manifest {file_name} {file_sha256}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Result<Arguments, String> {
        parse_arguments(values.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn out_dir_is_mandatory_and_the_policy_default_is_the_gate0_battery() {
        assert!(arguments(&["raw"]).is_err());
        assert!(arguments(&["--out-dir", "ledgers"]).is_err());
        let parsed = arguments(&["raw", "--out-dir", "ledgers"]).unwrap();
        assert_eq!(parsed.raw_directory, PathBuf::from("raw"));
        assert_eq!(parsed.out_directory, PathBuf::from("ledgers"));
        assert_eq!(parsed.policy_id, DEFAULT_INCUMBENT_POLICY_ID);
    }

    #[test]
    fn policy_ids_outside_the_incumbent_namespace_are_refused_at_parse_time() {
        assert!(
            arguments(&["raw", "--out-dir", "ledgers", "--policy-id", "cpu-proxy:x"]).is_err()
        );
        assert!(arguments(&["raw", "--out-dir", "ledgers", "--policy-id", "incumbent:"]).is_err());
        let parsed = arguments(&[
            "raw",
            "--out-dir",
            "ledgers",
            "--policy-id",
            "incumbent:gate0-alt",
        ])
        .unwrap();
        assert_eq!(parsed.policy_id, "incumbent:gate0-alt");
    }

    #[test]
    fn arguments_are_strict() {
        assert!(arguments(&["raw", "--out-dir"]).is_err());
        assert!(arguments(&["raw", "--out-dir", "a", "--out-dir", "b"]).is_err());
        assert!(arguments(&["raw", "--out-dir", "a", "--frobnicate", "1"]).is_err());
    }

    #[test]
    fn ledger_file_names_are_seed_deterministic() {
        assert_eq!(
            ledger_file_name(2027160000),
            "rival_incumbent_ledger_seed_2027160000.json"
        );
    }
}
