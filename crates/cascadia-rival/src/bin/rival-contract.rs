//! Emit Rust-authoritative, canonical Rival contract records.
//!
//! This utility deliberately writes only to stdout.  Callers can inspect the
//! exact Rust wire before updating a checked-in fixture with `apply_patch`.

use std::fs::{File, Metadata};
use std::io::Read;
use std::path::Path;
use std::{convert::Infallible, process::ExitCode};

use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude};
use cascadia_rival::{
    ACTION_CONTENT_ID_VERSION, BkIdentity, CANDIDATE_ACTION_OCCURRENCE_ID_VERSION,
    CANONICAL_SIMULATOR_ID, CertifiedScoreDifferenceBound, DENSE_COMPILER_ID, EvaluationBranch,
    FailureBehavior, FailureDisposition, Fidelity, ForbiddenCapabilities, FrozenPolicy,
    HonestWorldSampler, INCUMBENT_MENU_HASH_VERSION, INDEPENDENT_SCENARIO_SAMPLER_ID,
    IncumbentCandidateMenu, MAX_TERMINAL_PAIR_LEDGER_BYTES, MenuComposer, NumericalMode,
    POLICY_IDENTITY_SCHEMA_ID, PUBLIC_POLICY_OBSERVATION_SCHEMA_ID, PolicyDecision,
    PolicyIdentityFields, PolicyRng, Precision, ProxyTerminalPair, ProxyTerminalPairRequest,
    RNG_CONTRACT_ID, ROOT_ACTION_OCCURRENCE_ID_VERSION, RULES_MENU_HASH_VERSION,
    ResearchRulesetIdentity, RivalSeed, RngContractIdentity, RngFactory, RootKind, RulesDecision,
    RulesLegalMenu, SEAT_LOCAL_MEMORY_SCHEMA_ID, ScenarioCoordinate, SeatIndex, SeatLocalMemory,
    Sha256Digest, run_proxy_terminal_pair,
};

fn usage() -> &'static str {
    "usage: rival-contract <bound-certificate|ruleset-identity|policy-identity-bk-fixture>\n       rival-contract proxy-terminal-pair-fixture [<parent-manifest-sha256> [<panel-id-sha256> <unit-index-u32>]]\n       rival-contract verify-terminal-pair <pair-ledger.json> <expected-pair-sha256> <expected-parent-manifest-sha256>"
}

fn fixture_policy_identity() -> Result<BkIdentity, String> {
    let digest: Sha256Digest = format!("sha256:{}", "a".repeat(64))
        .parse()
        .map_err(|error| format!("could not construct fixture digest: {error}"))?;
    BkIdentity::new(PolicyIdentityFields {
        ruleset: ResearchRulesetIdentity::canonical(),
        source_revision: "0123456789abcdef".to_owned(),
        source_digest: digest.clone(),
        executable_sha256: digest.clone(),
        model_manifest_sha256: digest.clone(),
        checkpoint_sha256: digest.clone(),
        weights_sha256: digest.clone(),
        bridge_protocol: "bridge.v1".to_owned(),
        tensor_schema: "tensor.v4".to_owned(),
        numerical_mode: NumericalMode::Deterministic,
        precision: Precision::Fp32,
        gumbel_config_sha256: digest.clone(),
        search_config_sha256: digest.clone(),
        refresh_config_sha256: digest.clone(),
        exact_endgame_config_sha256: digest,
        action_content_id_version: ACTION_CONTENT_ID_VERSION.to_owned(),
        rules_action_occurrence_id_version: ROOT_ACTION_OCCURRENCE_ID_VERSION.to_owned(),
        candidate_action_occurrence_id_version: CANDIDATE_ACTION_OCCURRENCE_ID_VERSION.to_owned(),
        rules_menu_hash_version: RULES_MENU_HASH_VERSION.to_owned(),
        incumbent_menu_hash_version: INCUMBENT_MENU_HASH_VERSION.to_owned(),
        rng_contracts: RngContractIdentity {
            physical: RNG_CONTRACT_ID.to_owned(),
            policy: RNG_CONTRACT_ID.to_owned(),
            redetermination: RNG_CONTRACT_ID.to_owned(),
            search: RNG_CONTRACT_ID.to_owned(),
            tie_break: RNG_CONTRACT_ID.to_owned(),
        },
        public_observation_schema: PUBLIC_POLICY_OBSERVATION_SCHEMA_ID.to_owned(),
        policy_memory_schema: SEAT_LOCAL_MEMORY_SCHEMA_ID.to_owned(),
        failure_behavior: FailureBehavior {
            timeout: FailureDisposition::RecordIncompleteNoLabel,
            incomplete_unit: FailureDisposition::RecordIncompleteNoLabel,
            oom: FailureDisposition::RecordIncompleteNoLabel,
            fallback: FailureDisposition::Forbidden,
        },
        compiler_identity: "compiler.v1".to_owned(),
        simulator_identity: "simulator.v1".to_owned(),
        sampler_identity: "sampler.v1".to_owned(),
        candidate_generator_identity: "candidate.v1".to_owned(),
        forbidden_capabilities: ForbiddenCapabilities {
            table_total_utility: false,
            table_native_q: false,
            true_hidden_peeking: false,
            model_fallback: false,
        },
    })
    .map_err(|error| format!("could not construct {POLICY_IDENTITY_SCHEMA_ID} fixture: {error}"))
}

#[derive(Clone)]
struct FixtureFirstLegalPolicy {
    identity: BkIdentity,
}

impl FrozenPolicy for FixtureFirstLegalPolicy {
    type Identity = BkIdentity;
    type Error = Infallible;

    fn identity(&self) -> &Self::Identity {
        &self.identity
    }

    fn fresh_instance(&self) -> Self {
        self.clone()
    }

    fn act(
        &mut self,
        observation: &cascadia_rival::PublicPolicyObs,
        menu: &RulesLegalMenu,
        _worlds: &HonestWorldSampler,
        _rng: &mut PolicyRng,
    ) -> Result<PolicyDecision, Self::Error> {
        let index = match menu.root_kind() {
            RootKind::PreludePolicyRoot => 0,
            RootKind::DraftPolicyRoot => menu
                .first_draft_index()
                .expect("canonical fixture draft menu contains a draft"),
        };
        Ok(
            PolicyDecision::new(index, menu, observation.memory().clone())
                .expect("fixture chooses an index from the supplied menu"),
        )
    }
}

fn fixture_proxy_policy_identity() -> Result<BkIdentity, String> {
    let mut fields = fixture_policy_identity()?.fields().clone();
    fields.compiler_identity = DENSE_COMPILER_ID.to_owned();
    fields.simulator_identity = CANONICAL_SIMULATOR_ID.to_owned();
    fields.sampler_identity = INDEPENDENT_SCENARIO_SAMPLER_ID.to_owned();
    fields.candidate_generator_identity = "rival-contract.first-legal-fixture.v1".to_owned();
    BkIdentity::new(fields).map_err(|error| format!("invalid fixture proxy identity: {error}"))
}

fn proxy_terminal_pair_fixture(
    parent_manifest_sha256: Sha256Digest,
    panel_id: Sha256Digest,
    unit_index: u32,
) -> Result<ProxyTerminalPair, String> {
    let mut source = GameState::new(
        GameConfig::research_aaaaa(4).map_err(|error| error.to_string())?,
        GameSeed::from_u64(123),
    )
    .map_err(|error| error.to_string())?;
    while source.completed_turns() < 75 {
        let action = source
            .legal_turn_actions(&MarketPrelude::default())
            .map_err(|error| error.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "fixture source unexpectedly has no legal action".to_owned())?;
        source.apply(&action).map_err(|error| error.to_string())?;
    }
    let rules_menu = MenuComposer::draft_root(&source, &MarketPrelude::default())
        .map_err(|error| error.to_string())?;
    let draft_indices: Vec<_> = rules_menu
        .decisions()
        .iter()
        .enumerate()
        .filter_map(|(index, decision)| {
            matches!(decision, RulesDecision::Draft(_)).then_some(index)
        })
        .take(2)
        .collect();
    if draft_indices.len() != 2 {
        return Err("fixture source does not expose two draft candidates".to_owned());
    }
    let candidate_menu = IncumbentCandidateMenu::from_rules_indices(&rules_menu, draft_indices)
        .map_err(|error| error.to_string())?;
    let sampler = cascadia_rival::IndependentScenarioSampler::new(
        source.clone(),
        std::array::from_fn(|seat| SeatLocalMemory::new(vec![10 + seat as u8])),
        RngFactory::new(RivalSeed::from_u64(501)),
    )
    .map_err(|error| error.to_string())?;
    let incumbent_coordinate = ScenarioCoordinate {
        panel_id: panel_id.clone(),
        unit_index,
        branch: EvaluationBranch::Incumbent,
        fidelity: Fidelity::High,
    };
    let challenger_coordinate = ScenarioCoordinate {
        panel_id,
        unit_index,
        branch: EvaluationBranch::Challenger(1),
        fidelity: Fidelity::High,
    };
    let identity = fixture_proxy_policy_identity()?;
    run_proxy_terminal_pair(ProxyTerminalPairRequest {
        sampler: &sampler,
        parent_manifest_sha256,
        incumbent_coordinate: &incumbent_coordinate,
        challenger_coordinate: &challenger_coordinate,
        candidate_menu: &candidate_menu,
        incumbent_candidate_index: 0,
        challenger_candidate_index: 1,
        incumbent_post_action_memory: SeatLocalMemory::new(vec![0xa1]),
        challenger_post_action_memory: SeatLocalMemory::new(vec![0xb1]),
        policy_prototype: FixtureFirstLegalPolicy {
            identity: identity.clone(),
        },
        rng_factory: &RngFactory::new(RivalSeed::from_u64(502)),
        target_seat: SeatIndex::new(source.current_player() as u8)
            .map_err(|error| error.to_string())?,
    })
    .map_err(|error| error.to_string())
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().ok_or_else(|| usage().to_owned())?;
    let value = match command.as_str() {
        "bound-certificate" => {
            require_no_more_arguments(&mut arguments)?;
            serde_json::to_value(CertifiedScoreDifferenceBound::global_research_aaaaa())
        }
        "ruleset-identity" => {
            require_no_more_arguments(&mut arguments)?;
            serde_json::to_value(ResearchRulesetIdentity::canonical())
        }
        "policy-identity-bk-fixture" => {
            require_no_more_arguments(&mut arguments)?;
            serde_json::to_value(fixture_policy_identity()?)
        }
        "proxy-terminal-pair-fixture" => {
            let (parent_manifest_sha256, panel_id, unit_index) =
                fixture_coordinates(&mut arguments)?;
            serde_json::to_value(proxy_terminal_pair_fixture(
                parent_manifest_sha256,
                panel_id,
                unit_index,
            )?)
        }
        "verify-terminal-pair" => {
            let path = arguments.next().ok_or_else(|| usage().to_owned())?;
            let (expected_pair, expected_parent) = required_verifier_pins(&mut arguments)?;
            let bytes = read_terminal_pair_ledger(Path::new(&path))?;
            let executable_path = std::env::current_exe()
                .map_err(|error| format!("could not identify verifier executable: {error}"))?;
            let executable_bytes = std::fs::read(&executable_path).map_err(|error| {
                format!(
                    "could not hash verifier executable {:?}: {error}",
                    executable_path
                )
            })?;
            let (_pair, receipt) = ProxyTerminalPair::verify_bytes_and_create_receipt(
                &bytes,
                &executable_bytes,
                &expected_pair,
                &expected_parent,
            )
            .map_err(|error| {
                format!("terminal-pair ledger {path:?} failed pinned Rust verification: {error}")
            })?;
            serde_json::to_value(receipt)
        }
        _ => return Err(format!("unknown command {command:?}; {}", usage())),
    }
    .map_err(|error| format!("could not serialize canonical contract: {error}"))?;
    let rendered = serde_json::to_string_pretty(&value)
        .map_err(|error| format!("could not render canonical contract: {error}"))?;
    println!("{rendered}");
    Ok(())
}

fn read_terminal_pair_ledger(path: &Path) -> Result<Vec<u8>, String> {
    let path_metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect terminal-pair ledger {path:?}: {error}"))?;
    if path_metadata.file_type().is_symlink() {
        return Err(format!(
            "terminal-pair ledger {path:?} must not be a symbolic link"
        ));
    }
    validate_terminal_pair_file_metadata(path, &path_metadata)?;

    let mut file = File::open(path)
        .map_err(|error| format!("could not open terminal-pair ledger {path:?}: {error}"))?;
    let opened_metadata = file.metadata().map_err(|error| {
        format!("could not inspect opened terminal-pair ledger {path:?}: {error}")
    })?;
    validate_terminal_pair_file_metadata(path, &opened_metadata)?;
    ensure_same_opened_file(path, &path_metadata, &opened_metadata)?;

    let capacity = usize::try_from(opened_metadata.len()).map_err(|_| {
        format!("terminal-pair ledger {path:?} length cannot be represented on this platform")
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take(MAX_TERMINAL_PAIR_LEDGER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("could not read terminal-pair ledger {path:?}: {error}"))?;
    let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual > MAX_TERMINAL_PAIR_LEDGER_BYTES {
        return Err(format!(
            "terminal-pair ledger {path:?} has more than the hard maximum of {MAX_TERMINAL_PAIR_LEDGER_BYTES} bytes"
        ));
    }
    let final_metadata = file
        .metadata()
        .map_err(|error| format!("could not re-inspect terminal-pair ledger {path:?}: {error}"))?;
    if final_metadata.len() != opened_metadata.len() || actual != opened_metadata.len() {
        return Err(format!(
            "terminal-pair ledger {path:?} changed length while it was being read"
        ));
    }
    Ok(bytes)
}

fn validate_terminal_pair_file_metadata(path: &Path, metadata: &Metadata) -> Result<(), String> {
    if !metadata.is_file() {
        return Err(format!(
            "terminal-pair ledger {path:?} must be a regular file"
        ));
    }
    if metadata.len() > MAX_TERMINAL_PAIR_LEDGER_BYTES {
        return Err(format!(
            "terminal-pair ledger {path:?} has {} bytes; hard maximum is {MAX_TERMINAL_PAIR_LEDGER_BYTES}",
            metadata.len()
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn ensure_same_opened_file(
    path: &Path,
    path_metadata: &Metadata,
    opened_metadata: &Metadata,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;

    if path_metadata.dev() != opened_metadata.dev() || path_metadata.ino() != opened_metadata.ino()
    {
        return Err(format!(
            "terminal-pair ledger {path:?} changed between inspection and open"
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_opened_file(
    _path: &Path,
    _path_metadata: &Metadata,
    _opened_metadata: &Metadata,
) -> Result<(), String> {
    Ok(())
}

fn fixture_coordinates(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(Sha256Digest, Sha256Digest, u32), String> {
    let default_panel = Sha256Digest::of_bytes(b"rival-contract-terminal-pair-fixture-panel");
    let parent = match arguments.next() {
        Some(value) => value
            .parse()
            .map_err(|error| format!("invalid fixture parent-manifest SHA-256: {error}"))?,
        None => Sha256Digest::of_bytes(b"fixture-parent-manifest"),
    };
    let Some(panel) = arguments.next() else {
        return Ok((parent, default_panel, 0));
    };
    let unit = arguments.next().ok_or_else(|| usage().to_owned())?;
    require_no_more_arguments(arguments)?;
    let panel = panel
        .parse()
        .map_err(|error| format!("invalid fixture panel ID SHA-256: {error}"))?;
    let unit_index: u32 = unit
        .parse()
        .map_err(|error| format!("invalid fixture unit index: {error}"))?;
    if unit_index.to_string() != unit {
        return Err("fixture unit index must be canonical unsigned decimal".to_owned());
    }
    Ok((parent, panel, unit_index))
}

fn required_verifier_pins(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(Sha256Digest, Sha256Digest), String> {
    let pair = arguments.next().ok_or_else(|| usage().to_owned())?;
    let parent = arguments.next().ok_or_else(|| usage().to_owned())?;
    require_no_more_arguments(arguments)?;
    let pair = pair
        .parse()
        .map_err(|error| format!("invalid expected pair SHA-256: {error}"))?;
    let parent = parent
        .parse()
        .map_err(|error| format!("invalid expected parent-manifest SHA-256: {error}"))?;
    Ok((pair, parent))
}

fn require_no_more_arguments(arguments: &mut impl Iterator<Item = String>) -> Result<(), String> {
    if arguments.next().is_some() {
        Err(usage().to_owned())
    } else {
        Ok(())
    }
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

    #[test]
    fn fixture_coordinates_allow_only_zero_one_or_three_strict_arguments() {
        let mut absent = Vec::<String>::new().into_iter();
        assert_eq!(
            fixture_coordinates(&mut absent).unwrap(),
            (
                Sha256Digest::of_bytes(b"fixture-parent-manifest"),
                Sha256Digest::of_bytes(b"rival-contract-terminal-pair-fixture-panel"),
                0,
            )
        );

        let parent = Sha256Digest::of_bytes(b"python-parent-manifest");
        let mut supplied = vec![parent.to_string()].into_iter();
        assert_eq!(
            fixture_coordinates(&mut supplied).unwrap(),
            (
                parent.clone(),
                Sha256Digest::of_bytes(b"rival-contract-terminal-pair-fixture-panel"),
                0,
            )
        );

        let panel = Sha256Digest::of_bytes(b"python-panel-plan");
        let mut complete = vec![
            parent.to_string(),
            panel.to_string(),
            "4294967295".to_owned(),
        ]
        .into_iter();
        assert_eq!(
            fixture_coordinates(&mut complete).unwrap(),
            (parent.clone(), panel.clone(), u32::MAX)
        );

        let mut unqualified = vec!["0".repeat(64)].into_iter();
        assert!(fixture_coordinates(&mut unqualified).is_err());
        let mut incomplete = vec![parent.to_string(), panel.to_string()].into_iter();
        assert!(fixture_coordinates(&mut incomplete).is_err());
        let mut overflow = vec![
            parent.to_string(),
            panel.to_string(),
            "4294967296".to_owned(),
        ]
        .into_iter();
        assert!(fixture_coordinates(&mut overflow).is_err());
        let mut noncanonical =
            vec![parent.to_string(), panel.to_string(), "01".to_owned()].into_iter();
        assert!(fixture_coordinates(&mut noncanonical).is_err());
        let mut extra = vec![
            parent.to_string(),
            panel.to_string(),
            "0".to_owned(),
            "extra".to_owned(),
        ]
        .into_iter();
        assert!(fixture_coordinates(&mut extra).is_err());
    }

    #[test]
    fn verifier_requires_one_complete_strict_pin_pair() {
        let mut absent = Vec::<String>::new().into_iter();
        assert!(required_verifier_pins(&mut absent).is_err());

        let pair = Sha256Digest::of_bytes(b"pair");
        let parent = Sha256Digest::of_bytes(b"parent");
        let mut supplied = vec![pair.to_string(), parent.to_string()].into_iter();
        assert_eq!(
            required_verifier_pins(&mut supplied).unwrap(),
            (pair.clone(), parent.clone())
        );

        let mut partial = vec![pair.to_string()].into_iter();
        assert!(required_verifier_pins(&mut partial).is_err());
        let mut unqualified = vec!["0".repeat(64), parent.to_string()].into_iter();
        assert!(required_verifier_pins(&mut unqualified).is_err());
        let mut extra = vec![pair.to_string(), parent.to_string(), "extra".to_owned()].into_iter();
        assert!(required_verifier_pins(&mut extra).is_err());
    }
}
