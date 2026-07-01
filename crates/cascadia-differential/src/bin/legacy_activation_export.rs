#[cfg(not(feature = "legacy-teacher"))]
fn main() {
    eprintln!(
        "legacy_activation_export requires --features legacy-teacher \
         (which enables the historical mid-features,v4-opp extractor)"
    );
    std::process::exit(2);
}

#[cfg(feature = "legacy-teacher")]
mod enabled {
    use std::{
        collections::BTreeMap,
        error::Error,
        fmt,
        fs::{self, File},
        io::{BufRead, BufReader, BufWriter, Write},
        path::{Path, PathBuf},
    };

    use blake3::Hasher;
    use cascadia_ai::{
        draft_opponents::{
            preference_draft_move, random_draft_move, sample_preferences, scarcity_draft_move,
        },
        nnue::{
            BagInfo, NUM_FEATURES, NUM_FEATURES_MID, NUM_FEATURES_MID_V4, OPP_DETAILED_BASE,
            extract_features_with_bag,
        },
        search::{execute_scored_move, greedy_move},
    };
    use cascadia_core::{game::GameState, types::ScoringCards};
    use cascadia_provenance::{checksum_file, repository_root, source_provenance};
    use clap::{Parser, Subcommand};
    use rand::{SeedableRng, rngs::StdRng};
    use rayon::prelude::*;
    use serde::{Deserialize, Serialize};

    const DATASET_ID: &str = "legacy-mid-v4opp-activation-v1";
    const FEATURE_SCHEMA: &str = "legacy-mid-v4opp-11231";
    const FEATURE_COUNT: usize = 11_231;
    const MID_FEATURE_COUNT: usize = 10_862;
    const PLAYERS: usize = 4;
    const TURNS_PER_PLAYER: usize = 20;
    const ROWS_PER_GAME: usize = PLAYERS * TURNS_PER_PLAYER;
    const DEFAULT_GAMES: usize = 2_500;
    const DEFAULT_SHARD_GAMES: usize = 250;
    const GAME_SEED_DOMAIN: &[u8] = b"legacy-mid-v4opp-activation-v1/game-seed/v1";
    const POLICY_SEED_DOMAIN: &[u8] = b"legacy-mid-v4opp-activation-v1/policy-seed/v1";
    const SOURCE_BUNDLE_ROOTS: &[&str] = &[
        "Cargo.lock",
        "crates/cascadia-differential/Cargo.toml",
        "crates/cascadia-differential/src/bin/legacy_activation_export.rs",
        "legacy/crates/cascadia-ai/Cargo.toml",
        "legacy/crates/cascadia-ai/src/nnue.rs",
        "legacy/crates/cascadia-ai/src/eval.rs",
        "legacy/crates/cascadia-ai/src/search.rs",
        "legacy/crates/cascadia-ai/src/draft_opponents.rs",
        "legacy/crates/cascadia-ai/src/potential.rs",
        "legacy/crates/cascadia-core/Cargo.toml",
        "legacy/crates/cascadia-core/src",
    ];

    type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

    #[derive(Debug)]
    struct ExportError(String);

    impl fmt::Display for ExportError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl Error for ExportError {}

    fn failure(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
        Box::new(ExportError(message.into()))
    }

    #[derive(Debug, Parser)]
    #[command(about = "Export and validate exact historical mid-features,v4-opp activation rows")]
    struct Cli {
        #[command(subcommand)]
        command: CommandKind,
    }

    #[derive(Debug, Subcommand)]
    enum CommandKind {
        Generate {
            #[arg(long)]
            output: PathBuf,
            #[arg(long, default_value_t = DEFAULT_GAMES)]
            games: usize,
            #[arg(long, default_value_t = DEFAULT_SHARD_GAMES)]
            shard_games: usize,
            #[arg(long, default_value_t = 0)]
            first_game_index: u64,
            #[arg(long, default_value_t = 0)]
            threads: usize,
        },
        Validate {
            #[arg(long)]
            root: PathBuf,
            #[arg(long)]
            content_only: bool,
        },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum TrajectoryPolicy {
        Greedy,
        RandomDraft,
        ScarcityDraft,
        PreferenceDraft,
    }

    impl TrajectoryPolicy {
        const ALL: [Self; 4] = [
            Self::Greedy,
            Self::RandomDraft,
            Self::ScarcityDraft,
            Self::PreferenceDraft,
        ];

        fn id(self) -> u64 {
            match self {
                Self::Greedy => 0,
                Self::RandomDraft => 1,
                Self::ScarcityDraft => 2,
                Self::PreferenceDraft => 3,
            }
        }

        fn as_str(self) -> &'static str {
            match self {
                Self::Greedy => "greedy",
                Self::RandomDraft => "random_draft",
                Self::ScarcityDraft => "scarcity_draft",
                Self::PreferenceDraft => "preference_draft",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    enum Phase {
        Opening,
        Early,
        Middle,
        Late,
    }

    impl Phase {
        const ALL: [Self; 4] = [Self::Opening, Self::Early, Self::Middle, Self::Late];

        fn as_str(self) -> &'static str {
            match self {
                Self::Opening => "opening",
                Self::Early => "early",
                Self::Middle => "middle",
                Self::Late => "late",
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct ActivationRow {
        game_index: u64,
        decision_index: u8,
        features: Vec<u16>,
        raw_feature_count: u16,
        focal_seat: u8,
        personal_turn: u8,
        phase: Phase,
        policy: TrajectoryPolicy,
        free_overflow_applied: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SeedSchedule {
        algorithm: String,
        game_domain: String,
        policy_domain: String,
        first_game_index: u64,
        games: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct PolicySchedule {
        assignment: String,
        policies: Vec<String>,
        preference_sampling: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct GenerationContract {
        players: usize,
        scoring_cards: String,
        score_outputs_collected: bool,
        habitat_bonus_labels_collected: bool,
        state_timing: String,
        free_overflow_prelude: String,
        paid_wildlife_wipes: bool,
        feature_extractor: String,
        cargo_features: Vec<String>,
        feature_count: usize,
        sparse_normalization: String,
        seed_schedule: SeedSchedule,
        policy_schedule: PolicySchedule,
        rows_per_game: usize,
        shard_games: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SourceFileIdentity {
        file: String,
        bytes: u64,
        blake3: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SourceIdentity {
        contract: String,
        git_revision: String,
        repository_v2_source_blake3: String,
        source_bundle_blake3: String,
        extractor_source_blake3: String,
        files: Vec<SourceFileIdentity>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct ExecutableIdentity {
        file_name: String,
        bytes: u64,
        blake3: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Provenance {
        source: SourceIdentity,
        executable: ExecutableIdentity,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct ShardManifest {
        file: String,
        row_count: usize,
        bytes: u64,
        blake3: String,
        first_game_index: u64,
        games: usize,
        first_game_seed: u64,
        last_game_seed: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct DatasetStatistics {
        games: usize,
        rows: usize,
        rows_by_phase: BTreeMap<String, u64>,
        rows_by_focal_seat: BTreeMap<String, u64>,
        rows_by_policy: BTreeMap<String, u64>,
        free_overflow_preludes: u64,
        raw_feature_emissions: u64,
        unique_feature_activations: u64,
        duplicate_feature_emissions_removed: u64,
        minimum_unique_features_per_row: usize,
        maximum_unique_features_per_row: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Manifest {
        schema_version: u32,
        dataset_id: String,
        feature_schema: String,
        feature_count: usize,
        split: String,
        rows: usize,
        shards: Vec<ShardManifest>,
        payload_blake3: String,
        generation: GenerationContract,
        statistics: DatasetStatistics,
        provenance: Provenance,
        scientific_blake3: String,
    }

    #[derive(Debug, Serialize)]
    struct ScientificIdentity<'a> {
        schema_version: u32,
        dataset_id: &'a str,
        feature_schema: &'a str,
        feature_count: usize,
        split: &'a str,
        rows: usize,
        shards: &'a [ShardManifest],
        payload_blake3: &'a str,
        generation: &'a GenerationContract,
        statistics: &'a DatasetStatistics,
        source_bundle_blake3: &'a str,
        extractor_source_blake3: &'a str,
        executable_blake3: &'a str,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    struct ValidationReport {
        dataset_id: String,
        manifest_blake3: String,
        scientific_blake3: String,
        payload_blake3: String,
        rows: usize,
        shards: usize,
        source_bundle_blake3: String,
        extractor_source_blake3: String,
        executable_blake3: String,
        relevant_source_matches: bool,
        executable_matches: bool,
        content_only: bool,
        status: String,
    }

    #[derive(Debug, Clone)]
    struct MutableStatistics {
        games: usize,
        rows: usize,
        phase: [u64; 4],
        seats: [u64; 4],
        policies: [u64; 4],
        free_overflow_preludes: u64,
        raw_feature_emissions: u64,
        unique_feature_activations: u64,
        minimum_unique_features_per_row: usize,
        maximum_unique_features_per_row: usize,
    }

    impl Default for MutableStatistics {
        fn default() -> Self {
            Self {
                games: 0,
                rows: 0,
                phase: [0; 4],
                seats: [0; 4],
                policies: [0; 4],
                free_overflow_preludes: 0,
                raw_feature_emissions: 0,
                unique_feature_activations: 0,
                minimum_unique_features_per_row: usize::MAX,
                maximum_unique_features_per_row: 0,
            }
        }
    }

    impl MutableStatistics {
        fn observe(&mut self, row: &ActivationRow) {
            self.rows += 1;
            self.phase[phase_index(row.phase)] += 1;
            self.seats[usize::from(row.focal_seat)] += 1;
            self.policies[row.policy.id() as usize] += 1;
            self.free_overflow_preludes += u64::from(row.free_overflow_applied);
            self.raw_feature_emissions += u64::from(row.raw_feature_count);
            self.unique_feature_activations += row.features.len() as u64;
            self.minimum_unique_features_per_row =
                self.minimum_unique_features_per_row.min(row.features.len());
            self.maximum_unique_features_per_row =
                self.maximum_unique_features_per_row.max(row.features.len());
        }

        fn merge(&mut self, other: &Self) {
            self.games += other.games;
            self.rows += other.rows;
            for index in 0..4 {
                self.phase[index] += other.phase[index];
                self.seats[index] += other.seats[index];
                self.policies[index] += other.policies[index];
            }
            self.free_overflow_preludes += other.free_overflow_preludes;
            self.raw_feature_emissions += other.raw_feature_emissions;
            self.unique_feature_activations += other.unique_feature_activations;
            self.minimum_unique_features_per_row = self
                .minimum_unique_features_per_row
                .min(other.minimum_unique_features_per_row);
            self.maximum_unique_features_per_row = self
                .maximum_unique_features_per_row
                .max(other.maximum_unique_features_per_row);
        }

        fn finish(self) -> DatasetStatistics {
            let rows_by_phase = Phase::ALL
                .into_iter()
                .enumerate()
                .map(|(index, phase)| (phase.as_str().to_owned(), self.phase[index]))
                .collect();
            let rows_by_focal_seat = (0..PLAYERS)
                .map(|seat| (seat.to_string(), self.seats[seat]))
                .collect();
            let rows_by_policy = TrajectoryPolicy::ALL
                .into_iter()
                .enumerate()
                .map(|(index, policy)| (policy.as_str().to_owned(), self.policies[index]))
                .collect();
            DatasetStatistics {
                games: self.games,
                rows: self.rows,
                rows_by_phase,
                rows_by_focal_seat,
                rows_by_policy,
                free_overflow_preludes: self.free_overflow_preludes,
                raw_feature_emissions: self.raw_feature_emissions,
                unique_feature_activations: self.unique_feature_activations,
                duplicate_feature_emissions_removed: self
                    .raw_feature_emissions
                    .saturating_sub(self.unique_feature_activations),
                minimum_unique_features_per_row: if self.rows == 0 {
                    0
                } else {
                    self.minimum_unique_features_per_row
                },
                maximum_unique_features_per_row: self.maximum_unique_features_per_row,
            }
        }
    }

    #[derive(Debug)]
    struct GameRows {
        game_index: u64,
        rows: Vec<ActivationRow>,
        statistics: MutableStatistics,
    }

    struct TemporaryRoot {
        path: PathBuf,
        published: bool,
    }

    impl Drop for TemporaryRoot {
        fn drop(&mut self) {
            if !self.published {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    pub fn run() -> Result<()> {
        let cli = Cli::parse();
        match cli.command {
            CommandKind::Generate {
                output,
                games,
                shard_games,
                first_game_index,
                threads,
            } => {
                let report =
                    generate_dataset(&output, games, shard_games, first_game_index, threads)?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            CommandKind::Validate { root, content_only } => {
                let report = validate_dataset(&root, content_only)?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        Ok(())
    }

    fn validate_compiled_feature_contract() -> Result<()> {
        if NUM_FEATURES != FEATURE_COUNT
            || NUM_FEATURES_MID != MID_FEATURE_COUNT
            || NUM_FEATURES_MID_V4 != FEATURE_COUNT
            || OPP_DETAILED_BASE != MID_FEATURE_COUNT
        {
            return Err(failure(format!(
                "historical extractor feature contract drifted: NUM_FEATURES={NUM_FEATURES}, \
                 NUM_FEATURES_MID={NUM_FEATURES_MID}, NUM_FEATURES_MID_V4={NUM_FEATURES_MID_V4}, \
                 OPP_DETAILED_BASE={OPP_DETAILED_BASE}"
            )));
        }
        Ok(())
    }

    fn phase_for_personal_turn(personal_turn: u8) -> Result<Phase> {
        match personal_turn {
            1 => Ok(Phase::Opening),
            2..=5 => Ok(Phase::Early),
            6..=13 => Ok(Phase::Middle),
            14..=20 => Ok(Phase::Late),
            _ => Err(failure(format!(
                "personal turn {personal_turn} is outside 1..=20"
            ))),
        }
    }

    fn phase_index(phase: Phase) -> usize {
        match phase {
            Phase::Opening => 0,
            Phase::Early => 1,
            Phase::Middle => 2,
            Phase::Late => 3,
        }
    }

    fn policy_for(game_index: u64, seat: usize) -> TrajectoryPolicy {
        TrajectoryPolicy::ALL[((game_index + seat as u64) % 4) as usize]
    }

    fn derive_seed(domain: &[u8], values: &[u64]) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(domain);
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
        let digest = hasher.finalize();
        u64::from_le_bytes(
            digest.as_bytes()[..8]
                .try_into()
                .expect("BLAKE3 output always has at least eight bytes"),
        )
    }

    fn game_seed(game_index: u64) -> u64 {
        derive_seed(GAME_SEED_DOMAIN, &[game_index])
    }

    fn policy_seed(game_seed: u64, seat: usize, policy: TrajectoryPolicy) -> u64 {
        derive_seed(POLICY_SEED_DOMAIN, &[game_seed, seat as u64, policy.id()])
    }

    fn generate_game(game_index: u64) -> Result<GameRows> {
        let seed = game_seed(game_index);
        let mut setup_rng = StdRng::seed_from_u64(seed);
        let mut game = GameState::new(PLAYERS, ScoringCards::all_a(), &mut setup_rng);
        let policies: [TrajectoryPolicy; PLAYERS] =
            std::array::from_fn(|seat| policy_for(game_index, seat));
        let mut policy_rngs: [StdRng; PLAYERS] = std::array::from_fn(|seat| {
            StdRng::seed_from_u64(policy_seed(seed, seat, policies[seat]))
        });
        let preferences: [[f32; 5]; PLAYERS] = std::array::from_fn(|seat| {
            if policies[seat] == TrajectoryPolicy::PreferenceDraft {
                sample_preferences(&mut policy_rngs[seat])
            } else {
                [0.2; 5]
            }
        });
        let mut rows = Vec::with_capacity(ROWS_PER_GAME);
        let mut statistics = MutableStatistics::default();

        for decision_index in 0..ROWS_PER_GAME {
            if game.is_game_over() {
                return Err(failure(format!(
                    "game {game_index} ended after {decision_index} decisions"
                )));
            }
            let focal_seat = game.current_player;
            let expected_seat = decision_index % PLAYERS;
            if focal_seat != expected_seat {
                return Err(failure(format!(
                    "game {game_index} decision {decision_index} expected seat \
                     {expected_seat}, observed {focal_seat}"
                )));
            }
            let free_overflow_applied = if game.can_replace_overflow().is_some() {
                if !game.replace_overflow() {
                    return Err(failure(format!(
                        "game {game_index} decision {decision_index} failed legal overflow prelude"
                    )));
                }
                true
            } else {
                false
            };
            let personal_turn = u8::try_from(decision_index / PLAYERS + 1)?;
            let phase = phase_for_personal_turn(personal_turn)?;
            let bag = BagInfo::from_game_for_player(&game, focal_seat);
            let mut features = extract_features_with_bag(&game.boards[focal_seat], Some(&bag));
            let raw_feature_count = u16::try_from(features.len())?;
            features.sort_unstable();
            features.dedup();
            if features.is_empty() {
                return Err(failure(format!(
                    "game {game_index} decision {decision_index} emitted no features"
                )));
            }
            if features
                .last()
                .is_some_and(|index| usize::from(*index) >= FEATURE_COUNT)
            {
                return Err(failure(format!(
                    "game {game_index} decision {decision_index} crossed feature boundary"
                )));
            }
            let policy = policies[focal_seat];
            let row = ActivationRow {
                game_index,
                decision_index: u8::try_from(decision_index)?,
                features,
                raw_feature_count,
                focal_seat: u8::try_from(focal_seat)?,
                personal_turn,
                phase,
                policy,
                free_overflow_applied,
            };
            validate_row(&row, game_index, decision_index)?;
            statistics.observe(&row);
            rows.push(row);

            let movement = match policy {
                TrajectoryPolicy::Greedy => greedy_move(&game),
                TrajectoryPolicy::RandomDraft => {
                    random_draft_move(&game, &mut policy_rngs[focal_seat])
                }
                TrajectoryPolicy::ScarcityDraft => {
                    scarcity_draft_move(&game, &mut policy_rngs[focal_seat])
                }
                TrajectoryPolicy::PreferenceDraft => preference_draft_move(
                    &game,
                    &preferences[focal_seat],
                    &mut policy_rngs[focal_seat],
                ),
            }
            .ok_or_else(|| {
                failure(format!(
                    "game {game_index} decision {decision_index} policy {} produced no move",
                    policy.as_str()
                ))
            })?;
            if !execute_scored_move(&mut game, &movement) {
                return Err(failure(format!(
                    "game {game_index} decision {decision_index} policy {} produced an illegal move",
                    policy.as_str()
                )));
            }
        }
        if !game.is_game_over() {
            return Err(failure(format!(
                "game {game_index} remained live after {ROWS_PER_GAME} decisions"
            )));
        }
        statistics.games = 1;
        Ok(GameRows {
            game_index,
            rows,
            statistics,
        })
    }

    fn validate_row(
        row: &ActivationRow,
        expected_game_index: u64,
        expected_decision_index: usize,
    ) -> Result<()> {
        if row.game_index != expected_game_index {
            return Err(failure("activation row game index drifted"));
        }
        if usize::from(row.decision_index) != expected_decision_index {
            return Err(failure("activation row decision index drifted"));
        }
        let expected_seat = expected_decision_index % PLAYERS;
        if usize::from(row.focal_seat) != expected_seat {
            return Err(failure("activation row focal seat drifted"));
        }
        let expected_turn = u8::try_from(expected_decision_index / PLAYERS + 1)?;
        if row.personal_turn != expected_turn
            || row.phase != phase_for_personal_turn(expected_turn)?
        {
            return Err(failure("activation row phase or personal turn drifted"));
        }
        if row.policy != policy_for(expected_game_index, expected_seat) {
            return Err(failure("activation row policy schedule drifted"));
        }
        if row.features.is_empty() {
            return Err(failure("activation row has no features"));
        }
        if usize::from(row.raw_feature_count) < row.features.len() {
            return Err(failure(
                "activation row raw feature count is below normalized count",
            ));
        }
        if row.features.windows(2).any(|window| window[0] >= window[1]) {
            return Err(failure(
                "activation row features must be strictly sorted and unique",
            ));
        }
        if row
            .features
            .last()
            .is_some_and(|index| usize::from(*index) >= FEATURE_COUNT)
        {
            return Err(failure("activation row feature index is out of range"));
        }
        Ok(())
    }

    fn generation_contract(
        games: usize,
        shard_games: usize,
        first_game_index: u64,
    ) -> GenerationContract {
        GenerationContract {
            players: PLAYERS,
            scoring_cards: "AAAAA".to_owned(),
            score_outputs_collected: false,
            habitat_bonus_labels_collected: false,
            state_timing:
                ("pre-decision after legal free three-of-a-kind replacement and before draft")
                    .to_owned(),
            free_overflow_prelude: "always apply once when legal".to_owned(),
            paid_wildlife_wipes: false,
            feature_extractor: ("cascadia_ai::nnue::BagInfo::from_game_for_player + \
                 cascadia_ai::nnue::extract_features_with_bag")
                .to_owned(),
            cargo_features: vec!["mid-features".to_owned(), "v4-opp".to_owned()],
            feature_count: FEATURE_COUNT,
            sparse_normalization:
                ("sort ascending and deduplicate exact Rust emissions as binary activations")
                    .to_owned(),
            seed_schedule: SeedSchedule {
                algorithm:
                    ("u64 little-endian first eight bytes of BLAKE3(domain || values_le_u64)")
                        .to_owned(),
                game_domain: String::from_utf8_lossy(GAME_SEED_DOMAIN).into_owned(),
                policy_domain: String::from_utf8_lossy(POLICY_SEED_DOMAIN).into_owned(),
                first_game_index,
                games,
            },
            policy_schedule: PolicySchedule {
                assignment: "(game_index + focal_seat) mod 4".to_owned(),
                policies: TrajectoryPolicy::ALL
                    .into_iter()
                    .map(|policy| policy.as_str().to_owned())
                    .collect(),
                preference_sampling: ("one historical sample_preferences draw per game and seat")
                    .to_owned(),
            },
            rows_per_game: ROWS_PER_GAME,
            shard_games,
        }
    }

    fn generate_dataset(
        output: &Path,
        games: usize,
        shard_games: usize,
        first_game_index: u64,
        threads: usize,
    ) -> Result<ValidationReport> {
        validate_compiled_feature_contract()?;
        if games == 0 || shard_games == 0 {
            return Err(failure("games and shard-games must both be positive"));
        }
        let last_game_index = first_game_index
            .checked_add(u64::try_from(games - 1)?)
            .ok_or_else(|| failure("game index range overflowed u64"))?;
        let _ = game_seed(last_game_index);
        if output.exists() {
            return Err(failure(format!(
                "output root already exists: {}",
                output.display()
            )));
        }
        let parent = output
            .parent()
            .ok_or_else(|| failure("output root must have a parent directory"))?;
        fs::create_dir_all(parent)?;
        let file_name = output
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| failure("output root must have a UTF-8 file name"))?;
        let temporary_path = parent.join(format!(".{file_name}.tmp-{}", std::process::id()));
        if temporary_path.exists() {
            return Err(failure(format!(
                "temporary output root already exists: {}",
                temporary_path.display()
            )));
        }
        fs::create_dir(&temporary_path)?;
        let mut temporary = TemporaryRoot {
            path: temporary_path,
            published: false,
        };
        let source_before = capture_source_identity()?;
        let executable = capture_executable_identity()?;
        let thread_count = if threads == 0 {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        } else {
            threads
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()?;
        let mut shards = Vec::new();
        let mut aggregate = MutableStatistics::default();

        for (shard_index, game_offset) in (0..games).step_by(shard_games).enumerate() {
            let count = shard_games.min(games - game_offset);
            let first = first_game_index + u64::try_from(game_offset)?;
            let mut game_rows = pool.install(|| {
                (0..count)
                    .into_par_iter()
                    .map(|offset| generate_game(first + offset as u64))
                    .collect::<Result<Vec<_>>>()
            })?;
            game_rows.sort_by_key(|game| game.game_index);
            let file = format!("part-{shard_index:05}.jsonl");
            let path = temporary.path.join(&file);
            let mut writer = BufWriter::new(File::create(&path)?);
            let mut shard_statistics = MutableStatistics::default();
            for game in &game_rows {
                if game.rows.len() != ROWS_PER_GAME {
                    return Err(failure(format!(
                        "game {} produced {} rows instead of {ROWS_PER_GAME}",
                        game.game_index,
                        game.rows.len()
                    )));
                }
                for row in &game.rows {
                    serde_json::to_writer(&mut writer, row)?;
                    writer.write_all(b"\n")?;
                }
                shard_statistics.merge(&game.statistics);
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
            let metadata = fs::metadata(&path)?;
            let last = first + u64::try_from(count - 1)?;
            let row_count = count * ROWS_PER_GAME;
            if shard_statistics.rows != row_count {
                return Err(failure(format!(
                    "shard {shard_index} accumulated {} rows instead of {row_count}",
                    shard_statistics.rows
                )));
            }
            aggregate.merge(&shard_statistics);
            shards.push(ShardManifest {
                file,
                row_count,
                bytes: metadata.len(),
                blake3: checksum_file(&path)?,
                first_game_index: first,
                games: count,
                first_game_seed: game_seed(first),
                last_game_seed: game_seed(last),
            });
            eprintln!(
                "legacy activation export: shard {}/{} complete ({} games, {} rows)",
                shard_index + 1,
                games.div_ceil(shard_games),
                count,
                row_count
            );
        }

        let statistics = aggregate.finish();
        let expected_rows = games
            .checked_mul(ROWS_PER_GAME)
            .ok_or_else(|| failure("row count overflowed usize"))?;
        if statistics.games != games || statistics.rows != expected_rows {
            return Err(failure("aggregate game or row count drifted"));
        }
        validate_expected_cohorts(&statistics, games)?;
        let payload_blake3 = payload_blake3(&shards);
        let source_after = capture_source_identity()?;
        if source_before.source_bundle_blake3 != source_after.source_bundle_blake3
            || source_before.files != source_after.files
        {
            return Err(failure(
                "relevant source files changed while the dataset was being generated",
            ));
        }
        let generation = generation_contract(games, shard_games, first_game_index);
        let provenance = Provenance {
            source: source_before,
            executable,
        };
        let mut manifest = Manifest {
            schema_version: 1,
            dataset_id: DATASET_ID.to_owned(),
            feature_schema: FEATURE_SCHEMA.to_owned(),
            feature_count: FEATURE_COUNT,
            split: "train".to_owned(),
            rows: expected_rows,
            shards,
            payload_blake3,
            generation,
            statistics,
            provenance,
            scientific_blake3: String::new(),
        };
        manifest.scientific_blake3 = manifest_scientific_blake3(&manifest)?;
        write_json_atomic(&temporary.path.join("manifest.json"), &manifest)?;
        let report = validate_dataset(&temporary.path, false)?;
        fs::rename(&temporary.path, output)?;
        temporary.published = true;
        Ok(report)
    }

    fn validate_expected_cohorts(statistics: &DatasetStatistics, games: usize) -> Result<()> {
        let per_seat = u64::try_from(games * TURNS_PER_PLAYER)?;
        for seat in 0..PLAYERS {
            if statistics.rows_by_focal_seat.get(&seat.to_string()) != Some(&per_seat) {
                return Err(failure(format!("focal seat {seat} row count drifted")));
            }
        }
        let game_count = u64::try_from(games)?;
        let expected_phase = [
            ("opening", 4 * game_count),
            ("early", 16 * game_count),
            ("middle", 32 * game_count),
            ("late", 28 * game_count),
        ];
        for (phase, rows) in expected_phase {
            if statistics.rows_by_phase.get(phase) != Some(&rows) {
                return Err(failure(format!("phase {phase} row count drifted")));
            }
        }
        let total_policy_rows = statistics.rows_by_policy.values().sum::<u64>();
        if total_policy_rows != u64::try_from(statistics.rows)? {
            return Err(failure("policy row counts do not sum to total rows"));
        }
        Ok(())
    }

    fn validate_dataset(root: &Path, content_only: bool) -> Result<ValidationReport> {
        validate_compiled_feature_contract()?;
        let manifest_path = root.join("manifest.json");
        let manifest: Manifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        if manifest.schema_version != 1
            || manifest.dataset_id != DATASET_ID
            || manifest.feature_schema != FEATURE_SCHEMA
            || manifest.feature_count != FEATURE_COUNT
            || manifest.split != "train"
        {
            return Err(failure("legacy activation manifest contract drifted"));
        }
        if manifest.generation.players != PLAYERS
            || manifest.generation.scoring_cards != "AAAAA"
            || manifest.generation.score_outputs_collected
            || manifest.generation.habitat_bonus_labels_collected
            || manifest.generation.paid_wildlife_wipes
            || manifest.generation.rows_per_game != ROWS_PER_GAME
            || manifest.generation.feature_count != FEATURE_COUNT
        {
            return Err(failure("legacy activation generation contract drifted"));
        }
        if manifest.shards.is_empty()
            || manifest
                .shards
                .iter()
                .map(|shard| shard.row_count)
                .sum::<usize>()
                != manifest.rows
            || manifest.statistics.rows != manifest.rows
            || manifest.statistics.games != manifest.generation.seed_schedule.games
        {
            return Err(failure("legacy activation aggregate counts drifted"));
        }
        if manifest.payload_blake3 != payload_blake3(&manifest.shards) {
            return Err(failure("legacy activation payload identity drifted"));
        }
        if manifest.scientific_blake3 != manifest_scientific_blake3(&manifest)? {
            return Err(failure("legacy activation scientific identity drifted"));
        }
        let mut expected_game_index = manifest.generation.seed_schedule.first_game_index;
        let mut aggregate = MutableStatistics::default();
        let mut declared_files = Vec::new();
        for shard in &manifest.shards {
            declared_files.push(shard.file.clone());
            if shard.first_game_index != expected_game_index || shard.games == 0 {
                return Err(failure(
                    "legacy activation shard game ranges are not contiguous",
                ));
            }
            if shard.row_count != shard.games * ROWS_PER_GAME {
                return Err(failure("legacy activation shard row count drifted"));
            }
            let last_game_index = shard.first_game_index + u64::try_from(shard.games - 1)?;
            if shard.first_game_seed != game_seed(shard.first_game_index)
                || shard.last_game_seed != game_seed(last_game_index)
            {
                return Err(failure("legacy activation shard seed range drifted"));
            }
            let path = root.join(&shard.file);
            let metadata = fs::metadata(&path)?;
            if metadata.len() != shard.bytes || checksum_file(&path)? != shard.blake3 {
                return Err(failure(format!(
                    "legacy activation shard payload drifted: {}",
                    shard.file
                )));
            }
            let reader = BufReader::new(File::open(&path)?);
            let mut rows = 0usize;
            let mut shard_statistics = MutableStatistics::default();
            for line in reader.lines() {
                let row: ActivationRow = serde_json::from_str(&line?)?;
                let expected_game = shard.first_game_index + u64::try_from(rows / ROWS_PER_GAME)?;
                let expected_decision = rows % ROWS_PER_GAME;
                validate_row(&row, expected_game, expected_decision)?;
                shard_statistics.observe(&row);
                rows += 1;
            }
            if rows != shard.row_count {
                return Err(failure(format!(
                    "legacy activation shard {} has {rows} rows, expected {}",
                    shard.file, shard.row_count
                )));
            }
            shard_statistics.games = shard.games;
            aggregate.merge(&shard_statistics);
            expected_game_index = last_game_index + 1;
        }
        declared_files.sort();
        let mut actual_shards = fs::read_dir(root)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                (name.starts_with("part-") && name.ends_with(".jsonl")).then_some(name)
            })
            .collect::<Vec<_>>();
        actual_shards.sort();
        if actual_shards != declared_files {
            return Err(failure(
                "legacy activation payload shard set differs from manifest",
            ));
        }
        let observed_statistics = aggregate.finish();
        if observed_statistics != manifest.statistics {
            return Err(failure("legacy activation statistics drifted"));
        }
        validate_expected_cohorts(&observed_statistics, manifest.statistics.games)?;
        let current_source = capture_source_identity()?;
        let current_executable = capture_executable_identity()?;
        let relevant_source_matches = current_source.source_bundle_blake3
            == manifest.provenance.source.source_bundle_blake3
            && current_source.files == manifest.provenance.source.files;
        let executable_matches = current_executable.blake3 == manifest.provenance.executable.blake3;
        if !content_only && (!relevant_source_matches || !executable_matches) {
            return Err(failure(format!(
                "strict provenance validation failed: source_matches={relevant_source_matches}, \
                 executable_matches={executable_matches}"
            )));
        }
        Ok(ValidationReport {
            dataset_id: manifest.dataset_id,
            manifest_blake3: checksum_file(&manifest_path)?,
            scientific_blake3: manifest.scientific_blake3,
            payload_blake3: manifest.payload_blake3,
            rows: manifest.rows,
            shards: manifest.shards.len(),
            source_bundle_blake3: manifest.provenance.source.source_bundle_blake3,
            extractor_source_blake3: manifest.provenance.source.extractor_source_blake3,
            executable_blake3: manifest.provenance.executable.blake3,
            relevant_source_matches,
            executable_matches,
            content_only,
            status: "valid".to_owned(),
        })
    }

    fn manifest_scientific_blake3(manifest: &Manifest) -> Result<String> {
        let identity = ScientificIdentity {
            schema_version: manifest.schema_version,
            dataset_id: &manifest.dataset_id,
            feature_schema: &manifest.feature_schema,
            feature_count: manifest.feature_count,
            split: &manifest.split,
            rows: manifest.rows,
            shards: &manifest.shards,
            payload_blake3: &manifest.payload_blake3,
            generation: &manifest.generation,
            statistics: &manifest.statistics,
            source_bundle_blake3: &manifest.provenance.source.source_bundle_blake3,
            extractor_source_blake3: &manifest.provenance.source.extractor_source_blake3,
            executable_blake3: &manifest.provenance.executable.blake3,
        };
        Ok(blake3::hash(&serde_json::to_vec(&identity)?)
            .to_hex()
            .to_string())
    }

    fn payload_blake3(shards: &[ShardManifest]) -> String {
        let mut hasher = Hasher::new();
        hasher.update(b"legacy-mid-v4opp-activation-v1/payload/v1");
        for shard in shards {
            hasher.update(&(shard.file.len() as u64).to_le_bytes());
            hasher.update(shard.file.as_bytes());
            hasher.update(&(shard.row_count as u64).to_le_bytes());
            hasher.update(&shard.bytes.to_le_bytes());
            hasher.update(shard.blake3.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    fn capture_executable_identity() -> Result<ExecutableIdentity> {
        let path = std::env::current_exe()?;
        Ok(ExecutableIdentity {
            file_name: path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("legacy_activation_export")
                .to_owned(),
            bytes: fs::metadata(&path)?.len(),
            blake3: checksum_file(&path)?,
        })
    }

    fn capture_source_identity() -> Result<SourceIdentity> {
        let repository = repository_root()?;
        let mut paths = Vec::new();
        for relative in SOURCE_BUNDLE_ROOTS {
            collect_source_files(&repository.join(relative), &mut paths)?;
        }
        paths.sort();
        paths.dedup();
        let mut files = Vec::with_capacity(paths.len());
        let mut bundle = Hasher::new();
        bundle.update(b"legacy-mid-v4opp-activation-v1/source-bundle/v1");
        for path in paths {
            let relative = path
                .strip_prefix(&repository)?
                .to_string_lossy()
                .replace('\\', "/");
            let metadata = fs::metadata(&path)?;
            let digest = checksum_file(&path)?;
            bundle.update(&(relative.len() as u64).to_le_bytes());
            bundle.update(relative.as_bytes());
            bundle.update(&metadata.len().to_le_bytes());
            bundle.update(digest.as_bytes());
            files.push(SourceFileIdentity {
                file: relative,
                bytes: metadata.len(),
                blake3: digest,
            });
        }
        let extractor_source_blake3 = files
            .iter()
            .find(|file| file.file == "legacy/crates/cascadia-ai/src/nnue.rs")
            .map(|file| file.blake3.clone())
            .ok_or_else(|| failure("historical extractor source was not in source bundle"))?;
        let repository_snapshot = source_provenance()?;
        Ok(SourceIdentity {
            contract: (
                "path-length + relative-path + byte-length + file-BLAKE3, sorted, domain-separated"
            )
                .to_owned(),
            git_revision: repository_snapshot.git_revision,
            repository_v2_source_blake3: repository_snapshot.v2_source_blake3,
            source_bundle_blake3: bundle.finalize().to_hex().to_string(),
            extractor_source_blake3,
            files,
        })
    }

    fn collect_source_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_file() {
            files.push(path.to_owned());
            return Ok(());
        }
        if !path.is_dir() {
            return Err(failure(format!(
                "source bundle path does not exist: {}",
                path.display()
            )));
        }
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            collect_source_files(&entry.path(), files)?;
        }
        Ok(())
    }

    fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
        let temporary = path.with_extension("json.tmp");
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{
            io::Read,
            sync::atomic::{AtomicU64, Ordering},
            time::{SystemTime, UNIX_EPOCH},
        };

        static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

        fn temporary_path(label: &str) -> PathBuf {
            let epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            std::env::temp_dir().join(format!(
                "cascadia-{label}-{}-{epoch}-{}",
                std::process::id(),
                TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ))
        }

        #[test]
        fn frozen_phase_boundaries_match_adr_0129() {
            assert_eq!(phase_for_personal_turn(1).unwrap(), Phase::Opening);
            assert_eq!(phase_for_personal_turn(2).unwrap(), Phase::Early);
            assert_eq!(phase_for_personal_turn(5).unwrap(), Phase::Early);
            assert_eq!(phase_for_personal_turn(6).unwrap(), Phase::Middle);
            assert_eq!(phase_for_personal_turn(13).unwrap(), Phase::Middle);
            assert_eq!(phase_for_personal_turn(14).unwrap(), Phase::Late);
            assert_eq!(phase_for_personal_turn(20).unwrap(), Phase::Late);
            assert!(phase_for_personal_turn(0).is_err());
            assert!(phase_for_personal_turn(21).is_err());
        }

        #[test]
        fn seeds_and_policy_rotation_are_deterministic_and_balanced() {
            assert_eq!(game_seed(17), game_seed(17));
            assert_ne!(game_seed(17), game_seed(18));
            let mut counts = [[0usize; 4]; PLAYERS];
            for game_index in 0..2_500 {
                for (seat, row) in counts.iter_mut().enumerate() {
                    row[policy_for(game_index, seat).id() as usize] += 1;
                }
            }
            assert!(counts.iter().all(|row| *row == [625; 4]));
        }

        #[test]
        fn exact_extractor_rows_are_sorted_unique_and_in_range() {
            validate_compiled_feature_contract().unwrap();
            let game = generate_game(3).unwrap();
            assert_eq!(game.rows.len(), ROWS_PER_GAME);
            for (decision, row) in game.rows.iter().enumerate() {
                validate_row(row, 3, decision).unwrap();
                assert!(
                    row.features
                        .iter()
                        .all(|feature| usize::from(*feature) < FEATURE_COUNT)
                );
            }
        }

        #[test]
        fn generated_dataset_round_trips_through_strict_validator() {
            let root = temporary_path("legacy-activation-roundtrip");
            let report = generate_dataset(&root, 4, 2, 100, 2).unwrap();
            assert_eq!(report.rows, 4 * ROWS_PER_GAME);
            assert_eq!(report.shards, 2);
            assert!(report.relevant_source_matches);
            assert!(report.executable_matches);
            let repeated = validate_dataset(&root, false).unwrap();
            assert_eq!(report, repeated);
            let manifest: Manifest =
                serde_json::from_reader(File::open(root.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest.split, "train");
            assert_eq!(manifest.feature_count, FEATURE_COUNT);
            assert_eq!(manifest.statistics.rows_by_phase["opening"], 16);
            assert_eq!(manifest.statistics.rows_by_phase["early"], 64);
            assert_eq!(manifest.statistics.rows_by_phase["middle"], 128);
            assert_eq!(manifest.statistics.rows_by_phase["late"], 112);
            fs::remove_dir_all(root).unwrap();
        }

        #[test]
        fn validator_rejects_payload_corruption() {
            let root = temporary_path("legacy-activation-corruption");
            generate_dataset(&root, 1, 1, 900, 1).unwrap();
            let shard = root.join("part-00000.jsonl");
            let mut payload = Vec::new();
            File::open(&shard)
                .unwrap()
                .read_to_end(&mut payload)
                .unwrap();
            payload[0] ^= 1;
            File::create(&shard).unwrap().write_all(&payload).unwrap();
            assert!(validate_dataset(&root, true).is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[cfg(feature = "legacy-teacher")]
fn main() {
    if let Err(error) = enabled::run() {
        eprintln!("legacy-activation-export: {error}");
        std::process::exit(2);
    }
}
