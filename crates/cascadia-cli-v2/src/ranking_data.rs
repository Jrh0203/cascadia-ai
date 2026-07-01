use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

use cascadia_data::{
    ActionPositionRecord, ActionRankingDatasetConfig, ActionRankingDatasetManifest,
    ActionRankingDatasetWriter, ActionRankingRecord, ConservativeAdvantageRecord, DatasetSplit,
    PositionRecord, RankingCandidateFamily, RankingDatasetManifest, RankingRecord,
    read_ranking_shard_records, validate_ranking_dataset,
};
use cascadia_game::{GameConfig, GameState, MarketPrelude};
use cascadia_search::{
    BearCandidateLookaheadStrategy, HabitatCandidateLookaheadStrategy,
    LateConservativeBasePolicyImprovementStrategy, MlxHabitatRankingStrategy, RolloutCandidate,
    SearchError, TerminalPolicyImprovementConfig, TerminalPolicyImprovementStrategy,
};
use cascadia_sim::{
    PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, rank_pattern_frontier_actions,
    select_pattern_action, strategy_rng,
};

pub fn collect_ranking_game(
    teacher: &RankingTeacherStrategy,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<RankingRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut records = Vec::new();
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let (ranked, action) = teacher.rank_and_select_deterministic(&game)?;
        let candidate_count = u16::try_from(ranked.len())?;
        let group_id = ranking_group_id(split, game_index, turn, active_seat);
        for (candidate_index, candidate) in ranked.iter().enumerate() {
            let serialized_action = serde_json::to_vec(&candidate.action)?;
            records.push(RankingRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                immediate_rank: u16::try_from(candidate.immediate_rank)?,
                immediate_score: candidate.immediate_score,
                teacher_mean: candidate.mean_leaf_score as f32,
                teacher_stddev: candidate.leaf_score_stddev as f32,
                action_hash: *blake3::hash(&serialized_action).as_bytes(),
                position: PositionRecord::observable_afterstate(
                    &game,
                    &candidate.action,
                    game_index,
                )?,
            });
        }
        game.apply(&action)?;
    }
    Ok(records)
}

pub fn collect_terminal_ranking_game(
    teacher: &TerminalPolicyImprovementStrategy,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<RankingRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut records = Vec::new();
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let (ranked, action) = teacher.rank_and_select_deterministic(&game)?;
        let candidate_count = u16::try_from(ranked.len())?;
        let group_id = ranking_group_id(split, game_index, turn, active_seat);
        for (candidate_index, candidate) in ranked.iter().enumerate() {
            let serialized_action = serde_json::to_vec(&candidate.action)?;
            records.push(RankingRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                immediate_rank: u16::try_from(candidate.immediate_rank)?,
                immediate_score: candidate.immediate_score,
                teacher_mean: candidate.mean_leaf_score as f32,
                teacher_stddev: candidate.leaf_score_stddev as f32,
                action_hash: *blake3::hash(&serialized_action).as_bytes(),
                position: PositionRecord::observable_afterstate(
                    &game,
                    &candidate.action,
                    game_index,
                )?,
            });
        }
        game.apply(&action)?;
    }
    Ok(records)
}

pub fn collect_conservative_advantage_game(
    teacher: &LateConservativeBasePolicyImprovementStrategy,
    blueprint: PatternAwareConfig,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<ConservativeAdvantageRecord>, Box<dyn std::error::Error>> {
    let seed = split.game_seed(game_index);
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, seed)?;
    let mut blueprint_rngs = (0..usize::from(game.config().player_count))
        .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let anchor =
            select_pattern_action(&game, &prelude, blueprint, &mut blueprint_rngs[active_seat])?;
        let action = if teacher.uses_terminal_search(&game) {
            let (evaluations, selected) =
                teacher.evaluate_and_select_terminal_deterministic(&game, &anchor)?;
            let anchor_evaluation = evaluations
                .iter()
                .find(|evaluation| evaluation.is_anchor)
                .ok_or_else(|| std::io::Error::other("strong evaluation omitted its anchor"))?;
            if anchor_evaluation.action != anchor {
                return Err(
                    std::io::Error::other("strong evaluation anchor action drifted").into(),
                );
            }
            let anchor_input = ActionPositionRecord::observe(
                &game,
                &anchor,
                game_index,
                u16::try_from(anchor_evaluation.immediate_rank)?,
                anchor_evaluation.immediate_score,
            )?;
            let anchor_hash = *blake3::hash(&serde_json::to_vec(&anchor)?).as_bytes();
            let candidate_count = u16::try_from(
                evaluations
                    .iter()
                    .filter(|evaluation| !evaluation.is_anchor)
                    .count(),
            )?;
            let group_id = ranking_group_id(split, game_index, turn, active_seat);
            for (candidate_index, evaluation) in evaluations
                .iter()
                .filter(|evaluation| !evaluation.is_anchor)
                .enumerate()
            {
                let candidate_hash =
                    *blake3::hash(&serde_json::to_vec(&evaluation.action)?).as_bytes();
                records.push(ConservativeAdvantageRecord {
                    group_id,
                    candidate_index: u16::try_from(candidate_index)?,
                    candidate_count,
                    selected: evaluation.action == selected,
                    mean_advantage: evaluation.mean_advantage as f32,
                    advantage_standard_error: evaluation.advantage_standard_error as f32,
                    lower_bound: evaluation.lower_bound as f32,
                    anchor_hash,
                    candidate_hash,
                    anchor: anchor_input.clone(),
                    candidate: ActionPositionRecord::observe(
                        &game,
                        &evaluation.action,
                        game_index,
                        u16::try_from(evaluation.immediate_rank)?,
                        evaluation.immediate_score,
                    )?,
                });
            }
            selected
        } else {
            anchor
        };
        game.apply(&action)?;
    }
    if game.completed_turns() != 80 {
        return Err(
            std::io::Error::other("strong advantage trajectory ended before 80 decisions").into(),
        );
    }
    Ok(records)
}

pub fn enrich_action_ranking_dataset(
    source_root: &Path,
    output: PathBuf,
    resume: bool,
    policy_market_draws: usize,
) -> Result<ActionRankingDatasetManifest, Box<dyn std::error::Error>> {
    let source_manifest: RankingDatasetManifest =
        serde_json::from_reader(std::fs::File::open(source_root.join("dataset.json"))?)?;
    validate_ranking_dataset(source_root, &source_manifest)?;
    if source_manifest.teacher.candidate_family != RankingCandidateFamily::Pattern {
        return Err(std::io::Error::other(
            "action-delta enrichment requires a terminal pattern-ranking source dataset",
        )
        .into());
    }

    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: source_manifest.teacher.immediate_candidates,
        habitat_candidate_limit: source_manifest.teacher.habitat_candidates,
        bear_candidate_limit: source_manifest.teacher.bear_candidates,
        future_market_draws: policy_market_draws,
    };
    let continuation_id = blueprint.strategy_id();
    if source_manifest
        .teacher
        .terminal_continuation_strategy_id
        .as_deref()
        != Some(continuation_id.as_str())
    {
        return Err(std::io::Error::other(
            "source terminal continuation metadata does not match the reconstructed pattern policy",
        )
        .into());
    }
    let teacher = TerminalPolicyImprovementStrategy::new(TerminalPolicyImprovementConfig {
        determinizations: source_manifest.teacher.determinizations,
        blueprint,
    })?;
    if teacher.strategy_id() != source_manifest.teacher.strategy_id {
        return Err(std::io::Error::other(format!(
            "source teacher {} does not match reconstructed {}",
            source_manifest.teacher.strategy_id,
            teacher.strategy_id(),
        ))
        .into());
    }

    let mut writer = ActionRankingDatasetWriter::open(&ActionRankingDatasetConfig {
        output,
        source_root: source_root.to_owned(),
        source_manifest: source_manifest.clone(),
        resume,
    })?;
    let mut next_game_index =
        source_manifest.first_game_index + writer.manifest().completed_games as u64;
    for shard in &source_manifest.shards {
        let shard_end = shard.first_game_index + shard.game_count as u64;
        if shard_end <= next_game_index {
            continue;
        }
        if shard.first_game_index != next_game_index {
            return Err(std::io::Error::other(
                "action-ranking resume point does not align with a source shard",
            )
            .into());
        }
        let records = read_ranking_shard_records(source_root, source_manifest.split, shard)?;
        let mut by_game = BTreeMap::<u64, Vec<RankingRecord>>::new();
        for record in records {
            by_game
                .entry(record.position.game_index)
                .or_default()
                .push(record);
        }
        if by_game.len() != shard.game_count
            || by_game.keys().copied().next() != Some(shard.first_game_index)
        {
            return Err(std::io::Error::other(
                "source ranking shard does not contain its declared contiguous games",
            )
            .into());
        }
        let mut enriched = Vec::with_capacity(shard.record_count);
        for (offset, (game_index, records)) in by_game.into_iter().enumerate() {
            if game_index != shard.first_game_index + offset as u64 {
                return Err(std::io::Error::other(
                    "source ranking shard game indices are not contiguous",
                )
                .into());
            }
            enriched.extend(enrich_action_ranking_game(
                &teacher,
                blueprint,
                source_manifest.split,
                game_index,
                &records,
            )?);
        }
        if enriched.len() != shard.record_count {
            return Err(std::io::Error::other(
                "action-ranking enrichment changed the source candidate count",
            )
            .into());
        }
        writer.append_shard(shard.first_game_index, shard.game_count, &enriched)?;
        next_game_index = shard_end;
        eprintln!(
            "action ranking dataset: {}/{} games, {} groups, {} candidates",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().total_groups,
            writer.manifest().total_records,
        );
    }
    if writer.manifest().completed_games != source_manifest.completed_games {
        return Err(std::io::Error::other(
            "action-ranking enrichment did not consume every completed source game",
        )
        .into());
    }
    Ok(writer.manifest().clone())
}

fn enrich_action_ranking_game(
    teacher: &TerminalPolicyImprovementStrategy,
    blueprint: PatternAwareConfig,
    split: DatasetSplit,
    game_index: u64,
    source_records: &[RankingRecord],
) -> Result<Vec<ActionRankingRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut enriched = Vec::with_capacity(source_records.len());
    let mut start = 0;
    while start < source_records.len() {
        let group_id = source_records[start].group_id;
        let end = source_records[start..]
            .iter()
            .position(|record| record.group_id != group_id)
            .map_or(source_records.len(), |relative| start + relative);
        let group = &source_records[start..end];
        let expected_group_id = ranking_group_id(
            split,
            game_index,
            game.completed_turns(),
            game.current_player(),
        );
        if group_id != expected_group_id
            || group.len() != usize::from(group[0].candidate_count)
            || group.iter().enumerate().any(|(index, record)| {
                record.position.game_index != game_index
                    || record.position.turn != game.completed_turns() as u8 + 1
                    || usize::from(record.position.active_seat) != game.current_player()
                    || usize::from(record.candidate_index) != index
                    || usize::from(record.candidate_count) != group.len()
            })
        {
            return Err(std::io::Error::other(
                "source ranking group does not match deterministic replay state",
            )
            .into());
        }

        let prelude = cascadia_game::MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let frontier = rank_pattern_frontier_actions(
            &staged,
            &cascadia_game::MarketPrelude::default(),
            blueprint,
        )?;
        let mut actions_by_hash = HashMap::with_capacity(frontier.len());
        for candidate in frontier {
            let mut action = candidate.action;
            action.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
            action.wildlife_wipes.clone_from(&prelude.wildlife_wipes);
            let action_hash = *blake3::hash(&serde_json::to_vec(&action)?).as_bytes();
            if actions_by_hash
                .insert(
                    action_hash,
                    (
                        action,
                        candidate.immediate_rank,
                        candidate.resulting_base_score,
                    ),
                )
                .is_some()
            {
                return Err(std::io::Error::other(
                    "reconstructed pattern frontier contains duplicate action hashes",
                )
                .into());
            }
        }
        if actions_by_hash.len() != group.len() {
            return Err(std::io::Error::other(format!(
                "reconstructed frontier has {} candidates, source group has {}",
                actions_by_hash.len(),
                group.len(),
            ))
            .into());
        }

        let mut recorded_ranking = Vec::with_capacity(group.len());
        for record in group {
            let (action, immediate_rank, immediate_score) =
                actions_by_hash.remove(&record.action_hash).ok_or_else(|| {
                    std::io::Error::other(
                        "source candidate action hash is absent from reconstructed frontier",
                    )
                })?;
            if usize::from(record.immediate_rank) != immediate_rank
                || record.immediate_score != immediate_score
            {
                return Err(std::io::Error::other(
                    "source candidate immediate metadata does not match replay",
                )
                .into());
            }
            let input = ActionPositionRecord::observe(
                &game,
                &action,
                game_index,
                record.immediate_rank,
                record.immediate_score,
            )?;
            if input.position != record.position {
                return Err(std::io::Error::other(
                    "reconstructed candidate afterstate does not match source bytes",
                )
                .into());
            }
            recorded_ranking.push(RolloutCandidate {
                action: action.clone(),
                immediate_rank,
                immediate_score,
                mean_leaf_score: f64::from(record.teacher_mean),
                leaf_score_stddev: f64::from(record.teacher_stddev),
            });
            enriched.push(ActionRankingRecord {
                group_id: record.group_id,
                candidate_index: record.candidate_index,
                candidate_count: record.candidate_count,
                immediate_rank: record.immediate_rank,
                immediate_score: record.immediate_score,
                teacher_mean: record.teacher_mean,
                teacher_stddev: record.teacher_stddev,
                action_hash: record.action_hash,
                input,
            });
        }
        if !actions_by_hash.is_empty() {
            return Err(std::io::Error::other(
                "reconstructed frontier contains actions absent from the source group",
            )
            .into());
        }
        let selected =
            teacher.select_from_recorded_ranking_deterministic(&game, &recorded_ranking)?;
        game.apply(&selected)?;
        start = end;
    }
    if !game.is_game_over() || game.completed_turns() != 80 {
        return Err(std::io::Error::other(
            "action-ranking source game ended before all 80 decisions were replayed",
        )
        .into());
    }
    Ok(enriched)
}

pub fn collect_ranking_iteration_game(
    teacher: &HabitatCandidateLookaheadStrategy,
    apprentice: &mut MlxHabitatRankingStrategy,
    split: DatasetSplit,
    game_index: u64,
) -> Result<Vec<RankingRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let mut records = Vec::new();
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let (ranked, _) = teacher.rank_and_select_deterministic(&game)?;
        let action = apprentice.select_from_teacher_candidates(&game, &ranked, game_index)?;
        let candidate_count = u16::try_from(ranked.len())?;
        let group_id = ranking_group_id(split, game_index, turn, active_seat);
        for (candidate_index, candidate) in ranked.iter().enumerate() {
            let serialized_action = serde_json::to_vec(&candidate.action)?;
            records.push(RankingRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                immediate_rank: u16::try_from(candidate.immediate_rank)?,
                immediate_score: candidate.immediate_score,
                teacher_mean: candidate.mean_leaf_score as f32,
                teacher_stddev: candidate.leaf_score_stddev as f32,
                action_hash: *blake3::hash(&serialized_action).as_bytes(),
                position: PositionRecord::observable_afterstate(
                    &game,
                    &candidate.action,
                    game_index,
                )?,
            });
        }
        game.apply(&action)?;
    }
    Ok(records)
}

pub enum RankingTeacherStrategy {
    Bear(BearCandidateLookaheadStrategy),
    Habitat(HabitatCandidateLookaheadStrategy),
}

impl RankingTeacherStrategy {
    pub fn strategy_id(&self) -> &str {
        match self {
            Self::Bear(strategy) => strategy.strategy_id(),
            Self::Habitat(strategy) => strategy.strategy_id(),
        }
    }

    fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, cascadia_game::TurnAction), SearchError> {
        match self {
            Self::Bear(strategy) => strategy.rank_and_select_deterministic(game),
            Self::Habitat(strategy) => strategy.rank_and_select_deterministic(game),
        }
    }
}

pub(crate) fn ranking_group_id(
    split: DatasetSplit,
    game_index: u64,
    turn: u16,
    active_seat: usize,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-ranking-group");
    hasher.update(split.id().as_bytes());
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    hasher.update(&(active_seat as u64).to_le_bytes());
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 output contains eight bytes"),
    )
}
