use cascadia_data::PositionRecord;
use cascadia_game::{GameState, MarketPrelude};
use cascadia_model::MAX_BATCH;
use cascadia_sim::GreedyCandidate;

use super::{RankingPredictor, SearchError, with_prelude};

pub(crate) fn predict_ranking_scores<P: RankingPredictor>(
    predictor: &mut P,
    game: &GameState,
    prelude: &MarketPrelude,
    candidates: &[GreedyCandidate],
) -> Result<Vec<f32>, SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let records = candidates
        .iter()
        .map(|candidate| {
            PositionRecord::observable_afterstate(
                game,
                &with_prelude(candidate.action.clone(), prelude),
                0,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut scores = Vec::with_capacity(records.len());
    for chunk in records.chunks(MAX_BATCH) {
        scores.extend(predictor.predict_scores(chunk)?);
    }
    if scores.len() != records.len() {
        return Err(SearchError::PredictionCount {
            expected: records.len(),
            actual: scores.len(),
        });
    }
    if let Some((index, _)) = scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(SearchError::NonFinitePrediction { index });
    }
    Ok(scores)
}
