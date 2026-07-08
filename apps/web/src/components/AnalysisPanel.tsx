import { BrainCircuit, ChevronDown, Play, RefreshCw } from "lucide-react";

import { actionSummary } from "../game";
import type { SuggestionCandidate } from "../types";

export function AnalysisPanel({
  open,
  loading,
  candidates,
  onToggle,
  onAnalyze,
  onPlay,
}: {
  open: boolean;
  loading: boolean;
  candidates: SuggestionCandidate[];
  onToggle: () => void;
  onAnalyze: () => void;
  onPlay: (candidate: SuggestionCandidate) => void;
}) {
  return (
    <section className={`analysis-panel ${open ? "is-open" : ""}`}>
      <button className="analysis-handle" type="button" onClick={onToggle}>
        <BrainCircuit aria-hidden="true" />
        <span>Move analysis</span>
        <small>
          {candidates.length > 0
            ? `${candidates.length} candidates`
            : "Research search"}
        </small>
        <ChevronDown aria-hidden="true" />
      </button>
      {open && (
        <div className="analysis-body">
          <div className="analysis-toolbar">
            <div>
              <span className="eyebrow">Confidence-gated search</span>
              <h3>Candidate moves</h3>
            </div>
            <button
              className="command-button"
              type="button"
              onClick={onAnalyze}
              disabled={loading}
            >
              <RefreshCw aria-hidden="true" />
              {loading ? "Evaluating" : "Refresh"}
            </button>
          </div>
          {candidates.length === 0 ? (
            <div className="analysis-empty">
              <BrainCircuit aria-hidden="true" />
              <span>No analysis loaded</span>
            </div>
          ) : (
            <div className="candidate-table">
              <div className="candidate-header">
                <span>Rank</span>
                <span>Draft</span>
                <span>Placement</span>
                <span>Delta</span>
                <span>Total</span>
                <span>Value</span>
                <span />
              </div>
              {candidates.map((candidate) => (
                <div className="candidate-row" key={candidate.rank}>
                  <strong>#{candidate.rank}</strong>
                  <span>{actionSummary(candidate.action.draft)}</span>
                  <span>
                    q {candidate.action.tile.coord.q}, r {candidate.action.tile.coord.r}
                  </span>
                  <b className={candidate.base_score_delta >= 0 ? "positive" : "negative"}>
                    {candidate.base_score_delta >= 0 ? "+" : ""}
                    {candidate.base_score_delta}
                  </b>
                  <strong>{candidate.resulting_score.base_total}</strong>
                  <strong
                    title={`Estimated value; uncertainty ${candidate.evaluation_stddev.toFixed(2)}`}
                  >
                    {candidate.search_value.toFixed(1)}
                  </strong>
                  <button
                    className="icon-button"
                    type="button"
                    title={`Play candidate ${candidate.rank}`}
                    onClick={() => onPlay(candidate)}
                  >
                    <Play aria-hidden="true" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
