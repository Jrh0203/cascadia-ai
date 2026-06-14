import { Leaf, Mountain, Trees, Waves, Wind } from "lucide-react";

import { TERRAINS, WILDLIFE, titleCase } from "../game";
import type { BoardView, ScoringCards } from "../types";
import { WildlifeMark } from "./WildlifeMark";

const TERRAIN_ICONS = [Mountain, Trees, Wind, Leaf, Waves];

export function ScorePanel({
  boards,
  selectedPlayer,
  currentPlayer,
  cards,
  onSelectPlayer,
}: {
  boards: BoardView[];
  selectedPlayer: number;
  currentPlayer: number;
  cards: ScoringCards;
  onSelectPlayer: (player: number) => void;
}) {
  const board = boards[selectedPlayer];
  return (
    <aside className="score-panel" aria-label="Scores and scoring cards">
      <div className="player-tabs" role="tablist" aria-label="Player boards">
        {boards.map((candidate) => (
          <button
            key={candidate.player}
            type="button"
            role="tab"
            aria-selected={candidate.player === selectedPlayer}
            className={`${candidate.player === selectedPlayer ? "is-active" : ""} ${
              candidate.player === currentPlayer ? "is-current" : ""
            }`}
            onClick={() => onSelectPlayer(candidate.player)}
          >
            <span>P{candidate.player + 1}</span>
            <strong>{candidate.score.total}</strong>
          </button>
        ))}
      </div>

      <section className="score-section">
        <div className="section-heading">
          <span>Player {selectedPlayer + 1}</span>
          <strong>{board.score.total}</strong>
        </div>
        <div className="score-rows">
          {TERRAINS.map((terrain, index) => {
            const Icon = TERRAIN_ICONS[index];
            return (
              <div key={terrain}>
                <Icon className={`terrain-icon terrain-${terrain}-ink`} aria-hidden="true" />
                <span>{titleCase(terrain)}</span>
                <strong>{board.score.habitat[index]}</strong>
                {board.score.habitat_bonus[index] > 0 && (
                  <small>+{board.score.habitat_bonus[index]}</small>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="score-section wildlife-score-section">
        <div className="section-heading">
          <span>Wildlife cards</span>
          <strong>{board.score.wildlife.reduce((sum, score) => sum + score, 0)}</strong>
        </div>
        <div className="score-rows">
          {WILDLIFE.map((wildlife, index) => (
            <div key={wildlife}>
              <WildlifeMark wildlife={wildlife} size="small" />
              <span>{titleCase(wildlife)}</span>
              <b>{cards[wildlife]}</b>
              <strong>{board.score.wildlife[index]}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="token-summary">
        <span>Nature tokens</span>
        <strong>{board.nature_tokens}</strong>
      </section>
    </aside>
  );
}
