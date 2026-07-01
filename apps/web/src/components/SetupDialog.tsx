import { Dices, X } from "lucide-react";
import { useEffect, useState } from "react";

import { DEFAULT_CARDS, WILDLIFE, defaultSeats, randomSeed, titleCase } from "../game";
import type {
  Capabilities,
  GameConfig,
  ScoringCards,
  SeatConfig,
} from "../types";

export interface SetupResult {
  seed: number;
  config: GameConfig;
  seats: SeatConfig[];
}

export function SetupDialog({
  open,
  capabilities,
  initialConfig,
  initialSeed,
  initialSeats,
  onClose,
  onCreate,
}: {
  open: boolean;
  capabilities: Capabilities | null;
  initialConfig: GameConfig | null;
  initialSeed: number;
  initialSeats: SeatConfig[];
  onClose: () => void;
  onCreate: (result: SetupResult) => void;
}) {
  const [playerCount, setPlayerCount] = useState(initialConfig?.player_count ?? 4);
  const [seed, setSeed] = useState(initialSeed);
  const [cards, setCards] = useState<ScoringCards>(
    initialConfig?.scoring_cards ?? DEFAULT_CARDS,
  );
  const [habitatBonuses, setHabitatBonuses] = useState(
    initialConfig?.habitat_bonuses ?? false,
  );
  const [seats, setSeats] = useState<SeatConfig[]>(
    initialSeats.length > 0 ? initialSeats : defaultSeats(playerCount),
  );

  useEffect(() => {
    setSeats((current) =>
      Array.from(
        { length: playerCount },
        (_, index) => current[index] ?? defaultSeats(playerCount)[index],
      ),
    );
  }, [playerCount]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className="setup-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="setup-title"
      >
        <div className="dialog-heading">
          <div>
            <span className="eyebrow">New local game</span>
            <h2 id="setup-title">Table setup</h2>
          </div>
          <button
            className="icon-button quiet"
            type="button"
            title="Close setup"
            onClick={onClose}
          >
            <X aria-hidden="true" />
          </button>
        </div>

        <div className="setup-grid">
          <label>
            Players
            <div className="segmented-control">
              {[1, 2, 3, 4].map((count) => (
                <button
                  key={count}
                  type="button"
                  className={count === playerCount ? "is-active" : ""}
                  onClick={() => setPlayerCount(count)}
                >
                  {count}
                </button>
              ))}
            </div>
          </label>
          <label>
            Deterministic seed
            <div className="seed-control">
              <input
                type="number"
                min="0"
                max="4294967295"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
              <button
                className="icon-button"
                type="button"
                title="Generate seed"
                onClick={() => setSeed(randomSeed())}
              >
                <Dices aria-hidden="true" />
              </button>
            </div>
          </label>
        </div>

        <div className="seat-editor">
          <span className="field-label">Seats</span>
          {seats.map((seat, player) => (
            <div className="seat-row" key={player}>
              <strong>P{player + 1}</strong>
              <div className="segmented-control">
                <button
                  type="button"
                  className={seat.kind === "human" ? "is-active" : ""}
                  onClick={() =>
                    setSeats((current) =>
                      current.map((candidate, index) =>
                        index === player ? { ...candidate, kind: "human" } : candidate,
                      ),
                    )
                  }
                >
                  Human
                </button>
                <button
                  type="button"
                  className={seat.kind === "ai" ? "is-active" : ""}
                  onClick={() =>
                    setSeats((current) =>
                      current.map((candidate, index) =>
                        index === player ? { ...candidate, kind: "ai" } : candidate,
                      ),
                    )
                  }
                >
                  AI
                </button>
              </div>
              <select
                value={seat.strength}
                disabled={seat.kind === "human"}
                onChange={(event) =>
                  setSeats((current) =>
                    current.map((candidate, index) =>
                      index === player
                        ? {
                            ...candidate,
                            strength: event.target.value as SeatConfig["strength"],
                          }
                        : candidate,
                    ),
                  )
                }
              >
                {capabilities?.strengths.map((strength) => (
                  <option
                    key={strength.id}
                    value={strength.id}
                    disabled={!strength.available}
                  >
                    {strength.label}
                    {!strength.available ? " (unavailable)" : ""}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>

        <div className="scoring-editor">
          <span className="field-label">Wildlife scoring cards</span>
          <div className="scoring-card-grid">
            {WILDLIFE.map((wildlife) => (
              <label key={wildlife}>
                {titleCase(wildlife)}
                <select
                  value={cards[wildlife]}
                  onChange={(event) =>
                    setCards((current) => ({
                      ...current,
                      [wildlife]: event.target.value,
                    }))
                  }
                >
                  {["A", "B", "C", "D"].map((variant) => (
                    <option key={variant} value={variant}>
                      Card {variant}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </div>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={habitatBonuses}
            onChange={(event) => setHabitatBonuses(event.target.checked)}
          />
          <span>Habitat majority bonuses</span>
        </label>

        <div className="dialog-actions">
          <button className="secondary-command" type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="primary-command"
            type="button"
            onClick={() =>
              onCreate({
                seed,
                seats,
                config: {
                  player_count: playerCount,
                  mode: playerCount === 1 ? "Solo" : "Standard",
                  scoring_cards: cards,
                  habitat_bonuses: habitatBonuses,
                },
              })
            }
          >
            Start game
          </button>
        </div>
      </section>
    </div>
  );
}
