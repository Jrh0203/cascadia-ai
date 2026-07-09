import {
  LoaderCircle,
  X,
} from "lucide-react";
import {
  type ChangeEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { api } from "./api";
import { HexBoard } from "./components/HexBoard";
import { EventLog } from "./components/EventLog";
import { MarketRail } from "./components/MarketRail";
import { WildlifeMark } from "./components/WildlifeMark";
import { ScorePanel } from "./components/ScorePanel";
import { SetupDialog, type SetupResult } from "./components/SetupDialog";
import {
  DEFAULT_CARDS,
  EMPTY_PRELUDE,
  actionSummary,
  defaultSeats,
  independentDraft,
  pairedDraft,
  sameCoord,
  titleCase,
} from "./game";
import type {
  StrengthCapability,
  Capabilities,
  DraftChoice,
  GameConfig,
  GameDocument,
  HexCoord,
  MarketPrelude,
  PlacementOptions,
  SavedGame,
  SeatConfig,
  SuggestionCandidate,
  TurnOptions,
  Wildlife,
} from "./types";

const GAME_STORAGE_KEY = "cascadia-v2.saved-game";
const SEATS_STORAGE_KEY = "cascadia-v2.seats";

const DEFAULT_CONFIG: GameConfig = {
  player_count: 4,
  mode: "Standard",
  scoring_cards: DEFAULT_CARDS,
  habitat_bonuses: false,
};

type MobilePanel = "board" | "market" | "scores" | "analysis";

export default function App() {
  const [document, setDocument] = useState<GameDocument | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [seats, setSeats] = useState<SeatConfig[]>(defaultSeats(4));
  const [selectedPlayer, setSelectedPlayer] = useState(0);
  const [prelude, setPrelude] = useState<MarketPrelude>(EMPTY_PRELUDE);
  const [turnOptions, setTurnOptions] = useState<TurnOptions | null>(null);
  const [independent, setIndependent] = useState(false);
  const [selectedTileSlot, setSelectedTileSlot] = useState<number | null>(null);
  const [selectedWildlifeSlot, setSelectedWildlifeSlot] = useState<number | null>(
    null,
  );
  const [draft, setDraft] = useState<DraftChoice | null>(null);
  const [placements, setPlacements] = useState<PlacementOptions | null>(null);
  const [selectedCoord, setSelectedCoord] = useState<HexCoord | null>(null);
  const [selectedRotation, setSelectedRotation] = useState(0);
  const [selectedWildlife, setSelectedWildlife] = useState<HexCoord | null>(null);
  const [wildlifeDecision, setWildlifeDecision] = useState<
    "pending" | "placed" | "skipped"
  >("pending");
  const [redoStack, setRedoStack] = useState<SavedGame[]>([]);
  const [setupOpen, setSetupOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [candidates, setCandidates] = useState<SuggestionCandidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("board");
  const [hintStrength, setHintStrength] = useState<StrengthCapability["id"]>("research");
  const [selectedCandidateIdx, setSelectedCandidateIdx] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const loadInput = useRef<HTMLInputElement>(null);
  const aiState = useRef<string | null>(null);
  const autoState = useRef<string | null>(null);
  const turnFailures = useRef(new Map<string, number>());
  const [aiRetryTick, setAiRetryTick] = useState(0);

  const view = document?.view ?? null;
  const savedGame = document?.game ?? null;
  const stateHash = view?.state_hash;
  const gameOver = view?.game_over ?? false;
  const currentPlayer = view?.current_player ?? 0;
  const currentSeat = seats[currentPlayer];
  const isHumanTurn = currentSeat?.kind !== "ai";
  // Legacy v1 three-phase turn flow: pick a market pair, place the tile
  // (R rotates), then place or skip the wildlife token. Wildlife/skip
  // clicks apply the move immediately — no separate confirm step.
  const phase: "select_market" | "place_tile" | "place_wildlife" =
    draft === null
      ? "select_market"
      : selectedCoord === null
        ? "place_tile"
        : "place_wildlife";
  // Active suggestion preview (legacy v1 "suggest glow"): the selected
  // candidate highlights its market pair (or split halves) and target hex.
  const suggestion = candidates[selectedCandidateIdx] ?? null;
  const suggestionDraft = suggestion?.action.draft ?? null;
  const suggestionTileSlot = suggestionDraft
    ? "Paired" in suggestionDraft
      ? suggestionDraft.Paired.slot
      : suggestionDraft.Independent.tile_slot
    : null;
  const suggestionWildlifeSlot = suggestionDraft
    ? "Paired" in suggestionDraft
      ? suggestionDraft.Paired.slot
      : suggestionDraft.Independent.wildlife_slot
    : null;
  const suggestionSplit =
    suggestionTileSlot !== null && suggestionTileSlot !== suggestionWildlifeSlot;
  const selectedBoard = view?.boards[selectedPlayer] ?? null;
  const stagedMarket = turnOptions?.market ?? view?.market ?? [];
  const selectedPlacement = useMemo(
    () =>
      placements?.placements.find((option) =>
        sameCoord(option.coord, selectedCoord),
      ) ?? null,
    [placements, selectedCoord],
  );
  const selectedRotationOption = selectedPlacement?.rotations.find(
    (option) => option.rotation === selectedRotation,
  );
  const draftedTile =
    selectedTileSlot === null ? null : stagedMarket[selectedTileSlot]?.tile ?? null;
  const draftedWildlife =
    selectedWildlifeSlot === null
      ? null
      : stagedMarket[selectedWildlifeSlot]?.wildlife ?? null;
  const preview =
    selectedCoord && draftedTile
      ? {
          coord: selectedCoord,
          tile: draftedTile,
          rotation: selectedRotation,
          wildlife:
            selectedWildlife &&
            sameCoord(selectedCoord, selectedWildlife) &&
            draftedWildlife
              ? draftedWildlife
              : null,
        }
      : null;

  const resetTurnSelection = useCallback(() => {
    setPrelude(EMPTY_PRELUDE);
    setTurnOptions(null);
    setIndependent(false);
    setSelectedTileSlot(null);
    setSelectedWildlifeSlot(null);
    setDraft(null);
    setPlacements(null);
    setSelectedCoord(null);
    setSelectedRotation(0);
    setSelectedWildlife(null);
    setWildlifeDecision("pending");
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const nextCapabilities = await api.capabilities();
        if (!cancelled) setCapabilities(nextCapabilities);
        const storedSeats = localStorage.getItem(SEATS_STORAGE_KEY);
        if (storedSeats && !cancelled) {
          setSeats(JSON.parse(storedSeats) as SeatConfig[]);
        }
        const storedGame = localStorage.getItem(GAME_STORAGE_KEY);
        const next = storedGame
          ? await api.viewGame(JSON.parse(storedGame) as SavedGame)
          : await api.newGame(20260610, DEFAULT_CONFIG);
        if (!cancelled) {
          setDocument(next);
          setSelectedPlayer(next.view.current_player);
        }
      } catch (reason) {
        if (!cancelled) setError(messageFrom(reason));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!document) return;
    localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify(document.game));
  }, [document]);

  useEffect(() => {
    localStorage.setItem(SEATS_STORAGE_KEY, JSON.stringify(seats));
  }, [seats]);

  useEffect(() => {
    resetTurnSelection();
    setCandidates([]);
    setSelectedPlayer(currentPlayer);
  }, [currentPlayer, resetTurnSelection, stateHash]);

  useEffect(() => {
    if (!savedGame || gameOver) {
      setTurnOptions(null);
      return;
    }
    let cancelled = false;
    void api
      .turnOptions(savedGame, prelude)
      .then((options) => {
        if (!cancelled) setTurnOptions(options);
      })
      .catch((reason) => {
        if (!cancelled) setError(messageFrom(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [gameOver, prelude, savedGame]);

  useEffect(() => {
    if (!savedGame || !draft || gameOver) {
      setPlacements(null);
      return;
    }
    let cancelled = false;
    void api
      .placementOptions(savedGame, prelude, draft)
      .then((options) => {
        if (!cancelled) setPlacements(options);
      })
      .catch((reason) => {
        if (!cancelled) setError(messageFrom(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [draft, gameOver, prelude, savedGame]);

  useEffect(() => {
    if (
      !savedGame ||
      !stateHash ||
      gameOver ||
      currentSeat?.kind !== "ai"
    ) {
      return;
    }
    const key = `${stateHash}:${currentPlayer}`;
    if (aiState.current === key) return;
    aiState.current = key;
    let cancelled = false;
    setBusy(true);
    setError(null);
    void api
      .suggest(savedGame, 1, currentSeat.strength)
      .then((response) => {
        const action = response.candidates[0]?.action;
        if (!action) throw new Error("The AI returned no legal move.");
        return api.apply(savedGame, action);
      })
      .then((next) => {
        if (!cancelled) {
          setDocument(next);
          setRedoStack([]);
        }
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(messageFrom(reason));
          // Allow the turn to retry (bounded) instead of wedging the game:
          // clear the attempt mark and schedule a re-render tick.
          const failures = (turnFailures.current.get(key) ?? 0) + 1;
          turnFailures.current.set(key, failures);
          if (failures < 3) {
            aiState.current = null;
            window.setTimeout(() => setAiRetryTick((tick) => tick + 1), 1500);
          }
        }
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    aiRetryTick,
    currentPlayer,
    currentSeat?.kind,
    currentSeat?.strength,
    gameOver,
    savedGame,
    stateHash,
  ]);

  // Legacy v1 keyboard: R cycles the placed tile's legal rotations.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "r" && event.key !== "R") return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (!selectedPlacement || selectedPlacement.rotations.length < 2) return;
      const index = selectedPlacement.rotations.findIndex(
        (option) => option.rotation === selectedRotation,
      );
      const next =
        selectedPlacement.rotations[(index + 1) % selectedPlacement.rotations.length];
      setSelectedRotation(next.rotation);
      setSelectedWildlife(null);
      setWildlifeDecision("pending");
      event.preventDefault();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectedPlacement, selectedRotation]);

  // Legacy v1 auto-play: when enabled, the hint engine plays human seats too.
  useEffect(() => {
    if (!autoPlay || !savedGame || !stateHash || gameOver || busy) return;
    if (currentSeat?.kind === "ai") return;
    const key = `${stateHash}:${currentPlayer}`;
    if (autoState.current === key) return;
    autoState.current = key;
    void bestMove();
  }, [autoPlay, busy, currentPlayer, currentSeat?.kind, gameOver, savedGame, stateHash]);

  function clearPlacement() {
    setSelectedCoord(null);
    setSelectedRotation(0);
    setSelectedWildlife(null);
    setWildlifeDecision("pending");
  }

  function selectDraftComponent(slot: number, component: "tile" | "wildlife") {
    if (!independent) {
      setSelectedTileSlot(slot);
      setSelectedWildlifeSlot(slot);
      setDraft(pairedDraft(slot));
      clearPlacement();
      return;
    }
    const nextTile = component === "tile" ? slot : selectedTileSlot;
    const nextWildlife =
      component === "wildlife" ? slot : selectedWildlifeSlot;
    setSelectedTileSlot(nextTile);
    setSelectedWildlifeSlot(nextWildlife);
    setDraft(
      nextTile !== null && nextWildlife !== null
        ? independentDraft(nextTile, nextWildlife)
        : null,
    );
    clearPlacement();
  }

  function selectPlacement(coord: HexCoord) {
    const placement = placements?.placements.find((candidate) =>
      sameCoord(candidate.coord, coord),
    );
    if (!placement) return;
    setSelectedCoord(coord);
    const suggestedRotation =
      suggestion &&
      sameCoord(suggestion.action.tile.coord, coord) &&
      placement.rotations.some(
        (option) => option.rotation === suggestion.action.tile.rotation,
      )
        ? suggestion.action.tile.rotation
        : (placement.rotations[0]?.rotation ?? 0);
    setSelectedRotation(suggestedRotation);
    setSelectedWildlife(null);
    setWildlifeDecision("pending");
  }

  async function commitAction(action = buildAction()) {
    if (!document || !action) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.apply(document.game, action);
      setDocument(next);
      setRedoStack([]);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setBusy(false);
    }
  }

  function buildAction() {
    if (!draft || !selectedCoord || wildlifeDecision === "pending") return null;
    return {
      replace_three_of_a_kind: prelude.replace_three_of_a_kind,
      wildlife_wipes: prelude.wildlife_wipes,
      draft,
      tile: {
        coord: selectedCoord,
        rotation: selectedRotation,
      },
      wildlife: wildlifeDecision === "placed" ? selectedWildlife : null,
    };
  }

  /** Applies the drafted tile at the selected coord with the given
   * wildlife decision, immediately (legacy v1 flow — no confirm step). */
  async function commitWildlife(wildlife: HexCoord | null) {
    if (!draft || !selectedCoord) return;
    await commitAction({
      replace_three_of_a_kind: prelude.replace_three_of_a_kind,
      wildlife_wipes: prelude.wildlife_wipes,
      draft,
      tile: { coord: selectedCoord, rotation: selectedRotation },
      wildlife,
    });
  }

  /** Legacy "Best Move": ask the configured hint engine for its top action
   * and play it for the current seat. */
  async function bestMove() {
    if (!document || view?.game_over || busy) return;
    setBusy(true);
    setError(null);
    try {
      const response = await api.suggest(document.game, 1, hintStrength);
      const action = response.candidates[0]?.action;
      if (!action) throw new Error("The engine returned no legal move.");
      const next = await api.apply(document.game, action);
      setDocument(next);
      setRedoStack([]);
    } catch (reason) {
      setError(messageFrom(reason));
      if (stateHash) {
        const key = `auto:${stateHash}:${currentPlayer}`;
        const failures = (turnFailures.current.get(key) ?? 0) + 1;
        turnFailures.current.set(key, failures);
        if (failures < 3) autoState.current = null;
      }
    } finally {
      setBusy(false);
    }
  }

  async function undo() {
    if (!document || document.game.replay.turns.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.undo(document.game);
      setRedoStack((stack) => [...stack, document.game]);
      setDocument(next);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setBusy(false);
    }
  }

  async function redo() {
    const saved = redoStack.at(-1);
    if (!saved || busy) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.viewGame(saved);
      setRedoStack((stack) => stack.slice(0, -1));
      setDocument(next);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    if (!document || view?.game_over) return;
    setAnalysisLoading(true);
    setError(null);
    try {
      const response = await api.suggest(document.game, 8, hintStrength);
      setCandidates(response.candidates);
      setSelectedCandidateIdx(0);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setAnalysisLoading(false);
    }
  }

  async function createGame(result: SetupResult) {
    setBusy(true);
    setError(null);
    try {
      const next = await api.newGame(result.seed, result.config);
      setDocument(next);
      setSeats(result.seats);
      setRedoStack([]);
      setSetupOpen(false);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setBusy(false);
    }
  }

  function saveGame() {
    if (!document) return;
    const blob = new Blob([`${JSON.stringify(document.game, null, 2)}\n`], {
      type: "application/json",
    });
    const href = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = href;
    anchor.download = `cascadia-seed-${document.game.seed}-turn-${document.view.completed_turns}.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  async function loadGame(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const saved = JSON.parse(await file.text()) as SavedGame;
      const next = await api.viewGame(saved);
      setDocument(next);
      setSeats((current) =>
        current.length === next.view.config.player_count
          ? current
          : defaultSeats(next.view.config.player_count),
      );
      setRedoStack([]);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setBusy(false);
    }
  }

  if (!document || !view || !selectedBoard) {
    return (
      <main className="boot-screen">
        <LoaderCircle className="spinner" aria-hidden="true" />
        <span>Opening local table</span>
        {error && <p>{error}</p>}
      </main>
    );
  }

  const activePlacement = selectedPlayer === currentPlayer && isHumanTurn;
  const freeReplacementSelected = prelude.replace_three_of_a_kind;
  const preludeActive =
    freeReplacementSelected || prelude.wildlife_wipes.length > 0;

  return (
    <div className="app-shell" data-mobile-panel={mobilePanel}>
      <header className="topbar v1-header">
        <h1 className="v1-title">CASCADIA</h1>
        <div className="header-controls">
          <div className="ctrl-group">
            <button type="button" onClick={() => void analyze()} disabled={busy || gameOver || !view}>
              Suggest
            </button>
            <button
              type="button"
              className="icon-btn"
              title="Clear suggestions"
              onClick={() => {
                setCandidates([]);
                setSelectedCandidateIdx(0);
              }}
              disabled={candidates.length === 0}
            >
              ✕
            </button>
            <button type="button" onClick={() => void bestMove()} disabled={busy || gameOver || !view}>
              Best Move
            </button>
            <button
              type="button"
              className={autoPlay ? "active" : ""}
              onClick={() => {
                autoState.current = null;
                setAutoPlay((current) => !current);
              }}
              disabled={!view}
            >
              Auto: {autoPlay ? "ON" : "OFF"}
            </button>
            <button
              type="button"
              className="icon-btn"
              title="Undo"
              onClick={() => void undo()}
              disabled={!document || document.game.replay.turns.length === 0 || busy}
            >
              ↶
            </button>
            <button
              type="button"
              className="icon-btn"
              title="Redo"
              onClick={() => void redo()}
              disabled={redoStack.length === 0 || busy}
            >
              ↷
            </button>
          </div>
          <div className="ctrl-group">
            <span className="ctrl-label">Engine</span>
            <select
              className="strength-select"
              value={hintStrength}
              onChange={(event) =>
                setHintStrength(event.target.value as StrengthCapability["id"])
              }
            >
              {(capabilities?.strengths ?? [])
                .filter((strength) => strength.available)
                .map((strength) => (
                  <option key={strength.id} value={strength.id}>
                    {strength.label}
                  </option>
                ))}
            </select>
          </div>
          <div className="ctrl-group">
            <span className="turn-info">
              {view
                ? gameOver
                  ? "game over"
                  : `P${currentPlayer + 1} · turn ${Math.floor(view.completed_turns / view.config.player_count) + 1} · seed ${view.seed}`
                : "no game"}
            </span>
          </div>
          <div className="ctrl-group">
            <button type="button" onClick={() => setHistoryOpen(true)} disabled={!view}>
              History
            </button>
            <button type="button" onClick={saveGame} disabled={!view}>
              Save
            </button>
            <button type="button" onClick={() => loadInput.current?.click()}>
              Load
            </button>
            <button type="button" className="active" onClick={() => setSetupOpen(true)}>
              New Game
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button
            className="icon-button quiet"
            type="button"
            title="Dismiss error"
            onClick={() => setError(null)}
          >
            <X aria-hidden="true" />
          </button>
        </div>
      )}

      <main className="game-layout v1-layout">
        <div className="left-panel">
          <div className="panel-section">
            <h2>Suggestions</h2>
            {analysisLoading ? (
              <div className="panel-note">Computing…</div>
            ) : candidates.length === 0 ? (
              <div className="panel-note">
                Press Suggest for {hintStrength} hints.
              </div>
            ) : (
              <div className="candidates-list">
                {view.free_overpopulation_replacement && (
                  <div
                    className={`candidate-row preaction${candidates[0]?.action.replace_three_of_a_kind ? " recommended" : ""}`}
                    onClick={() => {
                      if (busy || !isHumanTurn) return;
                      setPrelude((current) => ({
                        ...current,
                        replace_three_of_a_kind:
                          !current.replace_three_of_a_kind,
                      }));
                      setDraft(null);
                      setSelectedTileSlot(null);
                      setSelectedWildlifeSlot(null);
                      clearPlacement();
                    }}
                    role="button"
                    title="Free three-of-a-kind market refresh"
                  >
                    <span className="candidate-rank">♻️</span>
                    <span className="candidate-desc">
                      Free replace{freeReplacementSelected ? " ✓" : ""}
                    </span>
                    <span
                      className={`preaction-label${candidates[0]?.action.replace_three_of_a_kind ? " do-first" : ""}`}
                    >
                      {candidates[0]?.action.replace_three_of_a_kind
                        ? "DO FIRST"
                        : "SKIP"}
                    </span>
                  </div>
                )}
                {candidates.map((candidate, index) => {
                  const draftChoice = candidate.action.draft;
                  const split = !("Paired" in draftChoice);
                  const tileSlot =
                    "Paired" in draftChoice
                      ? draftChoice.Paired.slot
                      : draftChoice.Independent.tile_slot;
                  const delta =
                    index === 0
                      ? null
                      : candidate.search_value - candidates[0].search_value;
                  return (
                    <div
                      key={index}
                      className={`candidate-row${index === selectedCandidateIdx ? " selected" : ""}`}
                      onClick={() => setSelectedCandidateIdx(index)}
                      role="button"
                      title="Preview this move (glows on board and market)"
                    >
                      <span className="candidate-rank">#{index + 1}</span>
                      <span className="candidate-desc">
                        {split ? "🌲 " : ""}pair {tileSlot + 1} → (
                        {candidate.action.tile.coord.q},
                        {candidate.action.tile.coord.r})
                        {candidate.action.wildlife ? " +tok" : ""}
                        {delta !== null && (
                          <span className="candidate-delta">
                            {" "}
                            ({delta.toFixed(1)})
                          </span>
                        )}
                      </span>
                      <span className="candidate-value">
                        {candidate.search_value.toFixed(1)}
                      </span>
                      <button
                        type="button"
                        className="candidate-play"
                        disabled={busy || !isHumanTurn}
                        onClick={(event) => {
                          event.stopPropagation();
                          void commitAction(candidate.action);
                        }}
                        title="Play this move now"
                      >
                        ▶
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          <div className="panel-section">
            <h2>Scoring — P{selectedPlayer + 1}</h2>
            <div className="score-breakdown">
              {(["bear", "elk", "salmon", "hawk", "fox"] as const).map(
                (animal, index) => (
                  <div key={animal} className="score-line">
                    <span className="score-name">
                      <WildlifeMark wildlife={animal} size="small" /> {titleCase(animal)}{" "}
                      <span className="score-card">
                        card {view.config.scoring_cards[animal]}
                      </span>
                    </span>
                    <span className="score-points">
                      {selectedBoard.score.wildlife[index]}
                    </span>
                  </div>
                ),
              )}
              <div className="score-divider" />
              {(
                ["mountain", "forest", "prairie", "wetland", "river"] as const
              ).map((terrain, index) => (
                <div key={terrain} className="score-line">
                  <span className="score-name">
                    <i className={`terrain-dot terrain-${terrain}`} />{" "}
                    {titleCase(terrain)}
                    {selectedBoard.score.habitat_bonus[index] > 0 && (
                      <span className="score-card">
                        {" "}
                        +{selectedBoard.score.habitat_bonus[index]} bonus
                      </span>
                    )}
                  </span>
                  <span className="score-points">
                    {selectedBoard.score.habitat[index]}
                  </span>
                </div>
              ))}
              <div className="score-divider" />
              <div className="score-line">
                <span className="score-name">🌲 Nature tokens</span>
                <span className="score-points">
                  {selectedBoard.score.nature_tokens}
                </span>
              </div>
              <div className="score-line total">
                <span className="score-name">Total</span>
                <span className="score-points">{selectedBoard.score.total}</span>
              </div>
            </div>
          </div>
          <div className="panel-section">
            <h2>Event log</h2>
            <EventLog
              turns={document?.game.replay.turns ?? []}
              playerCount={view.config.player_count}
            />
          </div>
        </div>

        <section className="board-column" aria-label="Cascadia board">
          <div className="board-tabs">
            {view.boards.map((board, index) => (
              <button
                key={index}
                type="button"
                className={index === selectedPlayer ? "active" : ""}
                onClick={() => setSelectedPlayer(index)}
              >
                P{index + 1}
                {index === currentPlayer ? " ●" : ""}
                <span className="tab-score">{board.score.total}</span>
              </button>
            ))}
          </div>
          <HexBoard
            board={selectedBoard}
            active={activePlacement}
            placements={activePlacement ? placements?.placements ?? [] : []}
            selectedCoord={selectedCoord}
            wildlifeTargets={
              activePlacement ? selectedRotationOption?.wildlife ?? [] : []
            }
            selectedWildlife={selectedWildlife}
            draftedWildlife={draftedWildlife as Wildlife | null}
            preview={activePlacement ? preview : null}
            suggestedCoord={
              suggestion && selectedPlayer === currentPlayer && isHumanTurn
                ? suggestion.action.tile.coord
                : null
            }
            suggestedWildlife={
              suggestion && selectedPlayer === currentPlayer && isHumanTurn
                ? suggestion.action.wildlife
                : null
            }
            onSelectCoord={selectPlacement}
            onSelectWildlife={(coord) => {
              setSelectedWildlife(coord);
              setWildlifeDecision("placed");
              void commitWildlife(coord);
            }}
          />
        </section>

        <div className="side-panel">
          <div className="panel-section">
            <h2>Market</h2>
            <MarketRail
              market={stagedMarket}
              turnOptions={turnOptions}
              draft={draft}
              independent={independent}
              selectedTileSlot={selectedTileSlot}
              selectedWildlifeSlot={selectedWildlifeSlot}
              phase={phase}
              glowTileSlot={isHumanTurn ? suggestionTileSlot : null}
              glowWildlifeSlot={isHumanTurn ? suggestionWildlifeSlot : null}
              glowSplit={suggestionSplit}
              busy={busy || !isHumanTurn}
              freeReplacement={view.free_overpopulation_replacement}
              freeReplacementSelected={freeReplacementSelected}
              preludeActive={preludeActive}
              natureTokens={turnOptions?.nature_tokens_remaining ?? 0}
              onSelectDraftComponent={selectDraftComponent}
              onSetIndependent={(next) => {
                setIndependent(next);
                setDraft(null);
                setSelectedTileSlot(null);
                setSelectedWildlifeSlot(null);
                clearPlacement();
              }}
              onWipeAll={() => {
                const counts = new Map<string, number[]>();
                stagedMarket.forEach((pair) => {
                  if (!pair.wildlife) return;
                  counts.set(pair.wildlife, [
                    ...(counts.get(pair.wildlife) ?? []),
                    pair.slot,
                  ]);
                });
                const repeated = [...counts.values()]
                  .filter((slots) => slots.length > 1)
                  .sort((left, right) => right.length - left.length)[0];
                if (!repeated) return;
                setPrelude((current) => ({
                  ...current,
                  wildlife_wipes: [...current.wildlife_wipes, { slots: repeated }],
                }));
                setDraft(null);
                setSelectedTileSlot(null);
                setSelectedWildlifeSlot(null);
                clearPlacement();
              }}
              onToggleFreeReplacement={() => {
                setPrelude((current) => ({
                  ...current,
                  replace_three_of_a_kind: !current.replace_three_of_a_kind,
                }));
                setDraft(null);
                setSelectedTileSlot(null);
                setSelectedWildlifeSlot(null);
                clearPlacement();
              }}
              onResetPrelude={resetTurnSelection}
            />
          </div>
          <div className="panel-section">
            <h2>Scores</h2>
            <ScorePanel
              boards={view.boards}
              selectedPlayer={selectedPlayer}
              currentPlayer={currentPlayer}
              cards={view.config.scoring_cards}
              onSelectPlayer={setSelectedPlayer}
            />
          </div>
          <div className="panel-section">
            <h2>How to play</h2>
            <div className="instructions">
              1. Click a market pair on the right
              <br />
              2. Click a highlighted hex — press <b>R</b> to rotate
              <br />
              3. Click a tile for the wildlife token, or skip
            </div>
            {phase === "place_wildlife" && (
              <button
                type="button"
                className="rail-button"
                onClick={() => {
                  setSelectedWildlife(null);
                  setWildlifeDecision("skipped");
                  void commitWildlife(null);
                }}
                disabled={busy || !isHumanTurn}
              >
                Skip Wildlife
              </button>
            )}
          </div>
        </div>
      </main>

      <div className="status-bar">
        {error
          ? `⚠ ${error}`
          : gameOver
            ? "Game over — press New Game to play again."
            : !isHumanTurn
              ? busy
                ? `P${currentPlayer + 1} (${currentSeat?.strength}) is thinking…`
                : `Waiting for P${currentPlayer + 1}…`
              : phase === "select_market"
                ? "Your turn: click a market pair on the right."
                : phase === "place_tile"
                  ? "Click a highlighted frontier hex. Press R to rotate after placing."
                  : "Click a highlighted tile to place the wildlife token, press R to rotate the tile, or Skip Wildlife."}
      </div>

      <input
        ref={loadInput}
        type="file"
        accept="application/json"
        className="visually-hidden"
        onChange={(event) => void loadGame(event)}
      />

      {busy && !isHumanTurn && (
        <div className="spinner-overlay">
          <div className="spinner" />
          <div className="spinner-text">Computing best move…</div>
        </div>
      )}

      <nav className="mobile-tabs" aria-label="Mobile views">
        {(["board", "market", "scores", "analysis"] as MobilePanel[]).map((panel) => (
          <button
            key={panel}
            type="button"
            className={mobilePanel === panel ? "is-active" : ""}
            onClick={() => setMobilePanel(panel)}
          >
            {titleCase(panel)}
          </button>
        ))}
      </nav>

      {view.game_over && (
        <section className="final-banner" aria-label="Final results">
          <span className="eyebrow">Game complete</span>
          <h2>Final scores</h2>
          <div>
            {[...view.boards]
              .sort((left, right) => right.score.total - left.score.total)
              .map((board, index) => (
                <span key={board.player} className={index === 0 ? "is-winner" : ""}>
                  P{board.player + 1}
                  <strong>{board.score.total}</strong>
                </span>
              ))}
          </div>
        </section>
      )}

      {historyOpen && (
        <div className="sheet-backdrop" role="presentation">
          <aside className="history-sheet" role="dialog" aria-modal="true">
            <div className="dialog-heading">
              <div>
                <span className="eyebrow">Replay</span>
                <h2>Game history</h2>
              </div>
              <button
                className="icon-button quiet"
                type="button"
                title="Close history"
                onClick={() => setHistoryOpen(false)}
              >
                <X aria-hidden="true" />
              </button>
            </div>
            <ol>
              {document.game.replay.turns.map((action, index) => (
                <li key={`${index}-${action.tile.coord.q}-${action.tile.coord.r}`}>
                  <span>P{(index % view.config.player_count) + 1}</span>
                  <div>
                    <strong>{actionSummary(action.draft)}</strong>
                    <small>
                      q {action.tile.coord.q}, r {action.tile.coord.r} · rotation{" "}
                      {action.tile.rotation + 1}
                    </small>
                  </div>
                </li>
              ))}
            </ol>
          </aside>
        </div>
      )}

      <SetupDialog
        open={setupOpen}
        capabilities={capabilities}
        initialConfig={view.config}
        initialSeed={view.seed}
        initialSeats={seats}
        onClose={() => setSetupOpen(false)}
        onCreate={(result) => void createGame(result)}
      />
    </div>
  );
}

function messageFrom(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}
