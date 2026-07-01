import {
  Download,
  History,
  LoaderCircle,
  Menu,
  Redo2,
  Server,
  Settings2,
  Undo2,
  Upload,
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
import { AnalysisPanel } from "./components/AnalysisPanel";
import { HexBoard } from "./components/HexBoard";
import { MarketWorkbench } from "./components/MarketWorkbench";
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
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [candidates, setCandidates] = useState<SuggestionCandidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("board");
  const loadInput = useRef<HTMLInputElement>(null);
  const aiState = useRef<string | null>(null);

  const view = document?.view ?? null;
  const savedGame = document?.game ?? null;
  const stateHash = view?.state_hash;
  const gameOver = view?.game_over ?? false;
  const currentPlayer = view?.current_player ?? 0;
  const currentSeat = seats[currentPlayer];
  const isHumanTurn = currentSeat?.kind !== "ai";
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
        if (!cancelled) setError(messageFrom(reason));
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    currentPlayer,
    currentSeat?.kind,
    currentSeat?.strength,
    gameOver,
    savedGame,
    stateHash,
  ]);

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
    setSelectedRotation(placement.rotations[0]?.rotation ?? 0);
    setSelectedWildlife(null);
    setWildlifeDecision("pending");
  }

  function selectRotation(rotation: number) {
    setSelectedRotation(rotation);
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
      const response = await api.suggest(document.game, 8, "research");
      setCandidates(response.candidates);
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
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <div>
            <h1>Cascadia Lab</h1>
            <span>AAAAA · Local</span>
          </div>
        </div>
        <div className="turn-status">
          <span className="turn-player">P{currentPlayer + 1}</span>
          <div>
            <strong>
              {view.game_over
                ? "Final scores"
                : busy && currentSeat?.kind === "ai"
                  ? "AI is choosing"
                  : `Turn ${Math.floor(view.completed_turns / view.config.player_count) + 1}`}
            </strong>
            <small>
              {view.completed_turns} / {view.total_turns} actions · seed {view.seed}
            </small>
          </div>
          {busy && <LoaderCircle className="spinner" aria-label="Working" />}
        </div>
        <nav className="top-actions" aria-label="Game actions">
          <a
            className="icon-button"
            href="/cluster"
            title="Cluster dashboard"
            aria-label="Cluster dashboard"
          >
            <Server aria-hidden="true" />
          </a>
          <button
            className="icon-button"
            type="button"
            title="Undo"
            onClick={undo}
            disabled={document.game.replay.turns.length === 0 || busy}
          >
            <Undo2 aria-hidden="true" />
          </button>
          <button
            className="icon-button"
            type="button"
            title="Redo"
            onClick={redo}
            disabled={redoStack.length === 0 || busy}
          >
            <Redo2 aria-hidden="true" />
          </button>
          <span className="action-divider" />
          <button
            className="icon-button"
            type="button"
            title="Game history"
            onClick={() => setHistoryOpen(true)}
          >
            <History aria-hidden="true" />
          </button>
          <button
            className="icon-button"
            type="button"
            title="Save game"
            onClick={saveGame}
          >
            <Download aria-hidden="true" />
          </button>
          <button
            className="icon-button"
            type="button"
            title="Load game"
            onClick={() => loadInput.current?.click()}
          >
            <Upload aria-hidden="true" />
          </button>
          <button
            className="command-button setup-button"
            type="button"
            onClick={() => setSetupOpen(true)}
          >
            <Settings2 aria-hidden="true" />
            New game
          </button>
          <button
            className="icon-button mobile-menu-button"
            type="button"
            title="New game"
            onClick={() => setSetupOpen(true)}
          >
            <Menu aria-hidden="true" />
          </button>
          <input
            ref={loadInput}
            className="visually-hidden"
            type="file"
            accept="application/json,.json"
            onChange={loadGame}
          />
        </nav>
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

      <main className="game-layout">
        <div className="desktop-score-column">
          <ScorePanel
            boards={view.boards}
            selectedPlayer={selectedPlayer}
            currentPlayer={currentPlayer}
            cards={view.config.scoring_cards}
            onSelectPlayer={setSelectedPlayer}
          />
        </div>

        <section className="board-column" aria-label="Cascadia board">
          <div className="board-caption">
            <div>
              <span className="eyebrow">Player {selectedPlayer + 1}</span>
              <h2>
                {selectedPlayer === currentPlayer
                  ? isHumanTurn
                    ? "Your environment"
                    : "AI environment"
                  : "Environment review"}
              </h2>
            </div>
            <div className="board-score">
              <span>Base</span>
              <strong>{selectedBoard.score.base_total}</strong>
            </div>
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
            onSelectCoord={selectPlacement}
            onSelectWildlife={(coord) => {
              setSelectedWildlife(coord);
              setWildlifeDecision("placed");
            }}
          />
        </section>

        <div className="desktop-workbench-column">
          {isHumanTurn && !view.game_over ? (
            <MarketWorkbench
              market={view.market}
              turnOptions={turnOptions}
              draft={draft}
              independent={independent}
              selectedTileSlot={selectedTileSlot}
              selectedWildlifeSlot={selectedWildlifeSlot}
              selectedCoord={selectedCoord}
              selectedRotation={selectedRotation}
              wildlifeDecision={wildlifeDecision}
              placement={selectedPlacement}
              busy={busy}
              freeReplacement={view.free_overpopulation_replacement}
              freeReplacementSelected={freeReplacementSelected}
              preludeActive={preludeActive}
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
              onAddWipe={(slots) => {
                setPrelude((current) => ({
                  ...current,
                  wildlife_wipes: [...current.wildlife_wipes, { slots }],
                }));
                setDraft(null);
                setSelectedTileSlot(null);
                setSelectedWildlifeSlot(null);
                clearPlacement();
              }}
              onResetPrelude={resetTurnSelection}
              onSetIndependent={(next) => {
                setIndependent(next);
                setDraft(null);
                setSelectedTileSlot(null);
                setSelectedWildlifeSlot(null);
                clearPlacement();
              }}
              onSelectDraftComponent={selectDraftComponent}
              onSetRotation={selectRotation}
              onSkipWildlife={() => {
                setSelectedWildlife(null);
                setWildlifeDecision("skipped");
              }}
              onCommit={() => void commitAction()}
            />
          ) : (
            <aside className="workbench passive-workbench">
              <LoaderCircle className={busy ? "spinner" : ""} aria-hidden="true" />
              <span>
                {view.game_over ? "Game complete" : `Player ${currentPlayer + 1} is choosing`}
              </span>
            </aside>
          )}
        </div>
      </main>

      <AnalysisPanel
        open={analysisOpen}
        loading={analysisLoading}
        candidates={candidates}
        onToggle={() => setAnalysisOpen((open) => !open)}
        onAnalyze={() => void analyze()}
        onPlay={(candidate) => void commitAction(candidate.action)}
      />

      <nav className="mobile-tabs" aria-label="Mobile views">
        {(["board", "market", "scores", "analysis"] as MobilePanel[]).map((panel) => (
          <button
            key={panel}
            type="button"
            className={mobilePanel === panel ? "is-active" : ""}
            onClick={() => {
              setMobilePanel(panel);
              if (panel === "analysis") setAnalysisOpen(true);
            }}
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
