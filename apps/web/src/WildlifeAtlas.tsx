import {
  ArrowLeft,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Dice5,
  LoaderCircle,
  Search,
  ShieldCheck,
  Telescope,
} from "lucide-react";
import {
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import { WildlifeAtlasBoard } from "./components/WildlifeAtlasBoard";
import { WildlifeMark } from "./components/WildlifeMark";
import type { ScoringVariant, Wildlife } from "./types";
import {
  SCORING_CARD_RULES,
  SCORING_VARIANTS,
  WILDLIFE_ORDER,
  WILDLIFE_PROFILES,
  assertAtlasDocument,
  indexToRuleset,
  isRuleset,
  rankedRows,
  replaceRulesetCard,
  requestedRuleset,
  rulesetToIndex,
  type AtlasListFilter,
  type WildlifeAtlasDocument,
} from "./wildlifeCatalogData";

const DEFAULT_RULESET = "ADCCB";

export default function WildlifeAtlas() {
  const [document, setDocument] = useState<WildlifeAtlasDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ruleset, setRuleset] = useState(() =>
    requestedRuleset(window.location.search, DEFAULT_RULESET),
  );
  const [rulesetDraft, setRulesetDraft] = useState(ruleset);
  const [listFilter, setListFilter] = useState<AtlasListFilter>("all");
  const [focusedWildlife, setFocusedWildlife] = useState<Wildlife | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void fetch("/wildlife-atlas.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Catalog request failed (${response.status})`);
        }
        return response.json() as Promise<unknown>;
      })
      .then(assertAtlasDocument)
      .then((nextDocument) => {
        if (!cancelled) setDocument(nextDocument);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectRuleset = useCallback((nextRuleset: string) => {
    if (!isRuleset(nextRuleset)) return;
    setRuleset(nextRuleset);
    setRulesetDraft(nextRuleset);
    setFocusedWildlife(null);
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set("rules", nextRuleset);
    window.history.replaceState(null, "", nextUrl);
  }, []);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        (event.target instanceof HTMLElement &&
          ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName))
      ) {
        return;
      }
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? -1 : 1;
      const nextIndex = (rulesetToIndex(ruleset) + delta + 1024) % 1024;
      selectRuleset(indexToRuleset(nextIndex));
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [ruleset, selectRuleset]);

  const selected = document?.rows[rulesetToIndex(ruleset)] ?? null;
  const sortedRows = useMemo(
    () => (document ? rankedRows(document.rows, listFilter) : []),
    [document, listFilter],
  );
  const globalRanking = useMemo(
    () => (document ? rankedRows(document.rows, "all") : []),
    [document],
  );
  const scoreRanks = useMemo(() => {
    const ranks = new Map<number, number>();
    globalRanking.forEach((candidate, position) => {
      if (!ranks.has(candidate.score)) ranks.set(candidate.score, position + 1);
    });
    return ranks;
  }, [globalRanking]);
  const rank = selected === null ? null : scoreRanks.get(selected.score);

  function submitRuleset(event: FormEvent) {
    event.preventDefault();
    selectRuleset(rulesetDraft);
  }

  function stepRuleset(delta: number) {
    const nextIndex = (rulesetToIndex(ruleset) + delta + 1024) % 1024;
    selectRuleset(indexToRuleset(nextIndex));
  }

  function surpriseMe() {
    const sample = new Uint32Array(1);
    crypto.getRandomValues(sample);
    selectRuleset(indexToRuleset(sample[0] % 1024));
  }

  async function copyLink() {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  if (error) {
    return (
      <main className="atlas-boot-screen">
        <Telescope aria-hidden="true" />
        <h1>Wildlife atlas unavailable</h1>
        <p>{error}</p>
        <a href="/">Return to Cascadia Lab</a>
      </main>
    );
  }

  if (!document || !selected) {
    return (
      <main className="atlas-boot-screen" aria-live="polite">
        <LoaderCircle className="atlas-spinner" aria-hidden="true" />
        <p>Opening the 1,024-board field ledger…</p>
      </main>
    );
  }

  const proofGap = selected.upper - selected.score;
  const selectedIndex = rulesetToIndex(ruleset);

  return (
    <main className="atlas-shell">
      <header className="atlas-header">
        <a className="atlas-back" href="/" aria-label="Return to Cascadia Lab">
          <ArrowLeft aria-hidden="true" />
          <span>Lab</span>
        </a>
        <div className="atlas-title">
          <span className="atlas-tree-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <div>
            <span>Pure wildlife research / cap 6</span>
            <h1>Cascadia Wildlife Atlas</h1>
          </div>
        </div>
        <div className="atlas-global-stats" aria-label="Catalog summary">
          <div>
            <strong>{document.rulesetCount.toLocaleString()}</strong>
            <span>boards</span>
          </div>
          <div>
            <strong>{document.completedRulesets}</strong>
            <span>proven</span>
          </div>
          <div>
            <strong>
              {document.incumbentHolisticMaximum}–{document.holisticSoundUpper}
            </strong>
            <span>global range</span>
          </div>
        </div>
      </header>

      <div className="atlas-layout">
        <aside className="atlas-index" aria-label="Scoring card selector and catalog index">
          <section className="atlas-selector">
            <div className="atlas-section-kicker">
              <span>Scoring cards</span>
              <b>{selectedIndex + 1} / {document.rulesetCount}</b>
            </div>
            <form className="atlas-ruleset-form" onSubmit={submitRuleset}>
              <label htmlFor="atlas-ruleset">Card set ID</label>
              <div>
                <Search aria-hidden="true" />
                <input
                  id="atlas-ruleset"
                  value={rulesetDraft}
                  maxLength={5}
                  spellCheck={false}
                  autoComplete="off"
                  aria-invalid={!isRuleset(rulesetDraft)}
                  onChange={(event) =>
                    setRulesetDraft(
                      event.target.value.toUpperCase().replace(/[^A-D]/g, ""),
                    )
                  }
                />
                <button
                  type="submit"
                  disabled={!isRuleset(rulesetDraft) || rulesetDraft === ruleset}
                >
                  Go
                </button>
              </div>
            </form>
            <div className="atlas-card-selector">
              {WILDLIFE_PROFILES.map((profile, position) => (
                <div className="atlas-card-row" key={profile.id}>
                  <div className="atlas-card-animal">
                    <WildlifeMark wildlife={profile.id} size="small" />
                    <span>{profile.label}</span>
                  </div>
                  <div className="atlas-variant-switch" aria-label={`${profile.label} card`}>
                    {SCORING_VARIANTS.map((variant) => (
                      <button
                        type="button"
                        key={variant}
                        className={ruleset[position] === variant ? "is-active" : ""}
                        aria-pressed={ruleset[position] === variant}
                        aria-label={`${profile.label} card ${variant}`}
                        onClick={() =>
                          selectRuleset(replaceRulesetCard(ruleset, position, variant))
                        }
                      >
                        {variant}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div className="atlas-sequence-controls">
              <button type="button" onClick={() => stepRuleset(-1)}>
                <ChevronLeft aria-hidden="true" /> Previous
              </button>
              <button type="button" onClick={surpriseMe}>
                <Dice5 aria-hidden="true" /> Random
              </button>
              <button type="button" onClick={() => stepRuleset(1)}>
                Next <ChevronRight aria-hidden="true" />
              </button>
            </div>
          </section>

          <section className="atlas-ledger">
            <div className="atlas-ledger-heading">
              <div>
                <span>Ranked ledger</span>
                <b>{sortedRows.length}</b>
              </div>
              <div className="atlas-ledger-filters" aria-label="Catalog filters">
                {(["all", "exact", "leaders"] as const).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className={listFilter === filter ? "is-active" : ""}
                    aria-pressed={listFilter === filter}
                    onClick={() => setListFilter(filter)}
                  >
                    {filter === "all" ? "All" : filter === "exact" ? "Proven" : "85s"}
                  </button>
                ))}
              </div>
            </div>
            <div className="atlas-ledger-list">
              {sortedRows.map((row, position) => (
                <button
                  type="button"
                  key={row.id}
                  className={row.id === ruleset ? "is-active" : ""}
                  aria-current={row.id === ruleset ? "true" : undefined}
                  onClick={() => selectRuleset(row.id)}
                >
                  <span className="atlas-ledger-rank">
                    {listFilter === "all"
                      ? scoreRanks.get(row.score)
                      : position + 1}
                  </span>
                  <code>{row.id}</code>
                  {row.exact ? (
                    <ShieldCheck aria-label="Proven optimum" />
                  ) : (
                    <span className="atlas-gap-tick" title={`Upper bound ${row.upper}`}>
                      +{row.upper - row.score}
                    </span>
                  )}
                  <strong>{row.score}</strong>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <section className="atlas-board-column" aria-label="Selected wildlife board">
          <div className="atlas-board-caption">
            <div>
              <span className="atlas-eyebrow">
                {selected.exact ? "Certified optimum" : "Best board found"}
              </span>
              <h2>{ruleset}</h2>
            </div>
            <div className="atlas-board-rank">
              <span>catalog rank</span>
              <strong>#{rank}</strong>
            </div>
          </div>
          <WildlifeAtlasBoard
            ruleset={ruleset}
            tokens={selected.tokens}
            focusedWildlife={focusedWildlife}
            onFocusWildlife={setFocusedWildlife}
          />
          <div className="atlas-board-legend" aria-label="Highlight an animal">
            <span>Trace a species</span>
            <button
              type="button"
              className={focusedWildlife === null ? "is-active" : ""}
              aria-pressed={focusedWildlife === null}
              onClick={() => setFocusedWildlife(null)}
            >
              All
            </button>
            {WILDLIFE_PROFILES.map((profile, position) => (
              <button
                type="button"
                key={profile.id}
                className={focusedWildlife === profile.id ? "is-active" : ""}
                aria-pressed={focusedWildlife === profile.id}
                onClick={() =>
                  setFocusedWildlife(
                    focusedWildlife === profile.id ? null : profile.id,
                  )
                }
              >
                <span aria-hidden="true">{profile.emoji}</span>
                {selected.counts[position]}
              </button>
            ))}
          </div>
        </section>

        <aside className="atlas-dossier" aria-label="Score and proof dossier">
          <section className="atlas-score-hero">
            <div className="atlas-proof-line">
              {selected.exact ? (
                <>
                  <ShieldCheck aria-hidden="true" />
                  <span>Optimality proven</span>
                </>
              ) : (
                <>
                  <Telescope aria-hidden="true" />
                  <span>Validated incumbent</span>
                </>
              )}
            </div>
            <div className="atlas-score-number">
              <strong>{selected.score}</strong>
              <span>animal points</span>
            </div>
            <div className="atlas-score-range">
              <div>
                <span>best found</span>
                <b>{selected.score}</b>
              </div>
              <div className="atlas-range-track" aria-hidden="true">
                <i
                  style={{
                    width: selected.exact
                      ? "100%"
                      : `${Math.max(18, (selected.score / selected.upper) * 100)}%`,
                  }}
                />
              </div>
              <div>
                <span>sound upper</span>
                <b>{selected.upper}</b>
              </div>
            </div>
            <p>
              {selected.exact
                ? "The displayed board reaches the certified ceiling."
                : `The true optimum is in [${selected.score}, ${selected.upper}]. The remaining proof gap is ${proofGap} point${proofGap === 1 ? "" : "s"}.`}
            </p>
          </section>

          <section className="atlas-breakdown">
            <div className="atlas-dossier-heading">
              <span>Score anatomy</span>
              <b>{document.tokenCount} tokens</b>
            </div>
            {WILDLIFE_PROFILES.map((profile, position) => (
              <button
                type="button"
                key={profile.id}
                className={focusedWildlife === profile.id ? "is-active" : ""}
                onClick={() =>
                  setFocusedWildlife(
                    focusedWildlife === profile.id ? null : profile.id,
                  )
                }
              >
                <WildlifeMark wildlife={profile.id} size="small" />
                <span>
                  <b>{profile.label}</b>
                  <small>
                    card {ruleset[position]} · {selected.counts[position]} placed
                  </small>
                </span>
                <strong>{selected.parts[position]}</strong>
              </button>
            ))}
          </section>

          <section className="atlas-rules">
            <div className="atlas-dossier-heading">
              <span>Card notes</span>
              <b>{ruleset}</b>
            </div>
            {WILDLIFE_ORDER.map((wildlife, position) => {
              const variant = ruleset[position] as ScoringVariant;
              const profile = WILDLIFE_PROFILES[position];
              return (
                <div className="atlas-rule-note" key={wildlife}>
                  <span className={`atlas-rule-letter wildlife-${wildlife}`}>
                    {variant}
                  </span>
                  <p>
                    <b>{profile.label}</b>
                    {SCORING_CARD_RULES[wildlife][variant]}
                  </p>
                </div>
              );
            })}
          </section>

          <section className="atlas-notes">
            <button type="button" onClick={() => void copyLink()}>
              {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
              {copied ? "Copied" : "Copy board link"}
            </button>
            <p>
              Connected 20-token boards, maximum six of any animal. Habitat,
              Nature tokens, and all non-wildlife scoring are excluded.
            </p>
          </section>
        </aside>
      </div>
    </main>
  );
}
