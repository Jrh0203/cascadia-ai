import {
  Check,
  CircleDollarSign,
  Dices,
  RotateCw,
  Sparkles,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import { titleCase } from "../game";
import type {
  DraftChoice,
  HexCoord,
  MarketPairView,
  PlacementOption,
  TurnOptions,
} from "../types";
import { TileToken } from "./TileToken";
import { WildlifeMark } from "./WildlifeMark";

interface MarketWorkbenchProps {
  market: MarketPairView[];
  turnOptions: TurnOptions | null;
  draft: DraftChoice | null;
  independent: boolean;
  selectedTileSlot: number | null;
  selectedWildlifeSlot: number | null;
  selectedCoord: HexCoord | null;
  selectedRotation: number;
  wildlifeDecision: "pending" | "placed" | "skipped";
  placement: PlacementOption | null;
  busy: boolean;
  freeReplacement: boolean;
  freeReplacementSelected: boolean;
  preludeActive: boolean;
  onToggleFreeReplacement: () => void;
  onAddWipe: (slots: number[]) => void;
  onResetPrelude: () => void;
  onSetIndependent: (independent: boolean) => void;
  onSelectDraftComponent: (slot: number, component: "tile" | "wildlife") => void;
  onSetRotation: (rotation: number) => void;
  onSkipWildlife: () => void;
  onCommit: () => void;
}

export function MarketWorkbench({
  market,
  turnOptions,
  draft,
  independent,
  selectedTileSlot,
  selectedWildlifeSlot,
  selectedCoord,
  selectedRotation,
  wildlifeDecision,
  placement,
  busy,
  freeReplacement,
  freeReplacementSelected,
  preludeActive,
  onToggleFreeReplacement,
  onAddWipe,
  onResetPrelude,
  onSetIndependent,
  onSelectDraftComponent,
  onSetRotation,
  onSkipWildlife,
  onCommit,
}: MarketWorkbenchProps) {
  const [wipeOpen, setWipeOpen] = useState(false);
  const [wipeSlots, setWipeSlots] = useState<number[]>([]);
  const rotation = placement?.rotations.find(
    (option) => option.rotation === selectedRotation,
  );
  const canCommit =
    draft !== null &&
    selectedCoord !== null &&
    rotation !== undefined &&
    wildlifeDecision !== "pending";
  const step =
    draft === null
      ? "draft"
      : selectedCoord === null
        ? "habitat"
        : wildlifeDecision === "pending"
          ? "wildlife"
          : "ready";
  const stagedMarket = turnOptions?.market ?? market;

  const availableRotations = useMemo(
    () => placement?.rotations.map((option) => option.rotation) ?? [],
    [placement],
  );

  function toggleWipeSlot(slot: number) {
    setWipeSlots((current) =>
      current.includes(slot)
        ? current.filter((candidate) => candidate !== slot)
        : [...current, slot].sort(),
    );
  }

  return (
    <aside className="workbench" aria-label="Turn workbench">
      <div className="workbench-heading">
        <div>
          <span className="eyebrow">Turn workbench</span>
          <h2>
            {step === "draft" && "Choose a pair"}
            {step === "habitat" && "Place the habitat"}
            {step === "wildlife" && "Place the wildlife"}
            {step === "ready" && "Confirm the turn"}
          </h2>
        </div>
        {preludeActive && (
          <button
            className="icon-button quiet"
            type="button"
            title="Reset market actions"
            onClick={onResetPrelude}
          >
            <X aria-hidden="true" />
          </button>
        )}
      </div>

      {(freeReplacement || turnOptions?.can_add_paid_wipe) && (
        <div className="market-actions">
          {freeReplacement && (
            <button
              className={`market-action ${freeReplacementSelected ? "is-active" : ""}`}
              type="button"
              onClick={onToggleFreeReplacement}
            >
              <Sparkles aria-hidden="true" />
              Replace matching three
              <span>Free</span>
            </button>
          )}
          {turnOptions?.can_add_paid_wipe && (
            <button
              className={`market-action ${wipeOpen ? "is-active" : ""}`}
              type="button"
              onClick={() => setWipeOpen((open) => !open)}
            >
              <Dices aria-hidden="true" />
              Refresh wildlife
              <span>1 token</span>
            </button>
          )}
        </div>
      )}

      {wipeOpen && (
        <div className="wipe-editor">
          <span>Select wildlife to return</span>
          <div className="wipe-slots">
            {stagedMarket.map((pair) => (
              <button
                key={pair.slot}
                type="button"
                className={wipeSlots.includes(pair.slot) ? "is-selected" : ""}
                onClick={() => toggleWipeSlot(pair.slot)}
                aria-pressed={wipeSlots.includes(pair.slot)}
                title={`Select market wildlife ${pair.slot + 1}`}
              >
                {pair.wildlife && <WildlifeMark wildlife={pair.wildlife} size="small" />}
              </button>
            ))}
          </div>
          <button
            className="command-button"
            type="button"
            disabled={wipeSlots.length === 0}
            onClick={() => {
              onAddWipe(wipeSlots);
              setWipeSlots([]);
              setWipeOpen(false);
            }}
          >
            <Check aria-hidden="true" />
            Refresh selected
          </button>
        </div>
      )}

      <div className="draft-mode" role="group" aria-label="Draft mode">
        <button
          type="button"
          className={!independent ? "is-active" : ""}
          onClick={() => onSetIndependent(false)}
          aria-pressed={!independent}
        >
          Paired
        </button>
        <button
          type="button"
          className={independent ? "is-active" : ""}
          onClick={() => onSetIndependent(true)}
          aria-pressed={independent}
          disabled={(turnOptions?.nature_tokens_remaining ?? 0) === 0}
        >
          <CircleDollarSign aria-hidden="true" />
          Independent
        </button>
      </div>

      <div className="market-grid">
        {stagedMarket.map((pair) => {
          const tileSelected = selectedTileSlot === pair.slot;
          const wildlifeSelected = selectedWildlifeSlot === pair.slot;
          return (
            <div
              key={pair.slot}
              className={`market-pair ${tileSelected || wildlifeSelected ? "is-selected" : ""}`}
            >
              <button
                className={`market-tile-button ${tileSelected ? "is-selected" : ""}`}
                type="button"
                onClick={() => onSelectDraftComponent(pair.slot, "tile")}
                aria-pressed={tileSelected}
                title={`Choose habitat tile ${pair.slot + 1}`}
                disabled={!pair.tile}
              >
                {pair.tile && <TileToken tile={pair.tile} compact />}
              </button>
              <button
                className={`market-wildlife-button ${
                  wildlifeSelected ? "is-selected" : ""
                }`}
                type="button"
                onClick={() => onSelectDraftComponent(pair.slot, "wildlife")}
                aria-pressed={wildlifeSelected}
                title={`Choose ${pair.wildlife ?? "wildlife"} token ${pair.slot + 1}`}
                disabled={!pair.wildlife || !independent}
              >
                {pair.wildlife && <WildlifeMark wildlife={pair.wildlife} />}
              </button>
              <span className="market-index">{pair.slot + 1}</span>
            </div>
          );
        })}
      </div>

      {selectedCoord && availableRotations.length > 1 && (
        <div className="rotation-control">
          <span>Orientation</span>
          <div>
            {availableRotations.map((candidate) => (
              <button
                key={candidate}
                className={candidate === selectedRotation ? "is-active" : ""}
                type="button"
                onClick={() => onSetRotation(candidate)}
                aria-pressed={candidate === selectedRotation}
                title={`Rotation ${candidate + 1}`}
              >
                <RotateCw aria-hidden="true" />
                {candidate + 1}
              </button>
            ))}
          </div>
        </div>
      )}

      {selectedCoord && rotation?.may_skip_wildlife && (
        <button className="skip-wildlife" type="button" onClick={onSkipWildlife}>
          Skip wildlife placement
        </button>
      )}

      <div className="turn-checklist">
        <TurnStep complete={draft !== null} current={step === "draft"} label="Market draft" />
        <TurnStep
          complete={selectedCoord !== null}
          current={step === "habitat"}
          label="Habitat placement"
        />
        <TurnStep
          complete={
            selectedCoord !== null && wildlifeDecision !== "pending"
          }
          current={step === "wildlife"}
          label="Wildlife placement"
        />
      </div>

      <button
        className="primary-command"
        type="button"
        disabled={!canCommit || busy}
        onClick={onCommit}
      >
        <Check aria-hidden="true" />
        {busy ? "Applying turn" : "Complete turn"}
      </button>

      {draft && (
        <p className="selection-summary">
          {independent ? "Independent draft" : "Paired draft"} selected
          {selectedCoord
            ? ` at q ${selectedCoord.q}, r ${selectedCoord.r}`
            : " · placement pending"}
        </p>
      )}
    </aside>
  );
}

function TurnStep({
  complete,
  current,
  label,
}: {
  complete: boolean;
  current: boolean;
  label: string;
}) {
  return (
    <div className={`${complete ? "is-complete" : ""} ${current ? "is-current" : ""}`}>
      <span>{complete ? <Check aria-hidden="true" /> : null}</span>
      {titleCase(label)}
    </div>
  );
}
