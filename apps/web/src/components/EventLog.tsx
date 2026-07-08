import type { TurnAction } from "../types";

/** Legacy-v1 style inline event log: one compact line per completed turn,
 * newest first. Derived entirely from the replay, so undo/redo stay
 * consistent for free. */
interface EventLogProps {
  turns: TurnAction[];
  playerCount: number;
}

function describe(action: TurnAction): string {
  const slot =
    "Paired" in action.draft
      ? `pair ${action.draft.Paired.slot + 1}`
      : `tile ${action.draft.Independent.tile_slot + 1} + token ${action.draft.Independent.wildlife_slot + 1}`;
  const coord = `(${action.tile.coord.q},${action.tile.coord.r})`;
  const wildlife = action.wildlife
    ? ` · token (${action.wildlife.q},${action.wildlife.r})`
    : " · token skipped";
  const prelude = action.replace_three_of_a_kind
    ? " · refreshed market"
    : action.wildlife_wipes.length > 0
      ? " · wiped market"
      : "";
  return `${slot} → ${coord}${wildlife}${prelude}`;
}

export function EventLog({ turns, playerCount }: EventLogProps) {
  if (turns.length === 0) {
    return <div className="event-log empty">No moves yet.</div>;
  }
  return (
    <div className="event-log">
      {turns
        .map((action, index) => ({
          action,
          index,
          player: index % playerCount,
        }))
        .reverse()
        .slice(0, 40)
        .map(({ action, index, player }) => (
          <div key={index} className="event-log-row">
            <span className={`event-player seat-${player}`}>P{player + 1}</span>
            <span className="event-text">{describe(action)}</span>
          </div>
        ))}
    </div>
  );
}
