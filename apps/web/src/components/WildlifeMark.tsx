import type { Wildlife } from "../types";

const LABELS: Record<Wildlife, string> = {
  bear: "🐻",
  elk: "🫎",
  salmon: "🐟",
  hawk: "🦅",
  fox: "🦊",
};

export function WildlifeMark({
  wildlife,
  size = "medium",
}: {
  wildlife: Wildlife;
  size?: "small" | "medium" | "large";
}) {
  return (
    <span
      className={`wildlife-mark wildlife-${wildlife} wildlife-mark-${size}`}
      aria-label={wildlife}
      title={wildlife}
    >
      {LABELS[wildlife]}
    </span>
  );
}
