import type { Wildlife } from "../types";

const LABELS: Record<Wildlife, string> = {
  bear: "BR",
  elk: "EL",
  salmon: "SA",
  hawk: "HK",
  fox: "FX",
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
