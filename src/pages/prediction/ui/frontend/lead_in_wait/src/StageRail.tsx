import { motion } from "framer-motion";

type Props = {
  index: number;
  count: number;
  accent?: string;
  /** Stop the flowing "in progress" shimmer (completion / cancelled). */
  settled?: boolean;
  /** Task finished — full, solid progress bar. */
  complete?: boolean;
};

export function StageRail({
  index,
  count,
  accent = "#21fff4",
  settled = false,
  complete = false,
}: Props) {
  const n = Math.max(1, count);
  const idx = Math.min(Math.max(0, index), n - 1);
  const fillClass = complete
    ? "rail-fill rail-fill-complete"
    : settled
      ? "rail-fill rail-fill-static"
      : "rail-fill";

  return (
    <div className="rail" role="progressbar" aria-valuenow={idx + 1} aria-valuemin={1} aria-valuemax={n}>
      <div className="rail-track">
        <div className={fillClass} />
      </div>
      <div className="rail-dots">
        {Array.from({ length: n }, (_, i) => {
          const state = i < idx ? "done" : i === idx ? "cur" : "idle";
          return (
            <motion.span
              key={i}
              className={`rail-dot ${state}`}
              initial={false}
              animate={
                state === "cur"
                  ? { scale: 1.15, boxShadow: `0 0 8px ${accent}` }
                  : { scale: state === "done" ? 1 : 0.85, boxShadow: "0 0 0px transparent" }
              }
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
            />
          );
        })}
      </div>
    </div>
  );
}
