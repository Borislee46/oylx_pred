import { motion } from "framer-motion";

export default function Chips({ variants, activeId, pendingId, onSelect, onCompare }) {
  return (
    <div className="chips-row">
      <span className="chips-label">我的 GPA 档位</span>
      <div className="chips">
        {variants.map((v) => {
          const active = v.key === activeId;
          const pending = v.key === pendingId;
          const notReady = !v.ready;
          return (
            <motion.button
              key={v.key}
              className={`chip ${active ? "chip-active" : ""} ${notReady ? "chip-notready" : ""} ${pending ? "chip-pending" : ""}`}
              onClick={() => onSelect(v.key)}
              whileTap={{ scale: 0.96 }}
              layout
            >
              {v.label}
              {pending ? (
                <span className="chip-spin" />
              ) : v.best_pct != null ? (
                <span className="chip-pct">最优 {v.best_pct}%</span>
              ) : (
                <span className="chip-pct chip-pct-muted">待重算</span>
              )}
            </motion.button>
          );
        })}
      </div>
      <label className="compare-toggle">
        <input
          type="checkbox"
          defaultChecked
          onChange={(e) => onCompare?.(e.target.checked)}
        />
        对比上一个 GPA 档
      </label>
    </div>
  );
}
