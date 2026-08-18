import { motion } from "framer-motion";

const TIER_LABEL = { 稳: "保底", 偏稳: "适中", 冲: "冲刺" };

export default function Portfolio({
  schools,
  cells,
  baseCells,
  compareOn,
  hoverId,
  onHover,
  bandSummary,
}) {
  const sorted = [...schools].sort(
    (a, b) => (cells[b.name]?.p ?? 0) - (cells[a.name]?.p ?? 0),
  );

  return (
    <div className="portfolio">
      <div className="portfolio-head">
        <span className="portfolio-title">院校组合清单</span>
        {bandSummary ? <span className="portfolio-bands">{bandSummary}</span> : null}
        <span className="portfolio-hint">按把握从高到低 · 悬停/点击查看模型拆解</span>
      </div>
      <div className="portfolio-rows">
        {sorted.map((s, i) => {
          const c = cells[s.name];
          if (!c) return null;
          const pct = c.p * 100;
          const base = compareOn ? baseCells[s.name] : null;
          const delta = base ? (c.p - base.p) * 100 : null;
          const active = hoverId === s.name;
          return (
            <motion.button
              key={s.name}
              type="button"
              layout
              className={`prow ${active ? "prow-active" : ""}`}
              onMouseEnter={() => onHover?.(s.name)}
              onMouseLeave={() => onHover?.(null)}
              onFocus={() => onHover?.(s.name)}
              onBlur={() => onHover?.(null)}
              onClick={() => onHover?.(s.name)}
            >
              <span className="prow-rank">{i + 1}</span>
              <span className="prow-dot" style={{ background: s.color }} />
              <span className="prow-name" title={s.name}>
                {s.name}
              </span>
              <span className="prow-bar">
                <i
                  style={{
                    width: `${pct}%`,
                    background: s.color,
                  }}
                />
              </span>
              <span className="prow-pct">{Math.round(pct)}%</span>
              {compareOn ? (
                <span className={`prow-delta ${delta == null ? "na" : delta >= 0 ? "up" : "down"}`}>
                  {delta == null ? "—" : `${delta >= 0 ? "+" : ""}${Math.round(delta)}pp`}
                </span>
              ) : null}
              <span className={`prow-tier tier-${c.tier}`}>
                {TIER_LABEL[c.tier] ?? c.tier}
              </span>
              <span className={`prow-sample ${c.sample_ok ? "ok" : "low"}`}>
                {c.sample_ok
                  ? `样本 n=${c.n ?? 0}`
                  : `样本偏少 n=${c.n ?? 0} · 已收缩`}
              </span>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
