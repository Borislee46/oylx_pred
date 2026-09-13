import { useState } from "react";
import { motion } from "framer-motion";

const SPRING = { type: "spring", stiffness: 110, damping: 17, mass: 0.9 };
const TIER_COLOR = { 稳: "#22c55e", 偏稳: "#f59e0b", 冲: "#f43f5e" };
const TIER_LABEL = { 稳: "保底", 偏稳: "适中", 冲: "冲刺" };
const ZONES = [
  { label: "冲刺", from: 0, to: 40, color: "rgba(244, 63, 94, 0.09)", line: "#f43f5e" },
  { label: "适中", from: 40, to: 55, color: "rgba(245, 158, 11, 0.08)", line: "#f59e0b" },
  { label: "保底", from: 55, to: 100, color: "rgba(34, 197, 94, 0.09)", line: "#22c55e" },
];

function SchoolMarker({ school, cell, onHover, onTip }) {
  const pct = cell.p * 100;
  const tierColor = TIER_COLOR[cell.tier] ?? "#94a3b8";
  return (
    <motion.div
      className="mk-dot"
      animate={{ left: `${pct}%` }}
      transition={SPRING}
      style={{ background: school.color, borderColor: tierColor }}
      onMouseEnter={() => {
        onHover?.(school.name);
        onTip?.({ school, cell });
      }}
      onMouseLeave={() => {
        onHover?.(null);
        onTip?.(null);
      }}
      aria-label={`${school.name} ${Math.round(pct)}%`}
      role="img"
    />
  );
}

export default function Axis({ grid, variantId, prevVariantId, schools, onHover, awaiting, awaitingLabel }) {
  const [tip, setTip] = useState(null);
  const cell = grid[variantId] ?? {};
  const prevCells = prevVariantId ? grid[prevVariantId] : null;
  const sorted = [...schools].sort((a, b) => (cell[b.name]?.p ?? 0) - (cell[a.name]?.p ?? 0));
  const tipPct = tip ? tip.cell.p * 100 : 0;
  const tipLeft = Math.min(88, Math.max(12, tipPct));

  return (
    <div className="axis-wrap">
      <div className="axis">
        <div className="axis-track" />
        {awaiting && (
          <div className="axis-await">
            <span className="axis-await-spin" />
            正在用完整模型重算 {awaitingLabel}…
          </div>
        )}
        {ZONES.map((z) => (
          <div
            key={z.label}
            className="axis-zone"
            style={{
              left: `${z.from}%`,
              width: `${z.to - z.from}%`,
              background: z.color,
              borderLeft: z.from === 0 ? "none" : `1px solid ${z.line}`,
            }}
          >
            <span className="axis-zone-label">{z.label}</span>
          </div>
        ))}
        <div className="axis-grid">
          {[0, 20, 40, 60, 80, 100].map((t) => (
            <span key={t} className="axis-tick" style={{ left: `${t}%` }}>
              {t}%
            </span>
          ))}
        </div>

        {prevCells && (
          <div className="mk-ghost-layer">
            {sorted.map((s) => {
              const prev = prevCells[s.name];
              if (!prev) return null;
              return (
                <div
                  key={s.name}
                  className="mk-dot mk-dot-ghost"
                  style={{ left: `${prev.p * 100}%`, borderColor: s.color }}
                />
              );
            })}
          </div>
        )}

        {sorted.map((s, i) => {
          const c = cell[s.name];
          if (!c) return null;
          return (
            <div key={s.name} className="mk-layer" style={{ zIndex: sorted.length - i }}>
              <SchoolMarker
                school={s}
                cell={c}
                onHover={onHover}
                onTip={setTip}
              />
            </div>
          );
        })}

        {tip && (
          <div className="mk-tip" style={{ left: `${tipLeft}%` }}>
            <span className="mk-tip-name" style={{ color: tip.school.color }}>
              {tip.school.name}
            </span>
            <span className="mk-tip-pct">{Math.round(tipPct)}%</span>
            <span className={`mk-tip-tier tier-${tip.cell.tier}`}>
              {TIER_LABEL[tip.cell.tier] ?? tip.cell.tier}
            </span>
            {tip.cell.n ? <span className="mk-tip-n">样本 n={tip.cell.n}</span> : null}
          </div>
        )}
      </div>

      <div className="axis-legend">
        <span className="legend-item">
          <i style={{ background: TIER_COLOR.稳 }} />
          保底 ≥55%
        </span>
        <span className="legend-item">
          <i style={{ background: TIER_COLOR.偏稳 }} />
          适中 40–55%
        </span>
        <span className="legend-item">
          <i style={{ background: TIER_COLOR.冲 }} />
          冲刺 &lt;40%
        </span>
        <span className="legend-note">虚线圆 = 上一个 GPA 档的位置</span>
      </div>
    </div>
  );
}
