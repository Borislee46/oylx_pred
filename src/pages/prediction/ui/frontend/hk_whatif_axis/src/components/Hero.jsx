import { motion } from "framer-motion";
import { useAnimatedNumber } from "../hooks.js";

const TIER_COLOR = { 稳: "#34d399", 偏稳: "#fbbf24", 冲: "#fb7185" };
const TIER_LABEL = { 稳: "保底", 偏稳: "适中", 冲: "冲刺" };
const TIER_RANGE = { 稳: "≥55%", 偏稳: "40–55%", 冲: "<40%" };

export default function Hero({ school, variant, baseVariant, basePct }) {
  const pct = useAnimatedNumber(school ? school.p * 100 : 0);
  if (!school) return null;
  const tierColor = TIER_COLOR[school.tier] ?? "#94a3b8";
  const tierLabel = TIER_LABEL[school.tier] ?? school.tier;
  const tierRange = TIER_RANGE[school.tier] ?? "";
  const compareOn = baseVariant && variant && baseVariant.key !== variant.key;
  const delta =
    compareOn && basePct != null ? Math.round(school.p * 100 - basePct) : null;

  return (
    <motion.div
      className="hero"
      key={variant?.label ?? ""}
      initial={{ opacity: 0.6, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="hero-eyebrow">
        <span className="hero-dot" style={{ background: school.color }} />
        当前档位 {variant?.label ?? ""} · 把握最大的院校
      </div>
      <div className="hero-main">
        <div className="hero-num">
          <span className="hero-val">{Math.round(pct)}</span>
          <span className="hero-unit">%</span>
        </div>
        <div className="hero-info">
          <div className="hero-school">{school.name}</div>
          <div className="hero-band">
            <span style={{ color: tierColor }}>{tierLabel}</span>
            {tierRange ? <span className="hero-range">区间 {tierRange}</span> : null}
            {school.n ? <span>同方向历史样本 n={school.n}</span> : null}
            {compareOn && delta != null ? (
              <span className="hero-delta">
                vs {baseVariant.label} {delta >= 0 ? "+" : ""}
                {delta}pp
              </span>
            ) : null}
          </div>
        </div>
      </div>
      <div className="hero-bar">
        <motion.div
          className="hero-bar-caret"
          animate={{ left: `${school.p * 100}%` }}
          transition={{ type: "spring", stiffness: 110, damping: 18 }}
          style={{ background: school.color }}
        />
      </div>
    </motion.div>
  );
}
