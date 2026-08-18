import { motion } from "framer-motion";

const X0 = 3.0;
const X1 = 4.0;
const xOf = (g) => ((g - X0) / (X1 - X0)) * 100;
const yOf = (p) => (1 - p) * 88 + 8;
const TIER_COLOR = { 稳: "#22c55e", 偏稳: "#f59e0b", 冲: "#f43f5e" };
const TIER_LABEL = { 稳: "保底", 偏稳: "适中", 冲: "冲刺" };
const TRACE_LABELS = {
  "penalty_GPA Penalty": "GPA 惩罚",
  "penalty_Language Penalty": "语言成绩惩罚",
  "penalty_Language Requirement": "语言要求未达标",
  "penalty_Faculty Out of Scope Penalty": "跨学部惩罚",
  "penalty_Professional Major Penalty": "专业项目实习要求",
  "boost_Text Boost": "经历含金量加成",
  "bayesian_shrinkage": "小样本贝叶斯收缩",
};

function tierOf(p) {
  return p >= 0.55 ? "稳" : p >= 0.4 ? "偏稳" : "冲";
}

function pctOf(v) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

function smoothPath(points) {
  if (points.length < 2) return "";
  // Catmull-Rom → 贝塞尔，保留真实档位点，中间平滑过渡
  const p = [...points];
  let d = `M${p[0].x.toFixed(1)},${p[0].y.toFixed(1)}`;
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[i - 1] ?? p[i];
    const p1 = p[i];
    const p2 = p[i + 1];
    const p3 = p[i + 2] ?? p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return d;
}

export default function CurvePanel({ schoolName, variant, baseVariant, grid, tiers, schools }) {
  const school = schools.find((s) => s.name === schoolName) || schools[0];
  if (!school) return null;

  const points = tiers
    .map((t) => {
      const cell = grid[t.key]?.[school.name];
      return cell ? { gpa: t.gpa, p: cell.p, key: t.key } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.gpa - b.gpa);

  const pts = points.map((pt) => ({ x: xOf(pt.gpa), y: yOf(pt.p) }));
  const d = smoothPath(pts);
  const cur = points.find((pt) => Math.abs(pt.gpa - variant.gpa) < 0.001) ?? points[points.length - 1];
  const dotX = xOf(cur?.gpa ?? variant.gpa);
  const dotY = yOf(cur?.p ?? 0);
  const tier = cur ? TIER_COLOR[tierOf(cur.p)] : "#94a3b8";
  const cell = grid[variant.key]?.[school.name];
  const trace = cell?.trace ?? null;
  const baselineRate = cell?.baseline_rate ?? null;
  const cf = cell?.counterfactuals ?? null;
  const baseCell = baseVariant ? grid[baseVariant.key]?.[school.name] : null;
  const delta = baseCell ? (cur?.p ?? 0) - baseCell.p : null;
  const adjRows = trace
    ? Object.entries(trace)
        .filter(
          ([k, v]) =>
            k !== "base" &&
            k !== "final" &&
            k !== "quality_signals" &&
            Math.abs(v) >= 0.0005,
        )
        .map(([k, v]) => ({ label: TRACE_LABELS[k] ?? k, delta: v }))
    : [];
  const cfRows = cf
    ? [
        { label: "GPA +0.2", p: cf.gpa_up },
        { label: "语言提升", p: cf.lang_up },
        { label: "实习 +1 段", p: cf.intern_up },
      ].filter((r) => r.p != null)
    : [];

  return (
    <motion.div
      className="panel curve-panel"
      key={school.name}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className="panel-head">
        <span className="hero-dot" style={{ background: school.color }} />
        <span>{school.name} · GPA 每档概率</span>
      </div>
      <div className="curve-meta">
        <span>同方向历史样本 n={school.n ?? 0}</span>
        {cur && (
          <span style={{ color: tier }}>
            {TIER_LABEL[tierOf(cur.p)] ?? cur.p}
          </span>
        )}
      </div>
      <svg className="curve-svg" viewBox="0 0 100 104" preserveAspectRatio="none">
        {[0.25, 0.5, 0.75].map((g) => (
          <line
            key={g}
            x1="0"
            y1={yOf(g)}
            x2="100"
            y2={yOf(g)}
            className="curve-gridline"
          />
        ))}
        {d && (
          <path
            d={d}
            fill="none"
            stroke={school.color}
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
        )}
        {points.map((pt) => (
          <circle
            key={pt.key}
            cx={xOf(pt.gpa)}
            cy={yOf(pt.p)}
            r="1.7"
            fill={school.color}
            opacity="0.75"
          />
        ))}
        <circle cx={dotX} cy={dotY} r="3" fill={school.color} className="curve-dot" />
        <line
          x1={dotX}
          y1={dotY}
          x2={dotX}
          y2="104"
          stroke={school.color}
          strokeDasharray="1.5 2"
          strokeWidth="0.5"
          opacity="0.5"
        />
      </svg>
      <div className="curve-axis">
        <span>3.0</span>
        <span>3.5</span>
        <span>4.0</span>
      </div>
      <div className="curve-current">
        GPA {variant.gpa?.toFixed(1)} →{" "}
        <b style={{ color: school.color }}>
          {cur ? Math.round(cur.p * 100) : "—"}%
        </b>
        {delta != null && Math.abs(delta) >= 0.005 ? (
          <span className={`curve-delta ${delta >= 0 ? "up" : "down"}`}>
            vs {baseVariant.label} {delta >= 0 ? "+" : ""}
            {Math.round(delta * 100)}pp
          </span>
        ) : null}
      </div>
      <div className="curve-note">
        曲线只经过 4 个真实重算点（GPA 3.2 / 3.4 / 3.6 / 3.8），中间为平滑连线
      </div>

      {(trace || baselineRate != null) && (
        <div className="breakdown">
          <div className="breakdown-title">模型怎么算出这个数</div>
          {baselineRate != null && (
            <div className="bd-row">
              <span className="bd-label">历史同组合基线录取率</span>
              <span className="bd-val">{pctOf(baselineRate)}</span>
            </div>
          )}
          {trace?.base != null && (
            <div className="bd-row">
              <span className="bd-label">相似案例基础概率（KNN）</span>
              <span className="bd-val">{pctOf(trace.base)}</span>
            </div>
          )}
          {adjRows.map((r) => (
            <div className="bd-row" key={r.label}>
              <span className="bd-label">{r.label}</span>
              <span className={`bd-val bd-delta ${r.delta >= 0 ? "up" : "down"}`}>
                {r.delta >= 0 ? "+" : ""}
                {Math.round(r.delta * 100)}pp
              </span>
            </div>
          ))}
          {trace?.final != null && (
            <div className="bd-row bd-final">
              <span className="bd-label">最终概率</span>
              <span className="bd-val" style={{ color: school.color }}>
                {pctOf(trace.final)}
              </span>
            </div>
          )}
          {cell?.sample_ok === false && (
            <div className="bd-note">该组合历史样本偏少，已做贝叶斯收缩</div>
          )}
          {cfRows.length > 0 && (
            <div className="bd-cf">
              <div className="bd-cf-title">反事实模拟（同一模型重算）</div>
              {cfRows.map((r) => (
                <div className="bd-row" key={r.label}>
                  <span className="bd-label">如果{r.label}</span>
                  <span className="bd-val">{pctOf(r.p)}</span>
                </div>
              ))}
            </div>
          )}
          <div className="bd-foot">审计链来自模型内部计算记录 · 可复核</div>
        </div>
      )}
      {!trace && baselineRate == null && (
        <div className="curve-note">该档位暂无模型审计链（可能来自兜底结果）</div>
      )}
    </motion.div>
  );
}
