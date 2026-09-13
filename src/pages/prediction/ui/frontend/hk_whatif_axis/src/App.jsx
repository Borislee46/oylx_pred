import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { onRender, post, setFrameHeight } from "./streamlit.js";
import Hero from "./components/Hero.jsx";
import Axis from "./components/Axis.jsx";
import Chips from "./components/Chips.jsx";
import Portfolio from "./components/Portfolio.jsx";
import CurvePanel from "./components/CurvePanel.jsx";

const TIER_LABEL = { 稳: "保底", 偏稳: "适中", 冲: "冲刺" };

function formatCorpus(n) {
  if (!n || n <= 0) return "真实申请样本";
  return n >= 10000 ? `${(n / 10000).toFixed(0)} 万+ 条真实申请样本` : `${n} 条真实申请样本`;
}

export default function App() {
  const [args, setArgs] = useState(null);
  const [variantId, setVariantId] = useState(null);
  const [prevVariantId, setPrevVariantId] = useState(null);
  const [showCompare, setShowCompare] = useState(true);
  const [hoverId, setHoverId] = useState(null);
  const [pendingKey, setPendingKey] = useState(null);
  const lastArgsRef = useRef(null);

  useEffect(() => {
    const off = onRender((next) => {
      const prev = lastArgsRef.current;
      lastArgsRef.current = next;
      if (JSON.stringify(prev) !== JSON.stringify(next)) {
        setArgs(next);
        setPendingKey((pk) => {
          if (pk && next.grid?.[pk]) {
            setVariantId(pk);
            setPrevVariantId(next.base_key ?? null);
            return null;
          }
          return pk;
        });
        setVariantId((cur) => {
          if (cur && next.grid?.[cur]) return cur;
          return next.base_key ?? null;
        });
        setPrevVariantId(null);
      }
    });
    return () => off();
  }, []);

  const grid = args?.grid ?? {};
  const tiers = args?.tiers ?? [];
  const schools = args?.schools ?? [];
  const baseKey = args?.base_key ?? (tiers[0]?.key ?? null);
  const active = variantId && grid[variantId] ? variantId : baseKey;
  const awaiting = pendingKey && !grid[pendingKey];

  const variant = tiers.find((t) => t.key === active) ?? tiers[0];
  const baseVariant = tiers.find((t) => t.key === baseKey) ?? variant;
  const cells = grid[active] ?? {};
  const baseCells = grid[baseKey] ?? {};
  const top = useMemo(() => {
    if (!schools.length) return null;
    const sorted = [...schools].sort((a, b) => (cells[b.name]?.p ?? 0) - (cells[a.name]?.p ?? 0));
    const s = sorted[0];
    return s ? { ...s, ...cells[s.name] } : null;
  }, [schools, cells]);
  const corpusN = args?.corpus_n ?? 0;
  const bands = variant?.bands ?? {};
  const bandSummary = ["稳", "偏稳", "冲"]
    .filter((k) => (bands[k] ?? 0) > 0)
    .map((k) => `${TIER_LABEL[k]} ${bands[k]}`)
    .join(" · ");

  const selectVariant = (id) => {
    if (id === active) return;
    if (grid[id]) {
      setPrevVariantId(active);
      setVariantId(id);
    } else {
      setPendingKey(id);
    }
    post("streamlit:setComponentValue", { value: id });
  };

  const ready = Boolean(args && grid && tiers.length && schools.length);
  setFrameHeight(ready ? 760 : 120);

  return (
    <div className="app">
      <div className="bg-glow" />
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">S</span>
          SIGNALS · 港校录取概率推演
        </div>
        <div className="topbar-right">
          <span className="status-dot" />
          {ready ? "预测已就绪 · 档位切换无需重新提交资料" : "正在读取历史申请样本…"}
        </div>
      </header>

      {!ready ? (
        <div className="boot">
          <motion.span
            className="boot-dot"
            animate={{ opacity: [0.25, 1, 0.25], scale: [0.9, 1.1, 0.9] }}
            transition={{ repeat: Infinity, duration: 1.4 }}
          />
          正在读取历史申请样本，装配你的预测…
        </div>
      ) : (
        <main className="main">
          <section className="stage">
            <div className="card-title">
              <h1>GPA 每提高 0.2，你的把握怎么变？</h1>
              <p>
                把 GPA 换成每个档位后，由同一套统计模型完整重算每所学校的录取概率
                ——不是大模型凭感觉估算。
              </p>
            </div>
            <div className="fact-strip">
              <span className="fact-item">
                <i className="fact-ico fact-ico-data" />
                基于 {formatCorpus(corpusN)}
              </span>
              <span className="fact-item">
                <i className="fact-ico fact-ico-recompute" />
                每个档位完整重算 · 相同输入结果可复现
              </span>
              <span className="fact-item">
                <i className="fact-ico fact-ico-llm" />
                LLM 仅核验经历含金量，不产生概率
              </span>
            </div>
            <Hero
              school={top}
              variant={variant}
              baseVariant={baseVariant}
              basePct={
                baseKey !== active && top
                  ? Math.round((baseCells[top.name]?.p ?? 0) * 100)
                  : null
              }
            />
            <Chips
              variants={tiers}
              activeId={active}
              pendingId={awaiting ? pendingKey : null}
              onSelect={selectVariant}
              onCompare={setShowCompare}
            />
            <Axis
              grid={grid}
              variantId={active}
              prevVariantId={showCompare ? prevVariantId : null}
              schools={schools}
              onHover={setHoverId}
              awaiting={awaiting}
              awaitingLabel={tiers.find((t) => t.key === pendingKey)?.label ?? ""}
            />
            <Portfolio
              schools={schools}
              cells={cells}
              baseCells={baseCells}
              compareOn={Boolean(baseKey && baseKey !== active)}
              hoverId={hoverId}
              onHover={setHoverId}
              bandSummary={bandSummary}
            />
          </section>

          <aside className="side">
            <AnimatePresence mode="wait">
              <CurvePanel
                key={hoverId ?? top?.name ?? "none"}
                schoolName={hoverId ?? top?.name ?? ""}
                variant={variant}
                baseVariant={baseVariant}
                grid={grid}
                tiers={tiers}
                schools={schools}
              />
            </AnimatePresence>
          </aside>
        </main>
      )}

      <footer className="foot">
        每个数字都来自可复现的统计计算链：相似案例 → 背景调整 → 小样本收缩。
        历史数据估算仅作参考，录取结果受多种因素影响。
      </footer>
    </div>
  );
}
