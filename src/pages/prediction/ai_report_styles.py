REPORT_STYLE = """<style>
.ar-card {
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.92));
  box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
  padding: 1rem 1.1rem 0.95rem;
  margin-bottom: 0;
}
.ar-ai-card { margin-top: 0.55rem; }
.ar-ai-card.is-streaming { min-height: 96px; }
.ar-ai-card.is-pinned { position: sticky; top: 0.75rem; z-index: 20; }
.ar-grid { display: grid; grid-template-columns: minmax(128px, 0.9fr) minmax(0, 2.1fr); gap: 1.1rem; }
.ar-score-panel {
  text-align: center;
  border-radius: 15px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(226, 232, 240, 0.85);
  padding: 0.75rem 0.6rem;
}
.ar-main-panel { min-width: 0; }

.ar-ring { width: 104px; height: 104px; margin: 0.15rem auto 0; position: relative; }
.ar-ring svg { transform: rotate(-90deg); filter: drop-shadow(0 7px 12px rgba(15,23,42,0.08)); }
.ar-ring-bg { fill: none; stroke: var(--hk-slate-100); stroke-width: 3.5; }
.ar-ring-fill {
  fill: none;
  stroke-width: 3.8;
  stroke-linecap: round;
  transition: stroke-dashoffset 1s cubic-bezier(0.34,1.56,0.64,1);
}
.ar-ring-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.ar-ring-score {
  font-family: var(--hk-font-display);
  font-size: 1.45rem;
  font-weight: 800;
  line-height: 1;
  color: var(--hk-slate-900);
}
.ar-ring-label {
  font-size: 0.58rem;
  font-weight: 600;
  color: var(--hk-slate-400);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 2px;
}
.ar-profile-line {
  display: inline-flex;
  gap: 0.35rem;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 0.45rem;
}
.ar-profile-pill {
  border-radius: 999px;
  background: var(--hk-slate-50);
  color: var(--hk-slate-600);
  font-size: 0.7rem;
  font-weight: 650;
  padding: 0.16rem 0.45rem;
}

.ar-bar-row { display: flex; align-items: center; gap: 0.45rem; margin: 0.18rem 0; }
.ar-bar-label { width: 36px; font-size: 0.7rem; font-weight: 700; color: var(--hk-slate-600); text-align: right; flex-shrink: 0; }
.ar-bar-track { flex: 1; height: 6px; background: var(--hk-slate-100); border-radius: 999px; overflow: hidden; }
.ar-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1); }
.ar-bar-count { font-size: 0.68rem; color: var(--hk-slate-500); min-width: 24px; text-align: left; font-weight: 600; }

.ar-product-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.42rem; }
.ar-product {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.42rem;
  padding: 0.45rem 0.5rem;
  border: 1px solid rgba(226,232,240,0.9);
  border-radius: 12px;
  background: rgba(255,255,255,0.78);
  align-items: start;
}
.ar-product-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 0.34rem; flex-shrink: 0; }
.ar-product-name { display: block; font-size: 0.76rem; font-weight: 750; color: var(--hk-slate-850, var(--hk-slate-800)); line-height: 1.2; }
.ar-product-meta { display: block; font-size: 0.64rem; color: var(--hk-slate-400); line-height: 1.35; margin-top: 0.12rem; }
.ar-product-price { color: var(--hk-slate-700); font-weight: 700; }

.ar-divider { height: 1px; background: var(--hk-slate-100); margin: 0 0 0.55rem; border: none; }
.ar-section-label {
  font-size: 0.64rem;
  font-weight: 700;
  color: var(--hk-slate-350, var(--hk-slate-400));
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.26rem;
}
.ar-overview {
  font-size: 0.83rem;
  color: var(--hk-slate-600);
  line-height: 1.68;
  letter-spacing: 0.01em;
  margin: 0.18rem 0;
}
.ar-muted { color: var(--hk-slate-400); }
.ar-insight-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.55rem; margin-top: 0.35rem; }
.ar-insight-card { border-radius: 13px; padding: 0.52rem 0.62rem; border: 1px solid; }
.ar-insight-card.is-strength { background: rgba(16,185,129,0.055); border-color: rgba(16,185,129,0.13); }
.ar-insight-card.is-concern { background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.14); }
.ar-list { margin: 0; padding-left: 0.9rem; }
.ar-list li {
  font-size: 0.78rem;
  color: var(--hk-slate-600);
  line-height: 1.55;
  letter-spacing: 0.005em;
  margin-bottom: 0.12rem;
  animation: ar-fade-in 0.4s ease-out both;
}
.ar-list li:nth-child(1) { animation-delay: 0.05s; }
.ar-list li:nth-child(2) { animation-delay: 0.18s; }
.ar-list li:nth-child(3) { animation-delay: 0.31s; }
.ar-list li:nth-child(4) { animation-delay: 0.44s; }
.ar-list li::marker { color: var(--hk-cyan); }

@keyframes ar-blink { 50% { opacity: 0; } }
@keyframes ar-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.ar-streaming { animation: none; }
.ar-streaming::after { content: "|"; animation: ar-blink 0.8s infinite; color: var(--hk-cyan); font-weight: 700; }
.ar-reveal { animation: ar-fade-in 0.5s ease-out both; }

@media (max-width: 760px) {
  .ar-card { padding: 0.85rem; }
  .ar-grid { grid-template-columns: 1fr; gap: 0.75rem; }
  .ar-product-grid, .ar-insight-grid { grid-template-columns: 1fr; }
}
</style>"""
