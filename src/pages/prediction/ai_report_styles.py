REPORT_STYLE = """<style>
/* ── Card foundation ─────────────────────────────────────────── */
.ar-card {
  font-family: var(--hk-font-sans);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.92));
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05), 0 2px 8px rgba(15, 23, 42, 0.03);
  padding: 1.1rem 1.2rem 1.05rem;
  margin-bottom: 0;
}
.ar-ai-card {
  margin-top: 0.6rem;
  padding: 1rem 1.15rem 0.95rem;
}
.ar-ai-card.is-streaming { min-height: 100px; }
.ar-ai-card.is-pinned { position: sticky; top: 0.75rem; z-index: 20; }

/* ── 2-col grid for static frame ────────────────────────────── */
.ar-grid {
  display: grid;
  grid-template-columns: minmax(178px, 1fr) minmax(0, 2fr);
  gap: 1.15rem;
}
.ar-score-panel {
  text-align: center;
  border-radius: 15px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(226, 232, 240, 0.85);
  padding: 0.8rem 0.65rem;
}
.ar-main-panel { min-width: 0; }

/* ── Radar chart ────────────────────────────────────────────── */
.ar-radar-wrap { width: 175px; height: 175px; margin: 0.15rem auto 0; }
.ar-radar-svg { width: 175px; height: 175px; display: block; }
.ar-radar-svg text { font-family: var(--hk-font-sans), "Microsoft YaHei", sans-serif; }
.ar-radar-grid { fill: none; stroke: var(--hk-slate-150); stroke-width: 0.8; }
.ar-radar-axis { stroke: var(--hk-slate-200); stroke-width: 0.6; }
.ar-radar-fill { fill: rgba(6, 182, 212, 0.10); stroke: #06b6d4; stroke-width: 1.3; }
.ar-radar-dot { fill: #06b6d4; stroke: #fff; stroke-width: 1.2; }
.ar-radar-label { font-size: 11px; fill: var(--hk-slate-450); }

/* ── Profile tags ───────────────────────────────────────────── */
.ar-tag-row {
  display: flex; gap: 0.3rem; flex-wrap: wrap;
  justify-content: center; margin-top: 0.3rem;
}
.ar-tag {
  font-size: 0.625rem; font-weight: 600;
  padding: 0.1rem 0.45rem; border-radius: 999px; line-height: 1.5;
}
.ar-tag-good { background: rgba(16,185,129,0.10); color: #059669; }
.ar-tag-weak { background: rgba(245,158,11,0.10); color: #b45309; }

/* ── Ring (match score) ─────────────────────────────────────── */
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
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}
.ar-ring-score {
  font-family: var(--hk-font-display);
  font-size: 1.125rem; font-weight: 700; line-height: 1;
  color: var(--hk-slate-900);
}
.ar-ring-label {
  font-size: 0.625rem; font-weight: 600;
  color: var(--hk-slate-300);
  letter-spacing: 0.08em; text-transform: uppercase;
  margin-top: 2px;
}

/* ── Profile pills ──────────────────────────────────────────── */
.ar-profile-line {
  display: inline-flex; gap: 0.35rem;
  align-items: center; justify-content: center;
  flex-wrap: wrap; margin-top: 0.45rem;
}
.ar-profile-pill {
  border-radius: 999px;
  background: var(--hk-slate-50);
  color: var(--hk-slate-500);
  font-size: 0.625rem; font-weight: 600;
  padding: 0.16rem 0.45rem;
}

/* ── Bar chart (tier distribution) ──────────────────────────── */
.ar-bar-row { display: flex; align-items: center; gap: 0.45rem; margin: 0.18rem 0; }
.ar-bar-label { width: 36px; font-size: 0.625rem; font-weight: 700; color: var(--hk-slate-500); text-align: right; flex-shrink: 0; }
.ar-bar-track { flex: 1; height: 6px; background: var(--hk-slate-100); border-radius: 999px; overflow: hidden; }
.ar-bar-fill { height: 100%; border-radius: 999px; transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1); }
.ar-bar-count { font-size: 0.625rem; color: var(--hk-slate-400); min-width: 24px; text-align: left; font-weight: 600; }

/* ── Product grid ───────────────────────────────────────────── */
.ar-product-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.42rem; }
.ar-product {
  display: grid; grid-template-columns: auto 1fr; gap: 0.42rem;
  padding: 0.45rem 0.5rem;
  border: 1px solid rgba(226,232,240,0.9);
  border-radius: 12px;
  background: rgba(255,255,255,0.78);
  align-items: start;
}
.ar-product-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 0.34rem; flex-shrink: 0; }
.ar-product-name { display: block; font-size: 0.75rem; font-weight: 700; color: var(--hk-slate-700); line-height: 1.2; }
.ar-product-meta { display: block; font-size: 0.625rem; color: var(--hk-slate-400); line-height: 1.35; margin-top: 0.12rem; }
.ar-product-price { color: var(--hk-slate-700); font-weight: 700; }

/* ── Tier note (distribution annotation) ────────────────────── */
.ar-tier-note {
  font-size: 0.625rem; color: var(--hk-slate-400);
  text-align: right; margin-top: 0.2rem; font-style: italic;
}

/* ── Divider ────────────────────────────────────────────────── */
.ar-divider { height: 1px; background: var(--hk-slate-100); margin: 0 0 0.6rem; border: none; }

/* ── Section label ──────────────────────────────────────────── */
.ar-section-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--hk-slate-600);
  margin-bottom: 0.4rem;
  margin-top: 0.45rem;
  line-height: 1.3;
}
.ar-section-label:first-child { margin-top: 0; }
/* main title */
.ar-score-panel .ar-section-label {
  font-size: 0.85rem;
  text-align: center;
}

/* ── Overview paragraph ─────────────────────────────────────── */
.ar-overview {
  font-size: 0.75rem;
  color: var(--hk-slate-500);
  line-height: 1.7;
  letter-spacing: 0.01em;
  margin: 0.2rem 0;
}
.ar-muted { color: var(--hk-slate-400); }

/* ── Insight grid (strengths / concerns) ────────────────────── */
.ar-insight-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; margin-top: 0.4rem; }
.ar-insight-card {
  border-radius: 13px;
  padding: 0.55rem 0.65rem;
  border: 1px solid;
}
.ar-insight-card.is-strength { background: rgba(16,185,129,0.05); border-color: rgba(16,185,129,0.12); }
.ar-insight-card.is-concern { background: rgba(245,158,11,0.05); border-color: rgba(245,158,11,0.13); }

/* ── List (strengths, concerns, product reasons) ────────────── */
.ar-list { margin: 0; padding-left: 0.95rem; }
.ar-list li {
  font-size: 0.75rem;
  color: var(--hk-slate-500);
  line-height: 1.6;
  letter-spacing: 0.005em;
  margin-bottom: 0.15rem;
  animation: ar-fade-in 0.4s ease-out both;
}
.ar-list li:nth-child(1) { animation-delay: 0.05s; }
.ar-list li:nth-child(2) { animation-delay: 0.18s; }
.ar-list li:nth-child(3) { animation-delay: 0.31s; }
.ar-list li:nth-child(4) { animation-delay: 0.44s; }
.ar-list li::marker { color: var(--hk-cyan); }
.ar-list li strong {
  font-weight: 700;
  color: #0f172a;
  text-decoration: underline;
  text-decoration-color: #06b6d4;
  text-decoration-thickness: 1.5px;
  text-underline-offset: 2px;
}

/* ── Keyframes ──────────────────────────────────────────────── */
@keyframes ar-blink { 50% { opacity: 0; } }
@keyframes ar-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ar-cursor-pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.15; }
}
@keyframes ar-shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes ar-section-reveal {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Streaming indicator ────────────────────────────────────── */
.ar-streaming { animation: none; }

/* ── Processing indicator at end of streaming card ─────────── */
.ar-wait {
  display: inline-block;
  color: var(--hk-slate-400);
  font-style: italic;
  font-size: 0.75rem;
  margin-left: 0.25rem;
  line-height: 1.7;
}

/* ── Section reveal for static/complete render ──────────────── */
.ar-reveal { animation: ar-fade-in 0.5s ease-out both; }
.ar-section-enter {
  animation: ar-section-reveal 0.45s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

/* ── School notes ───────────────────────────────────────────── */
.ar-school-notes { margin-top: 0.5rem; }
.ar-school-note {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.4rem;
  padding: 0.35rem 0.45rem;
  margin: 0.14rem 0;
  border-radius: 10px;
  background: rgba(241,245,249,0.55);
  font-size: 0.75rem;
  line-height: 1.55;
}
.ar-school-note-uni { font-weight: 700; color: var(--hk-slate-700); white-space: nowrap; }
.ar-school-note-text { color: var(--hk-slate-450); }

/* ── Product reasons ────────────────────────────────────────── */
.ar-product-reasons { margin-top: 0.5rem; }

/* ── Comparison cards (deprecated, kept for compatibility) ─── */
.ar-compare-card { margin-top: 0.6rem; padding: 0.65rem 0.8rem; }
.ar-compare-entry { margin: 0.3rem 0; }
.ar-compare-label { font-weight: 700; font-size: 0.75rem; color: var(--hk-slate-700); }
.ar-compare-prob { font-size: 0.625rem; color: var(--hk-slate-450); margin-left: 0.35rem; }
.ar-compare-text { font-size: 0.75rem; color: var(--hk-slate-400); line-height: 1.55; }

/* ── Unified School Cards ──────────────────────────────────── */
.usc-section { margin-top: 0.5rem; }
.usc-cards-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.4rem;
}
.usc-card {
  border-radius: 12px;
  background: rgba(248,250,252,0.85);
  border: 1px solid rgba(226,232,240,0.9);
  padding: 0.5rem 0.6rem;
  margin: 0;
}
.usc-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.3rem;
}
.usc-uni {
  font-weight: 700; font-size: 0.875rem; color: var(--hk-slate-700);
  line-height: 1.3;
}
.usc-prob-badge {
  font-size: 0.625rem; font-weight: 700;
  padding: 0.06rem 0.45rem; border-radius: 999px;
  line-height: 1.5; white-space: nowrap; flex-shrink: 0;
}
.usc-prob-high { background: rgba(16,185,129,0.12); color: #059669; }
.usc-prob-mid  { background: rgba(245,158,11,0.12); color: #b45309; }
.usc-prob-low  { background: rgba(239,68,68,0.10);  color: #dc2626; }

.usc-prob-track {
  height: 4px; background: var(--hk-slate-100);
  border-radius: 999px; overflow: hidden; margin-bottom: 0.45rem;
}
.usc-prob-fill {
  height: 100%; border-radius: 999px;
  transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
}

.usc-note {
  font-size: 0.75rem; line-height: 1.6;
  color: var(--hk-slate-500); margin-bottom: 0.4rem;
}

.usc-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.3rem;
}
.usc-stat {
  border-radius: 8px;
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(226,232,240,0.85);
  padding: 0.3rem 0.45rem;
}
.usc-stat-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.18rem;
}
.usc-stat-name {
  font-size: 0.625rem; font-weight: 600; color: var(--hk-slate-400);
}
.usc-stat-pct {
  font-size: 0.625rem; font-weight: 700; color: var(--hk-slate-500);
}
.usc-stat-bar {
  height: 3px; background: var(--hk-slate-100);
  border-radius: 999px; overflow: hidden; margin-bottom: 0.12rem;
}
.usc-stat-fill {
  height: 100%; border-radius: 999px;
  transition: width 0.4s ease-out;
}
.usc-stat-value {
  font-size: 0.625rem; color: var(--hk-slate-500); font-weight: 600;
}
.usc-no-data {
  font-size: 0.625rem; color: var(--hk-slate-400);
  font-style: italic; margin-top: 0.15rem;
}
.usc-samples {
  font-size: 0.625rem; color: var(--hk-slate-400);
  margin-top: 0.25rem; text-align: right;
}

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 760px) {
  .ar-card { padding: 0.9rem; }
  .ar-grid { grid-template-columns: 1fr; gap: 0.75rem; }
  .ar-product-grid, .ar-insight-grid, .usc-cards-grid { grid-template-columns: 1fr; }
}
</style>"""
