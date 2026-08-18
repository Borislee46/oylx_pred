import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { CursorAura } from "./CursorAura";
import { LlmRibbon } from "./LlmRibbon";
import { Mark } from "./Mark";
import { OrbitRing } from "./OrbitRing";
import { Particles } from "./Particles";
import { PERSONAS, asPersona, type Persona } from "./persona";
import { DEMO } from "./demoData";
import { StageRail } from "./StageRail";
import { TagOrbit } from "./TagOrbit";
import { onRender, setFrameHeight, type WaitArgs } from "./streamlit";
import { usePointer } from "./usePointer";
import { loadWaitState, saveWaitState } from "./waitState";
import { WebGLField } from "./WebGLField";
import type { WaitMeta } from "./useTokenStream";

type Args = {
  title: string;
  stage: string;
  subtitle: string;
  hint: string;
  elapsed: number;
  retry: number;
  stage_index: number;
  stage_count: number;
  persona: Persona;
  llm_text: string;
  details: string[];
  tags: string[];
  gpa_norm: number;
  lang_norm: number;
  gpa_label: string;
  lang_label: string;
  sse_url: string;
  sse_port: number;
  sse_run_id: string;
  started_at: number;
  retry_max: number;
  dark: boolean;
  bg_color: string;
};

const DEFAULTS: Args = {
  title: "Signals 正在解读",
  stage: "读背景",
  subtitle: "正在接通顾问引擎…",
  hint: "",
  elapsed: 0,
  retry: 0,
  stage_index: 0,
  stage_count: 4,
  persona: "signal",
  llm_text: "",
  details: [],
  tags: [],
  gpa_norm: 0.7,
  lang_norm: 0.65,
  gpa_label: "",
  lang_label: "",
  sse_url: "",
  sse_port: 0,
  sse_run_id: "",
  started_at: 0,
  retry_max: 3,
  dark: false,
  bg_color: "#0b1220",
};

function parseList(raw: unknown, limit = 8): string[] {
  if (Array.isArray(raw)) return raw.map(String).filter(Boolean).slice(0, limit);
  if (typeof raw === "string" && raw.trim()) {
    try {
      const j = JSON.parse(raw);
      if (Array.isArray(j)) return j.map(String).filter(Boolean).slice(0, limit);
    } catch {
      return raw
        .split(/[,，、|]/)
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, limit);
    }
  }
  return [];
}

function num(v: unknown, fallback: number) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : fallback;
}

function fromStreamlit(next: WaitArgs): Args {
  return {
    title: String(next.title ?? DEFAULTS.title),
    stage: String(next.stage ?? DEFAULTS.stage),
    subtitle: String(next.subtitle ?? ""),
    hint: String(next.hint ?? ""),
    elapsed: Number(next.elapsed ?? 0),
    retry: Number(next.retry ?? 0),
    stage_index: Math.max(0, Number(next.stage_index ?? 0)),
    stage_count: Math.max(1, Number(next.stage_count ?? 4)),
    persona: asPersona(next.persona),
    llm_text: String(next.llm_text ?? next.subtitle ?? ""),
    details: parseList(next.details, 8),
    tags: parseList(next.tags, 6),
    gpa_norm: num(next.gpa_norm, 0.65),
    lang_norm: num(next.lang_norm, 0.65),
    gpa_label: String(next.gpa_label ?? ""),
    lang_label: String(next.lang_label ?? ""),
    sse_url: String(next.sse_url ?? ""),
    sse_port: Math.max(0, Number(next.sse_port ?? 0) || 0),
    sse_run_id: String(next.sse_run_id ?? ""),
    started_at: Number(next.started_at ?? 0) || 0,
    retry_max: Math.max(1, Number(next.retry_max ?? 3) || 3),
    dark: Boolean(next.dark),
    bg_color: String(next.bg_color ?? DEFAULTS.bg_color) || DEFAULTS.bg_color,
  };
}

function shallowEq(a: Args, b: Args) {
  return (
    a.title === b.title &&
    a.stage === b.stage &&
    a.subtitle === b.subtitle &&
    a.hint === b.hint &&
    a.stage_index === b.stage_index &&
    a.stage_count === b.stage_count &&
    a.persona === b.persona &&
    a.llm_text === b.llm_text &&
    a.gpa_norm === b.gpa_norm &&
    a.lang_norm === b.lang_norm &&
    a.gpa_label === b.gpa_label &&
    a.lang_label === b.lang_label &&
    a.sse_url === b.sse_url &&
    a.sse_port === b.sse_port &&
    a.sse_run_id === b.sse_run_id &&
    a.started_at === b.started_at &&
    a.retry_max === b.retry_max &&
    a.dark === b.dark &&
    a.bg_color === b.bg_color &&
    a.retry === b.retry &&
    a.tags.join("\0") === b.tags.join("\0") &&
    a.details.join("\0") === b.details.join("\0")
  );
}

export function App() {
  const reduce = useReducedMotion();
  const [args, setArgs] = useState(DEFAULTS);
  const [demo, setDemo] = useState(false);
  const [burst, setBurst] = useState(0);
  const [runComplete, setRunComplete] = useState(false);
  const [cancelled, setCancelled] = useState(false);
  const [liveMeta, setLiveMeta] = useState<WaitMeta | null>(null);
  const [showAllSteps, setShowAllSteps] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const gotStreamlit = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const islandRef = useRef<HTMLDivElement>(null);
  const pointer = usePointer(islandRef);
  const prevStage = useRef(args.stage_index);
  const maxHeight = useRef(Math.max(360, loadWaitState()?.height || 0));
  const prevRunId = useRef(args.sse_run_id);
  // A new run must start at the base height — the grow-only persistence is
  // per-run and would otherwise leak the previous run's taller iframe.
  if (prevRunId.current !== args.sse_run_id) {
    prevRunId.current = args.sse_run_id;
    maxHeight.current = 360;
  }
  const [nowTs, setNowTs] = useState(() => Date.now());
  const settled = runComplete || cancelled;

  // SSE meta 驱动的最新进度（stage/明细），mount 后由 meta 原地更新。
  const meta = liveMeta;
  const effStage = meta?.stage ?? args.stage;
  const effStageIndex = meta?.stage_index ?? args.stage_index;
  const effStageCount = meta?.stage_count ?? args.stage_count;
  const effDetails = meta?.details ?? args.details;
  const effRetry = meta?.retry ?? args.retry;

  // A new wait run must not inherit the previous run's completion/cancel state.
  useEffect(() => {
    setRunComplete(false);
    setCancelled(false);
    setLiveMeta(null);
    setShowAllSteps(false);
    setStreaming(false);
  }, [args.sse_run_id]);

  useEffect(() => {
    const off = onRender((next) => {
      gotStreamlit.current = true;
      setDemo(false);
      const nextArgs = fromStreamlit(next);
      setArgs((prev) => (shallowEq(prev, nextArgs) ? prev : nextArgs));
    });
    const timer = window.setTimeout(() => {
      if (!gotStreamlit.current) setDemo(true);
    }, 2000);
    return () => {
      off();
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!demo) return;
    let i = 0;
    const t0 = Date.now();
    const apply = () => {
      const s = DEMO[i];
      const sse = `/sse/wait-demo?q=${encodeURIComponent(s.llm)}`;
      const details = DEMO.slice(0, i + 1).map((d) => d.subtitle);
      setArgs({
        ...DEFAULTS,
        persona: s.persona,
        stage: s.stage,
        subtitle: s.subtitle,
        stage_index: s.index,
        llm_text: s.llm,
        details,
        tags: s.tags,
        gpa_norm: s.gpa,
        lang_norm: s.lang,
        gpa_label: s.gpaLabel,
        lang_label: s.langLabel,
        sse_url: sse,
        started_at: t0 / 1000,
      });
      setBurst((b) => b + 1);
    };
    apply();
    const id = window.setInterval(() => {
      i = (i + 1) % DEMO.length;
      apply();
    }, 4200);
    return () => window.clearInterval(id);
  }, [demo]);

  useEffect(() => {
    if (effStageIndex !== prevStage.current) {
      prevStage.current = effStageIndex;
      setBurst((b) => b + 1);
    }
  }, [effStageIndex]);

  useEffect(() => {
    const id = window.setInterval(() => setNowTs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout>;
    const sync = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!rootRef.current) return;
        const h = Math.ceil(rootRef.current.getBoundingClientRect().height);
        // Only grow during wait — shrinking iframe height causes visible jump.
        // Once the run settles (completed / cancelled) the frame may shrink
        // to its true content height again.
        maxHeight.current = settled ? h : Math.max(maxHeight.current, h, 360);
        setFrameHeight(maxHeight.current);
        if (!demo) saveWaitState({ height: maxHeight.current });
      }, 80);
    };
    sync();
    const ro = new ResizeObserver(() => sync());
    ro.observe(el);
    return () => {
      ro.disconnect();
      clearTimeout(timer);
    };
  }, [args.title, args.subtitle, args.hint, effStage, args.llm_text, effDetails.join("|"), args.tags.join("|"), demo, settled]);

  const baseTheme = PERSONAS[args.persona];
  const theme = args.dark
    ? {
        ...baseTheme,
        accent: baseTheme.accentDark ?? baseTheme.accent,
        accent2: baseTheme.accent2Dark ?? baseTheme.accent2,
        glow: baseTheme.glowDark ?? baseTheme.glow,
      }
    : baseTheme;
  const accentText = args.dark ? theme.accent : (theme.accentText ?? theme.accent);
  const n = Math.max(1, effStageCount);
  const idx = Math.min(effStageIndex, n - 1);
  const progress = n <= 1 ? 1 : idx / (n - 1);
  const elapsedSec =
    args.started_at > 0 ? Math.max(0, nowTs / 1000 - args.started_at) : args.elapsed;
  const elapsedLabel = elapsedSec >= 1 ? `已等待 ${Math.floor(elapsedSec)}s` : "";
  const sseUrl =
    args.sse_url ||
    (args.sse_port > 0 && args.sse_run_id
      ? `http://${window.location.hostname}:${args.sse_port}/sse/${args.sse_run_id}`
      : "");
  const cancelBase = sseUrl ? sseUrl.slice(0, sseUrl.indexOf("/sse/")) : "";
  const showCancel =
    Boolean(cancelBase) && !demo && !runComplete && !cancelled && elapsedSec >= 8;
  const handleCancel = () => {
    setCancelled(true);
    fetch(`${cancelBase}/cancel/${args.sse_run_id}`, {
      method: "POST",
      keepalive: true,
    }).catch(() => {});
  };
  const title = runComplete ? "已完成" : effRetry > 0 ? "页面闪断，正在自动续跑" : args.title;
  const tilt = useMemo(
    () => ({
      rotateX: pointer.active ? pointer.ny * -10 : 0,
      rotateY: pointer.active ? pointer.nx * 12 : 0,
    }),
    [pointer],
  );
  const feed = effDetails.length ? effDetails : args.subtitle ? [args.subtitle] : [];
  const feedShown = showAllSteps ? feed : feed.slice(-3);
  const hiddenSteps = Math.max(0, feed.length - 3);

  return (
    <div className="root" ref={rootRef} data-theme={args.dark ? "dark" : "light"}>
      <motion.div
        ref={islandRef}
        className={`island persona-${args.persona}${cancelled ? " is-cancelled" : ""}`}
        style={
          {
            "--accent": theme.accent,
            "--accent2": theme.accent2,
            "--glow": theme.glow,
          } as CSSProperties
        }
        initial={false}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: "spring", stiffness: 140, damping: 18 }}
        onClick={() => setBurst((b) => b + 1)}
      >
        <WebGLField
          accent={theme.accent}
          accent2={theme.accent2}
          bg={args.bg_color}
          pointer={pointer}
          intensity={0.7 + args.gpa_norm * 0.5}
          active={streaming && !settled}
          settled={settled}
        />
        {!settled && <CursorAura pointer={pointer} theme={theme} containerRef={islandRef} />}
        <Particles
          intensity={0.85 + progress * 0.7}
          seed={args.persona.length}
          theme={theme}
          pointer={pointer}
          settled={settled}
        />
        {!settled && <div className="scan" />}

        <motion.div
          className="hero"
          style={{ perspective: 700 }}
          animate={tilt}
          transition={{ type: "spring", stiffness: 180, damping: 18 }}
        >
          <TagOrbit tags={args.tags} theme={theme} progress={progress} settled={settled} />
          <OrbitRing progress={progress} theme={theme} settled={settled} />
          <Mark
            stageIndex={idx}
            persona={args.persona}
            theme={theme}
            burstKey={burst}
            gpaNorm={args.gpa_norm}
            langNorm={args.lang_norm}
            settled={settled}
          />
        </motion.div>

        {runComplete ? (
          <motion.div
            className="complete-badge"
            role="status"
            style={{ x: "-50%" }}
            initial={reduce ? false : { scale: 0.4, opacity: 0, y: 8 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            transition={reduce ? undefined : { type: "spring", stiffness: 300, damping: 14 }}
          >
            <span className="complete-check-wrap">
              <svg className="complete-check" viewBox="0 0 24 24" aria-hidden="true">
                <motion.circle
                  cx="12"
                  cy="12"
                  r="10"
                  fill={theme.accent}
                  initial={reduce ? false : { scale: 0.4, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={
                    reduce ? undefined : { type: "spring", stiffness: 260, damping: 13, delay: 0.08 }
                  }
                  style={{ transformOrigin: "12px 12px", transformBox: "fill-box" }}
                />
                <motion.path
                  d="M7 12.5l3.2 3.2L17 8.5"
                  stroke="#fff"
                  strokeWidth="2.4"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={reduce ? false : { pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={reduce ? undefined : { duration: 0.32, delay: 0.28, ease: "easeOut" }}
                />
              </svg>
              {!reduce && (
                <motion.span
                  className="complete-halo"
                  style={{ borderColor: theme.accent }}
                  initial={{ scale: 0.5, opacity: 0.65 }}
                  animate={{ scale: 1.9, opacity: 0 }}
                  transition={{ duration: 0.7, delay: 0.18, ease: "easeOut" }}
                />
              )}
            </span>
            <motion.span
              className="complete-text"
              initial={reduce ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={reduce ? undefined : { duration: 0.25, delay: 0.42 }}
            >
              已完成
            </motion.span>
          </motion.div>
        ) : null}

        {args.gpa_label || args.lang_label ? (
          <div className="score-row">
            {args.gpa_label ? <span>GPA {args.gpa_label}</span> : null}
            {args.lang_label ? <span>{args.lang_label}</span> : null}
          </div>
        ) : null}

        <motion.div className="persona-pill" style={{ color: accentText }}>
          {theme.label}
        </motion.div>

        <motion.div className="title" initial={false} animate={{ opacity: 1, y: 0 }}>
          {title}
        </motion.div>

        <div className="stage-row" role="status">
          <motion.span
            key={effStage}
            className="stage"
            initial={{ opacity: 0.35 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            {effStage}
          </motion.span>
          {elapsedLabel ? <span className="elapsed">{elapsedLabel}</span> : null}
          {effRetry > 0 ? (
            <span className="retry">续跑 {effRetry}/{args.retry_max}</span>
          ) : null}
        </div>

        {args.llm_text ? (
          <LlmRibbon
            text={args.llm_text}
            theme={theme}
            sseUrl={sseUrl || undefined}
            streamKey={args.llm_text}
            persist={!demo}
            labelColor={accentText}
            onStreaming={setStreaming}
            onComplete={setRunComplete}
            onMeta={setLiveMeta}
          />
        ) : null}

        {feed.length ? (
          <div className="detail-feed" aria-live="off">
            {feedShown.map((line, i) => (
              <div
                key={`${i}-${line}`}
                className={`detail-line${i === feedShown.length - 1 ? " latest" : ""}`}
                role={i === feedShown.length - 1 ? "status" : undefined}
              >
                {line}
              </div>
            ))}
            {hiddenSteps > 0 && !showAllSteps ? (
              <button type="button" className="more-steps" onClick={() => setShowAllSteps(true)}>
                更早 {hiddenSteps} 步
              </button>
            ) : null}
          </div>
        ) : (
          <div className="subtitle-slot">
            <AnimatePresence mode="popLayout">
              {args.subtitle ? (
                <motion.div
                  key={args.subtitle}
                  className="subtitle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.18 }}
                >
                  {args.subtitle}
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        )}

        <StageRail
          index={idx}
          count={n}
          accent={theme.accent}
          settled={settled}
          complete={runComplete}
        />

        <AnimatePresence>
          {args.hint ? (
            <motion.div key={args.hint} className="hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              {args.hint}
            </motion.div>
          ) : null}
        </AnimatePresence>

        {showCancel ? (
          <button type="button" className="cancel-btn" onClick={handleCancel}>
            取消并手动填写
          </button>
        ) : null}
        {cancelled ? (
          <div className="cancelled-note" role="status">
            已取消，请手动填写下方表单
          </div>
        ) : null}

        {demo ? <div className="demo-badge">demo · 移动光标 / 点击</div> : null}
      </motion.div>
    </div>
  );
}
