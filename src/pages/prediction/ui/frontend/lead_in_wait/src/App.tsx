import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { CursorAura } from "./CursorAura";
import { Mark } from "./Mark";
import { OrbitRing } from "./OrbitRing";
import { Particles } from "./Particles";
import { PERSONAS, asPersona, type Persona } from "./persona";
import { DEMO } from "./demoData";
import { TagOrbit } from "./TagOrbit";
import { onRender, setFrameHeight, type WaitArgs } from "./streamlit";
import { usePointer } from "./usePointer";
import { useTokenStream } from "./useTokenStream";
import { loadWaitState, saveWaitState } from "./waitState";
import { WebGLField } from "./WebGLField";

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
  const gotStreamlit = useRef(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const islandRef = useRef<HTMLDivElement>(null);
  const pointer = usePointer(islandRef);
  const prevStage = useRef(args.stage_index);
  const maxHeight = useRef(Math.max(240, loadWaitState()?.height || 0));
  const prevRunId = useRef(args.sse_run_id);
  // A new run must start at the base height — the grow-only persistence is
  // per-run and would otherwise leak the previous run's taller iframe.
  if (prevRunId.current !== args.sse_run_id) {
    prevRunId.current = args.sse_run_id;
    maxHeight.current = 240;
  }
  const settled = runComplete;

  const sseUrl =
    args.sse_url ||
    (args.sse_port > 0 && args.sse_run_id
      ? `http://${window.location.hostname}:${args.sse_port}/sse/${args.sse_run_id}`
      : "");

  const { status: streamStatus, sseOk, meta: sseMeta } = useTokenStream({
    text: args.llm_text,
    sseUrl: sseUrl || undefined,
    streamKey: args.llm_text,
    enabled: Boolean(args.llm_text),
    persist: !demo,
  });
  const streaming = streamStatus === "streaming";

  // SSE meta 驱动的最新进度（阶段/续跑），mount 后由 meta 原地更新。
  const effStageIndex = sseMeta?.stage_index ?? args.stage_index;
  const effStageCount = sseMeta?.stage_count ?? args.stage_count;

  // A new wait run must not inherit the previous run's completion state.
  useEffect(() => {
    setRunComplete(false);
  }, [args.sse_run_id]);

  // Mark the run complete when the active stream finishes. `args.llm_text` is
  // read fresh here (deliberately not a dependency): a text change leaves the
  // previous "done" status visible for one render, and we must not complete on
  // that stale value — the stream resets to "streaming" before finishing.
  useEffect(() => {
    if (streamStatus === "done" && args.llm_text) setRunComplete(sseOk !== false);
  }, [streamStatus, sseOk]);

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
      setRunComplete(false);
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
        dark: true,
        bg_color: "#0b1220",
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
    const el = rootRef.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout>;
    const sync = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!rootRef.current) return;
        const h = Math.ceil(rootRef.current.getBoundingClientRect().height);
        // Only grow during wait — shrinking iframe height causes visible jump.
        // Once the run settles (completed) the frame may shrink to its true
        // content height again.
        maxHeight.current = settled ? h : Math.max(maxHeight.current, h, 240);
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
  }, [demo, settled]);

  const baseTheme = PERSONAS[args.persona];
  const theme = args.dark
    ? {
        ...baseTheme,
        accent: baseTheme.accentDark ?? baseTheme.accent,
        accent2: baseTheme.accent2Dark ?? baseTheme.accent2,
        glow: baseTheme.glowDark ?? baseTheme.glow,
      }
    : baseTheme;
  const n = Math.max(1, effStageCount);
  const idx = Math.min(effStageIndex, n - 1);
  const progress = n <= 1 ? 1 : idx / (n - 1);
  const tilt = useMemo(
    () => ({
      rotateX: pointer.active ? pointer.ny * -10 : 0,
      rotateY: pointer.active ? pointer.nx * 12 : 0,
    }),
    [pointer],
  );
  return (
    <div className="root" ref={rootRef} data-theme={args.dark ? "dark" : "light"}>
      <motion.div
        ref={islandRef}
        className={`island persona-${args.persona}`}
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
            initial={reduce ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={reduce ? undefined : { duration: 0.3, ease: "easeOut" }}
          >
            <span className="complete-check-wrap">
              <svg className="complete-check" viewBox="0 0 24 24" aria-hidden="true">
                <defs>
                  <linearGradient id="completeGrad" x1="0" y1="0" x2="24" y2="24">
                    <stop offset="0%" stopColor={theme.accent2} />
                    <stop offset="100%" stopColor={theme.accent} />
                  </linearGradient>
                </defs>
                <motion.circle
                  cx="12"
                  cy="12"
                  r="10.5"
                  fill="none"
                  stroke="url(#completeGrad)"
                  strokeWidth="1.5"
                  initial={reduce ? false : { pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.9 }}
                  transition={reduce ? undefined : { duration: 0.55, ease: "easeOut" }}
                />
                <motion.path
                  d="M7 12.4l3.2 3.2L17 8.6"
                  stroke="url(#completeGrad)"
                  strokeWidth="2.4"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={reduce ? false : { pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={reduce ? undefined : { duration: 0.4, delay: 0.18, ease: "easeOut" }}
                />
              </svg>
              {!reduce && (
                <motion.span
                  className="complete-halo"
                  style={{ borderColor: theme.accent }}
                  initial={{ scale: 0.6, opacity: 0.5 }}
                  animate={{ scale: 1.6, opacity: 0 }}
                  transition={{ duration: 0.9, delay: 0.15, ease: "easeOut" }}
                />
              )}
            </span>
          </motion.div>
        ) : null}

      </motion.div>
    </div>
  );
}
