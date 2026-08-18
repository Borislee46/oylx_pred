import { motion, useReducedMotion } from "framer-motion";
import { useEffect } from "react";
import type { PersonaTheme } from "./persona";
import { useTokenStream, type WaitMeta } from "./useTokenStream";

type Props = {
  text: string;
  theme: PersonaTheme;
  sseUrl?: string;
  streamKey?: string | number;
  persist?: boolean;
  labelColor?: string;
  onStreaming?: (active: boolean) => void;
  onComplete?: (ok: boolean) => void;
  onMeta?: (meta: WaitMeta | null) => void;
};

export function LlmRibbon({
  text,
  theme,
  sseUrl,
  streamKey,
  persist = false,
  labelColor,
  onStreaming,
  onComplete,
  onMeta,
}: Props) {
  const clean = (text || "").trim();
  const reduce = useReducedMotion();
  const { shown, status, sseOk, meta, fallback } = useTokenStream({
    text: clean,
    sseUrl,
    streamKey: streamKey ?? clean,
    enabled: Boolean(clean),
    persist,
  });

  useEffect(() => {
    if (status === "done") onComplete?.(sseOk !== false);
  }, [status, sseOk, onComplete]);

  useEffect(() => {
    onMeta?.(meta);
  }, [meta, onMeta]);

  useEffect(() => {
    onStreaming?.(status === "streaming");
  }, [status, onStreaming]);

  return (
    <div className={`llm-ribbon${status === "streaming" ? " is-streaming" : ""}`}>
      <div className="llm-meta">
        <span className="llm-label" style={{ color: labelColor ?? theme.accent }}>
          正在分析
        </span>
      </div>
      <div className="llm-body">
        <span className="llm-text">
          {shown}
          {status === "streaming" ? (
            reduce ? (
              <span className="llm-caret" style={{ background: theme.accent }} />
            ) : (
              <motion.span
                className="llm-caret"
                style={{ background: theme.accent }}
                animate={{ opacity: [1, 0.15, 1] }}
                transition={{ duration: 0.7, repeat: Infinity }}
              />
            )
          ) : null}
        </span>
        {fallback ? <span className="llm-fallback">已切换本地模式</span> : null}
        {/* Announce the final text once instead of every token delta. */}
        {status === "done" && shown ? (
          <span className="sr-only" aria-live="polite">
            {shown}
          </span>
        ) : null}
      </div>
    </div>
  );
}
