import { useEffect, useRef, useState } from "react";
import { loadWaitState, saveWaitState } from "./waitState";

export type StreamStatus = "idle" | "streaming" | "done" | "error";

export type WaitMeta = {
  stage?: string;
  stage_index?: number;
  stage_count?: number;
  details?: string[];
  retry?: number;
  retry_max?: number;
};

type Options = {
  text?: string;
  sseUrl?: string;
  streamKey?: string | number;
  enabled?: boolean;
  persist?: boolean;
};

function tokenize(text: string): string[] {
  const t = text.trim();
  if (!t) return [];
  return t.match(/[\u4e00-\u9fff]|[A-Za-z0-9.]+|[^\s]/g) || t.split("");
}

async function* localTokenStream(text: string, prefix = ""): AsyncGenerator<string> {
  const parts = tokenize(text);
  let acc = "";
  let startIdx = 0;
  for (let i = 0; i < parts.length; i++) {
    if (acc.length >= prefix.length) {
      startIdx = i;
      break;
    }
    acc += parts[i];
  }
  for (let i = startIdx; i < parts.length; i++) {
    yield parts[i];
    const slow = /[。．.!？?]/.test(parts[i]);
    await new Promise((r) => setTimeout(r, slow ? 90 + Math.random() * 80 : 22 + Math.random() * 40));
  }
}

export function useTokenStream({
  text = "",
  sseUrl,
  streamKey,
  enabled = true,
  persist = false,
}: Options) {
  const [shown, setShown] = useState("");
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [sseOk, setSseOk] = useState<boolean | null>(null);
  const [meta, setMeta] = useState<WaitMeta | null>(null);
  const [fallback, setFallback] = useState(false);
  const gen = useRef(0);
  const shownRef = useRef("");
  const lastSaveRef = useRef(0);
  const metaRef = useRef<WaitMeta | null>(null);

  useEffect(() => {
    if (!enabled) {
      setShown(text);
      setStatus("done");
      return;
    }
    const id = ++gen.current;
    let cancelled = false;

    const commit = (value: string) => {
      shownRef.current = value;
      setShown(value);
      if (persist) {
        const now = Date.now();
        if (now - lastSaveRef.current >= 120) {
          lastSaveRef.current = now;
          saveWaitState({ text, shown: value });
        }
      }
    };

    const runLocal = async (prefix = "") => {
      let acc = prefix;
      commit(prefix || "");
      for await (const tok of localTokenStream(text, prefix)) {
        if (cancelled || gen.current !== id) return;
        acc += tok;
        commit(acc);
      }
      if (!cancelled && gen.current === id) setStatus("done");
    };

    const saved = persist ? loadWaitState() : null;
    const prev = shownRef.current;
    const prefix =
      saved && typeof saved.shown === "string" && text.startsWith(saved.shown)
        ? saved.shown
        : prev && text.startsWith(prev)
          ? prev
          : "";

    setStatus("streaming");
    setSseOk(null);
    setFallback(false);

    if (sseUrl) {
      let acc = prefix;
      if (prefix) commit(prefix);
      const es = new EventSource(sseUrl);
      es.onmessage = (ev) => {
        if (cancelled || gen.current !== id) return;
        try {
          const payload = JSON.parse(ev.data) as {
            token?: string;
            text?: string;
            reset?: boolean;
            done?: boolean;
            ok?: boolean;
            meta?: WaitMeta;
          };
          if (payload.meta && typeof payload.meta === "object") {
            const merged: WaitMeta = { ...(metaRef.current ?? {}), ...payload.meta };
            metaRef.current = merged;
            setMeta(merged);
          }
          if (payload.reset) {
            acc = "";
            commit("");
          }
          if (payload.token) {
            acc += payload.token;
            commit(acc);
          }
          if (typeof payload.text === "string") {
            acc = payload.text;
            commit(acc);
          }
          if (payload.done) {
            setSseOk(Boolean(payload.ok));
            setStatus("done");
            es.close();
          }
        } catch {
          acc += ev.data;
          commit(acc);
        }
      };
      es.onerror = () => {
        es.close();
        if (cancelled || gen.current !== id) return;
        setFallback(true);
        void runLocal(acc);
      };
      return () => {
        cancelled = true;
        es.close();
        if (persist) saveWaitState({ text, shown: shownRef.current });
      };
    }

    void runLocal(prefix);
    return () => {
      cancelled = true;
      if (persist) saveWaitState({ text, shown: shownRef.current });
    };
  }, [text, sseUrl, streamKey, enabled]);

  return { shown, status, sseOk, meta, fallback };
}
