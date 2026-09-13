/**
 * Session-scoped persistence so a remounted iframe (Streamlit progress ticks
 * mount a fresh iframe with a new key) can resume typing / height instead of
 * restarting from zero. sessionStorage is per-tab and cleared when the tab
 * closes, so no long-lived student data is retained.
 */

const KEY = "liw_wait_v1";

export type WaitState = {
  text?: string;
  shown?: string;
  height?: number;
};

export function loadWaitState(): WaitState | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const j = JSON.parse(raw) as WaitState;
    if (j && (typeof j.text === "string" || typeof j.height === "number")) return j;
    return null;
  } catch {
    return null;
  }
}

export function saveWaitState(patch: WaitState): void {
  try {
    const prev = loadWaitState() ?? {};
    sessionStorage.setItem(KEY, JSON.stringify({ ...prev, ...patch }));
  } catch {
    // Storage can be unavailable (private mode / quota) — degrade silently.
  }
}
