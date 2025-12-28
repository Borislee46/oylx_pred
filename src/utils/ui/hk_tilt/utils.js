function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function softClamp01(p) {
  const m = 0.08;
  const t = clamp((p - m) / (1 - 2 * m), 0, 1);
  const s = t * t * (3 - 2 * t);
  return m + s * (1 - 2 * m);
}

function findTarget() {
  const root = parentElement || document;
  const marker =
    root.querySelector(SELECTOR_MARKER_OUTER) ||
    document.querySelector(SELECTOR_MARKER_OUTER) ||
    root.querySelector(SELECTOR_MARKER) ||
    document.querySelector(SELECTOR_MARKER);
  if (!marker) return null;

  const target = marker.closest(SELECTOR_TARGET);
  return target || null;
}
