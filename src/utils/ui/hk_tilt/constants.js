const SELECTOR_MARKER_OUTER = ".hk-input-glass-marker--outer";
const SELECTOR_MARKER = ".hk-input-glass-marker";
const SELECTOR_TARGET = [
  '[data-testid="stVerticalBlockBorderWrapper"]',
  '[data-testid="stContainer"]',
  '[data-testid="stVerticalBlock"]',
  ".stVerticalBlockBorderWrapper",
].join(",");

const PERF_MONITOR = {
  frameTimes: [],
  lastCheck: 0,
  degraded: false,
  maxFramesToKeep: 60,
};

const prefersReducedMotion = (() => {
  try {
    return (
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  } catch (e) {
    return false;
  }
})();
