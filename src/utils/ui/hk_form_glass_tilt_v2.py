from __future__ import annotations

import streamlit as st

_HK_FORM_GLASS_TILT_JS = r"""
export default function(component) {
  const { parentElement } = component;

  const SELECTOR_MARKER_OUTER = '.hk-input-glass-marker--outer';
  const SELECTOR_MARKER = '.hk-input-glass-marker';
  const SELECTOR_TARGET = [
    '[data-testid="stVerticalBlockBorderWrapper"]',
    '[data-testid="stContainer"]',
    '[data-testid="stVerticalBlock"]',
    '.stVerticalBlockBorderWrapper',
  ].join(',');

  const prefersReducedMotion = (() => {
    try {
      return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      return false;
    }
  })();

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
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

  let targetEl = null;
  let cleanupFns = [];
  let observer = null;
  let observerTimeout = 0;

  function teardown() {
    cleanupFns.forEach((fn) => {
      try { fn(); } catch (e) {}
    });
    cleanupFns = [];
    if (observer) {
      try { observer.disconnect(); } catch (e) {}
      observer = null;
    }
    if (observerTimeout) {
      try { clearTimeout(observerTimeout); } catch (e) {}
      observerTimeout = 0;
    }
  }

  function installOn(el) {
    if (!el) return;
    if (el.dataset && el.dataset.hkTiltBound === '1') return;

    try { el.dataset.hkTiltBound = '1'; } catch (e) {}
    el.classList.add('hk-glass-card', 'hk-tilt-card');

    el.style.setProperty('--hk-tilt-x', '0deg');
    el.style.setProperty('--hk-tilt-y', '0deg');
    el.style.setProperty('--hk-glare-x', '50%');
    el.style.setProperty('--hk-glare-y', '50%');

    if (prefersReducedMotion) return;

    const MAX_TILT_DEG = 1.35;
    const MAX_GLARE_OPACITY = 0.26;
    const SMOOTHING = 0.075;
    const STOP_EPS = 0.02;
    const MOVE_EPS = 0.002;
    const LEAVE_DELAY_MS = 120;
    const ENTER_DELAY_MS = 15;

    const MASS = 1.2;
    const STIFFNESS = 65;
    const DAMPING_RATIO = 0.7;
    const CRITICAL_DAMPING = 2 * Math.sqrt(STIFFNESS * MASS);
    const DAMPING = DAMPING_RATIO * CRITICAL_DAMPING;

    const MAX_VEL = 220;

    const INTERACTIVE_SELECTOR = [
      'input',
      'textarea',
      'select',
      'button',
      '[role="textbox"]',
      '[role="combobox"]',
      '[role="listbox"]',
      '[role="slider"]',
      '[contenteditable="true"]',
      '[data-baseweb="select"]',
      '[data-baseweb="popover"]',
    ].join(',');

    let raf = 0;
    let rectRaf = 0;
    let hover = false;
    let leaveTimer = 0;
    let enterTimer = 0;
    let tiltEnabled = false;

    let targetTiltX = 0;
    let targetTiltY = 0;
    let currentTiltX = 0;
    let currentTiltY = 0;
    let velTiltX = 0;
    let velTiltY = 0;

    let targetGlareX = 50;
    let targetGlareY = 50;
    let currentGlareX = 50;
    let currentGlareY = 50;
    let lastPx = 0.5;
    let lastPy = 0.5;
    let baseRect = null;
    let lastTs = 0;
    let lastRectUpdate = 0;
    let lastClientX = null;
    let lastClientY = null;
    let lastClientT = 0;

    function updateBaseRect() {
      try { baseRect = el.getBoundingClientRect(); } catch (e) { baseRect = null; }
    }

    function scheduleBaseRectUpdate() {
      if (rectRaf) return;
      rectRaf = requestAnimationFrame(() => {
        rectRaf = 0;
        updateBaseRect();
      });
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function tick(ts) {
      raf = 0;
      if (!el.isConnected) return;

      const t = (typeof ts === 'number') ? ts : performance.now();
      if (!lastTs) lastTs = t;
      let dt = (t - lastTs) / 1000;
      lastTs = t;
      dt = clamp(dt, 0.0, 0.05);

      const ax = (((targetTiltX - currentTiltX) * STIFFNESS) - (velTiltX * DAMPING)) / MASS;
      const ay = (((targetTiltY - currentTiltY) * STIFFNESS) - (velTiltY * DAMPING)) / MASS;

      velTiltX = clamp(velTiltX + ax * dt, -MAX_VEL, MAX_VEL);
      velTiltY = clamp(velTiltY + ay * dt, -MAX_VEL, MAX_VEL);

      currentTiltX = currentTiltX + velTiltX * dt;
      currentTiltY = currentTiltY + velTiltY * dt;

      currentGlareX = lerp(currentGlareX, targetGlareX, SMOOTHING);
      currentGlareY = lerp(currentGlareY, targetGlareY, SMOOTHING);

      el.style.setProperty('--hk-tilt-x', currentTiltX.toFixed(3) + 'deg');
      el.style.setProperty('--hk-tilt-y', currentTiltY.toFixed(3) + 'deg');
      el.style.setProperty('--hk-glare-x', currentGlareX.toFixed(1) + '%');
      el.style.setProperty('--hk-glare-y', currentGlareY.toFixed(1) + '%');

      const dTilt =
        Math.abs(currentTiltX - targetTiltX) + Math.abs(currentTiltY - targetTiltY);
      const dGlare =
        Math.abs(currentGlareX - targetGlareX) + Math.abs(currentGlareY - targetGlareY);

      if (hover || dTilt > STOP_EPS || dGlare > 0.2) {
        raf = requestAnimationFrame(tick);
      }
    }

    function ensureTicking() {
      if (raf) return;
      raf = requestAnimationFrame(tick);
    }

    function onPointerEnter() {
      hover = true;
      updateBaseRect();
      lastRectUpdate = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      tiltEnabled = false;
      if (enterTimer) {
        try { clearTimeout(enterTimer); } catch (e) {}
        enterTimer = 0;
      }
      enterTimer = setTimeout(() => {
        enterTimer = 0;
        tiltEnabled = true;
      }, ENTER_DELAY_MS);
    }

    function softClamp01(p) {
      const m = 0.08;
      const t = clamp((p - m) / (1 - 2 * m), 0, 1);
      const s = t * t * (3 - 2 * t);
      return m + s * (1 - 2 * m);
    }

    function onPointerMove(e) {
      if (leaveTimer) {
        try { clearTimeout(leaveTimer); } catch (err) {}
        leaveTimer = 0;
      }
      hover = true;
      if (!tiltEnabled) return;

      const nowMs = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
      // Keep rect reasonably fresh even if Streamlit rerenders without a scroll/resize event.
      if (!baseRect || (nowMs - lastRectUpdate) > 250) {
        updateBaseRect();
        lastRectUpdate = nowMs;
      }

      const rect = baseRect || el.getBoundingClientRect();
      const rawPx = clamp((e.clientX - rect.left) / rect.width, 0, 1);
      const rawPy = clamp((e.clientY - rect.top) / rect.height, 0, 1);
      const px = softClamp01(rawPx);
      const py = softClamp01(rawPy);

      if (Math.abs(px - lastPx) < MOVE_EPS && Math.abs(py - lastPy) < MOVE_EPS) {
        return;
      }
      lastPx = px;
      lastPy = py;

      let isInteractive = false;
      try {
        const t = e.target;
        isInteractive = !!(t && t.closest && t.closest(INTERACTIVE_SELECTOR));
      } catch (err) { isInteractive = false; }

      const nx = Math.tanh(((px - 0.5) * 2) * 1.25);
      const ny = Math.tanh(((py - 0.5) * 2) * 1.25);

      let speedFactor = 1.0;
      try {
        const now = nowMs;
        if (lastClientX !== null && lastClientY !== null && lastClientT) {
          const dt = Math.max(0.001, (now - lastClientT) / 1000);
          const dx = e.clientX - lastClientX;
          const dy = e.clientY - lastClientY;
          const speed = Math.sqrt(dx * dx + dy * dy) / dt;
          speedFactor = clamp(1 - (speed - 400) / 1200, 0.55, 1.0);
        }
        lastClientX = e.clientX;
        lastClientY = e.clientY;
        lastClientT = now;
      } catch (err) {}

      targetTiltY = nx * (MAX_TILT_DEG * speedFactor);
      targetTiltX = -ny * (MAX_TILT_DEG * speedFactor);
      targetGlareX = px * 100;
      targetGlareY = py * 100;

      // UX: When hovering interactive controls (inputs/buttons/etc), keep tilt-follow but disable glare
      // to reduce distraction and avoid fighting focus/selection visuals.
      el.style.setProperty('--hk-glare-opacity', String(isInteractive ? 0 : (MAX_GLARE_OPACITY * speedFactor)));
      el.classList.add('hk-tilt-active');
      ensureTicking();
    }

    function onPointerLeave() {
      hover = false;
      if (leaveTimer) {
        try { clearTimeout(leaveTimer); } catch (err) {}
        leaveTimer = 0;
      }
      leaveTimer = setTimeout(() => {
        leaveTimer = 0;
        tiltEnabled = false;
        targetTiltX = 0;
        targetTiltY = 0;
        targetGlareX = 50;
        targetGlareY = 50;
        el.style.setProperty('--hk-glare-opacity', '0');
        el.classList.remove('hk-tilt-active');
        ensureTicking();
      }, LEAVE_DELAY_MS);
    }

    const onWindowResize = () => scheduleBaseRectUpdate();
    const onWindowScroll = () => scheduleBaseRectUpdate();

    el.addEventListener('pointerenter', onPointerEnter, { passive: true });
    el.addEventListener('pointermove', onPointerMove, { passive: true });
    el.addEventListener('pointerleave', onPointerLeave, { passive: true });
    window.addEventListener('resize', onWindowResize, { passive: true });
    window.addEventListener('scroll', onWindowScroll, { passive: true });

    cleanupFns.push(() => {
      if (raf) {
        try { cancelAnimationFrame(raf); } catch (e) {}
        raf = 0;
      }
      if (rectRaf) {
        try { cancelAnimationFrame(rectRaf); } catch (e) {}
        rectRaf = 0;
      }
      if (leaveTimer) {
        try { clearTimeout(leaveTimer); } catch (e) {}
        leaveTimer = 0;
      }
      if (enterTimer) {
        try { clearTimeout(enterTimer); } catch (e) {}
        enterTimer = 0;
      }
      try { el.removeEventListener('pointermove', onPointerMove); } catch (e) {}
      try { el.removeEventListener('pointerenter', onPointerEnter); } catch (e) {}
      try { el.removeEventListener('pointerleave', onPointerLeave); } catch (e) {}
      try { window.removeEventListener('resize', onWindowResize); } catch (e) {}
      try { window.removeEventListener('scroll', onWindowScroll); } catch (e) {}
      try {
        if (el.dataset) delete el.dataset.hkTiltBound;
      } catch (e) {}
    });
  }

  function tryInit() {
    const el = findTarget();
    if (el) {
      targetEl = el;
      installOn(targetEl);
      return true;
    }
    return false;
  }

  if (!tryInit()) {
    const OBSERVER_TIMEOUT_MS = 8000;
    observer = new MutationObserver(() => {
      if (tryInit()) {
        try { observer.disconnect(); } catch (e) {}
        observer = null;
        if (observerTimeout) {
          try { clearTimeout(observerTimeout); } catch (e) {}
          observerTimeout = 0;
        }
      }
    });
    try {
      observer.observe(document.body, { childList: true, subtree: true });
    } catch (e) {}

    // Safety: avoid observing forever if the marker/target never appears.
    observerTimeout = setTimeout(() => {
      if (observer) {
        try { observer.disconnect(); } catch (e) {}
        observer = null;
      }
      observerTimeout = 0;
    }, OBSERVER_TIMEOUT_MS);
  }

  return () => teardown();
}
"""


_HK_FORM_GLASS_TILT_CSS = r"""
:host, .hk-tilt-mount {
  display: block;
  width: 0;
  height: 0;
  overflow: hidden;
}
"""


@st.cache_resource(show_spinner=False)
def _get_hk_form_glass_tilt_component():
    return st.components.v2.component(
        "hk_form_glass_tilt_v2",
        css=_HK_FORM_GLASS_TILT_CSS,
        js=_HK_FORM_GLASS_TILT_JS,
        html='<div class="hk-tilt-mount"></div>',
    )


def mount_hk_form_glass_tilt(key: str = "hk_form_glass_tilt_v2"):
    comp = _get_hk_form_glass_tilt_component()
    return comp(key=key)
