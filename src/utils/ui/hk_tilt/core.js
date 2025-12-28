function installOn(el) {
  if (!el) return;
  if (el.dataset && el.dataset.hkTiltBound === "1") return;

  let profile = getPerformanceProfile();

  try {
    el.dataset.hkTiltBound = "1";
  } catch (e) {}

  el.style.willChange = "transform, opacity";
  el.classList.add("hk-glass-card", "hk-tilt-card");

  el.style.setProperty("--hk-tilt-x", (el.dataset.hkTiltCx || 0) + "deg");
  el.style.setProperty("--hk-tilt-y", (el.dataset.hkTiltCy || 0) + "deg");

  if (profile.useGlare) {
    el.style.setProperty("--hk-glare-x", (el.dataset.hkTiltGx || 50) + "%");
    el.style.setProperty("--hk-glare-y", (el.dataset.hkTiltGy || 50) + "%");
    el.style.setProperty("--hk-glare-opacity", "0");
  } else {
    el.style.setProperty("--hk-glare-opacity", "0", "important");
  }

  let interactionLock = 0;

  if (prefersReducedMotion) return;

  const MAX_GLARE_OPACITY = profile.useGlare ? 0.2 : 0;
  const STOP_EPS = 0.03;
  const MOVE_EPS = 0.003;
  const LEAVE_DELAY_MS = 100;
  const ENTER_DELAY_MS = 10;

  const MASS = 1.0;
  const STIFFNESS = 68;
  const DAMPING_RATIO = 0.85;
  const CRITICAL_DAMPING = 2 * Math.sqrt(STIFFNESS * MASS);
  const DAMPING = DAMPING_RATIO * CRITICAL_DAMPING;

  const MAX_VEL = 180;

  const INTERACTIVE_SELECTOR = [
    "input",
    "textarea",
    "select",
    "button",
    '[role="textbox"]',
    '[role="combobox"]',
    '[role="listbox"]',
    '[role="slider"]',
    '[contenteditable="true"]',
    '[data-baseweb="select"]',
    '[data-baseweb="popover"]',
  ].join(",");

  const POPOVER_SELECTOR = [
    '[data-baseweb="popover"]',
    '[role="listbox"]',
    ".stSelectbox-popover",
    ".stMultiSelect-popover",
  ].join(",");

  let lastPopoverCheck = 0;
  let cachedPopoverExists = false;
  let lastResultCheck = 0;
  let cachedResultsExist = false;
  let freezeWeight = 1.0;
  let targetFreezeWeight = 1.0;

  const RESULT_SELECTOR =
    '.prediction-result, [data-testid="stMetric"], .stMetric';

  function isInteracting(e) {
    try {
      if (e && e.target && e.target.closest(INTERACTIVE_SELECTOR)) return true;
      const now = Date.now();
      if (now - lastPopoverCheck > 400) {
        const popover = document.querySelector(POPOVER_SELECTOR);
        cachedPopoverExists = !!(
          popover && popover.getClientRects().length > 0
        );
        lastPopoverCheck = now;
      }
      if (cachedPopoverExists) return true;
    } catch (err) {}
    return false;
  }

  function checkResults() {
    const now = Date.now();
    if (now - lastResultCheck > 1000) {
      cachedResultsExist = !!document.querySelector(RESULT_SELECTOR);
      lastResultCheck = now;
    }
    return cachedResultsExist;
  }

  function updateFreezeWeight(dt) {
    const isResultMode = checkResults();
    const actualTarget = targetFreezeWeight;
    const speed = isResultMode
      ? 4.0
      : targetFreezeWeight < freezeWeight
        ? 9.0
        : 5.0;
    freezeWeight += (actualTarget - freezeWeight) * Math.min(1, dt * speed);
    return freezeWeight;
  }

  function checkInteraction(e, useTimeLock = false) {
    const interacting = isInteracting(e);
    const locked = useTimeLock ? Date.now() < interactionLock : false;
    targetFreezeWeight = interacting || locked ? 0.0 : 1.0;
    return interacting || locked;
  }

  function onPointerDown(e) {
    if (isInteracting(e)) {
      interactionLock = Date.now() + 500;
      ensureTicking();
    }
  }

  function onFocusIn(e) {
    if (isInteracting(e)) {
      interactionLock = Date.now() + 500;
      ensureTicking();
    }
  }

  function calculateGlareIntensity(tiltX, tiltY) {
    if (!profile.useGlare) return 0;
    const intensity = Math.sqrt(tiltX * tiltX + tiltY * tiltY);
    const MAX_INTENSITY = profile.maxTilt * 1.5;
    const curve = Math.pow(clamp(intensity / MAX_INTENSITY, 0, 1), 0.9);
    return curve * MAX_GLARE_OPACITY;
  }

  let raf = 0;
  let rectRaf = 0;
  let hover = false;
  let leaveTimer = 0;
  let enterTimer = 0;
  let tiltEnabled = false;
  let scrollLock = false;
  let scrollLockTimer = 0;
  let lastUpdate = 0;
  let frameCount = 0;

  let targetTiltX = parseFloat(el.dataset.hkTiltTx || 0);
  let targetTiltY = parseFloat(el.dataset.hkTiltTy || 0);
  let targetTiltZ = 0;
  let currentTiltX = parseFloat(el.dataset.hkTiltCx || 0);
  let currentTiltY = parseFloat(el.dataset.hkTiltCy || 0);
  let currentTiltZ = 0;
  let velTiltX = parseFloat(el.dataset.hkTiltVx || 0);
  let velTiltY = parseFloat(el.dataset.hkTiltVy || 0);
  let velTiltZ = 0;

  let targetGlareX = 50;
  let targetGlareY = 50;
  let currentGlareX = parseFloat(el.dataset.hkTiltGx || 50);
  let currentGlareY = parseFloat(el.dataset.hkTiltGy || 50);
  let lastPx = 0.5;
  let lastPy = 0.5;
  let baseRect = null;
  let lastTs = 0;
  let lastRectUpdate = 0;
  let lastClientX = null;
  let lastClientY = null;
  let lastClientT = 0;

  function updateBaseRect() {
    try {
      baseRect = el.getBoundingClientRect();
    } catch (e) {
      baseRect = null;
    }
  }

  function scheduleBaseRectUpdate() {
    if (rectRaf) return;
    rectRaf = requestAnimationFrame(() => {
      rectRaf = 0;
      updateBaseRect();
    });
  }

  function tick(ts) {
    const frameStart = performance.now();
    if (frameCount % 120 === 0) {
      const needsUpdate = monitorPerformance(frameStart - lastUpdate);
      if (needsUpdate) profile = getPerformanceProfile();
    }
    frameCount++;

    const t = typeof ts === "number" ? ts : performance.now();
    if (!lastTs) lastTs = t;
    let dt = (t - lastTs) / 1000;
    lastTs = t;
    dt = clamp(dt, 0.0, 0.05);

    const weight = updateFreezeWeight(dt);

    if (!profile.useSpring) {
      currentTiltX = lerp(currentTiltX, targetTiltX, profile.smoothing);
      currentTiltY = lerp(currentTiltY, targetTiltY, profile.smoothing);
      currentTiltZ = lerp(currentTiltZ, targetTiltZ, profile.smoothing);
      if (profile.useGlare) {
        currentGlareX = lerp(currentGlareX, targetGlareX, profile.smoothing);
        currentGlareY = lerp(currentGlareY, targetGlareY, profile.smoothing);
      }
      velTiltX = velTiltY = velTiltZ = 0;
    } else {
      const ax =
        ((targetTiltX - currentTiltX) * STIFFNESS - velTiltX * DAMPING) / MASS;
      const ay =
        ((targetTiltY - currentTiltY) * STIFFNESS - velTiltY * DAMPING) / MASS;
      const az =
        ((targetTiltZ - currentTiltZ) * STIFFNESS - velTiltZ * DAMPING) / MASS;

      velTiltX = clamp(velTiltX + ax * dt, -MAX_VEL, MAX_VEL);
      velTiltY = clamp(velTiltY + ay * dt, -MAX_VEL, MAX_VEL);
      velTiltZ = clamp(velTiltZ + az * dt, -MAX_VEL, MAX_VEL);

      currentTiltX += velTiltX * dt;
      currentTiltY += velTiltY * dt;
      currentTiltZ += velTiltZ * dt;

      if (profile.useGlare) {
        currentGlareX = lerp(currentGlareX, targetGlareX, profile.smoothing);
        currentGlareY = lerp(currentGlareY, targetGlareY, profile.smoothing);
      }
    }

    el.style.setProperty("--hk-tilt-x", currentTiltX + "deg");
    el.style.setProperty("--hk-tilt-y", currentTiltY + "deg");
    el.style.setProperty("--hk-tilt-z", currentTiltZ + "px");

    el.style.setProperty("--hk-shadow-x", currentTiltY * -1.2 + "px");
    el.style.setProperty("--hk-shadow-y", currentTiltX * 1.2 + 25 + "px");

    if (profile.useGlare) {
      const glareIntensity = calculateGlareIntensity(
        currentTiltX,
        currentTiltY,
      );
      el.style.setProperty("--hk-glare-x", currentGlareX + "%");
      el.style.setProperty("--hk-glare-y", currentGlareY + "%");
      el.style.setProperty(
        "--hk-glare-opacity",
        glareIntensity * (0.3 + 0.7 * weight),
      );
    }

    const dTilt =
      Math.abs(currentTiltX - targetTiltX) +
      Math.abs(currentTiltY - targetTiltY) +
      Math.abs(currentTiltZ - targetTiltZ);
    const dGlare = profile.useGlare
      ? Math.abs(currentGlareX - targetGlareX)
      : 0;

    if (
      hover ||
      dTilt > STOP_EPS ||
      dGlare > 0.2 ||
      Math.abs(freezeWeight - targetFreezeWeight) > 0.01
    ) {
      if (profile.updateRate < 60) {
        setTimeout(() => {
          raf = requestAnimationFrame(tick);
        }, 1000 / profile.updateRate);
      } else {
        raf = requestAnimationFrame(tick);
      }
    }
    lastUpdate = performance.now();
  }

  function ensureTicking() {
    if (raf) return;
    raf = requestAnimationFrame(tick);
  }

  function onPointerEnter() {
    hover = true;
    targetTiltZ = 10;
    updateBaseRect();
    lastRectUpdate = performance.now();
    tiltEnabled = false;
    if (enterTimer) {
      clearTimeout(enterTimer);
      enterTimer = 0;
    }
    enterTimer = setTimeout(() => {
      enterTimer = 0;
      tiltEnabled = true;
    }, ENTER_DELAY_MS);
    ensureTicking();
  }

  function onPointerMove(e) {
    if (leaveTimer) {
      clearTimeout(leaveTimer);
      leaveTimer = 0;
    }
    hover = true;
    targetTiltZ = 10;
    if (checkInteraction(e, false)) {
      ensureTicking();
      return;
    }
    if (!tiltEnabled || scrollLock) return;
    if (window.innerWidth < 768 && !("ontouchstart" in window)) return;

    const nowMs = performance.now();
    if (!baseRect || nowMs - lastRectUpdate > 300) {
      updateBaseRect();
      lastRectUpdate = nowMs;
    }

    const rect = baseRect || el.getBoundingClientRect();
    const px = softClamp01(clamp((e.clientX - rect.left) / rect.width, 0, 1));
    const py = softClamp01(clamp((e.clientY - rect.top) / rect.height, 0, 1));

    if (Math.abs(px - lastPx) < MOVE_EPS && Math.abs(py - lastPy) < MOVE_EPS)
      return;
    lastPx = px;
    lastPy = py;

    const nx = Math.tanh((px - 0.5) * 2 * 1.25);
    const ny = Math.tanh((py - 0.5) * 2 * 1.25);

    let speedFactor = 1.0;
    try {
      if (lastClientX !== null && lastClientY !== null && lastClientT) {
        const dt = Math.max(0.001, (nowMs - lastClientT) / 1000);
        const speed =
          Math.sqrt(
            Math.pow(e.clientX - lastClientX, 2) +
              Math.pow(e.clientY - lastClientY, 2),
          ) / dt;
        speedFactor = clamp(1 - (speed - 400) / 1200, 0.55, 1.0);
      }
      lastClientX = e.clientX;
      lastClientY = e.clientY;
      lastClientT = nowMs;
    } catch (err) {}

    const finalTiltMax = profile.maxTilt * (checkResults() ? 0.45 : 1.0);
    targetTiltY = nx * (finalTiltMax * speedFactor);
    targetTiltX = -ny * (finalTiltMax * speedFactor);

    if (profile.useGlare) {
      targetGlareX = px * 100;
      targetGlareY = py * 100;
    }
    el.classList.add("hk-tilt-active");
    ensureTicking();
  }

  function onPointerLeave() {
    hover = false;
    targetTiltZ = 0;
    if (leaveTimer) {
      clearTimeout(leaveTimer);
      leaveTimer = 0;
    }
    leaveTimer = setTimeout(() => {
      if (checkInteraction(undefined, true)) return;
      leaveTimer = 0;
      tiltEnabled = false;
      targetTiltX = 0;
      targetTiltY = 0;
      if (profile.useGlare) {
        targetGlareX = 50;
        targetGlareY = 50;
      }
      el.classList.remove("hk-tilt-active");
      ensureTicking();
    }, LEAVE_DELAY_MS);
    ensureTicking();
  }

  function onWindowResize() {
    scheduleBaseRectUpdate();
    PERF_MONITOR.frameTimes = [];
    PERF_MONITOR.lastCheck = 0;
    Object.assign(profile, getPerformanceProfile());
  }

  function onWindowScroll() {
    if (hover) {
      scrollLock = true;
      targetTiltX = targetTiltY = 0;
      targetTiltZ = 0;
      if (profile.useGlare) targetGlareX = targetGlareY = 50;
      clearTimeout(scrollLockTimer);
      scrollLockTimer = setTimeout(() => {
        scrollLock = false;
      }, 80);
      ensureTicking();
    }
    scheduleBaseRectUpdate();
  }

  function onTouchMove(e) {
    if (e.touches.length > 0) onPointerMove(e.touches[0]);
  }
  function onTouchEnd() {
    onPointerLeave();
  }

  const intersectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting && hover) {
          targetTiltX = targetTiltY = 0;
          if (profile.useGlare) {
            targetGlareX = targetGlareY = 50;
          }
          ensureTicking();
        }
      });
    },
    { threshold: 0.05 },
  );

  try {
    intersectionObserver.observe(el);
  } catch (e) {}

  el.addEventListener("pointerenter", onPointerEnter, { passive: true });
  el.addEventListener("pointermove", onPointerMove, { passive: true });
  el.addEventListener("pointerleave", onPointerLeave, { passive: true });
  el.addEventListener("pointerdown", onPointerDown, { passive: true });
  el.addEventListener("focusin", onFocusIn, { passive: true });

  if ("ontouchstart" in window) {
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("touchend", onTouchEnd, { passive: true });
    profile.maxTilt *= 0.8;
  }

  window.addEventListener("resize", onWindowResize, { passive: true });
  let scrollTimeout;
  window.addEventListener(
    "scroll",
    () => {
      if (scrollTimeout) return;
      scrollTimeout = setTimeout(() => {
        onWindowScroll();
        scrollTimeout = null;
      }, 50);
    },
    { passive: true },
  );

  cleanupFns.push(() => {
    if (raf) cancelAnimationFrame(raf);
    if (rectRaf) cancelAnimationFrame(rectRaf);
    if (leaveTimer) clearTimeout(leaveTimer);
    if (enterTimer) clearTimeout(enterTimer);
    if (scrollLockTimer) clearTimeout(scrollLockTimer);
    el.removeEventListener("pointermove", onPointerMove);
    el.removeEventListener("pointerenter", onPointerEnter);
    el.removeEventListener("pointerleave", onPointerLeave);
    el.removeEventListener("pointerdown", onPointerDown);
    el.removeEventListener("focusin", onFocusIn);
    el.removeEventListener("touchmove", onTouchMove);
    el.removeEventListener("touchend", onTouchEnd);
    window.removeEventListener("resize", onWindowResize);
    window.removeEventListener("scroll", onWindowScroll);
    intersectionObserver.disconnect();
    try {
      if (el.dataset) delete el.dataset.hkTiltBound;
    } catch (e) {}
  });
}
