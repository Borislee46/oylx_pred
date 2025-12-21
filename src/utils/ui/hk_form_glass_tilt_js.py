HK_FORM_GLASS_TILT_JS = r"""
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

  const PERF_MONITOR = {
    frameTimes: [],
    lastCheck: 0,
    degraded: false,
    maxFramesToKeep: 60
  };

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

  function getPerformanceProfile() {
    if (PERF_MONITOR.degraded) {
      return {
        useSpring: false,
        smoothing: 0.25,
        updateRate: 20,
        maxTilt: 0.8,
        useGlare: false
      };
    }

    const isLowEnd = 
      navigator.hardwareConcurrency <= 2 ||
      (navigator.deviceMemory && navigator.deviceMemory <= 2) ||
      /(android|ios|mobile)/i.test(navigator.userAgent) ||
      /(msie|trident|edge\/(1[0-7]))/i.test(navigator.userAgent) ||
      (navigator.connection && (
        navigator.connection.saveData === true ||
        navigator.connection.effectiveType === 'slow-2g' ||
        navigator.connection.effectiveType === '2g'
      ));
    
    if (isLowEnd) {
      return {
        useSpring: false,
        smoothing: 0.2,
        updateRate: 24,
        maxTilt: window.innerWidth < 768 ? 0.6 : 0.8,
        useGlare: false
      };
    }
    
    return {
      useSpring: true,
      smoothing: 0.1,
      updateRate: 48,
      maxTilt: window.innerWidth < 768 ? 0.9 : 1.1,
      useGlare: true
    };
  }

  function monitorPerformance(frameTime) {
    const now = performance.now();
    
    if (now - PERF_MONITOR.lastCheck < 2000) return false;
    
    PERF_MONITOR.frameTimes.push(frameTime);
    if (PERF_MONITOR.frameTimes.length > PERF_MONITOR.maxFramesToKeep) {
      PERF_MONITOR.frameTimes.shift();
    }
    
    if (PERF_MONITOR.frameTimes.length >= 30) {
      const avgFrameTime = PERF_MONITOR.frameTimes.reduce((a, b) => a + b) / PERF_MONITOR.frameTimes.length;
      const frameRate = 1000 / avgFrameTime;
      
      if (frameRate < 25 && !PERF_MONITOR.degraded) {
        PERF_MONITOR.degraded = true;
        return true;
      }
      
      if (frameRate > 45 && PERF_MONITOR.degraded) {
        PERF_MONITOR.degraded = false;
        return true;
      }
    }
    
    PERF_MONITOR.lastCheck = now;
    return false;
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

    let profile = getPerformanceProfile();
    
    try { el.dataset.hkTiltBound = '1'; } catch (e) {}
    el.classList.add('hk-glass-card', 'hk-tilt-card');

    el.style.setProperty('--hk-tilt-x', (el.dataset.hkTiltCx || 0) + 'deg');
    el.style.setProperty('--hk-tilt-y', (el.dataset.hkTiltCy || 0) + 'deg');
    
    if (profile.useGlare) {
      el.style.setProperty('--hk-glare-x', (el.dataset.hkTiltGx || 50) + '%');
      el.style.setProperty('--hk-glare-y', (el.dataset.hkTiltGy || 50) + '%');
      el.style.setProperty('--hk-glare-opacity', '0');
    } else {
      el.style.setProperty('--hk-glare-opacity', '0', 'important');
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

    const POPOVER_SELECTOR = [
      '[data-baseweb="popover"]',
      '[role="listbox"]',
      '.stSelectbox-popover',
      '.stMultiSelect-popover'
    ].join(',');

    let lastPopoverCheck = 0;
    let cachedPopoverExists = false;
    let lastResultCheck = 0;
    let cachedResultsExist = false;
    let freezeWeight = 1.0;
    let targetFreezeWeight = 1.0;

    const RESULT_SELECTOR = '.prediction-result, [data-testid="stMetric"], .stMetric';

    function isInteracting(e) {
      try {
        if (e && e.target && e.target.closest(INTERACTIVE_SELECTOR)) return true;
        const now = Date.now();
        if (now - lastPopoverCheck > 400) {
          const popover = document.querySelector(POPOVER_SELECTOR);
          cachedPopoverExists = !!(popover && popover.getClientRects().length > 0);
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

      const speed = isResultMode ? 4.0 : (targetFreezeWeight < freezeWeight ? 9.0 : 5.0);
      
      freezeWeight += (actualTarget - freezeWeight) * Math.min(1, dt * speed);
      return freezeWeight;
    }

    function checkInteraction(e, useTimeLock = false) {
      const interacting = isInteracting(e);
      const locked = useTimeLock ? (Date.now() < interactionLock) : false;
      
      targetFreezeWeight = (interacting || locked) ? 0.0 : 1.0;
      
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

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function tick(ts) {
      const frameStart = performance.now();
      
      if (frameCount % 120 === 0) {
        const needsUpdate = monitorPerformance(frameStart - lastUpdate);
        if (needsUpdate) {
          profile = getPerformanceProfile();
        }
      }
      frameCount++;

      const t = (typeof ts === 'number') ? ts : performance.now();
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
        const ax = (((targetTiltX - currentTiltX) * STIFFNESS) - (velTiltX * DAMPING)) / MASS;
        const ay = (((targetTiltY - currentTiltY) * STIFFNESS) - (velTiltY * DAMPING)) / MASS;
        const az = (((targetTiltZ - currentTiltZ) * STIFFNESS) - (velTiltZ * DAMPING)) / MASS;

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

      el.style.setProperty('--hk-tilt-x', currentTiltX.toFixed(3) + 'deg');
      el.style.setProperty('--hk-tilt-y', currentTiltY.toFixed(3) + 'deg');
      el.style.setProperty('--hk-tilt-z', currentTiltZ.toFixed(2) + 'px');
      
      const shadowX = (currentTiltY * -1.2).toFixed(2) + 'px';
      const shadowY = (currentTiltX * 1.2 + 25).toFixed(2) + 'px';
      el.style.setProperty('--hk-shadow-x', shadowX);
      el.style.setProperty('--hk-shadow-y', shadowY);

      if (profile.useGlare) {
        const glareIntensity = calculateGlareIntensity(currentTiltX, currentTiltY);
        el.style.setProperty('--hk-glare-x', currentGlareX.toFixed(1) + '%');
        el.style.setProperty('--hk-glare-y', currentGlareY.toFixed(1) + '%');
        el.style.setProperty('--hk-glare-opacity', (glareIntensity * (0.3 + 0.7 * weight)).toFixed(2));
      }

      const dTilt = Math.abs(currentTiltX - targetTiltX) + Math.abs(currentTiltY - targetTiltY) + Math.abs(currentTiltZ - targetTiltZ);
      const dGlare = profile.useGlare ? Math.abs(currentGlareX - targetGlareX) : 0;

      if (hover || dTilt > STOP_EPS || dGlare > 0.2 || Math.abs(freezeWeight - targetFreezeWeight) > 0.01) {
        if (profile.updateRate < 60) {
          setTimeout(() => { raf = requestAnimationFrame(tick); }, 1000 / profile.updateRate);
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

    function softClamp01(p) {
      const m = 0.08;
      const t = clamp((p - m) / (1 - 2 * m), 0, 1);
      const s = t * t * (3 - 2 * t);
      return m + s * (1 - 2 * m);
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
      if (window.innerWidth < 768 && !('ontouchstart' in window)) return;
      
      const nowMs = performance.now();
      if (!baseRect || (nowMs - lastRectUpdate) > 300) {
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

      const finalTiltMax = profile.maxTilt * (checkResults() ? 0.45 : 1.0);
      targetTiltY = nx * (finalTiltMax * speedFactor);
      targetTiltX = -ny * (finalTiltMax * speedFactor);
      
      if (profile.useGlare) {
        targetGlareX = px * 100;
        targetGlareY = py * 100;
      }

      el.classList.add('hk-tilt-active');
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
        if (checkInteraction(undefined, true)) {
          return;
        }
        leaveTimer = 0;
        tiltEnabled = false;
        targetTiltX = 0;
        targetTiltY = 0;
        if (profile.useGlare) {
          targetGlareX = 50;
          targetGlareY = 50;
        }
        el.classList.remove('hk-tilt-active');
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
        if (profile.useGlare) {
          targetGlareX = targetGlareY = 50;
        }
        
        clearTimeout(scrollLockTimer);
        scrollLockTimer = setTimeout(() => {
          scrollLock = false;
        }, 80);
        
        ensureTicking();
      }
      scheduleBaseRectUpdate();
    }

    function onTouchMove(e) {
      if (e.touches.length > 0) {
        onPointerMove(e.touches[0]);
      }
    }

    function onTouchEnd() {
      onPointerLeave();
    }

    const intersectionObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting && hover) {
          targetTiltX = targetTiltY = 0;
          if (profile.useGlare) {
            targetGlareX = targetGlareY = 50;
          }
          ensureTicking();
        }
      });
    }, { threshold: 0.05 });

    try {
      intersectionObserver.observe(el);
    } catch (e) {}

    el.addEventListener('pointerenter', onPointerEnter, { passive: true });
    el.addEventListener('pointermove', onPointerMove, { passive: true });
    el.addEventListener('pointerleave', onPointerLeave, { passive: true });
    el.addEventListener('pointerdown', onPointerDown, { passive: true });
    el.addEventListener('focusin', onFocusIn, { passive: true });
    
    if ('ontouchstart' in window) {
      el.addEventListener('touchmove', onTouchMove, { passive: true });
      el.addEventListener('touchend', onTouchEnd, { passive: true });
      profile.maxTilt *= 0.8;
    }
    
    window.addEventListener('resize', onWindowResize, { passive: true });
    let scrollTimeout;
    window.addEventListener('scroll', () => {
      if (scrollTimeout) return;
      scrollTimeout = setTimeout(() => {
        onWindowScroll();
        scrollTimeout = null;
      }, 50);
    }, { passive: true });

    cleanupFns.push(() => {
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
      if (rectRaf) {
        cancelAnimationFrame(rectRaf);
        rectRaf = 0;
      }
      if (leaveTimer) {
        clearTimeout(leaveTimer);
        leaveTimer = 0;
      }
      if (enterTimer) {
        clearTimeout(enterTimer);
        enterTimer = 0;
      }
      if (scrollLockTimer) {
        clearTimeout(scrollLockTimer);
        scrollLockTimer = 0;
      }
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerenter', onPointerEnter);
      el.removeEventListener('pointerleave', onPointerLeave);
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('focusin', onFocusIn);
      el.removeEventListener('touchmove', onTouchMove);
      el.removeEventListener('touchend', onTouchEnd);
      window.removeEventListener('resize', onWindowResize);
      window.removeEventListener('scroll', onWindowScroll);
      intersectionObserver.disconnect();
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
    const OBSERVER_TIMEOUT_MS = 5000;
    observer = new MutationObserver(() => {
      if (tryInit()) {
        try { observer.disconnect(); } catch (e) {}
        observer = null;
        if (observerTimeout) {
          clearTimeout(observerTimeout);
          observerTimeout = 0;
        }
      }
    });
    try {
      observer.observe(document.body, { childList: true, subtree: true });
    } catch (e) {}

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