function getPerformanceProfile() {
  if (PERF_MONITOR.degraded) {
    return {
      useSpring: false,
      smoothing: 0.25,
      updateRate: 20,
      maxTilt: 0.8,
      useGlare: false,
    };
  }

  const isLowEnd =
    navigator.hardwareConcurrency <= 2 ||
    (navigator.deviceMemory && navigator.deviceMemory <= 2) ||
    /(android|ios|mobile)/i.test(navigator.userAgent) ||
    /(msie|trident|edge\/(1[0-7]))/i.test(navigator.userAgent) ||
    (navigator.connection &&
      (navigator.connection.saveData === true ||
        navigator.connection.effectiveType === "slow-2g" ||
        navigator.connection.effectiveType === "2g"));

  if (isLowEnd) {
    return {
      useSpring: false,
      smoothing: 0.2,
      updateRate: 24,
      maxTilt: window.innerWidth < 768 ? 0.6 : 0.8,
      useGlare: false,
    };
  }

  return {
    useSpring: true,
    smoothing: 0.1,
    updateRate: 48,
    maxTilt: window.innerWidth < 768 ? 0.9 : 1.1,
    useGlare: true,
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
    const avgFrameTime =
      PERF_MONITOR.frameTimes.reduce((a, b) => a + b) /
      PERF_MONITOR.frameTimes.length;
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
