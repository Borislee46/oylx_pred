const { parentElement } = component;
let targetEl = null;
let cleanupFns = [];
let observer = null;
let observerTimeout = 0;

function teardown() {
  cleanupFns.forEach((fn) => {
    try {
      fn();
    } catch (e) {}
  });
  cleanupFns = [];
  if (observer) {
    try {
      observer.disconnect();
    } catch (e) {}
    observer = null;
  }
  if (observerTimeout) {
    try {
      clearTimeout(observerTimeout);
    } catch (e) {}
    observerTimeout = 0;
  }
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
      try {
        observer.disconnect();
      } catch (e) {}
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
      try {
        observer.disconnect();
      } catch (e) {}
      observer = null;
    }
    observerTimeout = 0;
  }, OBSERVER_TIMEOUT_MS);
}

return () => teardown();
