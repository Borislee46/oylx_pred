(function () {
  const fe = window.frameElement;
  if (fe) {
    fe.style.cssText =
      "position:absolute!important;left:-9999px!important;width:0!important;height:0!important;border:0!important;margin:0!important;padding:0!important;opacity:0!important;pointer-events:none!important;";
    const shell = fe.closest('[data-testid="stElementContainer"]');
    if (shell) {
      shell.style.cssText =
        "display:none!important;width:0!important;height:0!important;margin:0!important;padding:0!important;border:0!important;overflow:hidden!important;";
    }
  }

  const root = window.parent.document.documentElement;
  let ticking = false;
  let mouseX = 0;
  let mouseY = 0;

  const updateParallax = () => {
    const x = (mouseX / window.parent.innerWidth - 0.5) * 25;
    const y = (mouseY / window.parent.innerHeight - 0.5) * 25;
    root.style.setProperty("--bg-pos-x", `${x}px`);
    root.style.setProperty("--bg-pos-y", `${y}px`);
    ticking = false;
  };

  window.parent.addEventListener(
    "mousemove",
    (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      if (!ticking) {
        window.parent.requestAnimationFrame(updateParallax);
        ticking = true;
      }
    },
    { passive: true },
  );

  setTimeout(() => {
    const anchor = window.parent.document.getElementById(
      "main-page-header-anchor",
    );
    if (anchor) {
      anchor.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, 50);
})();
