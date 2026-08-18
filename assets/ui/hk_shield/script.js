export default function (component) {
  const doc = window.parent.document;

  function init() {
    const container = doc.querySelector(".hk-logo-container");
    const logo = doc.querySelector(".hk-header-logo");

    if (!container || !logo) {
      setTimeout(init, 200);
      return;
    }

    if (!container.querySelector(".hk-shield-svg-loader")) {
      const loaderDiv = doc.createElement("div");
      loaderDiv.className = "hk-shield-svg-loader";
      loaderDiv.innerHTML = `
<svg viewBox="0 0 800 800" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hkLaser" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#EAFDFF" stop-opacity="0.95"/>
      <stop offset="12%" stop-color="#7FFBF2" stop-opacity="0.9"/>
      <stop offset="45%" stop-color="#21FFF4" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#0891B2" stop-opacity="0.8"/>
    </linearGradient>
    <filter id="hkGlow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="8" result="outerGlow"/>
      <feGaussianBlur stdDeviation="2" result="innerGlow"/>
      <feComponentTransfer in="outerGlow" result="outerGlowBright">
        <feFuncA type="linear" slope="1.15"/>
      </feComponentTransfer>
      <feMerge>
        <feMergeNode in="outerGlowBright"/>
        <feMergeNode in="innerGlow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <path class="shield-path" d="M450 185C455 185 575 200 585 258C600 340 596 400 578 468C552 564 400 640 400 640C400 640 248 564 222 468C204 400 200 340 215 258C225 200 345 185 400 185Z" 
        fill="none" stroke="url(#hkLaser)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" filter="url(#hkGlow)"/>
  
  <g class="energy-bars" fill="#21FFF4" filter="url(#hkGlow)">
    <rect x="285" y="540" width="18" height="30" rx="9"/>
    <rect x="320" y="510" width="18" height="80" rx="9"/>
    <rect x="355" y="465" width="18" height="150" rx="9"/>
    <rect x="390" y="480" width="18" height="40" rx="9"/>
  </g>  
  <path class="arrow-path" d="M 252 514 L 370 395 L 398 434 L 494 343" 
        stroke="url(#hkLaser)" stroke-width="18" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#hkGlow)"/>
  <path class="arrow-head" d="M488 295H530V337" 
        stroke="url(#hkLaser)" stroke-width="18" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#hkGlow)"/>
</svg>
      `;
      container.appendChild(loaderDiv);
      const paths = loaderDiv.querySelectorAll("path");
      const energy = loaderDiv.querySelector(".energy-bars");

      paths.forEach((p) => {
        const len = Math.ceil(p.getTotalLength());
        p.style.strokeDasharray = `${len}`;
        p.style.strokeDashoffset = `${len}`;
      });

      const animatePath = (path, duration, delay) => {
        setTimeout(() => {
          path.style.transition = `stroke-dashoffset ${duration}s cubic-bezier(0.19, 1, 0.22, 1)`;
          path.style.strokeDashoffset = "0";
        }, delay);
      };

      const shieldPath = loaderDiv.querySelector(".shield-path");
      const arrowPath = loaderDiv.querySelector(".arrow-path");
      const arrowHead = loaderDiv.querySelector(".arrow-head");

      animatePath(shieldPath, 1.8, 100);
      animatePath(arrowPath, 1.2, 800);
      animatePath(arrowHead, 0.6, 1600);

      setTimeout(() => {
        energy.style.opacity = "1";
        energy.style.transition = "opacity 0.8s ease-out";
      }, 1200);

      setTimeout(() => {
        container.classList.add("animating-finish");
        logo.classList.add("shimmer");
        logo.style.transition = "opacity 1.2s cubic-bezier(0.4, 0, 0.2, 1)";
        logo.style.opacity = "1";

        loaderDiv.style.opacity = "0";
        loaderDiv.style.transform = "scale(1.1)";
        loaderDiv.style.transition = "opacity 0.8s ease, transform 0.8s ease";

        setTimeout(() => {
          loaderDiv.remove();
          container.classList.remove("animating-finish");
          logo.classList.remove("shimmer");
        }, 2500);
      }, 2200);
    } else {
      logo.style.opacity = "1";
    }

    let targetX = 0,
      targetY = 0;
    let curX = 0,
      curY = 0;
    let raf = null;

    const lerp = (a, b, n) => (1 - n) * a + n * b;

    const loop = () => {
      curX = lerp(curX, targetX, 0.12);
      curY = lerp(curY, targetY, 0.12);

      container.style.setProperty("--shield-tilt-x", `${curX}deg`);
      container.style.setProperty("--shield-tilt-y", `${curY}deg`);

      if (Math.abs(curX - targetX) > 0.01 || Math.abs(curY - targetY) > 0.01) {
        raf = requestAnimationFrame(loop);
      } else {
        raf = null;
      }
    };

    container.onmousemove = (e) => {
      const r = container.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width;
      const py = (e.clientY - r.top) / r.height;

      container.style.setProperty("--shield-x", `${px * 100}%`);
      container.style.setProperty("--shield-y", `${py * 100}%`);
      container.style.setProperty("--shield-glow", "1");

      targetX = (py - 0.5) * -7;
      targetY = (px - 0.5) * 7;

      if (!raf) raf = requestAnimationFrame(loop);
    };

    container.onmousedown = (e) => {
      if (e.detail > 1) e.preventDefault();
    };

    container.onmouseleave = () => {
      targetX = 0;
      targetY = 0;
      container.style.setProperty("--shield-glow", "0");
      if (!raf) raf = requestAnimationFrame(loop);
    };
  }

  init();
}
