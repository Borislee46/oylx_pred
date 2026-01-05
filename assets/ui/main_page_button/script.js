(function () {
  const wrapper = document.getElementById("cardsWrapper");
  const container = document.querySelector(".poker-container");
  const hint = document.getElementById("modeHint");
  const cardWrappers = wrapper.querySelectorAll(".card-wrapper");
  const numCards = Number("{{num_cards}}");
  let currentMode = "fan";
  let isTransitioning = false;
  let hoverTimeout = null;
  let activeWrapper = null;

  function calculateFanAngles() {
    const maxSpread = Math.min(52, 18 + numCards * 10);
    const angleStep = numCards > 1 ? maxSpread / (numCards - 1) : 0;
    const startAngle = -maxSpread / 2;
    const maxDepth = 55;
    const maxBlur = 1.5;
    const horizontalSpacing = 46;
    const centerIndex = (numCards - 1) / 2;

    cardWrappers.forEach((cw, index) => {
      const card = cw.querySelector(".card");
      const normalizedPos = numCards > 1 ? index / (numCards - 1) : 1;
      const eased = Math.pow(normalizedPos, 1.15);
      const angle = numCards > 1 ? startAngle + angleStep * index : 0;
      const depth = maxDepth * eased;
      const blur = 0.15 + maxBlur * Math.pow(1 - eased, 1.6);
      const brightness = 0.86 + 0.14 * eased;
      const zIndex = index + 1;
      const offsetX = (index - centerIndex) * horizontalSpacing;

      if (!cw.classList.contains("active")) {
        cw.style.transform = `translateX(${offsetX}px) rotate(${angle}deg) translateZ(${depth}px)`;
        cw.style.zIndex = zIndex;
        card.style.filter = `brightness(${brightness}) blur(${blur}px)`;
      }
      cw.dataset.baseAngle = angle;
      cw.dataset.baseDepth = depth;
      cw.dataset.baseZIndex = zIndex;
      cw.dataset.baseBlur = blur;
      cw.dataset.baseBrightness = brightness;
      cw.dataset.baseOffsetX = offsetX;
    });
  }

  function setActiveWrapper(cw) {
    if (activeWrapper === cw || (isTransitioning && cw !== null)) return;
    if (activeWrapper) {
      const prevCard = activeWrapper.querySelector(".card");
      activeWrapper.classList.remove("active");
      activeWrapper.style.transform = `translateX(${activeWrapper.dataset.baseOffsetX}px) rotate(${activeWrapper.dataset.baseAngle}deg) translateZ(${activeWrapper.dataset.baseDepth}px)`;
      activeWrapper.style.zIndex = activeWrapper.dataset.baseZIndex;
      prevCard.style.filter = `brightness(${activeWrapper.dataset.baseBrightness}) blur(${activeWrapper.dataset.baseBlur}px)`;
    }
    activeWrapper = cw;
    if (cw) {
      const card = cw.querySelector(".card");
      cw.classList.add("active");
      cw.style.zIndex = 100;
      const angle = parseFloat(cw.dataset.baseAngle);
      cw.style.transform = `translateX(${cw.dataset.baseOffsetX}px) rotate(${angle}deg) translateY(-55px) translateZ(100px) scale(1.08)`;
      card.style.filter = "brightness(1.05) blur(0px)";
    }
  }

  function switchToFan() {
    if (currentMode === "fan" || isTransitioning) return;
    isTransitioning = true;
    if (hoverTimeout) clearTimeout(hoverTimeout);

    cardWrappers.forEach((cw, index) => {
      cw.style.transition = `all 0.5s cubic-bezier(0.23, 1, 0.32, 1) ${
        index * 40
      }ms`;
      cw.classList.remove("hovered");
      const card = cw.querySelector(".card");
      card.style.transform = "";
      const glare = card.querySelector(".glare");
      if (glare) glare.style.opacity = "0";
    });

    wrapper.classList.remove("linear");
    wrapper.classList.add("fan");
    currentMode = "fan";
    hint.textContent = "滚轮切换布局";

    setTimeout(() => {
      calculateFanAngles();
      setTimeout(
        () => {
          cardWrappers.forEach((cw) => {
            cw.style.transition = "all 0.35s cubic-bezier(0.23, 1, 0.32, 1)";
          });
          isTransitioning = false;
        },
        300 + numCards * 40,
      );
    }, 50);
  }

  function switchToLinear() {
    if (currentMode === "linear" || isTransitioning) return;
    isTransitioning = true;
    if (hoverTimeout) clearTimeout(hoverTimeout);
    setActiveWrapper(null);

    const centerIndex = (numCards - 1) / 2;
    cardWrappers.forEach((cw, index) => {
      const card = cw.querySelector(".card");
      const distFromCenter = Math.abs(index - centerIndex);
      const delay = distFromCenter * 50;
      cw.style.transition = `all 0.5s cubic-bezier(0.23, 1, 0.32, 1) ${delay}ms`;
      cw.style.transform = "translateX(0) rotate(0deg) translateZ(0)";
      cw.style.zIndex = 1;
      card.style.filter = "brightness(1) blur(0px)";
      card.style.transform = "";
    });

    setTimeout(
      () => {
        wrapper.classList.remove("fan");
        wrapper.classList.add("linear");
        currentMode = "linear";
        hint.textContent = "滚轮切换布局";

        setTimeout(() => {
          cardWrappers.forEach((cw) => {
            cw.style.transition = "all 0.35s cubic-bezier(0.23, 1, 0.32, 1)";
          });
          isTransitioning = false;
        }, 100);
      },
      150 + Math.floor(centerIndex) * 50,
    );
  }

  let tiltTicking = false;

  function handleTilt(cw, e) {
    if (currentMode !== "linear" || isTransitioning) return;

    const rect = cw.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (!tiltTicking) {
      requestAnimationFrame(() => {
        const card = cw.querySelector(".card");
        const glare = card.querySelector(".glare");
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = ((y - centerY) / centerY) * -10;
        const rotateY = ((x - centerX) / centerX) * 10;

        card.style.transform = `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-12px) scale(1.04)`;

        if (glare) {
          const glareX = (x / rect.width) * 100;
          const glareY = (y / rect.height) * 100;
          glare.style.background = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 80%)`;
          glare.style.opacity = "0.6";
        }
        tiltTicking = false;
      });
      tiltTicking = true;
    }
  }

  function resetTilt(cw) {
    if (currentMode !== "linear") return;
    const card = cw.querySelector(".card");
    const glare = card.querySelector(".glare");
    card.style.transform = "";
    if (glare) {
      glare.style.opacity = "0";
    }
  }

  calculateFanAngles();

  let wheelTimeout = null;

  function handleWheel(e) {
    if (wheelTimeout) return;

    if (Math.abs(e.deltaY) < 5) return;

    if (e.deltaY > 0 && currentMode === "fan") {
      wheelTimeout = setTimeout(() => {
        wheelTimeout = null;
      }, 300);
      switchToLinear();
    } else if (e.deltaY < 0 && currentMode === "linear") {
      wheelTimeout = setTimeout(() => {
        wheelTimeout = null;
      }, 300);
      switchToFan();
    }
  }

  window.addEventListener("wheel", handleWheel, { passive: true });

  if (window.parent) {
    window.parent.addEventListener("wheel", handleWheel, { passive: true });

    window.addEventListener("unload", () => {
      window.parent.removeEventListener("wheel", handleWheel);
    });
  }

  cardWrappers.forEach((cw) => {
    cw.addEventListener("mouseenter", function () {
      if (isTransitioning) return;
      if (currentMode === "fan") {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        const delay = activeWrapper ? 25 : 80;
        hoverTimeout = setTimeout(() => {
          setActiveWrapper(this);
        }, delay);
      } else if (currentMode === "linear") {
        this.classList.add("hovered");
      }
    });

    cw.addEventListener("mousemove", function (e) {
      if (isTransitioning) return;
      if (currentMode === "linear") {
        handleTilt(this, e);
      }
    });

    cw.addEventListener("mouseleave", function () {
      if (isTransitioning) return;
      if (currentMode === "fan") {
        if (hoverTimeout) clearTimeout(hoverTimeout);
      } else if (currentMode === "linear") {
        this.classList.remove("hovered");
        resetTilt(this);
      }
      this.classList.remove("pressed");
    });

    cw.addEventListener("mousedown", function () {
      this.classList.add("pressed");
    });

    cw.addEventListener("mouseup", function () {
      this.classList.remove("pressed");
    });

    cw.addEventListener("touchstart", function () {
      this.classList.add("pressed");
    });

    cw.addEventListener("touchend", function () {
      this.classList.remove("pressed");
    });

    cw.addEventListener("click", function () {
      const url = this.getAttribute("data-url");
      window.open(url, "_blank");
    });
  });

  wrapper.addEventListener("mouseleave", function () {
    if (currentMode !== "fan") return;
    if (hoverTimeout) clearTimeout(hoverTimeout);
    hoverTimeout = setTimeout(() => {
      setActiveWrapper(null);
    }, 150);
  });
})();
