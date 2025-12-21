HK_FORM_GLASS_TILT_CSS = r"""
:host, .hk-tilt-mount {
  display: block;
  width: 0;
  height: 0;
  overflow: hidden;
}

.hk-glass-card {
  position: relative;
  transition: opacity 0.3s ease;
  transform-style: preserve-3d;
  backface-visibility: hidden;
  will-change: transform;
  transform: perspective(1200px) translateZ(var(--hk-tilt-z, 0px)) rotateX(var(--hk-tilt-x, 0deg)) rotateY(var(--hk-tilt-y, 0deg));
}

.hk-glass-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(
    circle at var(--hk-glare-x, 50%) var(--hk-glare-y, 50%),
    rgba(255, 255, 255, 0.22) 0%,
    rgba(255, 255, 255, 0) 75%
  );
  opacity: var(--hk-glare-opacity, 0);
  pointer-events: none;
  z-index: 1;
  border-radius: inherit;
  transition: opacity 0.3s ease;
  mix-blend-mode: overlay;
}

.hk-tilt-active {
  box-shadow: 
    var(--hk-shadow-x, 0px) var(--hk-shadow-y, 25px) 50px rgba(0, 0, 0, 0.12),
    0 0 0 1px rgba(255, 255, 255, 0.1) inset,
    0 0 60px rgba(255, 255, 255, 0.03);
}

@keyframes gentle-pulse {
  0%, 100% { 
    --hk-tint-opacity: 0.15;
    box-shadow: 
      0 20px 40px rgba(0, 0, 0, 0.08),
      0 0 0 1px rgba(255, 255, 255, 0.1) inset,
      0 0 60px rgba(255, 255, 255, 0.03);
  }
  50% { 
    --hk-tint-opacity: 0.22;
    box-shadow: 
      0 20px 40px rgba(0, 0, 0, 0.12),
      0 0 0 1px rgba(255, 255, 255, 0.15) inset,
      0 0 80px rgba(255, 255, 255, 0.05);
  }
}

@media (hover: none) and (pointer: coarse) {
  .hk-glass-card {
    --hk-tilt-scale: 0.8;
  }
  
  .hk-tilt-active {
    transform: translateZ(6px) rotateX(var(--hk-tilt-x, 0deg)) rotateY(var(--hk-tilt-y, 0deg)) scale(var(--hk-tilt-scale, 1));
  }
}

@media (prefers-reduced-motion: reduce) {
  .hk-glass-card,
  .hk-glass-card::before,
  .hk-tilt-active {
    transition: none !important;
    animation: none !important;
  }
  
  .hk-glass-card::before {
    opacity: 0.05 !important;
  }
}
"""