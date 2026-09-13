import { motion, useReducedMotion } from "framer-motion";
import type { RefObject } from "react";
import type { PersonaTheme } from "./persona";
import type { Pointer } from "./usePointer";

type Props = {
  pointer: Pointer;
  theme: PersonaTheme;
  /** Island container — used to dissolve the aura before it hits the edge. */
  containerRef: RefObject<HTMLElement | null>;
};

const FADE_MARGIN = 120;

function smooth01(t: number) {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

export function CursorAura({ pointer, theme, containerRef }: Props) {
  const reduce = useReducedMotion();
  if (reduce || !pointer.active) return null;

  const rect = containerRef.current?.getBoundingClientRect();
  let fade = 1;
  if (rect) {
    const edgeDist = Math.min(
      pointer.x,
      pointer.y,
      rect.width - pointer.x,
      rect.height - pointer.y,
    );
    fade = smooth01(edgeDist / FADE_MARGIN);
  }

  return (
    <motion.div
      className="cursor-aura"
      style={{ background: `radial-gradient(circle, ${theme.glow}, transparent 62%)` }}
      animate={{
        left: pointer.x,
        top: pointer.y,
        opacity: fade,
        scale: 0.88 + fade * 0.12,
      }}
      transition={{ type: "spring", stiffness: 280, damping: 28, mass: 0.4 }}
      initial={false}
    />
  );
}
