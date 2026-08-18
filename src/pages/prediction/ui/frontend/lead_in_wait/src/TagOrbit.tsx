import { motion, useReducedMotion } from "framer-motion";
import { useMemo } from "react";
import type { PersonaTheme } from "./persona";

type Props = { tags: string[]; theme: PersonaTheme; progress: number; settled?: boolean };

export function TagOrbit({ tags, theme, progress, settled = false }: Props) {
  const reduce = useReducedMotion();
  const items = useMemo(
    () => tags.map((t) => t.trim()).filter(Boolean).slice(0, 6),
    [tags],
  );
  if (!items.length) return null;

  const spin = 30 - progress * 8;

  return (
    <motion.div
      className="tag-orbit"
      aria-hidden="true"
      animate={settled ? { opacity: 0.25 } : reduce ? undefined : { rotate: -360 }}
      transition={
        settled
          ? { duration: 0.6 }
          : reduce
            ? undefined
            : { duration: spin, repeat: Infinity, ease: "linear" }
      }
    >
      {items.map((tag, i) => {
        const angle = (i / items.length) * Math.PI * 2 - Math.PI / 2;
        const radius = 58 + (i % 2) * 10 + progress * 6;
        return (
          <motion.span
            key={tag}
            className="tag-chip"
            style={{
              color: theme.accent,
              x: Math.cos(angle) * radius,
              y: Math.sin(angle) * radius,
            }}
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 0.9, scale: 1, rotate: reduce || settled ? 0 : 360 }}
            transition={
              reduce || settled
                ? {
                    opacity: { duration: 0.35, delay: i * 0.05 },
                    scale: { type: "spring", stiffness: 140, damping: 14, delay: i * 0.05 },
                  }
                : {
                    opacity: { duration: 0.35, delay: i * 0.05 },
                    scale: { type: "spring", stiffness: 140, damping: 14, delay: i * 0.05 },
                    rotate: { duration: spin, repeat: Infinity, ease: "linear" },
                  }
            }
          >
            {tag}
          </motion.span>
        );
      })}
    </motion.div>
  );
}
