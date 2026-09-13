import { motion, useReducedMotion } from "framer-motion";
import { useMemo } from "react";
import type { PersonaTheme } from "./persona";
import type { Pointer } from "./usePointer";

type Props = {
  intensity?: number;
  seed?: number;
  theme: PersonaTheme;
  pointer: Pointer;
  settled?: boolean;
};

function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function Particles({ intensity = 1, seed = 1, theme, pointer, settled = false }: Props) {
  const reduce = useReducedMotion();
  const dots = useMemo(() => {
    const rnd = mulberry32(42 + seed * 17);
    return Array.from({ length: 6 }, (_, i) => ({
      id: i,
      x: rnd() * 100,
      y: rnd() * 100,
      size: 1.1 + rnd() * 2.6,
      delay: rnd() * 2.2,
      dur: 3 + rnd() * 3.2,
      drift: 6 + rnd() * 16,
    }));
  }, [seed]);

  if (reduce) return null;

  const pullX = pointer.active ? pointer.nx * 18 : 0;
  const pullY = pointer.active ? pointer.ny * 14 : 0;

  return (
    <motion.div
      className="particles"
      aria-hidden="true"
      animate={{ opacity: settled ? 0 : 1 }}
      transition={{ duration: 0.5 }}
    >
      {dots.map((d) => (
        <motion.span
          key={d.id}
          className="particle"
          style={{
            left: `${d.x}%`,
            top: `${d.y}%`,
            width: d.size,
            height: d.size,
            background: theme.accent,
            boxShadow: `0 0 6px ${theme.glow}`,
          }}
          animate={{
            y: [0, -d.drift + pullY, 0],
            x: [0, d.drift * 0.35 + pullX, 0],
            opacity: [0.12, 0.42 * intensity, 0.12],
            scale: [1, 1.35, 1],
          }}
          transition={{
            duration: d.dur,
            delay: d.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </motion.div>
  );
}
