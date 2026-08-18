import { motion, useReducedMotion } from "framer-motion";
import type { PersonaTheme } from "./persona";

type Props = { progress: number; theme: PersonaTheme; settled?: boolean };

export function OrbitRing({ progress, theme, settled = false }: Props) {
  const reduce = useReducedMotion();
  const p = Math.max(0, Math.min(1, progress));
  const r = 46;
  const c = 2 * Math.PI * r;
  const dash = c * (0.18 + p * 0.72);

  return (
    <svg className="orbit" viewBox="0 0 120 120" aria-hidden="true">
      <defs>
        <linearGradient id="orbitGrad" x1="0" y1="0" x2="120" y2="120">
          <stop offset="0%" stopColor={theme.accent2} stopOpacity="0.15" />
          <stop offset="50%" stopColor={theme.accent} stopOpacity="0.95" />
          <stop offset="100%" stopColor={theme.accent} stopOpacity="0.45" />
        </linearGradient>
      </defs>
      <circle cx="60" cy="60" r={r} fill="none" stroke={`${theme.accent}14`} strokeWidth="1.2" />
      <motion.g
        style={{ transformOrigin: "60px 60px" }}
        animate={settled ? { opacity: 0.35 } : reduce ? undefined : { rotate: 360 }}
        transition={
          settled
            ? { duration: 0.6 }
            : reduce
              ? undefined
              : { duration: 8 - p * 3, repeat: Infinity, ease: "linear" }
        }
      >
        <motion.circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          stroke="url(#orbitGrad)"
          strokeWidth="1.6"
          strokeLinecap="round"
          initial={false}
          animate={{ strokeDasharray: `${dash} ${c}` }}
          transition={{ type: "spring", stiffness: 80, damping: 18 }}
        />
      </motion.g>
    </svg>
  );
}
