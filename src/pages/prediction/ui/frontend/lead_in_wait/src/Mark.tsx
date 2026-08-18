import { motion, useReducedMotion } from "framer-motion";
import type { Persona, PersonaTheme } from "./persona";

const BOLT = "M28 92 L42 70 L52 78 L68 48 L78 58 L96 30";
const WAVE = "M26 88 C40 60, 48 95, 62 55 S88 40, 98 28";
const SHIELD =
  "M60 6C78 14 96 16 108 18V68C108 92 88 112 60 126C32 112 12 92 12 68V18C24 16 42 14 60 6Z";

const BAR_BASE = [
  { x: 30, w: 0.55, delay: 0 },
  { x: 44, w: 0.7, delay: 0.08 },
  { x: 58, w: 0.85, delay: 0.16 },
  { x: 72, w: 1, delay: 0.24 },
] as const;

type Props = {
  stageIndex?: number;
  persona?: Persona;
  theme: PersonaTheme;
  burstKey?: number;
  /** 0..1 academic strength (from GPA / 均分). */
  gpaNorm?: number;
  /** 0..1 language strength (from IELTS/TOEFL). */
  langNorm?: number;
  /** Freeze decorative motion (completion / cancelled). */
  settled?: boolean;
};

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

export function Mark({
  stageIndex = 0,
  persona = "signal",
  theme,
  burstKey = 0,
  gpaNorm = 0.65,
  langNorm = 0.65,
  settled = false,
}: Props) {
  const reduce = useReducedMotion();
  const g = clamp01(gpaNorm);
  const L = clamp01(langNorm);
  const boost = 1 + stageIndex * 0.1;
  const path = persona === "creative" ? WAVE : BOLT;
  const maxH = 28 + g * 50;
  const boltDur = 3.5 - L * 1.8;
  const sparkDur = boltDur;

  return (
    <motion.div
      className="mark"
      aria-hidden="true"
      animate={
        reduce || settled ? undefined : { y: [0, -3, 0], scale: [1, 1.02 * Math.min(boost, 1.22), 1] }
      }
      transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut" }}
    >
      <motion.div
        className="mark-glow"
        style={{ background: `radial-gradient(circle, ${theme.glow}, transparent 70%)` }}
        animate={
          reduce || settled
            ? undefined
            : { opacity: [0.4, 0.55 + g * 0.4, 0.4], scale: [0.9, 1.1 + g * 0.12, 0.9] }
        }
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        key={burstKey}
        className="mark-burst"
        initial={{ scale: 0.4, opacity: 0.7 }}
        animate={{ scale: 1.8, opacity: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        style={{ borderColor: theme.accent }}
      />

      <svg className="mark-svg" viewBox="0 0 120 132" fill="none">
        <defs>
          <linearGradient id="shieldFill" x1="20" y1="8" x2="100" y2="120">
            <stop offset="0%" stopColor="#e0e7ff" />
            <stop offset="55%" stopColor="#c7d2fe" />
            <stop offset="100%" stopColor="#a5b4fc" />
          </linearGradient>
          <linearGradient id="boltGrad" x1="28" y1="90" x2="96" y2="28">
            <stop offset="0%" stopColor={theme.accent2} />
            <stop offset="55%" stopColor={theme.accent} />
            <stop offset="100%" stopColor={theme.accent} />
          </linearGradient>
          <filter id="boltGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.4" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <clipPath id="shieldClip">
            <path d={SHIELD} />
          </clipPath>
        </defs>

        <motion.path
          d={SHIELD}
          fill="url(#shieldFill)"
          stroke={`${theme.accent}66`}
          strokeWidth="1.2"
          initial={reduce ? false : { opacity: 0, scale: 0.86 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 160, damping: 16 }}
          style={{ transformOrigin: "60px 66px", transformBox: "fill-box" }}
        />

        {persona === "stem" && (
          <g opacity="0.35" stroke={theme.accent} strokeWidth="0.8" fill="none">
            <path d="M60 28 L84 42 L84 70 L60 84 L36 70 L36 42 Z" />
            <path d="M60 42 L72 49 L72 63 L60 70 L48 63 L48 49 Z" />
          </g>
        )}

        <g clipPath="url(#shieldClip)">
          {BAR_BASE.map((b, i) => {
            const h = Math.max(10, maxH * b.w);
            return (
              <motion.rect
                key={`${persona}-g${g.toFixed(2)}-${i}`}
                x={b.x}
                width={persona === "finance" ? 9 : 10}
                rx={persona === "creative" ? 5 : 3}
                fill={theme.bar}
                initial={reduce || settled ? false : { height: 0, y: 114, opacity: 0.2 }}
                animate={
                  reduce || settled
                    ? { height: h, y: 114 - h, opacity: 0.75 }
                    : {
                        height: [h * 0.72, h, h * 0.84, h],
                        y: [114 - h * 0.72, 114 - h, 114 - h * 0.84, 114 - h],
                        opacity: [0.4, 0.55 + g * 0.4, 0.5, 0.85],
                      }
                }
                transition={
                  reduce
                    ? { duration: 0 }
                    : settled
                      ? { duration: 0.5, ease: "easeOut" }
                      : {
                          duration: 1.5 + (1 - g) * 0.8,
                          delay: 0.3 + b.delay,
                          repeat: Infinity,
                          ease: "easeInOut",
                        }
                }
              />
            );
          })}

          <path
            d={path}
            stroke={`${theme.accent}22`}
            strokeWidth="3.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />

          <motion.path
            key={`bolt-${persona}-${L.toFixed(2)}`}
            d={path}
            stroke="url(#boltGrad)"
            strokeWidth={2.8 + L * 1.2}
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
            filter="url(#boltGlow)"
            initial={{ pathLength: 0, opacity: 0.4 }}
            animate={
              reduce || settled
                ? { pathLength: 1, opacity: 1 }
                : { pathLength: [0, 1, 1, 0], opacity: [0.35, 1, 1, 0.25] }
            }
            transition={
              reduce
                ? { duration: 0.4 }
                : settled
                  ? { duration: 0.5, ease: "easeOut" }
                  : {
                      duration: boltDur,
                      repeat: Infinity,
                      ease: [0.4, 0, 0.2, 1],
                      times: [0, 0.38, 0.72, 1],
                    }
            }
          />

          {!reduce && !settled && (
            <motion.circle
              r={2 + L * 1.2}
              fill={theme.accent}
              filter="url(#boltGlow)"
              animate={
                persona === "creative"
                  ? { cx: [26, 40, 62, 88, 98], cy: [88, 70, 55, 40, 28], opacity: [0, 1, 1, 1, 0] }
                  : {
                      cx: [28, 42, 52, 68, 78, 96],
                      cy: [92, 70, 78, 48, 58, 30],
                      opacity: [0, 1, 1, 1, 1, 0],
                    }
              }
              transition={{ duration: sparkDur, repeat: Infinity, ease: "easeInOut" }}
            />
          )}
        </g>
      </svg>
    </motion.div>
  );
}
