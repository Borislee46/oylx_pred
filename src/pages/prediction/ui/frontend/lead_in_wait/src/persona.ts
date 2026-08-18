export type Persona = "signal" | "finance" | "stem" | "creative";

export type PersonaTheme = {
  id: Persona;
  label: string;
  accent: string;
  accent2: string;
  glow: string;
  /** Readable accent for text on light backgrounds (darker than `accent`). */
  accentText?: string;
  bar: string;
  ring: string;
  accentDark?: string;
  accent2Dark?: string;
  glowDark?: string;
};

export const PERSONAS: Record<Persona, PersonaTheme> = {
  signal: {
    id: "signal",
    label: "Signals",
    accent: "#2563eb",
    accent2: "#7c3aed",
    glow: "rgba(37,99,235,0.28)",
    accentText: "#1d4ed8",
    bar: "#2563eb",
    ring: "#2563eb",
    accentDark: "#60a5fa",
    accent2Dark: "#a78bfa",
    glowDark: "rgba(96,165,250,0.3)",
  },
  finance: {
    id: "finance",
    label: "商科轨迹",
    accent: "#d97706",
    accent2: "#b45309",
    glow: "rgba(217,119,6,0.25)",
    accentText: "#b45309",
    bar: "#d97706",
    ring: "#d97706",
    accentDark: "#fbbf24",
    accent2Dark: "#f59e0b",
    glowDark: "rgba(251,191,36,0.28)",
  },
  stem: {
    id: "stem",
    label: "理工信号",
    accent: "#0284c7",
    accent2: "#0ea5e9",
    glow: "rgba(2,132,199,0.28)",
    accentText: "#0369a1",
    bar: "#0284c7",
    ring: "#0284c7",
    accentDark: "#38bdf8",
    accent2Dark: "#22d3ee",
    glowDark: "rgba(56,189,248,0.3)",
  },
  creative: {
    id: "creative",
    label: "人文波形",
    accent: "#c026d3",
    accent2: "#7c3aed",
    glow: "rgba(192,38,211,0.25)",
    accentText: "#a21caf",
    bar: "#c026d3",
    ring: "#c026d3",
    accentDark: "#e879f9",
    accent2Dark: "#a78bfa",
    glowDark: "rgba(232,121,249,0.28)",
  },
};

export function asPersona(v: unknown): Persona {
  const s = String(v || "").toLowerCase();
  if (s === "finance" || s === "stem" || s === "creative" || s === "signal") return s;
  return "signal";
}
