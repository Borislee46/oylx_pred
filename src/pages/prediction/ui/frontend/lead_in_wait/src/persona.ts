export type Persona = "signal" | "finance" | "stem" | "creative";

export type PersonaTheme = {
  id: Persona;
  accent: string;
  accent2: string;
  glow: string;
  bar: string;
  accentDark?: string;
  accent2Dark?: string;
  glowDark?: string;
};

export const PERSONAS: Record<Persona, PersonaTheme> = {
  signal: {
    id: "signal",
    accent: "#2563eb",
    accent2: "#7c3aed",
    glow: "rgba(37,99,235,0.28)",
    bar: "#2563eb",
    accentDark: "#60a5fa",
    accent2Dark: "#a78bfa",
    glowDark: "rgba(96,165,250,0.3)",
  },
  finance: {
    id: "finance",
    accent: "#d97706",
    accent2: "#b45309",
    glow: "rgba(217,119,6,0.25)",
    bar: "#d97706",
    accentDark: "#fbbf24",
    accent2Dark: "#f59e0b",
    glowDark: "rgba(251,191,36,0.28)",
  },
  stem: {
    id: "stem",
    accent: "#0284c7",
    accent2: "#0ea5e9",
    glow: "rgba(2,132,199,0.28)",
    bar: "#0284c7",
    accentDark: "#38bdf8",
    accent2Dark: "#22d3ee",
    glowDark: "rgba(56,189,248,0.3)",
  },
  creative: {
    id: "creative",
    accent: "#c026d3",
    accent2: "#7c3aed",
    glow: "rgba(192,38,211,0.25)",
    bar: "#c026d3",
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
