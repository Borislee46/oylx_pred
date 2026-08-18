import { useEffect, useRef } from "react";
import { useReducedMotion } from "framer-motion";
import type { Pointer } from "./usePointer";

type Props = {
  accent: string;
  accent2: string;
  bg?: string;
  pointer: Pointer;
  intensity?: number;
  /** LLM is streaming — smoke drifts a little faster and "breathes". */
  active?: boolean;
  /** Fade the field out when the wait settles (completion / cancelled). */
  settled?: boolean;
};

const VERT = `
attribute vec2 a;
void main(){ gl_Position = vec4(a,0.,1.); }
`;

/** Lightweight: 3-octave FBM, single sample, mediump. */
const FRAG = `
precision mediump float;
uniform vec2 uRes;
uniform float uTime;
uniform vec2 uPtr;
uniform vec3 uA;
uniform vec3 uB;
uniform vec3 uBg;
uniform float uInt;
uniform float uAct;
uniform float uSettle;

float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){
  vec2 i=floor(p), f=fract(p);
  float a=hash(i), b=hash(i+vec2(1.,0.)), c=hash(i+vec2(0.,1.)), d=hash(i+vec2(1.,1.));
  vec2 u=f*f*(3.-2.*f);
  return mix(a,b,u.x)+(c-a)*u.y*(1.-u.x)+(d-b)*u.x*u.y;
}
float fbm(vec2 p){
  float v=0., a=.5;
  for(int i=0;i<3;i++){ v+=a*noise(p); p*=2.05; a*=.5; }
  return v;
}

void main(){
  vec2 p = (gl_FragCoord.xy - .5*uRes)/min(uRes.x,uRes.y);
  vec2 q = gl_FragCoord.xy / uRes;
  vec2 ptr = (uPtr - .5*uRes)/min(uRes.x,uRes.y);
  float t = uTime*(.10 + .06*uAct);
  float dens = smoothstep(.3,.8, fbm(p*1.5 + vec2(t*.6, -t*.4)));
  float glow = exp(-4.0*length(p - ptr*.3)) * (.3 + .5*uInt) * uSettle;
  // A slow "thinking" breath — the field swells gently while tokens stream.
  float breathe = .88 + .2*uAct*(.5+.5*sin(uTime*1.7));

  // Anchor the smoke colour to the page background, tinted only slightly by
  // the accent — reads as natural fog instead of a coloured sheet.
  vec3 col = mix(uBg, uA, dens*.4) + uA * glow*.8;

  // Fade to fully transparent at the canvas border (works at any aspect
  // ratio), so the smoke never ends in a hard rectangle.
  float edge = smoothstep(0.0, .10, min(min(q.x, 1.-q.x), min(q.y, 1.-q.y)));
  float vign = smoothstep(1.2,.4, length(p)) * edge;
  float a = (.08 + dens*.13) * vign * uSettle * breathe;

  gl_FragColor = vec4(col * vign, a);
}
`;

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "").trim();
  if (!/^[0-9a-fA-F]{3,8}$/.test(h)) return [0.2, 0.8, 0.9]; // fallback cyan
  const expanded = h.length === 3 ? h[0] + h[0] + h[1] + h[1] + h[2] + h[2] : h.padEnd(6, "0");
  const n = parseInt(expanded.slice(0, 6), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const BG_FALLBACK = hexToRgb("#0b1220");

export function WebGLField({
  accent,
  accent2,
  bg,
  pointer,
  intensity = 1,
  active = false,
  settled = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();
  const ptrRef = useRef(pointer);
  const intenRef = useRef(intensity);
  const activeRef = useRef(active);
  const actRef = useRef(0);
  const settledRef = useRef(settled);
  const settleRef = useRef(1);
  const accentRef = useRef([0.2, 0.8, 0.9] as [number, number, number]);
  const accent2Ref = useRef([0.02, 0.7, 0.8] as [number, number, number]);
  const bgRef = useRef(BG_FALLBACK);
  ptrRef.current = pointer;
  intenRef.current = intensity;
  activeRef.current = active;
  settledRef.current = settled;
  accentRef.current = hexToRgb(accent);
  accent2Ref.current = hexToRgb(accent2);
  bgRef.current = bg ? hexToRgb(bg) : BG_FALLBACK;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let raf = 0;
    let lastDraw = 0;
    const FRAME_MS = 1000 / 30;
    let gl: WebGLRenderingContext | null = null;
    let prog: WebGLProgram | null = null;
    let buf: WebGLBuffer | null = null;
    let ro: ResizeObserver | null = null;
    let io: IntersectionObserver | null = null;
    let t0 = 0;
    let visible = true;
    let disposed = false;
    const uniforms: Record<string, WebGLUniformLocation | null> = {};

    const stopRaf = () => {
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    };

    const resize = () => {
      if (!gl || !prog) return;
      const scale = 0.5; // fixed half-res for perf; DPR is deliberately ignored
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.max(1, Math.floor(w * scale));
      canvas.height = Math.max(1, Math.floor(h * scale));
      gl.viewport(0, 0, canvas.width, canvas.height);
      if (uniforms.uRes) gl.uniform2f(uniforms.uRes, canvas.width, canvas.height);
    };

    const build = () => {
      gl = canvas.getContext("webgl", {
        alpha: true,
        antialias: false,
        premultipliedAlpha: true,
        powerPreference: "low-power",
      });
      if (!gl) return;

      const compile = (type: number, src: string): WebGLShader | null => {
        const s = gl!.createShader(type);
        if (!s) return null;
        gl!.shaderSource(s, src);
        gl!.compileShader(s);
        if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) {
          console.warn("[WebGLField] shader compile failed", gl!.getShaderInfoLog(s));
          gl!.deleteShader(s);
          return null;
        }
        return s;
      };
      const vs = compile(gl.VERTEX_SHADER, VERT);
      const fs = compile(gl.FRAGMENT_SHADER, FRAG);
      if (!vs || !fs) return;

      prog = gl.createProgram();
      if (!prog) return;
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.warn("[WebGLField] program link failed", gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        prog = null;
        return;
      }
      gl.useProgram(prog);

      buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, "a");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

      for (const name of ["uRes", "uTime", "uPtr", "uA", "uB", "uBg", "uInt", "uAct", "uSettle"]) {
        uniforms[name] = gl.getUniformLocation(prog, name);
      }

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      resize();
      if (!ro) {
        ro = new ResizeObserver(resize);
        ro.observe(canvas);
      }

      t0 = performance.now();
      lastDraw = 0;
      stopRaf();
      raf = requestAnimationFrame(frame);
    };

    const frame = (now: number) => {
      if (disposed) return;
      raf = requestAnimationFrame(frame);
      if (!visible || !gl || !prog) return;
      if (now - lastDraw < FRAME_MS) return;
      lastDraw = now;

      // Pull latest accent colours from refs each frame (no context teardown on prop change)
      if (uniforms.uA) gl.uniform3fv(uniforms.uA, accentRef.current);
      if (uniforms.uB) gl.uniform3fv(uniforms.uB, accent2Ref.current);
      if (uniforms.uBg) gl.uniform3fv(uniforms.uBg, bgRef.current);

      // Ease the whole field out on settle; snap instantly under reduced motion.
      const target = settledRef.current ? 0 : 1;
      const eased = settleRef.current + (target - settleRef.current) * 0.1;
      settleRef.current = Math.abs(target - eased) < 0.004 ? target : eased;
      const actTarget = activeRef.current ? 1 : 0;
      const actEased = actRef.current + (actTarget - actRef.current) * 0.07;
      actRef.current = Math.abs(actTarget - actEased) < 0.004 ? actTarget : actEased;

      const p = ptrRef.current;
      const sx = canvas.width / Math.max(canvas.clientWidth, 1);
      const sy = canvas.height / Math.max(canvas.clientHeight, 1);
      if (uniforms.uTime) gl.uniform1f(uniforms.uTime, reduce ? 0 : (now - t0) / 1000);
      if (uniforms.uPtr) {
        gl.uniform2f(
          uniforms.uPtr,
          (p.active ? p.x : canvas.clientWidth * 0.5) * sx,
          (canvas.clientHeight - (p.active ? p.y : canvas.clientHeight * 0.45)) * sy,
        );
      }
      if (uniforms.uInt) gl.uniform1f(uniforms.uInt, intenRef.current * settleRef.current);
      if (uniforms.uAct) gl.uniform1f(uniforms.uAct, actRef.current);
      if (uniforms.uSettle) gl.uniform1f(uniforms.uSettle, settleRef.current);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    const teardownGL = () => {
      stopRaf();
      ro?.disconnect();
      ro = null;
      if (gl && prog) {
        gl.deleteProgram(prog);
        prog = null;
      }
      if (gl && buf) {
        gl.deleteBuffer(buf);
        buf = null;
      }
    };

    build();

    io = new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting) && !document.hidden;
    });
    io.observe(canvas);

    const onVisibility = () => {
      if (document.hidden) visible = false;
    };
    document.addEventListener("visibilitychange", onVisibility);

    // After a context restore the GL state is fully reset — rebuild from scratch.
    const onContextLost = (e: Event) => {
      e.preventDefault();
      teardownGL();
    };
    const onContextRestored = () => {
      if (!disposed) build();
    };
    canvas.addEventListener("webglcontextlost", onContextLost);
    canvas.addEventListener("webglcontextrestored", onContextRestored);

    return () => {
      disposed = true;
      teardownGL();
      io?.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("webglcontextlost", onContextLost);
      canvas.removeEventListener("webglcontextrestored", onContextRestored);
    };
  }, []);

  return <canvas ref={canvasRef} className="webgl-field" aria-hidden="true" />;
}
