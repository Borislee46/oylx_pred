import { useEffect, useState, type RefObject } from "react";

export type Pointer = { x: number; y: number; nx: number; ny: number; active: boolean };

const IDLE: Pointer = { x: 0, y: 0, nx: 0, ny: 0, active: false };

/** Track pointer inside an element; nx/ny in [-1,1] from center. */
export function usePointer(ref: RefObject<HTMLElement | null>) {
  const [p, setP] = useState<Pointer>(IDLE);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      const nx = ((x / r.width) * 2 - 1) || 0;
      const ny = ((y / r.height) * 2 - 1) || 0;
      setP({ x, y, nx, ny, active: true });
    };
    const onLeave = () => setP((prev) => ({ ...prev, active: false, nx: 0, ny: 0 }));

    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [ref]);

  return p;
}
