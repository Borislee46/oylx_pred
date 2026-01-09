import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchResult:
    times: np.ndarray
    iters: int = 1

    @property
    def n(self) -> int:
        return int(self.times.size)

    def stats(self) -> dict:
        t = (self.times / self.iters).astype(np.float64, copy=False)
        mean = float(np.mean(t))
        p50 = float(np.percentile(t, 50))
        p95 = float(np.percentile(t, 95))
        std = float(np.std(t, ddof=1)) if t.size >= 2 else 0.0

        if mean < 1e-6:
            unit, factor = "ns", 1e9
        elif mean < 1e-3:
            unit, factor = "µs", 1e6
        elif mean < 1:
            unit, factor = "ms", 1e3
        else:
            unit, factor = "s", 1.0

        return {
            "n": int(t.size),
            "mean_human": f"{mean * factor:.2f} {unit}",
            "mean_s": mean,
            "std_s": std,
            "p50_s": p50,
            "p95_s": p95,
            "min_s": float(np.min(t)),
            "max_s": float(np.max(t)),
        }


def bench(func, *, warmup: int = 2, repeat: int = 10, iters: int = 1, progress_cb=None) -> BenchResult:
    total_steps = int(warmup) + int(repeat)
    
    if warmup > 0:
        for i in range(int(warmup)):
            func()
            if progress_cb:
                progress_cb((i + 1) / total_steps)

    times = np.empty(int(repeat), dtype=np.float64)
    for i in range(int(repeat)):
        start = time.perf_counter()
        for _ in range(int(iters)):
            func()
        times[i] = time.perf_counter() - start
        if progress_cb:
            progress_cb((int(warmup) + i + 1) / total_steps)
    
    return BenchResult(times=times, iters=int(iters))
