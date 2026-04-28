"""
命令行：在仓库根目录执行
  .venv\\Scripts\\python.exe src/pages/algorithm_lab/native_vs_numpy_benchmark.py
可选：--sizes 10000,100000 --warmup 2 --repeat 15 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pages.algorithm_lab.bench import bench


def _run_case(
    label: str,
    n: int,
    py_fn,
    np_fn,
    *,
    warmup: int,
    repeat: int,
) -> dict:
    r_py = bench(py_fn, warmup=warmup, repeat=repeat)
    r_np = bench(np_fn, warmup=warmup, repeat=repeat)
    t_py = r_py.stats()["mean_s"]
    t_np = r_np.stats()["mean_s"]
    speedup = t_py / t_np if t_np > 0 else float("inf")
    return {
        "op": label,
        "n": n,
        "python_mean_s": t_py,
        "numpy_mean_s": t_np,
        "numpy_faster_x": speedup,
    }


def benchmark_at_n(rng: np.random.Generator, n: int, warmup: int, repeat: int) -> list[dict]:
    rows: list[dict] = []

    a = rng.random(n, dtype=np.float64)
    b = rng.random(n, dtype=np.float64)
    a_list = a.tolist()
    b_list = b.tolist()

    rows.append(
        _run_case(
            "add",
            n,
            lambda: [x + y for x, y in zip(a_list, b_list)],
            lambda: a + b,
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "multiply",
            n,
            lambda: [x * y for x, y in zip(a_list, b_list)],
            lambda: a * b,
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "subtract",
            n,
            lambda: [x - y for x, y in zip(a_list, b_list)],
            lambda: a - b,
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "divide",
            n,
            lambda: [x / y if y != 0 else 0.0 for x, y in zip(a_list, b_list)],
            lambda: np.divide(a, np.where(b == 0, 1.0, b)),
            warmup=warmup,
            repeat=repeat,
        )
    )

    x = rng.random(n, dtype=np.float64) * 0.99 + 0.01
    x_list = x.tolist()
    rows.append(
        _run_case(
            "sqrt",
            n,
            lambda: [math.sqrt(v) for v in x_list],
            lambda: np.sqrt(x),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "exp",
            n,
            lambda: [math.exp(min(v, 20.0)) for v in x_list],
            lambda: np.exp(np.clip(x, None, 20.0)),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "log",
            n,
            lambda: [math.log(v) for v in x_list],
            lambda: np.log(x),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "sin",
            n,
            lambda: [math.sin(v) for v in x_list],
            lambda: np.sin(x),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "cos",
            n,
            lambda: [math.cos(v) for v in x_list],
            lambda: np.cos(x),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "power^2.2",
            n,
            lambda: [v**2.2 for v in x_list],
            lambda: np.power(x, 2.2),
            warmup=warmup,
            repeat=repeat,
        )
    )

    rows.append(
        _run_case(
            "sum",
            n,
            lambda: sum(a_list),
            lambda: np.sum(a),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "mean",
            n,
            lambda: sum(a_list) / n,
            lambda: np.mean(a),
            warmup=warmup,
            repeat=repeat,
        )
    )
    rows.append(
        _run_case(
            "min+max",
            n,
            lambda: (min(a_list), max(a_list)),
            lambda: (np.min(a), np.max(a)),
            warmup=warmup,
            repeat=repeat,
        )
    )

    rows.append(
        _run_case(
            "dot",
            n,
            lambda: sum(x * y for x, y in zip(a_list, b_list)),
            lambda: np.dot(a, b),
            warmup=warmup,
            repeat=repeat,
        )
    )

    def py_cumsum():
        out = []
        s = 0.0
        for v in a_list:
            s += v
            out.append(s)
        return out

    rows.append(
        _run_case(
            "cumsum",
            n,
            py_cumsum,
            lambda: np.cumsum(a),
            warmup=warmup,
            repeat=repeat,
        )
    )

    k = 64
    m = n // k
    if m >= 8:
        a2 = rng.random((m, k), dtype=np.float64)
        b2 = rng.random((k, m), dtype=np.float64)
        a2_rows = a2.tolist()
        b2_cols = [[b2[j, i] for j in range(k)] for i in range(m)]

        def py_matmul():
            return [
                [sum(a2_rows[i][t] * b2_cols[j][t] for t in range(k)) for j in range(m)]
                for i in range(m)
            ]

        rows.append(
            _run_case(
                f"matmul ({m}x{k})@({k}x{m})",
                m * k * m,
                py_matmul,
                lambda: a2 @ b2,
                warmup=warmup,
                repeat=repeat,
            )
        )

    return rows


def _fmt_s(t: float) -> str:
    if t < 1e-6:
        return f"{t * 1e9:.1f} ns"
    if t < 1e-3:
        return f"{t * 1e6:.2f} us"
    if t < 1:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.3f} s"


def main() -> None:
    p = argparse.ArgumentParser(description="Python 原生循环 vs NumPy 向量基准")
    p.add_argument(
        "--sizes",
        default="5000,50000,200000",
        help="逗号分隔数组长度（matmul 用子块时会自动调整）",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    all_rows: list[dict] = []
    for n in sizes:
        rng = np.random.default_rng(args.seed + n)
        all_rows.extend(benchmark_at_n(rng, n, args.warmup, args.repeat))

    w_op = max(len(r["op"]) for r in all_rows)
    print(f"warmup={args.warmup} repeat={args.repeat} seed={args.seed}")
    print()
    header = f"{'op':<{w_op}}  {'n':>10}  {'python':>12}  {'numpy':>12}  {'py/np':>10}"
    print(header)
    print("-" * len(header))
    for r in all_rows:
        print(
            f"{r['op']:<{w_op}}  {r['n']:>10}  "
            f"{_fmt_s(r['python_mean_s']):>12}  {_fmt_s(r['numpy_mean_s']):>12}  "
            f"{r['numpy_faster_x']:>9.1f}x"
        )

    ratios = [r["numpy_faster_x"] for r in all_rows if math.isfinite(r["numpy_faster_x"])]
    if ratios:
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        print()
        print(f"geometric_mean_speedup(all_cases)={geo:.1f}x")


if __name__ == "__main__":
    main()
