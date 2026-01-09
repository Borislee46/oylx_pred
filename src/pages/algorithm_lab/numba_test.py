import platform
import time

import numba
import numpy as np
import pandas as pd
import streamlit as st
from numba import float64, njit

from src.pages.algorithm_lab.bench import bench
from src.pages.algorithm_lab.numba_test_func import *


def run_numba_acceleration_test():
    st.header("Numba | NumPy 加速性能深度测试")

    if platform.system() == "Windows":
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    st.caption(f"numba=={numba.__version__} | numpy=={np.__version__}")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Numba 线程数", f"{numba.get_num_threads()}")
    tl = getattr(numba, "threading_layer", None)
    info_col2.metric("Threading layer", f"{tl() if callable(tl) else '-'}")
    info_col3.metric("NumPy 版本", f"{np.__version__}")

    cfg = st.expander("基准测试参数", expanded=False)
    with cfg:
        warmup = st.number_input("warmup 次数", 0, 20, 2, 1, key="nb_warmup")
        repeat = st.number_input("repeat 次数", 1, 50, 10, 1, key="nb_repeat")
        set_threads = st.checkbox("设置 Numba 线程数", value=False, key="nb_set_threads")
        max_threads = int(numba.config.NUMBA_NUM_THREADS)
        threads = st.number_input(
            f"线程数 (最大 {max_threads})",
            1,
            max_threads,
            int(numba.get_num_threads()),
            1,
            key="nb_threads",
        )
    if set_threads:
        numba.set_num_threads(int(threads))

    test_tabs = st.tabs(
        [
            "Monte Carlo pi 估算",
            "曼德博集合 (Mandelbrot)",
            "向量化 Ufunc 测试",
            "Sobel 卷积算子",
            "Guvectorize 矩阵乘法",
            "搜索算法测试",
            "FastMath & 内存布局 & 类型",
        ]
    )

    with test_tabs[0]:
        st.write("涉及大量循环和随机数生成的计算密集型任务。")

        n_points = st.select_slider(
            "样本点数 (N)", options=[10**4, 10**5, 10**6, 10**7, 10**8], value=10**6, key="pi_n"
        )
        seed = st.number_input(
            "随机种子", min_value=0, max_value=2**31 - 1, value=1, step=1, key="pi_seed"
        )

        if st.button("运行 Pi 估算对比", key="btn_pi"):
            progress_bar = st.progress(0, text="初始化测试数据...")
            rng = np.random.default_rng(int(seed))
            x = rng.random(int(n_points), dtype=np.float64)
            y = rng.random(int(n_points), dtype=np.float64)

            t0 = time.perf_counter()
            count_inside_circle(x[:1024], y[:1024])
            compile_single = time.perf_counter() - t0
            t0 = time.perf_counter()
            count_inside_circle_parallel(x[:1024], y[:1024])
            compile_parallel = time.perf_counter() - t0

            results = []
            if n_points <= 10**6:
                b = bench(
                    lambda: python_pi_from_arrays(x, y),
                    warmup=int(warmup),
                    repeat=int(repeat),
                    progress_cb=lambda p: progress_bar.progress(p, text="运行 Pure Python 测试..."),
                )
                python_res = python_pi_from_arrays(x, y)
                row = {"实现方式": "Pure Python", **b.stats(), "结果": float(python_res)}
                results.append(row)
            else:
                st.warning(f"由于 N={n_points:,} 过大，已跳过 Pure Python 测试。")

            def run_single():
                acc = count_inside_circle(x, y)
                return 4.0 * acc / int(n_points)

            b = bench(
                lambda: run_single(),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(p, text="运行 Numba JIT 测试..."),
            )
            numba_res = float(run_single())
            results.append(
                {
                    "实现方式": "Numba JIT",
                    "compile_s": float(compile_single),
                    **b.stats(),
                    "结果": numba_res,
                }
            )

            def run_parallel():
                acc_p = count_inside_circle_parallel(x, y)
                return 4.0 * acc_p / int(n_points)

            b = bench(
                lambda: run_parallel(),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(p, text="运行 Numba Parallel 测试..."),
            )
            numba_p_res = float(run_parallel())
            results.append(
                {
                    "实现方式": "Numba Parallel",
                    "compile_s": float(compile_parallel),
                    **b.stats(),
                    "结果": numba_p_res,
                }
            )
            progress_bar.empty()

            st.table(pd.DataFrame(results))
            st.download_button(
                "下载 Pi 估算压测数据 (CSV)",
                pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
                "numba_pi_perf.csv",
                "text/csv",
            )
            err_df = pd.DataFrame(
                [
                    {"实现方式": r["实现方式"], "abs_error": abs(float(r["结果"]) - float(np.pi))}
                    for r in results
                ]
            )
            st.dataframe(err_df)
            if len(results) >= 2 and results[0].get("mean_s", 0) > 0:
                speedup = float(results[0]["mean_s"]) / float(results[1]["mean_s"])
                st.success(f"Numba JIT 相比第一个方案加速了约 **{speedup:.1f}x** (mean)")

    with test_tabs[1]:
        st.write("嵌套循环计算，测试 JIT 对复杂分支逻辑的优化。")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            h_w = st.selectbox("图像分辨率", [256, 512, 1024, 2048], index=1, key="m_res")
        with col_m2:
            max_iter = st.slider("最大迭代次数", 10, 500, 100, key="m_iter")

        if st.button("生成曼德博集合", key="btn_m"):
            mandelbrot_numba(10, 10, 10)
            times = {}

            start = time.perf_counter()
            _ = mandelbrot_python(h_w, h_w, max_iter)
            times["NumPy Vectorized"] = time.perf_counter() - start

            start = time.perf_counter()
            _ = mandelbrot_numba(h_w, h_w, max_iter)
            times["Numba JIT"] = time.perf_counter() - start

            start = time.perf_counter()
            img_nb_p = mandelbrot_numba_parallel(h_w, h_w, max_iter)
            times["Numba Parallel"] = time.perf_counter() - start

            st.json(times)
            st.image(
                img_nb_p.astype(np.float32) / max_iter,
                caption="Mandelbrot (Numba Parallel)",
                use_container_width=True,
            )

    with test_tabs[2]:
        st.write("测试 `@vectorize` 对自定义数学函数的自动 SIMD 和并行化。")

        array_size = st.select_slider(
            "数组大小", options=[10**5, 10**6, 10**7, 5 * 10**7], value=10**6, key="u_size"
        )

        if st.button("开始 Ufunc 压测", key="btn_u"):
            progress_bar = st.progress(0, text="运行 Ufunc 测试...")
            a, b = np.random.random(array_size), np.random.random(array_size)
            numba_vectorize_add(a[:10], b[:10])

            perf = []
            ref = a + b
            b0 = bench(
                lambda: a + b,
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(p, text="运行 Native NumPy 测试..."),
            )
            perf.append({"方式": "Native NumPy (+)", **b0.stats()})

            b1 = bench(
                lambda: numba_vectorize_add(a, b),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 Numba Vectorize (Single) 测试..."
                ),
            )
            out1 = numba_vectorize_add(a, b)
            perf.append({"方式": "Numba Vectorize (Single)", **b1.stats()})

            b2 = bench(
                lambda: numba_vectorize_add_parallel(a, b),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 Numba Vectorize (Parallel) 测试..."
                ),
            )
            out2 = numba_vectorize_add_parallel(a, b)
            perf.append({"方式": "Numba Vectorize (Parallel)", **b2.stats()})
            progress_bar.empty()

            st.caption(
                f"正确性: single={np.allclose(ref, out1)} | parallel={np.allclose(ref, out2)}"
            )
            st.dataframe(pd.DataFrame(perf))

    with test_tabs[3]:
        st.write("模拟图像处理中的卷积操作，典型的局部内存访问。")

        img_size = st.selectbox("模拟图像尺寸", [512, 1024, 2048, 4096], index=1, key="s_size")

        if st.button("运行 Sobel 压测", key="btn_s"):
            progress_bar = st.progress(0, text="运行 Sobel 压测...")
            test_img = np.random.random((img_size, img_size)).astype(np.float32)
            sobel_numba_parallel(test_img[:10, :10])

            b0 = bench(
                lambda: sobel_numba_parallel(test_img),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 Numba Parallel 卷积测试..."
                ),
            )
            st.metric("Numba Parallel mean(s)", f"{b0.stats()['mean_s']:.4f}")
            st.metric("Numba Parallel p95(s)", f"{b0.stats()['p95_s']:.4f}")
            st.write(f"处理速度(mean): {(img_size**2 / b0.stats()['mean_s'] / 1e6):.2f} MPixels/s")
            sobel_res = sobel_numba_parallel(test_img)
            st.image(
                sobel_res / (sobel_res.max() + 1e-6),
                caption="Sobel Filter Output",
                use_container_width=True,
            )

            small = test_img[:64, :64]
            nb_small = sobel_numba_parallel(small)
            np_small = sobel_numpy(small)
            st.caption(
                f"小样本一致性 max_abs_diff={float(np.max(np.abs(nb_small - np_small))):.6g}"
            )
            progress_bar.empty()

    with test_tabs[4]:
        st.write("测试 `@guvectorize` 处理多维数组（矩阵乘法）的性能。")

        dim = st.selectbox("矩阵维度 (N x N)", [128, 256, 512, 1024], index=1, key="g_dim")

        if st.button("运行矩阵乘法测试", key="btn_g"):
            progress_bar = st.progress(0, text="运行矩阵乘法测试...")
            A = np.random.random((dim, dim))
            B = np.random.random((dim, dim))

            numba_matmul_gu(A[:10, :10], B[:10, :10])

            perf_g = []

            ref = np.dot(A, B)
            b0 = bench(
                lambda: np.dot(A, B),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 NumPy np.dot (BLAS) 测试..."
                ),
            )
            perf_g.append({"方式": "NumPy np.dot (BLAS)", **b0.stats()})

            b1 = bench(
                lambda: numba_matmul_gu(A, B),
                warmup=int(warmup),
                repeat=int(repeat),
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 Numba Guvectorize 测试..."
                ),
            )
            out = numba_matmul_gu(A, B)
            perf_g.append({"方式": "Numba Guvectorize", **b1.stats()})

            st.dataframe(pd.DataFrame(perf_g))
            st.caption(f"正确性 max_abs_diff={float(np.max(np.abs(ref - out))):.6g}")
            st.caption(
                "注：np.dot 通常调用高度优化的 BLAS 库，Numba 在此类密集线性代数运算中通常难以超越它，但展示了灵活实现的性能潜力。"
            )
            progress_bar.empty()

    with test_tabs[5]:
        st.write("测试 JIT 对分支逻辑密集的算法加速效果。")

        list_size = st.select_slider(
            "有序数组大小", options=[10**5, 10**6, 10**7], value=10**6, key="bs_size"
        )

        if st.button("运行二分查找测试", key="btn_bs"):
            progress_bar = st.progress(0, text="初始化数据...")
            data = np.sort(np.random.randint(0, list_size * 10, list_size))
            target = data[len(data) // 2]

            n_iters = 10000

            binary_search_numba(data[:10], target)

            perf_bs = []

            b0 = bench(
                lambda: binary_search_python(data, target),
                warmup=int(warmup),
                repeat=int(repeat),
                iters=n_iters,
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 Pure Python Loop 测试..."
                ),
            )
            perf_bs.append({"方式": "Pure Python Loop", **b0.stats()})

            @njit
            def run_numba_single(arr, item):
                return binary_search_numba(arr, item)

            run_numba_single(data[:10], target)

            b1 = bench(
                lambda: run_numba_single(data, target),
                warmup=int(warmup),
                repeat=int(repeat),
                iters=n_iters,
                progress_cb=lambda p: progress_bar.progress(p, text="运行 Numba JIT 测试..."),
            )
            perf_bs.append({"方式": "Numba JIT", **b1.stats()})

            b2 = bench(
                lambda: np.searchsorted(data, target),
                warmup=int(warmup),
                repeat=int(repeat),
                iters=n_iters,
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 NumPy searchsorted 测试..."
                ),
            )
            perf_bs.append({"方式": "NumPy searchsorted", **b2.stats()})

            st.write(f"注：结果已通过 `bench` 内部迭代 {n_iters:,} 次并平摊到单次查找耗时。")
            st.table(pd.DataFrame(perf_bs)[["方式", "mean_human", "mean_s", "p50_s", "p95_s"]])
            if float(perf_bs[0]["mean_s"]) > 0:
                bs_speedup = float(perf_bs[0]["mean_s"]) / float(perf_bs[1]["mean_s"])
                st.success(f"Numba 相比 Python 循环加速了 **{bs_speedup:.1f}x** (mean)")
            progress_bar.empty()

    with test_tabs[6]:
        st.write("FastMath 影响 (Exp Sum)")
        fm_size = st.select_slider(
            "测试规模", options=[10**5, 10**6, 10**7], value=10**6, key="fm_size"
        )
        if st.button("运行 FastMath 对比"):
            progress_bar = st.progress(0, text="运行 FastMath 对比...")
            fm_data = np.random.random(fm_size)
            fastmath_off(fm_data[:10])
            fastmath_on(fm_data[:10])

            b_off = bench(
                lambda: fastmath_off(fm_data),
                warmup=1,
                repeat=5,
                progress_cb=lambda p: progress_bar.progress(p, text="运行 FastMath=False 测试..."),
            )
            b_on = bench(
                lambda: fastmath_on(fm_data),
                warmup=1,
                repeat=5,
                progress_cb=lambda p: progress_bar.progress(p, text="运行 FastMath=True 测试..."),
            )

            st.table(
                pd.DataFrame(
                    [
                        {"Mode": "FastMath=False", **b_off.stats()},
                        {"Mode": "FastMath=True", **b_on.stats()},
                    ]
                )
            )
            st.info(
                "注：FastMath 允许 Numba 使用不严格遵循 IEEE 754 的浮点优化（如重排算术运算），在某些硬件上可显著加速。"
            )
            progress_bar.empty()

        st.write("---")
        st.write("内存布局 (C vs Fortran Order)")
        layout_dim = st.selectbox("矩阵尺寸", [1000, 2000, 4000], index=1)
        if st.button("运行布局对比"):
            progress_bar = st.progress(0, text="运行布局对比...")
            C_arr = np.ascontiguousarray(np.random.random((layout_dim, layout_dim)))
            F_arr = np.asfortranarray(np.random.random((layout_dim, layout_dim)))

            sum_slice(C_arr[:10, :10])

            b_c = bench(
                lambda: sum_slice(C_arr),
                warmup=1,
                repeat=3,
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 C-order (Row Major) 测试..."
                ),
            )
            b_f = bench(
                lambda: sum_slice(F_arr),
                warmup=1,
                repeat=3,
                progress_cb=lambda p: progress_bar.progress(
                    p, text="运行 F-order (Column Major) 测试..."
                ),
            )

            st.table(
                pd.DataFrame(
                    [
                        {"Layout": "C-order (Row Major)", **b_c.stats()},
                        {"Layout": "F-order (Column Major)", **b_f.stats()},
                    ]
                )
            )
            st.caption("注：按列求和时，F-order 布局由于内存连续访问，通常比 C-order 快得多。")
            progress_bar.empty()

        st.write("---")
        st.write("显式类型 vs 自动推断 (Compilation overhead)")
        type_size = 10**6
        if st.button("运行类型推断对比"):
            type_data = np.random.random(type_size)

            t0 = time.perf_counter()

            @njit(float64(float64[:]))
            def fresh_explicit(a):
                return np.sum(a)

            fresh_explicit(type_data)
            comp_explicit = time.perf_counter() - t0

            t0 = time.perf_counter()

            @njit
            def fresh_auto(a):
                return np.sum(a)

            fresh_auto(type_data)
            comp_auto = time.perf_counter() - t0

            st.metric("显式类型编译+运行", f"{comp_explicit:.4f} s")
            st.metric("自动推断编译+运行", f"{comp_auto:.4f} s")
            st.info(
                "注：显式签名有时能略微减少 Numba 在分发 (dispatch) 时的开销，但在现代 Numba 中差异通常极小。"
            )


if __name__ == "__main__":
    run_numba_acceleration_test()
