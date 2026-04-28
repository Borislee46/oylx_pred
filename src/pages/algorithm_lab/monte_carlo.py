import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc

from src.pages.algorithm_lab.bench import bench


def run_qmc_efficiency_test():
    st.header("Quasi-Monte Carlo 实验")

    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    elif platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False

    tab1, tab2, tab3, tab4 = st.tabs(["采样可视化", "收敛性测试", "效率压测", "MultinomialQMC"])

    with tab1:
        st.subheader("采样均匀性与差异度对比")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_points = st.slider("抽样点数", 16, 2048, 512, step=16)
        with col2:
            scramble = st.checkbox("Scramble", value=True)
        with col3:
            disc_method = st.selectbox("Discrepancy Type", ["L2-star", "CD", "WD", "MD"])

        engines = {
            "Random (MC)": None,
            "Sobol": qmc.Sobol(d=2, scramble=scramble),
            "Halton": qmc.Halton(d=2, scramble=scramble),
            "LHS": qmc.LatinHypercube(d=2),
            "Poisson Disk": qmc.PoissonDisk(d=2),
            "MV Normal": qmc.MultivariateNormalQMC(mean=[0.5, 0.5]),
        }

        cols = st.columns(2)
        for i, (name, engine) in enumerate(engines.items()):
            with cols[i % 2]:
                st.write(f"**{name}**")
                if name == "Random (MC)":
                    sample = np.random.uniform(0, 1, size=(n_points, 2))
                elif name == "Sobol":
                    n_sobol = 2 ** int(np.log2(n_points))
                    sample = engine.random(n=n_sobol)
                else:
                    try:
                        sample = engine.random(n=n_points)
                    except Exception:
                        sample = np.random.uniform(0, 1, size=(n_points, 2))

                df = pd.DataFrame(sample, columns=["x", "y"])
                st.scatter_chart(df, x="x", y="y", height=300)

                if name != "MV Normal":
                    disc = qmc.discrepancy(sample, method=disc_method)
                    st.caption(f"{disc_method.upper()} Discrepancy: {disc:.6f}")
                else:
                    st.caption("MV Normal 不在单位超立方体内，跳过差异度计算")

    with tab2:
        st.subheader("蒙特卡罗积分收敛性")
        st.write("测试 QMC 与传统 MC 在不同测试函数下的收敛速度。")

        col_func, col_dim, col_n = st.columns(3)
        with col_func:
            func_choice = st.selectbox(
                "测试函数",
                [
                    "Sum(x) (有真值)",
                    "Prod(x) (有真值)",
                    "Exp(-sum(x)) (有真值)",
                    "Sphere",
                    "Rosenbrock",
                    "Ackley",
                    "Genz's Oscillatory",
                    "Griewank",
                    "Rastrigin",
                ],
            )
        with col_dim:
            test_dims = st.selectbox("测试维度", [2, 5, 10, 20, 50, 100], index=2)
        with col_n:
            max_n_log = st.slider("最大样本量 (2^n)", 5, 14, 12)

        seed = st.number_input(
            "随机种子", min_value=0, max_value=2**31 - 1, value=1, step=1, key="qmc_seed"
        )
        repeats = st.number_input(
            "重复次数", min_value=1, max_value=20, value=3, step=1, key="qmc_repeats"
        )
        warmup = st.number_input("warmup 次数", 0, 20, 1, 1, key="qmc_warmup")

        def get_test_func(name):
            if name == "Sum(x) (有真值)":
                return lambda x: np.sum(x, axis=1)
            if name == "Prod(x) (有真值)":
                return lambda x: np.prod(x, axis=1)
            if name == "Exp(-sum(x)) (有真值)":
                return lambda x: np.exp(-np.sum(x, axis=1))
            if name == "Sphere":
                return lambda x: np.sum(x**2, axis=1)
            elif name == "Rosenbrock":
                return lambda x: np.sum(
                    100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1
                )
            elif name == "Ackley":
                return (
                    lambda x: -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1)))
                    - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1))
                    + 20
                    + np.e
                )
            elif name == "Genz's Oscillatory":
                return lambda x: np.cos(2 * np.pi * 0.5 + np.sum(x, axis=1))
            elif name == "Griewank":
                return (
                    lambda x: 1
                    + np.sum(x**2, axis=1) / 4000
                    - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
                )
            elif name == "Rastrigin":
                return lambda x: 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
            return lambda x: np.mean(x, axis=1)

        def get_true_value(name, d):
            if name == "Sum(x) (有真值)":
                return d / 2.0
            if name == "Prod(x) (有真值)":
                return 0.5**d
            if name == "Exp(-sum(x)) (有真值)":
                return (1.0 - np.e ** (-1.0)) ** d
            return None

        if st.button("开始收敛性实验"):
            n_vals = 2 ** np.arange(5, max_n_log + 1)
            results = []

            f = get_test_func(func_choice)
            true_val = get_true_value(func_choice, int(test_dims))

            progress_bar = st.progress(0)
            for i, n in enumerate(n_vals):
                mc_ests = []
                qmc_ests = []
                mc_times = []
                qmc_times = []
                for r in range(int(repeats)):
                    rng = np.random.default_rng(int(seed) + r)
                    mc_samples = rng.random((int(n), int(test_dims)), dtype=np.float64)

                    def run_mc():
                        return float(np.mean(f(mc_samples)))

                    b = bench(run_mc, warmup=int(warmup), repeat=1)
                    mc_times.append(float(b.times[0]))
                    mc_ests.append(float(run_mc()))

                    engine = qmc.Sobol(d=int(test_dims), scramble=True, seed=int(seed) + r)
                    n_sobol = int(n)
                    qmc_samples = engine.random(n=n_sobol)

                    def run_qmc():
                        return float(np.mean(f(qmc_samples)))

                    b = bench(run_qmc, warmup=int(warmup), repeat=1)
                    qmc_times.append(float(b.times[0]))
                    qmc_ests.append(float(run_qmc()))

                mc_est = float(np.mean(mc_ests))
                qmc_est = float(np.mean(qmc_ests))
                row = {"n": int(n), "Random (MC)": mc_est, "Sobol (QMC)": qmc_est}
                if true_val is not None:
                    row["MC abs_error"] = abs(mc_est - float(true_val))
                    row["QMC abs_error"] = abs(qmc_est - float(true_val))
                    row["MC abs_error_std"] = (
                        float(np.std(np.abs(np.asarray(mc_ests) - float(true_val)), ddof=1))
                        if len(mc_ests) >= 2
                        else 0.0
                    )
                    row["QMC abs_error_std"] = (
                        float(np.std(np.abs(np.asarray(qmc_ests) - float(true_val)), ddof=1))
                        if len(qmc_ests) >= 2
                        else 0.0
                    )
                row["MC time_mean_s"] = float(np.mean(mc_times))
                row["QMC time_mean_s"] = float(np.mean(qmc_times))
                results.append(row)
                progress_bar.progress((i + 1) / len(n_vals))

            res_df = pd.DataFrame(results).set_index("n")
            for col in ["MC time_mean_s", "QMC time_mean_s"]:
                new_col = col.replace("_s", "_human")
                res_df[new_col] = res_df[col].apply(
                    lambda x: f"{x * 1e3:.2f} ms" if x < 1 else f"{x:.2f} s"
                )

            if "MC abs_error" in res_df.columns:
                st.subheader("收敛曲线 (Log-Log Scale)")
                fig, ax = plt.subplots(figsize=(10, 6))

                x = res_df.index
                # MC
                ax.plot(x, res_df["MC abs_error"], "o-", label="Random (MC)", color="tab:blue")
                if "MC abs_error_std" in res_df.columns:
                    ax.fill_between(
                        x,
                        res_df["MC abs_error"] - res_df["MC abs_error_std"],
                        res_df["MC abs_error"] + res_df["MC abs_error_std"],
                        alpha=0.2,
                        color="tab:blue",
                    )

                # QMC
                ax.plot(x, res_df["QMC abs_error"], "s-", label="Sobol (QMC)", color="tab:orange")
                if "QMC abs_error_std" in res_df.columns:
                    ax.fill_between(
                        x,
                        res_df["QMC abs_error"] - res_df["QMC abs_error_std"],
                        res_df["QMC abs_error"] + res_df["QMC abs_error_std"],
                        alpha=0.2,
                        color="tab:orange",
                    )

                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xlabel("Sample Size (N)")
                ax.set_ylabel("Absolute Error")
                ax.grid(True, which="both", ls="-", alpha=0.5)
                ax.legend()
                st.pyplot(fig)
            else:
                st.subheader("估值演进曲线")
                st.line_chart(res_df[["Random (MC)", "Sobol (QMC)"]])

            st.table(res_df)
            st.download_button(
                "下载收敛实验数据 (CSV)",
                res_df.to_csv().encode("utf-8"),
                "qmc_convergence.csv",
                "text/csv",
            )

            st.write("**估值稳定性分析**")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("MC 最后估值", f"{results[-1]['Random (MC)']:.6f}")
            metrics_col2.metric("Sobol 最后估值", f"{results[-1]['Sobol (QMC)']:.6f}")
            if "MC abs_error" in res_df.columns:
                metrics_col3, metrics_col4 = st.columns(2)
                metrics_col3.metric("真值", f"{float(true_val):.6f}")
                metrics_col4.metric(
                    "最后 abs_error (MC/QMC)",
                    f"{results[-1]['MC abs_error']:.6g} / {results[-1]['QMC abs_error']:.6g}",
                )

                err = res_df[["MC abs_error", "QMC abs_error"]].replace(0.0, np.nan).dropna()
                if err.shape[0] >= 2:
                    x_log = np.log(err.index.to_numpy(dtype=np.float64))
                    y_mc = np.log(err["MC abs_error"].to_numpy(dtype=np.float64))
                    y_qmc = np.log(err["QMC abs_error"].to_numpy(dtype=np.float64))
                    slope_mc = float(np.polyfit(x_log, y_mc, 1)[0])
                    slope_qmc = float(np.polyfit(x_log, y_qmc, 1)[0])
                    st.caption(
                        f"log-log 斜率(越负越快收敛): MC={slope_mc:.3f} | QMC={slope_qmc:.3f}"
                    )

            st.caption("注：QMC 在高维积分中通常具有更快的收敛阶 O(N^-1)，而 MC 为 O(N^-0.5)。")

    with tab3:
        st.subheader("极端高维效率压测")
        with st.form("perf_form"):
            dims = st.number_input("维度", 1, 500, 100)
            n_power = st.number_input("样本数 (2^n)", 10, 20, 15)
            submitted = st.form_submit_button("运行压测")

        if submitted:
            n_samples = 2**n_power
            status_placeholder = st.empty()
            status_placeholder.write(f"正在压测: {dims} 维, {n_samples:,} 样本...")

            perf_data = []
            engines_to_test = [
                ("Sobol", qmc.Sobol),
                ("Halton", qmc.Halton),
                ("LHS", qmc.LatinHypercube),
            ]

            progress_bar_perf = st.progress(0)
            for i, (name, eng_cls) in enumerate(engines_to_test):
                status_placeholder.write(
                    f"正在运行引擎: **{name}** ({dims} 维, {n_samples:,} 样本)..."
                )
                start = time.perf_counter()
                eng = eng_cls(d=dims)
                _ = eng.random(n=n_samples)
                duration = time.perf_counter() - start
                perf_data.append({"引擎": name, "耗时 (s)": f"{duration:.4f}"})
                progress_bar_perf.progress((i + 1) / len(engines_to_test))

            status_placeholder.success(f"压测完成: {dims} 维, {n_samples:,} 样本")
            st.table(pd.DataFrame(perf_data))
        else:
            st.info("配置参数并点击“运行压测”开始实验。")

    with tab4:
        st.subheader("MultinomialQMC 离散分布测试")
        st.write("测试 QMC 在多项分布采样中的分布均匀性。")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            pvals = st.text_input("概率分布 (逗号分隔)", "0.2, 0.3, 0.5")
            m_n_points = st.number_input("总采样数", 10, 10000, 1000)
        with m_col2:
            m_engine_type = st.selectbox("底层引擎", ["Sobol", "Halton"])

        try:
            p = np.fromstring(pvals, sep=",")
            p /= p.sum()

            if st.button("运行 Multinomial 实验"):
                base_engine = qmc.Sobol(d=1) if m_engine_type == "Sobol" else qmc.Halton(d=1)
                n_trials_actual = (
                    m_n_points if m_engine_type != "Sobol" else 2 ** int(np.log2(m_n_points))
                )
                mqmc = qmc.MultinomialQMC(p, n_trials=n_trials_actual, engine=base_engine)

                qmc_counts = mqmc.random(n=1)[0]

                mc_counts = np.random.multinomial(m_n_points, p)

                m_df = pd.DataFrame(
                    {
                        "Category": [f"Cat {i}" for i in range(len(p))],
                        "Expected": p * m_n_points,
                        "QMC Actual": qmc_counts,
                        "MC Actual": mc_counts,
                    }
                )

                st.bar_chart(m_df.set_index("Category"))

                qmc_err = np.sum(np.abs(qmc_counts - p * m_n_points))
                mc_err = np.sum(np.abs(mc_counts - p * m_n_points))

                st.metric("QMC 总绝对误差", f"{qmc_err:.2f}")
                st.metric("MC 总绝对误差", f"{mc_err:.2f}")
                st.caption("注：QMC 在离散分布采样中通常能提供更接近理论分布的频数。")
        except Exception as e:
            st.error(f"参数错误: {e}")


if __name__ == "__main__":
    run_qmc_efficiency_test()
