import time

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc


def run_qmc_efficiency_test():
    st.header("Quasi-Monte Carlo 实验")
    st.caption("scipy==1.16.2")

    tab1, tab2, tab3 = st.tabs(["采样可视化", "收敛性测试", "效率压测"])

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
                    except:
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
                ["Sphere", "Rosenbrock", "Ackley", "Genz's Oscillatory", "Griewank", "Rastrigin"],
            )
        with col_dim:
            test_dims = st.selectbox("测试维度", [2, 5, 10, 20, 50, 100], index=2)
        with col_n:
            max_n_log = st.slider("最大样本量 (2^n)", 5, 14, 12)

        def get_test_func(name):
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

        if st.button("开始收敛性实验"):
            n_vals = 2 ** np.arange(5, max_n_log + 1)
            results = []

            f = get_test_func(func_choice)

            progress_bar = st.progress(0)
            for i, n in enumerate(n_vals):
                mc_samples = np.random.uniform(0, 1, size=(n, test_dims))
                mc_est = np.mean(f(mc_samples))

                engine = qmc.Sobol(d=test_dims, scramble=True)
                qmc_samples = engine.random(n=n)
                qmc_est = np.mean(f(qmc_samples))

                results.append({"n": n, "Random (MC)": mc_est, "Sobol (QMC)": qmc_est})
                progress_bar.progress((i + 1) / len(n_vals))

            res_df = pd.DataFrame(results).set_index("n")

            st.line_chart(res_df)

            st.write("**估值稳定性分析**")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("MC 最后估值", f"{results[-1]['Random (MC)']:.6f}")
            metrics_col2.metric("Sobol 最后估值", f"{results[-1]['Sobol (QMC)']:.6f}")

            st.caption("注：QMC 在高维积分中通常具有更快的收敛阶 O(N^-1)，而 MC 为 O(N^-0.5)。")

    with tab3:
        st.subheader("极端高维效率压测")
        with st.form("perf_form"):
            dims = st.number_input("维度", 1, 500, 100)
            n_power = st.number_input("样本数 (2^n)", 10, 20, 15)
            st.form_submit_button("运行压测")

        n_samples = 2**n_power
        st.write(f"压测中: {dims} 维, {n_samples:,} 样本...")

        perf_data = []
        for name, eng_cls in [
            ("Sobol", qmc.Sobol),
            ("Halton", qmc.Halton),
            ("LHS", qmc.LatinHypercube),
        ]:
            start = time.perf_counter()
            eng = eng_cls(d=dims)
            _ = eng.random(n=n_samples)
            perf_data.append({"引擎": name, "耗时 (s)": time.perf_counter() - start})

        st.table(pd.DataFrame(perf_data))


if __name__ == "__main__":
    run_qmc_efficiency_test()
