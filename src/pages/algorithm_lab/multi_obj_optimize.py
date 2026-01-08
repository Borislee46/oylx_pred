import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


def run_multi_obj_optimization_test():
    st.header("多目标优化实验")
    st.caption("pymoo==0.6.1.6")

    exp_config = st.expander("实验参数配置", expanded=True)
    with exp_config:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("全局参数")
            mode = st.radio("实验模式", ["单算法分析", "多算法对比"], horizontal=True)

            problem_group = st.selectbox(
                "问题系列", ["ZDT (2目标)", "DTLZ (多目标)", "WFG (复杂前沿)"]
            )

            if problem_group == "ZDT (2目标)":
                problem_name = st.selectbox("测试问题", ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"])
                n_obj = 2
            elif problem_group == "DTLZ (多目标)":
                problem_name = st.selectbox("测试问题", ["dtlz1", "dtlz2", "dtlz3", "dtlz4"])
                n_obj = st.slider("目标数量 (n_obj)", 3, 5, 3)
            else:
                problem_name = st.selectbox(
                    "测试问题",
                    ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"],
                )
                n_obj = st.slider("目标数量 (n_obj)", 2, 5, 2)

        with col2:
            st.subheader("算法与算子")
            algo_list = ["NSGA-II", "NSGA-III", "MOEA/D", "AGE-MOEA", "SMS-EMOA", "SPEA2", "RVEA"]
            if mode == "单算法分析":
                algo_choices = [st.selectbox("算法选择", algo_list)]
            else:
                algo_choices = st.multiselect(
                    "选择对比算法", algo_list, default=["NSGA-II", "NSGA-III"]
                )

            pop_size = st.number_input("种群大小", 20, 500, 100)
            n_gen = st.slider("迭代代数", 10, 500, 100)

            col_op1, col_op2 = st.columns(2)
            cx_type = col_op1.selectbox("交叉算子", ["SBX", "DE"])
            cx_prob = col_op1.slider("交叉概率", 0.0, 1.0, 0.9)
            mt_prob = col_op2.slider("变异概率", 0.0, 1.0, 0.1)

    problem = (
        get_problem(problem_name, n_obj=n_obj)
        if ("dtlz" in problem_name or "wfg" in problem_name)
        else get_problem(problem_name)
    )
    crossover = SBX(prob=cx_prob, eta=20) if cx_type == "SBX" else DEX(prob=cx_prob)
    mutation = PM(prob=mt_prob, eta=20)

    def get_algo(choice):
        if choice == "NSGA-II":
            return NSGA2(pop_size=pop_size, crossover=crossover, mutation=mutation)
        elif choice == "NSGA-III":
            ref_dirs = get_reference_directions("energy", n_obj, n_points=pop_size)
            return NSGA3(
                pop_size=pop_size, ref_dirs=ref_dirs, crossover=crossover, mutation=mutation
            )
        elif choice == "MOEA/D":
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            return MOEAD(ref_dirs, n_neighbors=15, crossover=crossover, mutation=mutation)
        elif choice == "AGE-MOEA":
            return AGEMOEA(pop_size=pop_size, crossover=crossover, mutation=mutation)
        elif choice == "SMS-EMOA":
            return SMSEMOA(pop_size=pop_size, crossover=crossover, mutation=mutation)
        elif choice == "SPEA2":
            return SPEA2(pop_size=pop_size, crossover=crossover, mutation=mutation)
        elif choice == "RVEA":
            ref_dirs = get_reference_directions("energy", n_obj, n_points=pop_size)
            return RVEA(
                pop_size=pop_size, ref_dirs=ref_dirs, crossover=crossover, mutation=mutation
            )
        return None

    if st.button("开始优化实验"):
        if not algo_choices:
            st.warning("请至少选择一个算法。")
            return

        all_res = {}
        progress_bar = st.progress(0)

        for i, name in enumerate(algo_choices):
            st.write(f"正在运行 {name}...")
            algo = get_algo(name)
            start_t = time.perf_counter()
            res = minimize(problem, algo, ("n_gen", n_gen), seed=1, verbose=False)
            end_t = time.perf_counter()
            all_res[name] = {"res": res, "time": end_t - start_t}
            progress_bar.progress((i + 1) / len(algo_choices))

        st.subheader("Pareto 前沿结果")
        fig = plt.figure(figsize=(10, 7))

        pf = problem.pareto_front()

        if n_obj == 2:
            ax = fig.add_subplot(111)
            if pf is not None:
                ax.scatter(pf[:, 0], pf[:, 1], c="gray", alpha=0.2, s=10, label="理论前沿")

            colors = plt.cm.rainbow(np.linspace(0, 1, len(algo_choices)))
            for (name, data), color in zip(all_res.items(), colors):
                f_data = data["res"].F
                if f_data is not None:
                    ax.scatter(f_data[:, 0], f_data[:, 1], color=color, s=30, label=f"{name}")
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
        else:
            ax = fig.add_subplot(111, projection="3d")
            colors = plt.cm.rainbow(np.linspace(0, 1, len(algo_choices)))
            for (name, data), color in zip(all_res.items(), colors):
                f_data = data["res"].F
                if f_data is not None:
                    ax.scatter(
                        f_data[:, 0], f_data[:, 1], f_data[:, 2], color=color, s=30, label=name
                    )
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
            ax.set_zlabel("$f_3$")

        ax.legend()
        st.pyplot(fig)

        if n_obj > 2:
            st.write("**并行坐标图 (High-D Visualization)**")
            pc_data = []
            for name, data in all_res.items():
                f_data = data["res"].F
                if f_data is not None:
                    temp_df = pd.DataFrame(f_data, columns=[f"f{i + 1}" for i in range(n_obj)])
                    temp_df["Algorithm"] = name
                    pc_data.append(temp_df)
            if pc_data:
                full_pc_df = pd.concat(pc_data)
                st.line_chart(full_pc_df, y=[f"f{i + 1}" for i in range(n_obj)], color="Algorithm")

        st.subheader("性能指标对比")
        metrics = []
        for name, data in all_res.items():
            f_data = data["res"].F
            if f_data is not None:
                ref_point = np.ones(n_obj) * 1.1
                hv_val = HV(ref_point=ref_point).do(f_data)

                igd_val = "-"
                if pf is not None:
                    igd_val = f"{IGD(pf).do(f_data):.4f}"

                metrics.append(
                    {
                        "算法": name,
                        "非支配解数量": len(f_data),
                        "Hypervolume (HV)": f"{hv_val:.4f}",
                        "IGD": igd_val,
                        "运行耗时 (s)": f"{data['time']:.2f}",
                    }
                )

        if metrics:
            st.table(pd.DataFrame(metrics))

            st.subheader("导出结果")
            selected_export = st.selectbox("选择导出结果的算法", algo_choices)
            export_df = pd.DataFrame(
                all_res[selected_export]["res"].F, columns=[f"f{i + 1}" for i in range(n_obj)]
            )
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载选定算法的解集 (CSV)", csv, f"{selected_export}_results.csv", "text/csv"
            )
        else:
            st.error("未能找到任何有效解。")


if __name__ == "__main__":
    run_multi_obj_optimization_test()
