import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pages.algorithm_lab.pymoo.algorithms.moo.age import AGEMOEA
from src.pages.algorithm_lab.pymoo.algorithms.moo.moead import MOEAD
from src.pages.algorithm_lab.pymoo.algorithms.moo.nsga2 import NSGA2
from src.pages.algorithm_lab.pymoo.algorithms.moo.nsga3 import NSGA3
from src.pages.algorithm_lab.pymoo.algorithms.moo.rvea import RVEA
from src.pages.algorithm_lab.pymoo.algorithms.moo.sms import SMSEMOA
from src.pages.algorithm_lab.pymoo.algorithms.moo.spea2 import SPEA2
from src.pages.algorithm_lab.pymoo.indicators.gd import GD
from src.pages.algorithm_lab.pymoo.indicators.gd_plus import GDPlus
from src.pages.algorithm_lab.pymoo.indicators.hv import HV
from src.pages.algorithm_lab.pymoo.indicators.igd import IGD
from src.pages.algorithm_lab.pymoo.mcdm.pseudo_weights import PseudoWeights
from src.pages.algorithm_lab.pymoo.operators.crossover.dex import DEX
from src.pages.algorithm_lab.pymoo.operators.crossover.sbx import SBX
from src.pages.algorithm_lab.pymoo.operators.mutation.pm import PM
from src.pages.algorithm_lab.pymoo.optimize import minimize
from src.pages.algorithm_lab.pymoo.problems import get_problem
from src.pages.algorithm_lab.pymoo.util.ref_dirs import get_reference_directions


def run_multi_obj_optimization_test():
    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    elif platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    exp_config = st.expander("实验参数配置", expanded=True)
    with exp_config:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("全局参数")
            mode = st.radio("实验模式", ["单算法分析", "多算法对比"], horizontal=True)

            problem_group = st.selectbox(
                "问题系列",
                ["ZDT (2目标)", "DTLZ (多目标)", "WFG (复杂前沿)", "Constrained (带约束)"],
            )

            if problem_group == "ZDT (2目标)":
                problem_name = st.selectbox("测试问题", ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"])
                n_obj = 2
                n_var = 30
            elif problem_group == "DTLZ (多目标)":
                problem_name = st.selectbox("测试问题", ["dtlz1", "dtlz2", "dtlz3", "dtlz4"])
                n_obj = st.slider("目标数量 (n_obj)", 3, 5, 3)
                n_var = st.number_input("变量数量 (n_var)", min_value=n_obj + 1, value=n_obj + 10)
            elif problem_group == "Constrained (带约束)":
                problem_name = st.selectbox("测试问题", ["bnh", "srn", "tnk", "osy"])
                n_obj = 2
                n_var = None
            else:
                problem_name = st.selectbox(
                    "测试问题",
                    ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"],
                )
                n_obj = st.slider("目标数量 (n_obj)", 2, 5, 2)
                n_var = st.number_input("变量数量 (n_var)", min_value=n_obj + 1, value=n_obj + 8)

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
            n_seeds = st.number_input("重复次数 (seed)", 1, 20, 3)

            col_op1, col_op2 = st.columns(2)
            cx_type = col_op1.selectbox("交叉算子", ["SBX", "DE"])
            cx_prob = col_op1.slider("交叉概率", 0.0, 1.0, 0.9)
            mt_prob = col_op2.slider("变异概率", 0.0, 1.0, 0.1)

    if "dtlz" in problem_name or "wfg" in problem_name:
        problem = get_problem(problem_name, n_obj=n_obj, n_var=n_var)
    elif "zdt" in problem_name:
        problem = get_problem(problem_name, n_var=n_var)
    else:
        problem = get_problem(problem_name)
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

        total_runs = int(n_seeds) * len(algo_choices)
        run_idx = 0
        for name in algo_choices:
            runs = []
            for s in range(int(n_seeds)):
                st.write(f"正在运行 {name} | seed={s + 1}")
                algo = get_algo(name)
                start_t = time.perf_counter()
                res = minimize(
                    problem, algo, ("n_gen", n_gen), seed=s + 1, verbose=False, save_history=True
                )
                end_t = time.perf_counter()
                runs.append({"res": res, "time": end_t - start_t, "seed": s + 1})
                run_idx += 1
                progress_bar.progress(run_idx / total_runs)
            all_res[name] = runs

        st.subheader("Pareto 前沿结果")
        fig = plt.figure(figsize=(10, 7))

        pf = problem.pareto_front()

        if n_obj == 2:
            ax = fig.add_subplot(111)
            if pf is not None:
                ax.scatter(pf[:, 0], pf[:, 1], c="gray", alpha=0.2, s=10, label="理论前沿")

            colors = plt.cm.rainbow(np.linspace(0, 1, len(algo_choices)))
            for name, color in zip(algo_choices, colors):
                f_data = all_res[name][0]["res"].F
                if f_data is not None:
                    ax.scatter(f_data[:, 0], f_data[:, 1], color=color, s=30, label=f"{name}")
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
        else:
            ax = fig.add_subplot(111, projection="3d")
            colors = plt.cm.rainbow(np.linspace(0, 1, len(algo_choices)))
            for name, color in zip(algo_choices, colors):
                f_data = all_res[name][0]["res"].F
                if f_data is not None:
                    ax.scatter(
                        f_data[:, 0], f_data[:, 1], f_data[:, 2], color=color, s=30, label=name
                    )
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
            ax.set_zlabel("$f_3$")

        ax.legend()
        st.pyplot(fig)

        st.subheader("收敛历史 (Hypervolume)")
        fig_hv, ax_hv = plt.subplots(figsize=(10, 5))

        all_final_F = []
        for name in algo_choices:
            for r in all_res[name]:
                if r["res"].F is not None:
                    all_final_F.append(r["res"].F)

        if all_final_F:
            combined_F = np.vstack(all_final_F)
            nadir_point = np.max(combined_F, axis=0)
            ref_point_history = nadir_point * 1.1
            ref_point_history = np.maximum(ref_point_history, 1.1)
        else:
            ref_point_history = np.ones(n_obj) * 1.1

        hv_indicator = HV(ref_point=ref_point_history)
        st.caption(f"当前使用的 HV 参考点: {ref_point_history}")

        for name in algo_choices:
            res_h = all_res[name][0]["res"]
            if res_h.history:
                hv_history = []
                for algo_state in res_h.history:
                    opt = algo_state.opt
                    if opt is not None and len(opt) > 0:
                        F = opt.get("F")
                        hv_history.append(hv_indicator.do(F))
                    else:
                        hv_history.append(0.0)
                ax_hv.plot(range(len(hv_history)), hv_history, label=name)

        ax_hv.set_xlabel("Generation")
        ax_hv.set_ylabel("Hypervolume")
        ax_hv.legend()
        ax_hv.grid(True, alpha=0.3)
        st.pyplot(fig_hv)

        if n_obj > 2:
            st.write("**并行坐标图 (High-D Visualization)**")
            pc_data = []
            for name, runs in all_res.items():
                f_data = runs[0]["res"].F
                if f_data is not None:
                    temp_df = pd.DataFrame(f_data, columns=[f"f{i + 1}" for i in range(n_obj)])
                    temp_df["Algorithm"] = name
                    pc_data.append(temp_df)
            if pc_data:
                full_pc_df = pd.concat(pc_data)
                st.line_chart(full_pc_df, y=[f"f{i + 1}" for i in range(n_obj)], color="Algorithm")

        st.subheader("性能指标对比")
        metrics = []
        ref_point = ref_point_history
        hv = HV(ref_point=ref_point)
        igd = IGD(pf) if pf is not None else None
        gd = GD(pf) if pf is not None else None
        gd_plus = GDPlus(pf) if pf is not None else None

        per_seed_rows = []
        for name, runs in all_res.items():
            hv_vals = []
            igd_vals = []
            gd_vals = []
            gd_plus_vals = []
            times = []
            nd_sizes = []
            cv_means = []

            for r in runs:
                res = r["res"]
                f_data = res.F
                if f_data is None:
                    continue
                nd_sizes.append(len(f_data))
                hv_v = float(hv.do(f_data))
                hv_vals.append(hv_v)
                if igd is not None:
                    igd_v = float(igd.do(f_data))
                    gd_v = float(gd.do(f_data))
                    gdp_v = float(gd_plus.do(f_data))
                    igd_vals.append(igd_v)
                    gd_vals.append(gd_v)
                    gd_plus_vals.append(gdp_v)
                else:
                    igd_v = None
                    gd_v = None
                    gdp_v = None
                t_v = float(r["time"])
                times.append(t_v)

                cv_m = np.nan
                if hasattr(res, "CV") and res.CV is not None:
                    cv = np.asarray(res.CV).astype(float)
                    cv_m = float(np.mean(cv))
                    cv_means.append(cv_m)
                else:
                    if res.X is not None:
                        cv = problem.evaluate(res.X, return_values_of=["CV"])[0]
                        cv_m = float(np.mean(np.asarray(cv).astype(float)))
                        cv_means.append(cv_m)

                per_seed_rows.append(
                    {
                        "算法": name,
                        "seed": int(r["seed"]),
                        "HV": hv_v,
                        "IGD": igd_v if igd_v is not None else np.nan,
                        "GD": gd_v if gd_v is not None else np.nan,
                        "GD+": gdp_v if gdp_v is not None else np.nan,
                        "CV_mean": cv_m,
                        "time_s": t_v,
                        "nd_size": int(len(f_data)),
                    }
                )

            if hv_vals:
                metrics.append(
                    {
                        "算法": name,
                        "重复次数": len(hv_vals),
                        "非支配解数量(均值)": f"{float(np.mean(nd_sizes)):.1f}",
                        "HV(均值)": f"{float(np.mean(hv_vals)):.4f}",
                        "IGD(均值)": f"{float(np.mean(igd_vals)):.4f}" if igd_vals else "-",
                        "GD(均值)": f"{float(np.mean(gd_vals)):.4f}" if gd_vals else "-",
                        "GD+(均值)": f"{float(np.mean(gd_plus_vals)):.4f}" if gd_plus_vals else "-",
                        "CV均值": f"{float(np.mean(cv_means)):.4g}" if cv_means else "-",
                        "运行耗时(均值 s)": f"{float(np.mean(times)):.2f}",
                    }
                )

        if metrics:
            st.table(pd.DataFrame(metrics))
            if per_seed_rows:
                st.subheader("每个 seed 的指标明细")
                per_df = pd.DataFrame(per_seed_rows)
                st.dataframe(per_df)
                st.subheader("指标分布(箱线图)")
                for col in ["HV", "IGD", "GD", "GD+", "CV_mean", "time_s"]:
                    if col in per_df.columns and per_df[col].notna().any():
                        fig2 = plt.figure(figsize=(10, 4))
                        ax2 = fig2.add_subplot(111)
                        groups = []
                        labels = []
                        for algo in algo_choices:
                            vals = (
                                per_df.loc[per_df["算法"] == algo, col]
                                .dropna()
                                .to_numpy(dtype=float)
                            )
                            if vals.size:
                                groups.append(vals)
                                labels.append(algo)
                        if groups:
                            ax2.boxplot(groups, labels=labels, showmeans=True)
                            ax2.set_title(col)
                            ax2.grid(True, axis="y", alpha=0.2)
                            st.pyplot(fig2)

            st.subheader("导出结果")
            selected_export = st.selectbox("选择导出结果的算法", algo_choices)
            export_df = pd.DataFrame(
                all_res[selected_export][0]["res"].F, columns=[f"f{i + 1}" for i in range(n_obj)]
            )
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载选定算法的解集 (CSV)", csv, f"{selected_export}_results.csv", "text/csv"
            )

            st.subheader("决策辅助 (Pseudo-Weights)")
            st.write("利用 Pseudo-Weights 方法从 Pareto 前沿中选择折中解。")
            best_res_f = all_res[selected_export][0]["res"].F
            if best_res_f is not None:
                weights = np.ones(n_obj) / n_obj
                i = PseudoWeights(weights).do(best_res_f)
                st.write(f"选定的折中解索引: `{i}`")
                st.write(f"目标函数值: `{best_res_f[i]}`")

                if n_obj == 2:
                    fig_highlight, ax_h = plt.subplots()
                    ax_h.scatter(
                        best_res_f[:, 0], best_res_f[:, 1], c="blue", s=30, label="Pareto Front"
                    )
                    ax_h.scatter(
                        best_res_f[i, 0],
                        best_res_f[i, 1],
                        c="red",
                        s=100,
                        marker="*",
                        label="Compromise Solution",
                    )
                    ax_h.set_xlabel("$f_1$")
                    ax_h.set_ylabel("$f_2$")
                    ax_h.legend()
                    st.pyplot(fig_highlight)
        else:
            st.error("未能找到任何有效解。")


if __name__ == "__main__":
    run_multi_obj_optimization_test()
