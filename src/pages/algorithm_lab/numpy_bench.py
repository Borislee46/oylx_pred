import time
import math
import heapq
import numpy as np
import pandas as pd
import streamlit as st
from src.pages.algorithm_lab.bench import bench

def run_numpy_performance_test():
    st.header("NumPy 深度性能对比测试")
    st.write("本模块用于对比 NumPy 与原生 Python 或其他工具库在特定场景下的执行效率。")

    cfg = st.expander("全局基准测试参数", expanded=False)
    with cfg:
        warmup = st.number_input("warmup 次数", 0, 20, 2, 1, key="np_warmup")
        repeat = st.number_input("repeat 次数", 1, 100, 10, 1, key="np_repeat")

    tabs = st.tabs([
        "指数运算 (exp)", 
        "数组初始化 (full/fill)", 
        "排序与 Top-K (sort/heapq)",
        "裁剪与条件 (clip/where)",
        "统计聚合 (sum/mean)",
        "去重与集合 (unique)"
    ])

    with tabs[0]:
        st.subheader("math.exp vs np.exp")
        st.write("NumPy 的优势在于向量化运算，但在处理单个标量时，原生 math 库通常更快。")
        
        test_size = st.select_slider("数据规模 (N)", options=[1, 10, 100, 1000, 10000, 100000], value=1000, key="exp_n")
        
        if st.button("运行指数运算对比", key="btn_exp"):
            data = np.random.uniform(-1, 1, test_size)
            py_data = data.tolist()
            
            results = []
            
            # 1. math.exp (Python list loop)
            def py_exp_loop():
                return [math.exp(x) for x in py_data]
            b1 = bench(py_exp_loop, warmup=warmup, repeat=repeat)
            results.append({"方式": "Python loop + math.exp", **b1.stats()})
            
            # 2. np.exp (Vectorized)
            def np_exp_vec():
                return np.exp(data)
            b2 = bench(np_exp_vec, warmup=warmup, repeat=repeat)
            results.append({"方式": "NumPy Vectorized (np.exp)", **b2.stats()})
            
            # 3. math.exp (Single scalar - if N=1)
            if test_size == 1:
                val = py_data[0]
                b3 = bench(lambda: math.exp(val), warmup=warmup, repeat=repeat, iters=1000)
                results.append({"方式": "math.exp (Single scalar, amortized)", **b3.stats()})
                
                b4 = bench(lambda: np.exp(val), warmup=warmup, repeat=repeat, iters=1000)
                results.append({"方式": "np.exp (Single scalar, amortized)", **b4.stats()})

            st.table(pd.DataFrame(results))
            if len(results) >= 2:
                speedup = results[0]["mean_s"] / results[1]["mean_s"]
                if speedup > 1:
                    st.success(f"NumPy 向量化加速了 **{speedup:.1f}x**")
                else:
                    st.info(f"原生 Python 在此规模下快了 **{1/speedup:.1f}x**")

    with tabs[1]:
        st.subheader("数组初始化性能")
        st.write("对比 `np.full`, `np.zeros` + fill, `np.empty` + fill 等方式创建填充数组。")
        
        init_size = st.select_slider("数组大小", options=[10**3, 10**4, 10**5, 10**6, 10**7], value=10**5, key="init_n")
        fill_val = 3.14
        
        if st.button("运行初始化对比", key="btn_init"):
            res_init = []
            
            # np.full
            b_full = bench(lambda: np.full(init_size, fill_val), warmup=warmup, repeat=repeat)
            res_init.append({"方式": "np.full", **b_full.stats()})
            
            # np.zeros + fill
            def zeros_fill():
                a = np.zeros(init_size)
                a.fill(fill_val)
                return a
            b_zf = bench(zeros_fill, warmup=warmup, repeat=repeat)
            res_init.append({"方式": "np.zeros + .fill()", **b_zf.stats()})
            
            # np.empty + fill
            def empty_fill():
                a = np.empty(init_size)
                a.fill(fill_val)
                return a
            b_ef = bench(empty_fill, warmup=warmup, repeat=repeat)
            res_init.append({"方式": "np.empty + .fill()", **b_ef.stats()})
            
            # np.ones * val
            b_ones = bench(lambda: np.ones(init_size) * fill_val, warmup=warmup, repeat=repeat)
            res_init.append({"方式": "np.ones * val", **b_ones.stats()})
            
            # Python list
            b_list = bench(lambda: [fill_val] * init_size, warmup=warmup, repeat=repeat)
            res_init.append({"方式": "Python List [v]*N", **b_list.stats()})
            
            st.table(pd.DataFrame(res_init))

    with tabs[2]:
        st.subheader("排序与选择 (Sort vs Argsort vs Heapq)")
        st.write("对于全排序或仅提取前 K 个元素，不同算法表现差异巨大。")
        
        sort_size = st.select_slider("数据大小", options=[10**3, 10**4, 10**5, 10**6], value=10**4, key="sort_n")
        k_val = st.number_input("Top-K 的 K 值", 1, 1000, 10, key="sort_k")
        
        if st.button("运行排序对比", key="btn_sort"):
            data = np.random.rand(sort_size)
            py_list = data.tolist()
            
            res_sort = []
            
            # np.sort
            b_npsort = bench(lambda: np.sort(data), warmup=warmup, repeat=repeat)
            res_sort.append({"方式": "np.sort (Full)", **b_npsort.stats()})
            
            # np.argsort
            b_npargsort = bench(lambda: np.argsort(data), warmup=warmup, repeat=repeat)
            res_sort.append({"方式": "np.argsort (Full indices)", **b_npargsort.stats()})
            
            # np.partition (Top-K)
            b_nppart = bench(lambda: np.partition(data, k_val)[:k_val], warmup=warmup, repeat=repeat)
            res_sort.append({"方式": "np.partition (Top-K)", **b_nppart.stats()})
            
            # heapq.nsmallest
            b_heapq = bench(lambda: heapq.nsmallest(k_val, py_list), warmup=warmup, repeat=repeat)
            res_sort.append({"方式": "heapq.nsmallest (Python List)", **b_heapq.stats()})
            
            # sorted (Python)
            b_pysort = bench(lambda: sorted(py_list), warmup=warmup, repeat=repeat)
            res_sort.append({"方式": "Python sorted()", **b_pysort.stats()})
            
            st.table(pd.DataFrame(res_sort))
            st.info("注：np.partition 是 O(N) 算法，对于极大的 N 和较小的 K 极其高效。")

    with tabs[3]:
        st.subheader("数值裁剪与条件选择")
        st.write("对比 `np.clip` 和 `np.where` 与原生 Python 循环/列表推导式的差异。")
        
        cw_size = st.select_slider("数据规模", options=[10**3, 10**4, 10**5, 10**6], value=10**5, key="cw_n")
        
        if st.button("运行裁剪/条件对比", key="btn_cw"):
            data = np.random.randn(cw_size)
            py_data = data.tolist()
            
            res_cw = []
            
            # np.clip
            b_clip = bench(lambda: np.clip(data, -0.5, 0.5), warmup=warmup, repeat=repeat)
            res_cw.append({"功能": "Clip (np.clip)", **b_clip.stats()})
            
            # Python clip
            def py_clip():
                return [max(-0.5, min(x, 0.5)) for x in py_data]
            b_pyclip = bench(py_clip, warmup=warmup, repeat=repeat)
            res_cw.append({"功能": "Clip (Python loop)", **b_pyclip.stats()})
            
            # np.where
            b_where = bench(lambda: np.where(data > 0, 1.0, -1.0), warmup=warmup, repeat=repeat)
            res_cw.append({"功能": "Where (np.where)", **b_where.stats()})
            
            # Python where
            def py_where():
                return [1.0 if x > 0 else -1.0 for x in py_data]
            b_pywhere = bench(py_where, warmup=warmup, repeat=repeat)
            res_cw.append({"功能": "Where (Python loop)", **b_pywhere.stats()})
            
            st.table(pd.DataFrame(res_cw))

    with tabs[4]:
        st.subheader("聚合运算性能")
        st.write("对比 NumPy 聚合函数与 Python 内置聚合函数的效率。")
        
        agg_size = st.select_slider("数据大小 ", options=[10**4, 10**5, 10**6, 10**7], value=10**6, key="agg_n")
        
        if st.button("运行聚合对比", key="btn_agg"):
            data = np.random.rand(agg_size)
            py_list = data.tolist()
            
            res_agg = []
            
            # np.sum
            b_npsum = bench(lambda: np.sum(data), warmup=warmup, repeat=repeat)
            res_agg.append({"功能": "Sum (np.sum)", **b_npsum.stats()})
            
            # Python sum
            b_pysum = bench(lambda: sum(py_list), warmup=warmup, repeat=repeat)
            res_agg.append({"功能": "Sum (Python builtin)", **b_pysum.stats()})
            
            # np.mean
            b_npmean = bench(lambda: np.mean(data), warmup=warmup, repeat=repeat)
            res_agg.append({"功能": "Mean (np.mean)", **b_npmean.stats()})
            
            # Python mean
            def py_mean():
                return sum(py_list) / len(py_list)
            b_pymean = bench(py_mean, warmup=warmup, repeat=repeat)
            res_agg.append({"功能": "Mean (Python loop)", **b_pymean.stats()})
            
            st.table(pd.DataFrame(res_agg))

    with tabs[5]:
        st.subheader("去重与集合运算")
        st.write("对比 `np.unique` 与 Python `set()` 的处理能力。")
        
        u_size = st.select_slider("去重规模", options=[10**3, 10**4, 10**5, 10**6], value=10**5, key="u_n")
        
        if st.button("运行去重对比", key="btn_np_u"):
            data = np.random.randint(0, u_size // 10, u_size)
            py_list = data.tolist()
            
            res_u = []
            
            # np.unique
            b_npuniq = bench(lambda: np.unique(data), warmup=warmup, repeat=repeat)
            res_u.append({"功能": "Unique (np.unique)", **b_npuniq.stats()})
            
            # Python set
            b_pyset = bench(lambda: sorted(list(set(py_list))), warmup=warmup, repeat=repeat)
            res_u.append({"功能": "Unique (Python set + sorted)", **b_pyset.stats()})
            
            st.table(pd.DataFrame(res_u))
            st.info("注：np.unique 内部通常涉及排序，而 Python set 使用哈希表，在某些情况下 set 可能更快，但 np.unique 返回的是有序数组。")
