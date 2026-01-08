import time
import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process, utils, distance

def run_rapidfuzz_test():
    st.header("RapidFuzz 深度功能测试")
    st.caption("rapidfuzz==3.14.3")

    tab_fuzz, tab_process, tab_distance, tab_perf = st.tabs([
        "Fuzz 算法全集", 
        "Process 提取与矩阵", 
        "Distance & 编辑操作", 
        "性能压测"
    ])

    with tab_fuzz:
        st.subheader("Fuzz Module 评分对比")
        col1, col2 = st.columns(2)
        with col1:
            s1 = st.text_input("文本 1", "北京市海淀区中关村软件园", key="fuzz_s1")
        with col2:
            s2 = st.text_input("文本 2", "北京海淀中关村", key="fuzz_s2")

        fuzz_scorers = {
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "partial_token_set_ratio": fuzz.partial_token_set_ratio,
            "token_ratio": fuzz.token_ratio,
            "partial_token_ratio": fuzz.partial_token_ratio,
            "WRatio": fuzz.WRatio,
            "QRatio": fuzz.QRatio,
        }

        fuzz_results = []
        for name, scorer in fuzz_scorers.items():
            score = scorer(s1, s2)
            fuzz_results.append({"算法": name, "得分": f"{score:.2f}"})
        
        st.table(pd.DataFrame(fuzz_results))

        st.subheader("Partial Ratio Alignment (对齐分析)")
        alignment = fuzz.partial_ratio_alignment(s1, s2)
        if alignment:
            st.json({
                "score": alignment.score,
                "src_start": alignment.src_start,
                "src_end": alignment.src_end,
                "dest_start": alignment.dest_start,
                "dest_end": alignment.dest_end,
            })
            st.write(f"匹配片段: `{s1[alignment.src_start:alignment.src_end]}` <-> `{s2[alignment.dest_start:alignment.dest_end]}`")

    with tab_process:
        st.subheader("Process Module 批量操作")
        
        proc_col1, proc_col2 = st.columns([1, 2])
        with proc_col1:
            query = st.text_input("查询词", "清华大学", key="proc_query")
            limit = st.number_input("提取数量", 1, 10, 3)
            use_proc = st.checkbox("默认预处理 (utils.default_process)", value=True)
        with proc_col2:
            choices_str = st.text_area("候选词 (换行分隔)", "北京大学\n清华大学\n复旦大学\n上海交通大学\n清华", height=150)
            choices = [c.strip() for c in choices_str.split("\n") if c.strip()]

        st.write("**extractOne 最佳匹配:**")
        best = process.extractOne(query, choices, processor=utils.default_process if use_proc else None)
        if best:
            st.write(f"结果: `{best[0]}` | 得分: `{best[1]:.2f}` | 索引: `{best[2]}`")

        st.write("**extract 批量提取:**")
        matches = process.extract(query, choices, limit=limit, processor=utils.default_process if use_proc else None)
        st.table(pd.DataFrame([{"匹配项": m[0], "得分": f"{m[1]:.2f}", "索引": m[2]} for m in matches]))

        st.write("**extract_iter 迭代提取 (前 2 个):**")
        it = process.extract_iter(query, choices, processor=utils.default_process if use_proc else None)
        it_results = []
        try:
            for _ in range(2):
                m = next(it)
                it_results.append({"匹配项": m[0], "得分": f"{m[1]:.2f}"})
        except StopIteration:
            pass
        st.write(it_results)

        st.write("**cdist 距离矩阵 (多对多):**")
        queries = [query, "北京大学"]
        matrix = process.cdist(queries, choices, scorer=fuzz.ratio)
        df_matrix = pd.DataFrame(matrix, index=queries, columns=choices)
        st.dataframe(df_matrix)

        st.write("**cpdist 成对距离:**")
        if len(choices) >= len(queries):
            pairs = process.cpdist(queries, choices[:len(queries)], scorer=fuzz.ratio)
            st.write(f"成对结果: {pairs}")

    with tab_distance:
        st.subheader("Distance & Alignment (编辑距离与对齐)")
        
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            d1 = st.text_input("字符串 A", "kitten", key="dist_s1")
        with dist_col2:
            d2 = st.text_input("字符串 B", "sitting", key="dist_s2")

        dist_metrics = {
            "Levenshtein": distance.Levenshtein.distance,
            "DamerauLevenshtein": distance.DamerauLevenshtein.distance,
            "Hamming": distance.Hamming.distance,
            "Indel": distance.Indel.distance,
            "Jaro": distance.Jaro.distance,
            "JaroWinkler": distance.JaroWinkler.distance,
            "OSA (Optimal String Alignment)": distance.OSA.distance,
            "Prefix": distance.Prefix.distance,
            "Postfix": distance.Postfix.distance,
        }

        dist_results = []
        for name, func in dist_metrics.items():
            try:
                val = func(d1, d2)
                dist_results.append({"指标": name, "值": f"{val:.4f}"})
            except Exception as e:
                dist_results.append({"指标": name, "值": f"Error: {str(e)}"})
        
        st.table(pd.DataFrame(dist_results))

        st.subheader("Edit Operations (编辑步次)")
        ops = distance.Levenshtein.editops(d1, d2)
        st.write(f"Editops (Levenshtein): `{ops}`")
        
        opcodes = distance.Levenshtein.opcodes(d1, d2)
        st.write("Opcodes 详情:")
        st.table(pd.DataFrame([
            {"tag": op.tag, "src_start": op.src_start, "src_end": op.src_end, "dest_start": op.dest_start, "dest_end": op.dest_end}
            for op in opcodes
        ]))

    with tab_perf:
        st.subheader("高性能压力测试")
        count = st.select_slider("候选词规模", options=[1000, 10000, 50000, 100000, 500000], value=10000)
        
        if st.button("开始压测", key="btn_perf"):
            benchmark_query = "这是一个用于压测的示例字符串"
            dummy_choices = [f"这是第 {i} 个模拟生成的测试字符串，长度适中" for i in range(count)]
            
            st.write(f"正在对 {count} 条数据进行 `process.extract` (scorer=WRatio)...")
            
            start_time = time.perf_counter()
            _ = process.extract(benchmark_query, dummy_choices, limit=5, scorer=fuzz.WRatio)
            duration = time.perf_counter() - start_time
            
            st.success(f"耗时: {duration:.4f} 秒")
            st.metric("吞吐量", f"{int(count/duration)} items/sec")
            st.metric("平均耗时", f"{duration/count*1e6:.2f} μs/item")

if __name__ == "__main__":
    run_rapidfuzz_test()
