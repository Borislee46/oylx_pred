import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import distance, fuzz, process, utils

from src.pages.algorithm_lab.bench import bench


def run_rapidfuzz_test():
    st.header("RapidFuzz 深度功能测试")

    cfg = st.expander("基准测试参数", expanded=False)
    with cfg:
        warmup = st.number_input("warmup 次数", 0, 20, 2, 1, key="rf_warmup")
        repeat = st.number_input("repeat 次数", 1, 50, 10, 1, key="rf_repeat")
        seed = st.number_input(
            "随机种子", min_value=0, max_value=2**31 - 1, value=1, step=1, key="rf_seed"
        )

    tab_fuzz, tab_process, tab_distance, tab_perf = st.tabs(
        ["Fuzz 算法全集", "Process 提取与矩阵", "Distance & 编辑操作", "性能压测"]
    )

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
            st.json(
                {
                    "score": alignment.score,
                    "src_start": alignment.src_start,
                    "src_end": alignment.src_end,
                    "dest_start": alignment.dest_start,
                    "dest_end": alignment.dest_end,
                }
            )
            st.write(
                f"匹配片段: `{s1[alignment.src_start : alignment.src_end]}` <-> `{s2[alignment.dest_start : alignment.dest_end]}`"
            )

    with tab_process:
        st.subheader("Process Module 批量操作")

        proc_col1, proc_col2 = st.columns([1, 2])
        with proc_col1:
            query = st.text_input("查询词", "清华大学", key="proc_query")
            limit = st.number_input("提取数量", 1, 10, 3)
            score_cutoff = st.slider("得分阈值 (score_cutoff)", 0, 100, 0)
            use_proc = st.checkbox("默认预处理 (utils.default_process)", value=True)
            dedupe_threshold = st.slider("dedupe 阈值", 0, 100, 85)
        with proc_col2:
            choices_str = st.text_area(
                "候选词 (换行分隔)", "北京大学\n清华大学\n复旦大学\n上海交通大学\n清华", height=150
            )
            choices = [c.strip() for c in choices_str.split("\n") if c.strip()]

        st.write("**extractOne 最佳匹配:**")
        best = process.extractOne(
            query,
            choices,
            processor=utils.default_process if use_proc else None,
            score_cutoff=score_cutoff,
        )
        if best:
            st.write(f"结果: `{best[0]}` | 得分: `{best[1]:.2f}` | 索引: `{best[2]}`")
        else:
            st.write("未找到匹配项 (低于阈值)")

        st.write("**extract 批量提取:**")
        matches = process.extract(
            query,
            choices,
            limit=limit,
            processor=utils.default_process if use_proc else None,
            score_cutoff=score_cutoff,
        )
        st.table(
            pd.DataFrame([{"匹配项": m[0], "得分": f"{m[1]:.2f}", "索引": m[2]} for m in matches])
        )

        st.write("**extract_iter 迭代提取 (前 2 个):**")
        it = process.extract_iter(
            query, choices, processor=utils.default_process if use_proc else None
        )
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
            pairs = process.cpdist(queries, choices[: len(queries)], scorer=fuzz.ratio)
            st.write(f"成对结果: {pairs}")

        if hasattr(process, "dedupe"):
            st.write("**dedupe 去重/聚类:**")
            deduped = process.dedupe(
                choices,
                scorer=fuzz.WRatio,
                threshold=dedupe_threshold,
                processor=utils.default_process if use_proc else None,
            )
            st.write(deduped)

        st.subheader("合成噪声数据集评测 (Top-1 Accuracy)")
        st.write("评估不同算法在面对随机噪声（交换、删除、插入）时的鲁棒性。")

        noise_level = st.slider("噪声强度 (每个字符串最大修改次数)", 0, 5, 2)
        eval_n_val = st.number_input("评测样本量", 10, 1000, 100, 10, key="rf_eval_n_new")

        def apply_noise(s, level, rng):
            if not s or level == 0:
                return s
            s_list = list(s)
            for _ in range(rng.integers(0, level + 1)):
                if not s_list:
                    break
                op = rng.choice(["swap", "delete", "insert"])
                idx = rng.integers(0, len(s_list))
                if op == "swap" and len(s_list) > 1:
                    idx2 = (idx + 1) % len(s_list)
                    s_list[idx], s_list[idx2] = s_list[idx2], s_list[idx]
                elif op == "delete" and len(s_list) > 1:
                    s_list.pop(idx)
                elif op == "insert":
                    char = chr(rng.integers(0x4E00, 0x9FA5))
                    s_list.insert(idx, char)
            return "".join(s_list)

        if st.button("运行噪声稳健性实验"):
            rng_eval = np.random.default_rng(int(seed))
            base_corpus = [
                "清华大学",
                "北京大学",
                "复旦大学",
                "上海交通大学",
                "浙江大学",
                "南京大学",
                "武汉大学",
                "中山大学",
            ]
            eval_scorers = {
                "ratio": fuzz.ratio,
                "partial_ratio": fuzz.partial_ratio,
                "token_sort_ratio": fuzz.token_sort_ratio,
                "WRatio": fuzz.WRatio,
            }

            acc_results = []
            for sc_name, sc_func in eval_scorers.items():
                correct = 0
                for _ in range(int(eval_n_val)):
                    gold = rng_eval.choice(base_corpus)
                    noisy = apply_noise(gold, noise_level, rng_eval)
                    best = process.extractOne(noisy, base_corpus, scorer=sc_func)
                    if best and best[0] == gold:
                        correct += 1
                acc_results.append({"Scorer": sc_name, "Accuracy": correct / int(eval_n_val)})

            st.table(pd.DataFrame(acc_results))

    with tab_distance:
        st.subheader("Distance & Alignment (编辑距离与对齐)")

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            d1 = st.text_input("字符串 A", "kitten", key="dist_s1")
        with dist_col2:
            d2 = st.text_input("字符串 B", "sitting", key="dist_s2")

        dist_metrics = {
            "Levenshtein": distance.Levenshtein.distance,
            "Levenshtein (similarity)": distance.Levenshtein.similarity,
            "Levenshtein (norm_similarity)": distance.Levenshtein.normalized_similarity,
            "Levenshtein (norm_distance)": distance.Levenshtein.normalized_distance,
            "DamerauLevenshtein": distance.DamerauLevenshtein.distance,
            "DamerauLevenshtein (norm_similarity)": distance.DamerauLevenshtein.normalized_similarity,
            "Hamming": distance.Hamming.distance,
            "Indel": distance.Indel.distance,
            "Indel (norm_similarity)": distance.Indel.normalized_similarity,
            "Jaro": distance.Jaro.distance,
            "Jaro (similarity)": distance.Jaro.similarity,
            "JaroWinkler": distance.JaroWinkler.distance,
            "JaroWinkler (similarity)": distance.JaroWinkler.similarity,
            "LCSseq (Longest Common Subsequence)": distance.LCSseq.distance,
            "OSA (Optimal String Alignment)": distance.OSA.distance,
            "Prefix": distance.Prefix.distance,
            "Postfix": distance.Postfix.distance,
        }

        dist_results = []
        for name, func in dist_metrics.items():
            if "Hamming" in name and len(d1) != len(d2):
                dist_results.append({"指标": name, "值": "N/A (len mismatch)"})
                continue
            val = func(d1, d2)
            dist_results.append({"指标": name, "值": f"{val:.4f}"})

        st.table(pd.DataFrame(dist_results))

        st.subheader("Edit Operations (编辑步次)")
        ops = distance.Levenshtein.editops(d1, d2)
        st.write(f"Editops (Levenshtein): `{ops}`")

        opcodes = distance.Levenshtein.opcodes(d1, d2)
        st.write("Opcodes 详情:")
        st.table(
            pd.DataFrame(
                [
                    {
                        "tag": op.tag,
                        "src_start": op.src_start,
                        "src_end": op.src_end,
                        "dest_start": op.dest_start,
                        "dest_end": op.dest_end,
                    }
                    for op in opcodes
                ]
            )
        )

    with tab_perf:
        st.subheader("高性能压力测试")
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            count = st.select_slider(
                "候选词规模", options=[1000, 10000, 50000, 100000, 500000], value=10000
            )
        with perf_col2:
            perf_cutoff = st.slider("得分阈值 (开启后算法会剪枝加速)", 0, 90, 0, step=10)
            st.info("RapidFuzz 会根据 score_cutoff 自动跳过不可能达标的字符串，从而显著提升速度。")

        if st.button("开始压测", key="btn_perf"):
            benchmark_query = "这是一个用于压测的示例字符串，用于测试 RapidFuzz 的极限吞吐量"
            dummy_choices = [
                f"这是第 {i} 个模拟生成的测试字符串，包含一部分相似内容" for i in range(count)
            ]

            st.write(f"正在对 {count} 条数据进行 `process.extract` (cutoff={perf_cutoff})...")

            def run():
                process.extract(
                    benchmark_query,
                    dummy_choices,
                    limit=5,
                    scorer=fuzz.WRatio,
                    score_cutoff=perf_cutoff,
                )

            b = bench(run, warmup=int(warmup), repeat=int(repeat))
            stats = b.stats()
            st.dataframe(pd.DataFrame([stats])[["n", "mean_human", "mean_s", "p50_s", "p95_s"]])
            st.download_button(
                "下载 RapidFuzz 压测数据 (CSV)",
                pd.DataFrame([stats]).to_csv(index=False).encode("utf-8"),
                "rapidfuzz_perf.csv",
                "text/csv",
            )
            mean_s = float(stats["mean_s"])
            st.metric("吞吐量(mean)", f"{int(count / mean_s)} items/sec")
            st.metric("平均耗时(mean)", f"{mean_s / count * 1e6:.2f} μs/item")

            st.subheader("剪枝加速效果可视化")
            cutoffs = [0, 50, 80, 90, 95]
            perf_trend = []
            for co in cutoffs:

                def run_co():
                    process.extract(
                        benchmark_query, dummy_choices, limit=5, scorer=fuzz.WRatio, score_cutoff=co
                    )

                b_co = bench(run_co, warmup=1, repeat=3)
                perf_trend.append({"cutoff": co, "time_ms": float(b_co.stats()["mean_s"]) * 1000})

            st.line_chart(pd.DataFrame(perf_trend).set_index("cutoff"))
            st.caption(
                "注：随着 cutoff 增加，RapidFuzz 可以更早地跳过不相关的字符串，性能呈指数级提升。"
            )


if __name__ == "__main__":
    run_rapidfuzz_test()
