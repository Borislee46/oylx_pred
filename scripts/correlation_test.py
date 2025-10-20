import os

import numpy as np
import pandas as pd


def load_data(filepath="src/machine_learning_models/data/cases.feather"):
    df = pd.read_feather(filepath)
    print(f"成功加载 {len(df)} 条记录，来自 {filepath}")
    required_cols = ["target_university", "target_major", "admitted"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"缺少必要的列: {missing}")
    return df


def calculate_correlation_matrix(
    df,
    min_common_students: int = 1,
    shrinkage_k: int = 20,
    psd_lambda: float = 0.05,
):
    if df is None or not all(
        col in df.columns for col in ["target_university", "target_major", "admitted"]
    ):
        print("无效的 DataFrame 或缺少必要的列。")
        return None, None, None

    identifier_columns = [
        "background_university",
        "background_major",
        "background_major_original",
        "gpa",
        "research_count",
        "research_detail",
        "paper_count",
        "paper_detail",
        "internship_count",
        "internship_detail",
        "activity_count",
        "activity_detail",
        "award_count",
        "award_detail",
        "ielts",
        "toefl",
    ]
    selected_id_cols = [c for c in identifier_columns if c in df.columns]
    if selected_id_cols:
        print("\n正在按背景信息分组生成学生ID...")
        df["student_id"] = df.groupby(selected_id_cols, dropna=False).ngroup() + 1
        df["student_id"] = df["student_id"].map(lambda x: f"学生编号 {x}")
    else:
        print("\n警告: 未找到用于识别学生的字段，将每行视为不同学生。")
        df = df.reset_index(drop=True)
        df["student_id"] = (df.index + 1).map(lambda x: f"学生编号 {x}")

    print("\n正在透视数据（这可能需要一些时间）...")
    df["target_combined"] = df["target_university"] + " - " + df["target_major"]

    try:
        pivot_df = df.pivot_table(index="student_id", columns="target_combined", values="admitted")
    except Exception as e:
        print(f"透视数据时出错: {e}")
        print("正在尝试在透视前删除重复的学生-目标项...")
        df_unique = df.drop_duplicates(subset=["student_id", "target_combined"], keep="last")
        try:
            pivot_df = df_unique.pivot_table(
                index="student_id", columns="target_combined", values="admitted"
            )
        except Exception as e2:
            print(f"即使在删除重复项后，透视数据时仍然出错: {e2}")
            return None, None, None

    print(f"数据透视完成。形状: {pivot_df.shape}")
    keys = list(pivot_df.columns)
    print(f"找到 {len(keys)} 个独立的目标院校-专业组合。")

    present = (~pivot_df.isna()).astype(int)
    n_co = present.T.values @ present.values
    n_co_df = pd.DataFrame(n_co, index=keys, columns=keys)

    print(f"\n正在计算经验相关 (min_periods={min_common_students})...")
    r_emp = pivot_df.corr(method="pearson", min_periods=min_common_students)

    print("计算共申 Jaccard 相似度作为兜底...")
    counts = present.sum(axis=0).values
    union = counts[:, None] + counts[None, :] - n_co
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, n_co / union, 0.0)
    jaccard_df = pd.DataFrame(jaccard, index=keys, columns=keys)

    print("构造先验矩阵...")
    universities = [k.split(" - ")[0] if " - " in k else "" for k in keys]
    majors = [k.split(" - ")[1] if " - " in k else "" for k in keys]
    U = np.array(universities, dtype=object)
    M = np.array(majors, dtype=object)
    same_uni = U[:, None] == U[None, :]
    same_major = M[:, None] == M[None, :]
    r_prior = np.where(same_uni, 0.30, np.where(same_major, 0.20, 0.00))
    np.fill_diagonal(r_prior, 1.0)
    r_prior_df = pd.DataFrame(r_prior, index=keys, columns=keys)

    r_emp_filled = r_emp.copy()
    r_emp_filled = r_emp_filled.where(~r_emp_filled.isna(), jaccard_df)
    r_emp_filled = r_emp_filled.clip(-1.0, 1.0)

    print("应用贝叶斯收缩...")
    w = n_co_df / (n_co_df + float(shrinkage_k))
    w = w.clip(0.0, 1.0)

    r_shrunk = w * r_emp_filled + (1.0 - w) * r_prior_df
    r_shrunk = r_shrunk.clip(-1.0, 1.0)
    np.fill_diagonal(r_shrunk.values, 1.0)

    print("进行 PSD 正定化与对角收缩...")
    R = r_shrunk.values.astype(float)
    R = (1.0 - psd_lambda) * R + psd_lambda * np.eye(R.shape[0])
    try:
        eigvals, eigvecs = np.linalg.eigh(R)
        min_eig = 1e-6
        eigvals[eigvals < min_eig] = min_eig
        R_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.clip(np.diag(R_fixed), 1e-12, None))
        D_inv = np.diag(1.0 / d)
        R_corr = D_inv @ R_fixed @ D_inv
        np.fill_diagonal(R_corr, 1.0)
    except Exception as e:
        print(f"PSD 修复失败，回退到收缩矩阵: {e}")
        R_corr = R
        np.fill_diagonal(R_corr, 1.0)

    correlation_df = pd.DataFrame(R_corr, index=keys, columns=keys)
    weight_df = w

    approx_zero_ratio = (np.isclose(correlation_df.values, 0.0, atol=1e-8).sum() - len(keys)) / (
        correlation_df.size - len(keys)
    )
    print(f"相关性近 0 的比例（去除对角）: {approx_zero_ratio:.2%}")

    return correlation_df, weight_df, n_co_df


if __name__ == "__main__":
    df_cases = load_data()

    if df_cases is not None:
        correlation_mat, weight_mat, overlap_mat = calculate_correlation_matrix(
            df_cases,
            min_common_students=1,
            shrinkage_k=20,
            psd_lambda=0.05,
        )

        if correlation_mat is not None:
            print("\n相关性矩阵样本 (前 10x10):")
            print(correlation_mat.iloc[:10, :10])

            output_dir = "src/machine_learning_models/data"
            corr_path = os.path.join(output_dir, "correlation_matrix.feather")
            weight_path = os.path.join(output_dir, "correlation_pair_weight.feather")
            overlap_path = os.path.join(output_dir, "correlation_overlap_count.feather")

            try:
                os.makedirs(output_dir, exist_ok=True)
                corr_feather = correlation_mat.reset_index(names="index")
                corr_feather.to_feather(corr_path)
                print(f"\n已保存相关性 Feather: {corr_path}")

                if weight_mat is not None:
                    weight_feather = weight_mat.reset_index(names="index")
                    weight_feather.to_feather(weight_path)
                    print(f"已保存权重 Feather: {weight_path}")

                if overlap_mat is not None:
                    overlap_feather = overlap_mat.reset_index(names="index")
                    overlap_feather.to_feather(overlap_path)
                    print(f"已保存重叠计数 Feather: {overlap_path}")
            except Exception as e:
                print(f"\n保存 Feather 时出错: {e}")

        else:
            print("\n计算相关性矩阵失败。")
    else:
        print("\n未能加载数据，无法继续。")
