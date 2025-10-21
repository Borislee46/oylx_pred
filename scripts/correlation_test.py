import os

import numpy as np
import pandas as pd


def load_data(filepath="src/machine_learning_models/data/cases.feather"):
    df = pd.read_feather(filepath)
    print(f"成功加载 {len(df)} 条记录")

    required_cols = ["target_university", "target_major", "admitted"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要的列: {missing}")

    return df


def calculate_correlation_matrix(df, min_common_students=1, shrinkage_k=20, psd_lambda=0.05):
    identifier_columns = ["background_university", "background_major", "gpa"]
    selected_id_cols = [c for c in identifier_columns if c in df.columns]

    if selected_id_cols:
        df["student_id"] = df.groupby(selected_id_cols, dropna=False).ngroup() + 1
        df["student_id"] = df["student_id"].map(lambda x: f"学生编号 {x}")
    else:
        df = df.reset_index(drop=True)
        df["student_id"] = (df.index + 1).map(lambda x: f"学生编号 {x}")

    df["target_combined"] = df["target_university"] + " - " + df["target_major"]
    df_unique = df.drop_duplicates(subset=["student_id", "target_combined"], keep="last")
    pivot_df = df_unique.pivot_table(
        index="student_id", columns="target_combined", values="admitted"
    )

    print(f"数据透视完成。形状: {pivot_df.shape}")
    keys = list(pivot_df.columns)

    present = (~pivot_df.isna()).astype(int)
    n_co = present.T.values @ present.values
    n_co_df = pd.DataFrame(n_co, index=keys, columns=keys)

    r_emp = pivot_df.corr(method="pearson", min_periods=min_common_students)

    universities = [k.split(" - ")[0] for k in keys]
    majors = [k.split(" - ")[1] for k in keys]
    U, M = np.array(universities), np.array(majors)

    same_uni = U[:, None] == U[None, :]
    same_major = M[:, None] == M[None, :]
    r_prior = np.where(same_uni, 0.30, np.where(same_major, 0.20, 0.00))
    np.fill_diagonal(r_prior, 1.0)

    w = n_co_df / (n_co_df + float(shrinkage_k))
    w = w.clip(0.0, 1.0)

    r_emp_filled = r_emp.fillna(0).clip(-1.0, 1.0)
    r_shrunk = w * r_emp_filled + (1.0 - w) * r_prior
    r_shrunk = r_shrunk.clip(-1.0, 1.0)
    np.fill_diagonal(r_shrunk.values, 1.0)

    R = (1.0 - psd_lambda) * r_shrunk.values + psd_lambda * np.eye(r_shrunk.shape[0])

    try:
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals[eigvals < 1e-6] = 1e-6
        R_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.clip(np.diag(R_fixed), 1e-12, None))
        R_corr = np.diag(1.0 / d) @ R_fixed @ np.diag(1.0 / d)
        np.fill_diagonal(R_corr, 1.0)
    except Exception:
        R_corr = R

    correlation_df = pd.DataFrame(R_corr, index=keys, columns=keys)
    return correlation_df, w, n_co_df


if __name__ == "__main__":
    df_cases = load_data()

    if df_cases is not None:
        correlation_mat, weight_mat, overlap_mat = calculate_correlation_matrix(df_cases)

        if correlation_mat is not None:
            print("\n相关性矩阵样本 (前 5x5):")
            print(correlation_mat.iloc[:5, :5])

            output_dir = "src/machine_learning_models/data"
            os.makedirs(output_dir, exist_ok=True)

            correlation_mat.reset_index(names="index").to_feather(
                os.path.join(output_dir, "correlation_matrix.feather")
            )
            weight_mat.reset_index(names="index").to_feather(
                os.path.join(output_dir, "correlation_pair_weight.feather")
            )
            overlap_mat.reset_index(names="index").to_feather(
                os.path.join(output_dir, "correlation_overlap_count.feather")
            )

            print("所有文件保存完成")
