from pathlib import Path

import pandas as pd


def compare_feather_files(file1_path: str, file2_path: str):
    df1 = pd.read_feather(file1_path)
    df2 = pd.read_feather(file2_path)

    print("=" * 80)
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print("=" * 80)
    print()

    print("【基本信息对比】")
    print(f"文件1 - 行数: {len(df1)}, 列数: {len(df1.columns)}")
    print(f"文件2 - 行数: {len(df2)}, 列数: {len(df2.columns)}")
    print()

    print("【列名对比】")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    common_cols = cols1 & cols2

    if only_in_1:
        print(f"仅在文件1中的列: {sorted(only_in_1)}")
    if only_in_2:
        print(f"仅在文件2中的列: {sorted(only_in_2)}")
    if not only_in_1 and not only_in_2:
        print("两个文件的列名完全相同")
    print(f"共同列: {sorted(common_cols)}")
    print()

    if len(df1) != len(df2):
        print("【行数差异】")
        print(f"文件1有 {len(df1)} 行，文件2有 {len(df2)} 行，相差 {abs(len(df1) - len(df2))} 行")
        print()

    if common_cols:
        print("【数据类型对比】")
        for col in sorted(common_cols):
            dtype1 = df1[col].dtype
            dtype2 = df2[col].dtype
            if dtype1 != dtype2:
                print(f"  {col}: 文件1={dtype1}, 文件2={dtype2}")
        print()

        print("【数据值对比】")
        if len(df1) == len(df2):
            for col in sorted(common_cols):
                if df1[col].dtype == df2[col].dtype:
                    if not df1[col].equals(df2[col]):
                        diff_count = (df1[col] != df2[col]).sum()
                        print(f"  {col}: {diff_count} 个值不同")
                        if diff_count <= 10:
                            diff_indices = df1[col][df1[col] != df2[col]].index.tolist()
                            for idx in diff_indices[:5]:
                                print(f"    索引 {idx}: 文件1={df1[col].iloc[idx]}, 文件2={df2[col].iloc[idx]}")
                    else:
                        print(f"  {col}: 值完全相同")
        else:
            print("  由于行数不同，无法进行逐行对比")
            print("  统计信息对比:")
            for col in sorted(common_cols):
                if pd.api.types.is_numeric_dtype(df1[col]):
                    print(f"  {col}:")
                    print(f"    文件1 - 均值={df1[col].mean():.4f}, 标准差={df1[col].std():.4f}")
                    print(f"    文件2 - 均值={df2[col].mean():.4f}, 标准差={df2[col].std():.4f}")
                else:
                    unique1 = df1[col].nunique()
                    unique2 = df2[col].nunique()
                    print(f"  {col}: 文件1唯一值数={unique1}, 文件2唯一值数={unique2}")
        print()

    print("【重复行检查】")
    dup1 = df1.duplicated().sum()
    dup2 = df2.duplicated().sum()
    print(f"文件1重复行数: {dup1}")
    print(f"文件2重复行数: {dup2}")
    print()

    print("【空值统计】")
    if common_cols:
        print("文件1空值:")
        null1 = df1[list(common_cols)].isnull().sum()
        for col in sorted(common_cols):
            if null1[col] > 0:
                print(f"  {col}: {null1[col]}")
        print("文件2空值:")
        null2 = df2[list(common_cols)].isnull().sum()
        for col in sorted(common_cols):
            if null2[col] > 0:
                print(f"  {col}: {null2[col]}")


def main():
    file1 = Path("src/machine_learning_models/data/cases_min.feather")
    file2 = Path("src/machine_learning_models/data/cases_min1.feather")

    if not file1.exists():
        print(f"错误: 文件不存在 - {file1}")
        return

    if not file2.exists():
        print(f"错误: 文件不存在 - {file2}")
        return

    compare_feather_files(str(file1), str(file2))


if __name__ == "__main__":
    main()

