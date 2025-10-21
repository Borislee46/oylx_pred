from pathlib import Path

import pandas as pd


def main():
    src_path = Path("src/machine_learning_models/data/cases.feather")
    dst_path = Path("src/machine_learning_models/data/cases_min.feather")

    df = pd.read_feather(src_path)

    required_columns = [
        "background_university",
        "background_major_original",
        "background_major",
        "target_university",
        "target_major",
        "admitted",
        "gpa",
        "faculty",
    ]

    existing_columns = [c for c in required_columns if c in df.columns]
    df_min = df[existing_columns].copy()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df_min.reset_index(drop=True).to_feather(dst_path)
    print(f"精简文件已生成: {dst_path}，行数: {len(df_min)}，列: {list(df_min.columns)}")


if __name__ == "__main__":
    main()
