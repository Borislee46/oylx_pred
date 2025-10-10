import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="生成 cases_min.feather（仅保留在线必要列）")
    parser.add_argument(
        "--src",
        type=str,
        default="src/machine_learning_models/data/cases.feather",
        help="源 cases.feather 路径",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="src/machine_learning_models/data/cases_min.feather",
        help="输出 cases_min.feather 路径",
    )
    parser.add_argument(
        "--include-language-cols",
        action="store_true",
        help="是否包含 toefl/ielts 列（用于概率调整统计）；不加此参数则不强制包含",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if not src_path.exists():
        print(f"源文件不存在: {src_path}")
        return

    try:
        print(f"开始读取: {src_path}")
        df = pd.read_feather(src_path)
    except Exception as e:
        print(f"读取源文件失败: {e}")
        return

    base_required = [
        "background_university",
        "background_major_original",
        "background_major",
        "target_university",
        "target_major",
        "admitted",
        "gpa",
        "faculty",
    ]

    language_candidates = ["toefl", "ielts"] if args.include_language_cols else []
    required_columns = base_required + language_candidates

    existing_required = [c for c in required_columns if c in df.columns]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"警告: 下列必要列在源数据中缺失，将忽略: {missing}")

    if not existing_required:
        print("无法生成精简文件: 必要列均不存在")
        return

    df_min = df[existing_required].copy()

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df_min.reset_index(drop=True).to_feather(dst_path)
        print(f"精简文件已生成: {dst_path}，行数: {len(df_min)}，列: {list(df_min.columns)}")
    except Exception as e:
        print(f"写入精简文件失败: {e}")


if __name__ == "__main__":
    main()
