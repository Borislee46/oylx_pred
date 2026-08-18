from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CASES_PATH = os.path.join(_PROJECT_ROOT, "src", "ml", "data", "cases.feather")
_YEAR_PATH = os.path.join(_PROJECT_ROOT, "src", "ml", "data", "cases_year_recovered.feather")
_HELD_OUT_PATH = os.path.join(_PROJECT_ROOT, "data", "held_out_test.feather")
_HELD_OUT_META_PATH = os.path.join(_PROJECT_ROOT, "data", "held_out_test_meta.json")

SPLIT_YEAR = 2024


def compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    sha = hashlib.sha256()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    sha.update(csv_bytes)
    return sha.hexdigest()


def create_held_out_set(
    cases_path: str | None = None,
    year_path: str | None = None,
    output_feather: str | None = None,
    output_meta: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases_path = cases_path or _CASES_PATH
    year_path = year_path or _YEAR_PATH
    output_feather = output_feather or _HELD_OUT_PATH
    output_meta = output_meta or _HELD_OUT_META_PATH

    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"cases.feather 不存在: {cases_path}")
    if not os.path.exists(year_path):
        raise FileNotFoundError(
            f"cases_year_recovered.feather 不存在: {year_path}\n"
            f"请先从 cases_grad.feather 恢复年份信息。"
        )

    cases_df = pd.read_feather(cases_path)
    year_df = pd.read_feather(year_path)

    if "year" not in year_df.columns:
        raise ValueError(f"cases_year_recovered.feather 缺少 'year' 列: {list(year_df.columns)}")
    if len(year_df) != len(cases_df):
        raise ValueError(f"行数不匹配: cases={len(cases_df)}, year={len(year_df)}")

    years = year_df["year"].values

    test_mask = years >= SPLIT_YEAR
    train_mask = ~test_mask

    held_out_df = cases_df.iloc[test_mask].copy()
    train_df = cases_df.iloc[train_mask].copy()

    n_total = len(cases_df)
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if "admitted" in cases_df.columns:
        train_rate = float(train_df["admitted"].mean())
        test_rate = float(held_out_df["admitted"].mean())
    else:
        train_rate = test_rate = float("nan")

    test_years = years[test_mask]
    year_dist = {str(int(y)): int((test_years == y).sum()) for y in sorted(pd.unique(test_years))}

    data_hash = compute_file_hash(cases_path)
    held_out_hash = compute_dataframe_hash(held_out_df)

    meta: dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "data_hash": data_hash,
        "held_out_hash": held_out_hash,
        "split_criterion": f"year >= {SPLIT_YEAR}",
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "positive_rate_train": round(train_rate, 6),
        "positive_rate_test": round(test_rate, 6),
        "test_year_distribution": year_dist,
        "source_cases_file": os.path.relpath(cases_path, _PROJECT_ROOT),
        "source_year_file": os.path.relpath(year_path, _PROJECT_ROOT),
    }

    held_out_df.to_feather(output_feather)
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[held_out] 冻结留存集已创建:")
    print(f"  总样本: {n_total:,}  |  训练: {n_train:,}  |  测试: {n_test:,}")
    print(f"  训练正例率: {train_rate:.4f}  |  测试正例率: {test_rate:.4f}")
    print(f"  测试年份: {year_dist}")
    print(f"  data_hash: {data_hash[:16]}...")
    print(f"  held_out_hash: {held_out_hash[:16]}...")
    print(f"  输出: {output_feather}")
    print(f"  元数据: {output_meta}")

    return held_out_df, meta


def load_held_out_set(
    feather_path: str | None = None,
    meta_path: str | None = None,
    verify_integrity: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feather_path = feather_path or _HELD_OUT_PATH
    meta_path = meta_path or _HELD_OUT_META_PATH

    if not os.path.exists(feather_path):
        raise FileNotFoundError(
            f"冻结留存集不存在: {feather_path}\n请先运行: python -m src.ml.held_out --freeze"
        )

    df = pd.read_feather(feather_path)

    meta: dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    if verify_integrity and meta:
        current_hash = compute_dataframe_hash(df)
        expected_hash = meta.get("held_out_hash", "")
        if current_hash != expected_hash:
            raise ValueError(
                f"留存集完整性校验失败!\n"
                f"  当前 hash: {current_hash[:16]}...\n"
                f"  预期 hash: {expected_hash[:16]}...\n"
                f"  held_out_test.feather 可能被修改——请重新生成或恢复原始文件。"
            )
        print(
            f"[held_out] 完整性校验通过 "
            f"(hash={current_hash[:16]}..., n={len(df):,}, "
            f"正例率={df['admitted'].mean():.4f})"
        )

    return df, meta


def verify_integrity() -> bool:
    try:
        load_held_out_set(verify_integrity=True)
        print("[held_out] ✅ 完整性校验通过")
        return True
    except (FileNotFoundError, ValueError) as e:
        print(f"[held_out] ❌ 完整性校验失败: {e}", file=sys.stderr)
        return False


def print_info() -> None:
    try:
        _, meta = load_held_out_set(verify_integrity=False)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"[held_out] {e}")
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(description="冻结留存集管理 (Held-Out Test Set)")
    parser.add_argument(
        "--freeze", action="store_true", help="创建冻结留存集 (data/held_out_test.feather + meta)"
    )
    parser.add_argument("--verify", action="store_true", help="校验留存集完整性")
    parser.add_argument("--info", action="store_true", help="查看留存集元数据")
    args = parser.parse_args()

    if args.freeze:
        create_held_out_set()
    elif args.verify:
        ok = verify_integrity()
        sys.exit(0 if ok else 1)
    elif args.info:
        print_info()
    else:
        parser.print_help()
        print()
        try:
            print_info()
        except SystemExit:
            pass


if __name__ == "__main__":
    _main()
