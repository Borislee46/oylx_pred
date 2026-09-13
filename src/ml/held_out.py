from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HELD_OUT_PATH = os.path.join(_PROJECT_ROOT, "data", "held_out_test.feather")
_HELD_OUT_META_PATH = os.path.join(_PROJECT_ROOT, "data", "held_out_test_meta.json")


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


def load_held_out_set(
    feather_path: str | None = None,
    meta_path: str | None = None,
    verify_integrity: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feather_path = feather_path or _HELD_OUT_PATH
    meta_path = meta_path or _HELD_OUT_META_PATH

    if not os.path.exists(feather_path):
        raise FileNotFoundError(f"冻结留存集不存在: {feather_path}")

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
                f"  held_out_test.feather 可能被修改——请恢复原始文件。"
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
        print("完整性校验通过")
        return True
    except (FileNotFoundError, ValueError) as e:
        print(f"完整性校验失败: {e}", file=sys.stderr)
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
    parser.add_argument("--verify", action="store_true", help="校验留存集完整性")
    parser.add_argument("--info", action="store_true", help="查看留存集元数据")
    args = parser.parse_args()

    if args.verify:
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
