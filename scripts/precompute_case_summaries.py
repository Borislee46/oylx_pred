import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_background_summary(df):
    background_cols_map = {
        "工作数量": "工作",
        "发表数量": "论文",
        "科研数量": "科研",
        "实习数量": "实习",
        "活动数量": "活动",
        "获奖数量": "获奖",
    }
    
    summary_parts = []
    for col, prefix in background_cols_map.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            valid_values = values.where((values > 0) & values.notna(), np.nan)
            str_values = valid_values.apply(lambda x: f"{prefix}{int(x)}" if pd.notna(x) else "")
            summary_parts.append(str_values)
    
    if summary_parts:
        result = summary_parts[0]
        for part in summary_parts[1:]:
            result = result + part
        return result
    return pd.Series("", index=df.index)


def compute_uni_classification_summary(df):
    invalid_values = ["空空空", "nan", "NaN", "null", "NULL", "None", "无", "", "0.0", "0"]
    
    parts = []
    
    if "国本院校分类" in df.columns:
        domestic = df["国本院校分类"].astype(str)
        domestic = domestic.where(
            ~domestic.isin(invalid_values) & domestic.notna(),
            ""
        )
        parts.append(domestic)
    
    if "海本QS排名区间" in df.columns:
        overseas = df["海本QS排名区间"].astype(str)
        overseas = overseas.where(
            ~overseas.isin(invalid_values) & overseas.notna(),
            ""
        )
        parts.append(overseas)
    
    if len(parts) == 2:
        result = parts[0] + "/" + parts[1]
        result = result.str.strip("/")
        result = result.replace("/", "")
    elif len(parts) == 1:
        result = parts[0]
    else:
        result = pd.Series("", index=df.index)
    
    return result


def compute_gre_gmat_score(df):
    parts = []
    
    if "GRE分数" in df.columns:
        gre = pd.to_numeric(df["GRE分数"], errors="coerce")
        gre_str = gre.where((gre > 0) & gre.notna(), np.nan).apply(
            lambda x: str(int(x)) if pd.notna(x) else ""
        )
        parts.append(gre_str)
    
    if "GMAT分数" in df.columns:
        gmat = pd.to_numeric(df["GMAT分数"], errors="coerce")
        gmat_str = gmat.where((gmat > 0) & gmat.notna(), np.nan).apply(
            lambda x: str(int(x)) if pd.notna(x) else ""
        )
        parts.append(gmat_str)
    
    if len(parts) == 2:
        result = parts[0] + "/" + parts[1]
        result = result.str.strip("/")
        result = result.replace("/", "")
    elif len(parts) == 1:
        result = parts[0]
    else:
        result = pd.Series("", index=df.index)
    
    return result


def precompute_summaries_for_file(file_path: Path):
    if not file_path.exists():
        print(f"[WARNING] 文件不存在: {file_path}")
        return False
    
    print(f"\n[INFO] 处理文件: {file_path.name}")
    
    try:
        df = pd.read_feather(file_path)
        print(f"       读取 {len(df)} 行数据")
        
        modified = False
        
        background_cols = ["工作数量", "发表数量", "科研数量", "实习数量", "活动数量", "获奖数量"]
        if any(col in df.columns for col in background_cols):
            print("       计算 background_summary_precomputed...")
            df["background_summary_precomputed"] = compute_background_summary(df)
            modified = True
        
        if "国本院校分类" in df.columns or "海本QS排名区间" in df.columns:
            print("       计算 uni_classification_summary_precomputed...")
            df["uni_classification_summary_precomputed"] = compute_uni_classification_summary(df)
            modified = True
        
        if "GRE分数" in df.columns or "GMAT分数" in df.columns:
            print("       计算 gre_gmat_score_precomputed...")
            df["gre_gmat_score_precomputed"] = compute_gre_gmat_score(df)
            modified = True
        
        if modified:
            df.reset_index(drop=True).to_feather(file_path)
            precomputed_count = sum([1 for c in df.columns if '_precomputed' in c])
            print(f"       [SUCCESS] 已更新并保存，新增 {precomputed_count} 个预计算列")
            return True
        else:
            print("       无需添加预计算列")
            return False
    
    except Exception as e:
        print(f"       [ERROR] 处理失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="为案例库数据预计算汇总列以提升查询性能"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="src/machine_learning_models/data",
        help="数据目录路径",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="cases_*.feather",
        help="文件匹配模式",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return
    
    files = list(data_dir.glob(args.pattern))
    if not files:
        print(f"[ERROR] 未找到匹配的文件: {data_dir / args.pattern}")
        return
    
    print(f"[START] 开始预计算，共找到 {len(files)} 个文件\n")
    print("=" * 60)
    
    success_count = 0
    for file_path in sorted(files):
        if precompute_summaries_for_file(file_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"[COMPLETE] 预计算完成！成功处理 {success_count}/{len(files)} 个文件")


if __name__ == "__main__":
    main()

