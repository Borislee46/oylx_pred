import os
import sys

import numpy as np
import pandas as pd
import torch


def setup_environment():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)

    src_path = os.path.join(project_root, "src")
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root, script_dir


project_root, script_dir = setup_environment()

from scripts.model_utils import compute_embeddings_batch_local, get_model

CASES_DATA_PATH = os.path.join(
    project_root, "src", "machine_learning_models", "data", "cases.feather"
)
SCHOOL_MAJOR_DETAILS_PATH = os.path.join(
    project_root, "src", "machine_learning_models", "data", "school_major_details.feather"
)
BG_TARGET_CACHE_PATH = os.path.join(project_root, "cache", "background_target_similarity.feather")


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
BATCH_SIZE = 64


def precompute_similarities():
    raw_school_english_majors = set()
    raw_case_background_majors = set()
    raw_case_target_majors = set()
    eng_to_chi_map = {}

    if os.path.exists(SCHOOL_MAJOR_DETAILS_PATH):
        details_df = pd.read_feather(SCHOOL_MAJOR_DETAILS_PATH)
        if "专业英文名称" in details_df.columns:
            if "专业中文名称" in details_df.columns:
                valid_map = details_df.dropna(subset=["专业英文名称", "专业中文名称"])
                eng_to_chi_map = dict(
                    zip(valid_map["专业英文名称"], valid_map["专业中文名称"], strict=True)
                )

            raw_school_english_majors.update(details_df["专业英文名称"].dropna().astype(str))

    if os.path.exists(CASES_DATA_PATH):
        cases_df = pd.read_feather(CASES_DATA_PATH)
        if "background_major" in cases_df.columns:
            raw_case_background_majors.update(cases_df["background_major"].dropna().astype(str))
        if "target_major" in cases_df.columns:
            raw_case_target_majors.update(cases_df["target_major"].dropna().astype(str))

    for s in [raw_case_background_majors, raw_case_target_majors, raw_school_english_majors]:
        for val in ["无", "", "nan", "None"]:
            s.discard(val)

    bg_list = sorted(raw_case_background_majors)
    target_list = sorted(raw_case_target_majors.union(raw_school_english_majors))

    if not bg_list or not target_list:
        print("未找到有效的背景或目标专业数据。")
        return

    model = get_model(MODEL_NAME)
    task = (
        "Given an academic major, retrieve the most relevant target major for university admission"
    )

    bg_reps = [eng_to_chi_map.get(m, m) for m in bg_list]
    target_reps = [eng_to_chi_map.get(m, m) for m in target_list]

    print(f"计算 {len(bg_reps)} 个背景专业和 {len(target_reps)} 个目标专业的嵌入...")

    bg_inputs = [f"Instruct: {task}\nQuery: {text}" for text in bg_reps]

    bg_embeddings = torch.stack(compute_embeddings_batch_local(bg_inputs, model, BATCH_SIZE))
    target_embeddings = torch.stack(compute_embeddings_batch_local(target_reps, model, BATCH_SIZE))

    with torch.no_grad():
        bg_embeddings = bg_embeddings.to(model.device)
        target_embeddings = target_embeddings.to(model.device)
        similarity_matrix = (bg_embeddings @ target_embeddings.T).cpu().numpy()

    print("正在构建并保存相似度缓存...")
    num_bg = len(bg_list)
    num_target = len(target_list)

    cache_df = pd.DataFrame(
        {
            "bg_major": np.repeat(bg_list, num_target),
            "target_major": np.tile(target_list, num_bg),
            "similarity": similarity_matrix.ravel(),
        }
    )

    os.makedirs(os.path.dirname(BG_TARGET_CACHE_PATH), exist_ok=True)
    cache_df.to_feather(BG_TARGET_CACHE_PATH)
    print(f"成功将 {len(cache_df)} 条相似度记录保存至: {BG_TARGET_CACHE_PATH}")


if __name__ == "__main__":
    precompute_similarities()
