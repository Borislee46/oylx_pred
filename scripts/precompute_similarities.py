import logging
import os
import sys

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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

from model_utils import compute_embeddings_batch_local, compute_similarity_matrix, get_model


def _create_major_similarity_key(major1: str, major2: str) -> str:
    key_pair = tuple(sorted([major1, major2]))
    return f"{key_pair[0]}|{key_pair[1]}"


CASES_DATA_PATH = os.path.join(
    project_root, "src", "machine_learning_models", "data", "cases.feather"
)
SCHOOL_MAJOR_DETAILS_PATH = os.path.join(
    project_root, "src", "machine_learning_models", "data", "school_major_details.feather"
)
MAJOR_SIMILARITY_CACHE_PATH = os.path.join(
    project_root,
    "src",
    "machine_learning_models",
    "data",
    "cache",
    "major_similarity_cache.json",
)


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
BATCH_SIZE = 64


def precompute_similarities():
    raw_school_english_majors = set()
    raw_case_background_majors = set()
    raw_case_target_majors = set()
    english_to_chinese_major_map = {}

    def read_table_auto(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        lower = path.lower()
        if lower.endswith(".feather"):
            return pd.read_feather(path)
        return pd.read_feather(path)

    if os.path.exists(SCHOOL_MAJOR_DETAILS_PATH):
        details_df = read_table_auto(SCHOOL_MAJOR_DETAILS_PATH)
        if "专业英文名称" in details_df.columns and "专业中文名称" in details_df.columns:
            for _, row in details_df.iterrows():
                eng_major_val = row["专业英文名称"]
                chi_major_val = row["专业中文名称"]

                eng_major = str(eng_major_val) if pd.notna(eng_major_val) else None
                chi_major = str(chi_major_val) if pd.notna(chi_major_val) else None

                if eng_major:
                    raw_school_english_majors.add(eng_major)
                    if chi_major and chi_major.strip():
                        english_to_chinese_major_map[eng_major] = chi_major
        elif "专业英文名称" in details_df.columns:
            raw_school_english_majors.update(details_df["专业英文名称"].dropna().astype(str))

    if os.path.exists(CASES_DATA_PATH):
        cases_df = read_table_auto(CASES_DATA_PATH)
        raw_case_background_majors.update(cases_df["background_major"].dropna().astype(str))
        raw_case_target_majors.update(cases_df["target_major"].dropna().astype(str))

    raw_case_background_majors.discard("无")
    raw_case_background_majors.discard("")
    raw_case_target_majors.discard("无")
    raw_case_target_majors.discard("")
    raw_school_english_majors.discard("无")
    raw_school_english_majors.discard("")

    all_target_majors = raw_case_target_majors.union(raw_school_english_majors)

    background_major_list_orig = sorted(list(raw_case_background_majors))
    target_major_list_orig = sorted(list(all_target_majors))
    all_majors_for_embedding_orig = sorted(
        list(set(background_major_list_orig + target_major_list_orig))
    )

    original_to_embedding_representation_map = {}
    embedding_representation_set = set()

    for raw_major in all_majors_for_embedding_orig:
        representation = ""
        if raw_major in english_to_chinese_major_map:
            chi_major = english_to_chinese_major_map.get(raw_major)
            if chi_major and chi_major.strip():
                representation = chi_major
            else:
                representation = raw_major
        else:
            representation = raw_major

        if representation is None:
            representation = ""

        original_to_embedding_representation_map[raw_major] = representation
        if representation:
            embedding_representation_set.add(representation)

    embedding_input_list = sorted(list(embedding_representation_set))

    embedding_representation_to_idx_map = {name: i for i, name in enumerate(embedding_input_list)}
    model = get_model(MODEL_NAME)
    use_e5_style_prompt = "e5" in str(MODEL_NAME).lower()
    embedding_inputs = [
        f"query: {text}" if use_e5_style_prompt else text for text in embedding_input_list
    ]
    embeddings = compute_embeddings_batch_local(embedding_inputs, model, BATCH_SIZE)

    similarity_matrix = compute_similarity_matrix(embeddings, model_device=model.device)
    bg_target_cache = {}
    num_combinations_bg_target = len(background_major_list_orig) * len(target_major_list_orig)
    with tqdm(total=num_combinations_bg_target, desc="构建缓存") as pbar:
        for bg_major_orig in background_major_list_orig:
            for target_major_orig in target_major_list_orig:
                bg_major_repr = original_to_embedding_representation_map.get(bg_major_orig, "")
                target_major_repr = original_to_embedding_representation_map.get(
                    target_major_orig, ""
                )

                similarity = 0.0
                if (
                    bg_major_repr
                    and target_major_repr
                    and bg_major_repr in embedding_representation_to_idx_map
                    and target_major_repr in embedding_representation_to_idx_map
                ):
                    idx1 = embedding_representation_to_idx_map[bg_major_repr]
                    idx2 = embedding_representation_to_idx_map[target_major_repr]
                    similarity = similarity_matrix[idx1, idx2].item()

                key = _create_major_similarity_key(bg_major_orig, target_major_orig)
                bg_target_cache[key] = similarity
                pbar.update(1)

    root_cache_dir = os.path.join(project_root, "cache")
    bg_target_cache_path = os.path.join(root_cache_dir, "background_target_similarity.feather")

    def save_cache(cache_data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame({"key": list(cache_data.keys()), "similarity": list(cache_data.values())})
        df.to_feather(path)

    save_cache(bg_target_cache, bg_target_cache_path)


if __name__ == "__main__":
    precompute_similarities()
