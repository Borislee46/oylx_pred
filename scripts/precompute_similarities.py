import logging
import os
import re
import sys
import time

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_environment():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)

    logging.info(f"项目根目录: {project_root}")

    src_path = os.path.join(project_root, "src")
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
        logging.info(f"添加 {src_path} 到 Python 路径")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logging.info(f"添加 {project_root} 到 Python 路径")

    return project_root, script_dir


project_root, script_dir = setup_environment()

logging.info("导入本地脚本中的模型工具...")
from model_utils import compute_embeddings_batch_local, compute_similarity_matrix, get_model


def import_dependencies():
    try:
        logging.info("尝试相对导入...")
        from src.pages.prediction.prediction_utils import get_cached_major_similarity_key

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

        logging.info("相对导入成功")
        return (
            get_cached_major_similarity_key,
            CASES_DATA_PATH,
            SCHOOL_MAJOR_DETAILS_PATH,
            MAJOR_SIMILARITY_CACHE_PATH,
        )

    except ImportError as e:
        logging.warning(f"相对导入失败: {e}")


(
    get_cached_major_similarity_key,
    CASES_DATA_PATH,
    SCHOOL_MAJOR_DETAILS_PATH,
    MAJOR_SIMILARITY_CACHE_PATH,
) = import_dependencies()

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
BATCH_SIZE = 64


def clean_english_major_name_specifically(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name_no_chinese = re.sub(r"[\u4e00-\u9fff]+", "", name)
    cleaned_name = re.sub(r"[^a-zA-Z0-9\s\-\(\)\/\&\.\']+", "", name_no_chinese)

    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    return cleaned_name


def precompute_similarities():
    start_total_time = time.time()
    logging.info("开始相似度预计算...")

    logging.info("加载原始专业名称...")
    start_load_time = time.time()

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
        if lower.endswith(".csv"):
            return pd.read_csv(path)
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        try:
            return pd.read_feather(path)
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_parquet(path)

    try:
        if os.path.exists(SCHOOL_MAJOR_DETAILS_PATH):
            details_df = read_table_auto(SCHOOL_MAJOR_DETAILS_PATH)
            if "专业英文名称" in details_df.columns and "专业中文名称" in details_df.columns:
                logging.info(f"从 {SCHOOL_MAJOR_DETAILS_PATH} 加载专业英文名称和中文名称...")
                for _, row in details_df.iterrows():
                    eng_major_val = row["专业英文名称"]
                    chi_major_val = row["专业中文名称"]

                    eng_major = str(eng_major_val) if pd.notna(eng_major_val) else None
                    chi_major = str(chi_major_val) if pd.notna(chi_major_val) else None

                    if eng_major:
                        raw_school_english_majors.add(eng_major)
                        if chi_major and chi_major.strip():
                            english_to_chinese_major_map[eng_major] = chi_major
                logging.info(f"加载了 {len(raw_school_english_majors)} 个原始学校专业英文名称。")
            elif "专业英文名称" in details_df.columns:
                raw_school_english_majors.update(details_df["专业英文名称"].dropna().astype(str))
                logging.info(
                    f"从 {SCHOOL_MAJOR_DETAILS_PATH} 加载了 {len(raw_school_english_majors)} 个学校专业英文名称。"
                )
                logging.warning(
                    f"'专业中文名称' 列未在 {SCHOOL_MAJOR_DETAILS_PATH} 中找到，将无法使用中文名称进行匹配。"
                )
            else:
                logging.warning(
                    f"'专业英文名称' 和/或 '专业中文名称' 列未在 {SCHOOL_MAJOR_DETAILS_PATH} 中找到"
                )
        else:
            logging.warning(f"学校详情文件未找到: {SCHOOL_MAJOR_DETAILS_PATH}")

        if os.path.exists(CASES_DATA_PATH):
            cases_df = read_table_auto(CASES_DATA_PATH)
            raw_case_background_majors.update(cases_df["background_major"].dropna().astype(str))
            raw_case_target_majors.update(cases_df["target_major"].dropna().astype(str))
            logging.info(f"从 {CASES_DATA_PATH} 加载了背景专业和目标专业。")
        else:
            logging.warning(f"案例数据文件未找到: {CASES_DATA_PATH}")

        raw_case_background_majors.discard("无")
        raw_case_background_majors.discard("")
        raw_case_target_majors.discard("无")
        raw_case_target_majors.discard("")
        raw_school_english_majors.discard("无")
        raw_school_english_majors.discard("")

        all_target_majors = raw_case_target_majors.union(raw_school_english_majors)

        if not raw_case_background_majors:
            logging.error("未找到任何背景专业。无法计算相似度。退出。")
            sys.exit(1)
        if not all_target_majors:
            logging.error("未找到任何目标专业。无法计算相似度。退出。")
            sys.exit(1)

        background_major_list_orig = sorted(list(raw_case_background_majors))
        target_major_list_orig = sorted(list(all_target_majors))
        all_majors_for_embedding_orig = sorted(
            list(set(background_major_list_orig + target_major_list_orig))
        )

        logging.info(
            f"共找到 {len(background_major_list_orig)} 个背景专业, {len(target_major_list_orig)} 个目标专业。"
        )
        logging.info(
            f"共计 {len(all_majors_for_embedding_orig)} 个唯一专业需要计算嵌入。加载时间: {time.time() - start_load_time:.2f}s"
        )

    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        sys.exit(1)

    logging.info("准备用于嵌入计算的专业名称表示...")
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
                if raw_major in english_to_chinese_major_map:
                    logging.warning(
                        f"专业英文名称 '{raw_major}' 对应的中文名为空或无效，将使用原始英文名 '{representation}' 进行匹配。"
                    )
        else:
            representation = raw_major

        if representation is None:
            representation = ""

        original_to_embedding_representation_map[raw_major] = representation
        if representation:
            embedding_representation_set.add(representation)

    embedding_input_list = sorted(list(embedding_representation_set))

    if not embedding_input_list:
        logging.error("没有有效的专业名称表示用于嵌入计算。退出。")
        sys.exit(1)

    embedding_representation_to_idx_map = {name: i for i, name in enumerate(embedding_input_list)}

    logging.info(f"加载模型: {MODEL_NAME}...")
    start_model_time = time.time()
    try:
        model = get_model(MODEL_NAME)
        logging.info(f"模型加载成功。时间: {time.time() - start_model_time:.2f}s")
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        sys.exit(1)

    start_embed_time = time.time()
    try:
        instructed_embedding_input_list = []
        for major_name in embedding_input_list:
            prompt = f"Instruct: Represent this academic major for similarity comparison.\\nQuery: {major_name}"
            instructed_embedding_input_list.append(prompt)

        embeddings = compute_embeddings_batch_local(
            instructed_embedding_input_list, model, BATCH_SIZE
        )
        if len(embeddings) != len(instructed_embedding_input_list):
            logging.error("嵌入数量与带指令的专业名称表示数量不匹配。")
            sys.exit(1)
        logging.info(f"嵌入计算完成。时间: {time.time() - start_embed_time:.2f}s")
    except Exception as e:
        logging.error(f"计算嵌入失败: {e}")
        sys.exit(1)

    logging.info("基于专业名称表示的嵌入计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(embeddings, model_device=model.device)

    logging.info("构建背景专业-背景专业相似度缓存 (用于相似案例)...")
    bg_bg_cache = {}
    num_combinations_bg_bg = len(background_major_list_orig) * (
        len(background_major_list_orig) - 1
    ) // 2 + len(background_major_list_orig)
    with tqdm(total=num_combinations_bg_bg, desc="Building BG-BG Cache") as pbar:
        for i, major1_orig in enumerate(background_major_list_orig):
            key_self = get_cached_major_similarity_key(major1_orig, major1_orig)
            major1_repr = original_to_embedding_representation_map.get(major1_orig, "")
            bg_bg_cache[key_self] = (
                1.0 if major1_repr and major1_repr in embedding_representation_to_idx_map else 0.0
            )
            pbar.update(1)

            for j in range(i + 1, len(background_major_list_orig)):
                major2_orig = background_major_list_orig[j]

                major1_repr = original_to_embedding_representation_map.get(major1_orig, "")
                major2_repr = original_to_embedding_representation_map.get(major2_orig, "")

                similarity = 0.0
                if (
                    major1_repr
                    and major2_repr
                    and major1_repr in embedding_representation_to_idx_map
                    and major2_repr in embedding_representation_to_idx_map
                ):
                    idx1 = embedding_representation_to_idx_map[major1_repr]
                    idx2 = embedding_representation_to_idx_map[major2_repr]
                    similarity = similarity_matrix[idx1, idx2].item()

                key = get_cached_major_similarity_key(major1_orig, major2_orig)
                bg_bg_cache[key] = similarity
                pbar.update(1)

    logging.info("构建背景专业-目标专业相似度缓存 (用于预测)...")
    bg_target_cache = {}
    num_combinations_bg_target = len(background_major_list_orig) * len(target_major_list_orig)
    with tqdm(total=num_combinations_bg_target, desc="Building BG-Target Cache") as pbar:
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

                key = get_cached_major_similarity_key(bg_major_orig, target_major_orig)
                bg_target_cache[key] = similarity
                pbar.update(1)

    # 将缓存统一保存到项目根目录下的 cache 目录，格式为 feather
    root_cache_dir = os.path.join(project_root, "cache")
    bg_bg_cache_path = os.path.join(root_cache_dir, "background_background_similarity.feather")
    bg_target_cache_path = os.path.join(root_cache_dir, "background_target_similarity.feather")

    for path in [MAJOR_SIMILARITY_CACHE_PATH, MAJOR_SIMILARITY_CACHE_PATH.replace(".json", ".pkl")]:
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"已删除旧的缓存文件: {path}")
            except Exception as e:
                logging.warning(f"删除旧缓存文件失败: {path} - {e}")

    def save_cache(cache_data, path, name):
        logging.info(f"保存{name}到 {path} ...")
        start_save_time = time.time()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 将字典转换为 DataFrame 后保存为 feather
            df = pd.DataFrame(
                {"key": list(cache_data.keys()), "similarity": list(cache_data.values())}
            )
            df.to_feather(path)
            logging.info(
                f"{name}保存成功。{len(cache_data)} 个条目。时间: {time.time() - start_save_time:.2f}s"
            )
        except Exception as e:
            logging.error(f"保存{name}失败: {e}")

    save_cache(bg_bg_cache, bg_bg_cache_path, "背景专业-背景专业相似度缓存")
    save_cache(bg_target_cache, bg_target_cache_path, "背景专业-目标专业相似度缓存")

    logging.info(f"预计算完成。总时间: {time.time() - start_total_time:.2f}s")


if __name__ == "__main__":
    precompute_similarities()
