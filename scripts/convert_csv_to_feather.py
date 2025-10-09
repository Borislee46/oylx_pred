import logging
import os
import time

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    return project_root


def convert_csv_to_feather(project_root, csv_relative_path, feather_relative_path):
    csv_full_path = os.path.join(project_root, csv_relative_path)
    feather_full_path = os.path.join(project_root, feather_relative_path)

    if not os.path.exists(csv_full_path):
        logging.error(f"源 CSV 文件未找到: {csv_full_path}")
        return False

    logging.info(f"正在转换 {csv_relative_path} 为 {feather_relative_path}...")
    start_time = time.time()
    try:
        df = pd.read_csv(csv_full_path)
        os.makedirs(os.path.dirname(feather_full_path), exist_ok=True)
        df.to_feather(feather_full_path)

        end_time = time.time()
        logging.info(f"成功转换为 {feather_relative_path}。时间: {end_time - start_time:.2f}s")
        return True
    except Exception as e:
        logging.error(f"转换 {csv_relative_path} 时出错: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    project_root = get_project_root()
    logging.info(f"项目根目录识别为: {project_root}")

    files_to_convert = [
        {
            "csv": "src/machine_learning_models/data/cases.csv",
            "feather": "src/machine_learning_models/data/cases.feather",
        },
        {
            "csv": "src/machine_learning_models/data/school_major_details.csv",
            "feather": "src/machine_learning_models/data/school_major_details.feather",
        },
        {
            "csv": "src/machine_learning_models/data/school_base.csv",
            "feather": "src/machine_learning_models/data/school_base.feather",
        },
    ]
    success_count = 0
    total_start_time = time.time()
    for file_pair in files_to_convert:
        if convert_csv_to_feather(project_root, file_pair["csv"], file_pair["feather"]):
            success_count += 1
    total_end_time = time.time()

    if success_count == len(files_to_convert):
        logging.info(
            f"所有 {success_count} 个指定的 CSV 文件转换成功。总脚本时间: {total_end_time - total_start_time:.2f}s"
        )
    else:
        logging.warning(
            f"只有 {success_count}/{len(files_to_convert)} 个 CSV 文件转换成功。请检查日志。总脚本时间: {total_end_time - total_start_time:.2f}s"
        )
