import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def convert_csv_to_feather(project_root, csv_relative_path, feather_relative_path):
    csv_full_path = os.path.join(project_root, csv_relative_path)
    feather_full_path = os.path.join(project_root, feather_relative_path)

    if not os.path.exists(csv_full_path):
        logging.error(f"源 CSV 文件未找到: {csv_full_path}")
        return False

    df = pd.read_csv(csv_full_path)
    os.makedirs(os.path.dirname(feather_full_path), exist_ok=True)
    df.to_feather(feather_full_path)
    logging.info(f"成功转换: {csv_relative_path} -> {feather_relative_path}")
    return True


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    files_to_convert = [
        (
            "src/machine_learning_models/data/cases.csv",
            "src/machine_learning_models/data/cases.feather",
        ),
        (
            "src/machine_learning_models/data/school_major_details.csv",
            "src/machine_learning_models/data/school_major_details.feather",
        ),
        (
            "src/machine_learning_models/data/school_base.csv",
            "src/machine_learning_models/data/school_base.feather",
        ),
    ]

    for csv_path, feather_path in files_to_convert:
        convert_csv_to_feather(project_root, csv_path, feather_path)
