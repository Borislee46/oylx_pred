import os

import pandas as pd


def feather_to_excel(feather_files: list[str], xlsx_files: list[str]):
    for feather_file, xlsx_file in zip(feather_files, xlsx_files, strict=True):
        if not os.path.exists(feather_file):
            print(f"警告: 文件不存在，跳过: {feather_file}")
            continue

        print(f"正在转换: {feather_file} -> {xlsx_file}")
        try:
            df = pd.read_feather(feather_file)
            df.to_excel(xlsx_file, index=False)
        except Exception as e:
            print(f"转换失败 {feather_file}: {e}")
    return True


feather_files = [
    "src/machine_learning_models/data/school_base.feather",
    "src/machine_learning_models/data/cases.feather",
    "src/machine_learning_models/data/school_major_details.feather",
]
xlsx_files = [
    "src/machine_learning_models/data/school_base.xlsx",
    "src/machine_learning_models/data/cases.xlsx",
    "src/machine_learning_models/data/school_major_details.xlsx",
]

feather_to_excel(feather_files, xlsx_files)
