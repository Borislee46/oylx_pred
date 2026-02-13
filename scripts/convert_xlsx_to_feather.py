import os

import pandas as pd


def xlsx_to_feather(xlsx_files: list[str], feather_files: list[str]):
    for xlsx_file, feather_file in zip(xlsx_files, feather_files, strict=True):
        if not os.path.exists(xlsx_file):
            print(f"警告: 文件不存在，跳过: {xlsx_file}")
            continue

        print(f"正在转换: {xlsx_file} -> {feather_file}")
        try:
            df = pd.read_excel(xlsx_file)

            df.columns = [str(c).strip() for c in df.columns]

            df.to_feather(feather_file)
        except Exception as e:
            print(f"转换失败 {xlsx_file}: {e}")
    return True


xlsx_files = [
    "src/machine_learning_models/data/school_base.xlsx",
    "src/machine_learning_models/data/cases.xlsx",
    "src/machine_learning_models/data/school_major_details.xlsx",
]

feather_files = [
    "src/machine_learning_models/data/school_base.feather",
    "src/machine_learning_models/data/cases.feather",
    "src/machine_learning_models/data/school_major_details.feather",
]

xlsx_to_feather(xlsx_files, feather_files)
