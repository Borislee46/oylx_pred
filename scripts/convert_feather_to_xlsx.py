import pandas as pd


def feather_to_excel(feather_files: list[str], xlsx_files: list[str]):
    for feather_file, xlsx_file in zip(feather_files, xlsx_files):
        df = pd.read_feather(feather_file)
        df.to_excel(xlsx_file, index=False)
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
