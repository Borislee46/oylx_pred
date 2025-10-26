import pandas as pd


def xlsx_to_feather(xlsx_files: list[str], feather_files: list[str]):
    for feather_file, xlsx_file in zip(feather_files, xlsx_files):
        df = pd.read_xlsx(xlsx_file)
        df.to_feather(feather_file)
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
