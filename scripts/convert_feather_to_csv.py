import pandas as pd


def feather_to_csv(feather_file, csv_file):
    df = pd.read_feather(feather_file)
    df.to_csv(csv_file, index=False, encoding="utf-8")


feather_file = "src/machine_learning_models/data/school_base.feather"
csv_file = "src/machine_learning_models/data/school_base.csv"
feather_to_csv(feather_file, csv_file)
