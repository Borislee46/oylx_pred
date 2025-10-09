import pandas as pd


def xlsx_to_csv(xlsx_file, csv_file, sheet_name=0):
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    df.to_csv(csv_file, index=False, encoding="utf-8")


xlsx_file = "src/machine_learning_models/data/school_major_details.xlsx"
csv_file = "src/machine_learning_models/data/school_major_details.csv"
xlsx_to_csv(xlsx_file, csv_file)
