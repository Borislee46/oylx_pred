import pandas as pd

from src.pages.prediction.data_sort_config.top_result_school_order import (
    UNIVERSITY_ORDER_MAP,
    UNIVERSITY_SORT_ORDER,
)


class DataFrameBuilder:
    @staticmethod
    def get_probability_value(probability):
        return float(probability) if probability is not None else 0.0

    @staticmethod
    def clean_and_reorder_columns(df):
        if "±%" in df.columns and df["±%"].astype(str).str.strip().eq("").all():
            df = df.drop(columns=["±%"])
            return df

        if "±%" in df.columns and "录取概率" in df.columns:
            cols = list(df.columns)
            cols.remove("±%")
            insert_pos = cols.index("录取概率") + 1
            cols = cols[:insert_pos] + ["±%"] + cols[insert_pos:]
            df = df[cols]

        return df

    @classmethod
    def create_results_dataframe(
        cls,
        results: list,
        prev_prob_map: dict | None = None,
        show_delta: bool = False,
        max_items: int | None = None,
        delta_calculator=None,
    ):
        if not results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业中文名称"])

        results.sort(
            key=lambda item: (
                UNIVERSITY_ORDER_MAP.get(item.get("university"), len(UNIVERSITY_SORT_ORDER)),
                -cls.get_probability_value(item.get("probability")),
            )
        )

        if isinstance(max_items, int) and max_items > 0:
            results = results[:max_items]

        data = {
            "目标院校": [result["university"] for result in results],
            "目标专业": [
                (
                    f"{result['major']} (New!)"
                    if result.get("is_new_major", False)
                    else result["major"]
                )
                for result in results
            ],
            "录取概率": [
                cls.get_probability_value(result.get("probability")) for result in results
            ],
            "专业中文名称": [result.get("chinese_name", "") for result in results],
        }

        if show_delta and delta_calculator:
            data["±%"] = [
                delta_calculator.calculate_delta(result, prev_prob_map) for result in results
            ]

        return pd.DataFrame(data)
