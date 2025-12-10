import pandas as pd

from src.pages.prediction.data_sort_config.top_result_school_order import (
    UNIVERSITY_ORDER_MAP,
    UNIVERSITY_SORT_ORDER,
)
from src.pages.prediction.prediction_utils import get_school_major_details


class DataFrameBuilder:
    @staticmethod
    def get_probability_value(probability):
        return float(probability) if probability is not None else 0.0

    @classmethod
    def create_results_dataframe(
        cls,
        results: list,
        max_items: int | None = None,
    ):
        if not results:
            return pd.DataFrame(columns=["目标院校", "目标专业", "录取概率", "专业详情"])

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
            "专业详情": [
                get_school_major_details(result.get("university"), result.get("major")) or ""
                for result in results
            ],
        }

        return pd.DataFrame(data)
