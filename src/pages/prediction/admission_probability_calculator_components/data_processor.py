from typing import Any

import pandas as pd


class DataProcessor:
    def __init__(self):
        pass

    def prepare_selected_schools_data(
        self,
        similarity_results: list[dict[str, Any]],
        cross_major_results: list[dict[str, Any]],
        user_specified_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        data_for_df = []
        seen_combinations = set()

        all_results = [
            ("指定选择", user_specified_results),
            ("相似专业", similarity_results),
            ("跨专业", cross_major_results),
        ]

        for result_type, results_list in all_results:
            if not results_list:
                continue

            for result in results_list:
                school = result.get("university")
                major = result.get("major")
                prob = result.get("probability")

                if school is None or major is None or prob is None:
                    continue

                if (school, major) not in seen_combinations:
                    display_major = (
                        f"{major} (New!)" if result.get("is_new_major", False) else major
                    )
                    data_for_df.append(
                        {
                            "目标院校": school,
                            "目标专业": display_major,
                            "原始专业名称": major,
                            "原始概率": prob,
                            "调整后概率": prob,
                            "类型": result_type,
                            "is_new_major": result.get("is_new_major", False),
                        }
                    )
                    seen_combinations.add((school, major))

        if not data_for_df:
            return pd.DataFrame(
                columns=[
                    "目标院校",
                    "目标专业",
                    "原始专业名称",
                    "原始概率",
                    "调整后概率",
                    "类型",
                    "is_new_major",
                ]
            )

        return pd.DataFrame(data_for_df)

    def prepare_optimizer_input(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        columns_for_optimizer = {
            "目标院校": "university",
            "原始专业名称": "major",
            "调整后概率": "probability",
        }
        if "类型" in df.columns:
            columns_for_optimizer["类型"] = "type"
        if "is_new_major" in df.columns:
            columns_for_optimizer["is_new_major"] = "is_new_major"

        all_schools_data = (
            df[list(columns_for_optimizer.keys())]
            .rename(columns=columns_for_optimizer)
            .to_dict("records")
        )
        for item in all_schools_data:
            uni = item.get("university")
            if isinstance(uni, str):
                cleaned = "".join(str(uni).split())
                item["university"] = cleaned
        return all_schools_data
