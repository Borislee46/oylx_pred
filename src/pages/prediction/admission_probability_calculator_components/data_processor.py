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
        result_types = [
            ("指定选择", user_specified_results),
            ("相似专业", similarity_results),
            ("跨专业", cross_major_results),
        ]

        data_for_df = []
        seen_combinations = set()

        for result_type, results in result_types:
            if not results:
                continue

            for result in results:
                school = result.get("university")
                major = result.get("major")
                prob = result.get("probability")

                if None in (school, major, prob) or (school, major) in seen_combinations:
                    continue

                seen_combinations.add((school, major))
                is_new_major = result.get("is_new_major", False)

                data_for_df.append(
                    {
                        "目标院校": school,
                        "目标专业": f"{major} (New!)" if is_new_major else major,
                        "原始专业名称": major,
                        "原始概率": prob,
                        "调整后概率": prob,
                        "类型": result_type,
                        "is_new_major": is_new_major,
                        "专业中文名称": result.get("chinese_name", ""),
                        "相似度": result.get("similarity", 0.0),
                        "专业大类": result.get("faculty", ""),
                    }
                )

        columns = [
            "目标院校",
            "目标专业",
            "原始专业名称",
            "原始概率",
            "调整后概率",
            "类型",
            "is_new_major",
            "专业中文名称",
            "相似度",
            "专业大类",
        ]

        return (
            pd.DataFrame(data_for_df, columns=columns)
            if data_for_df
            else pd.DataFrame(columns=columns)
        )

    def prepare_optimizer_input(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        column_mapping = {
            "目标院校": "university",
            "原始专业名称": "major",
            "调整后概率": "probability",
            "类型": "type",
            "is_new_major": "is_new_major",
            "专业中文名称": "chinese_name",
            "相似度": "similarity",
            "专业大类": "faculty",
        }

        available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        result = df[list(available_columns)].rename(columns=available_columns).to_dict("records")

        for item in result:
            if uni := item.get("university"):
                item["university"] = "".join(str(uni).split())

        return result
