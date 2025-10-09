import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _get_gpa_rules_config_path() -> str:
    return os.path.join(os.getcwd(), "config", "gpa_conversion_rules.json")


class GPAConverter:
    def __init__(self, school_base_df):
        if school_base_df is not None and not school_base_df.empty:
            self.school_country_map = pd.Series(
                school_base_df["国家"].values, index=school_base_df["学校名称"]
            ).to_dict()
        else:
            self.school_country_map = {}

        self._country_cache = {}

    def get_university_country(self, university_name):
        if not self.school_country_map or university_name is None:
            return None

        if university_name in self._country_cache:
            return self._country_cache[university_name]

        country = self.school_country_map.get(university_name)
        self._country_cache[university_name] = country
        return country

    @staticmethod
    @st.cache_data
    def load_gpa_conversion_rules(config_path: str, file_mtime: float):
        try:
            if not os.path.exists(config_path):
                logger.warning(f"警告：GPA转换规则文件不存在: {config_path}")
                return None
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            if not isinstance(config, dict):
                logger.warning("警告：GPA转换规则文件格式错误，应为JSON对象")
                return None
            required_keys = ["conversion_rules", "country_rules"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                logger.warning(f"警告：GPA转换规则文件缺少必要字段: {missing_keys}")
                return None
            return config
        except json.JSONDecodeError as e:
            logger.error(f"错误：GPA转换规则文件JSON格式错误: {e}")
            return None
        except Exception as e:
            logger.error(f"错误：无法加载GPA转换规则文件: {e}")
            return None

    @staticmethod
    def convert_gpa_by_rules(
        raw_gpa: float, scale_key: str, background_university: str = None, country: str = None
    ) -> float | None:
        if raw_gpa is None or raw_gpa == "":
            return None

        try:
            raw_gpa = float(raw_gpa)
        except (ValueError, TypeError):
            logger.warning(f"警告：无法转换GPA值为数字: {raw_gpa}")
            return None

        config_path = _get_gpa_rules_config_path()
        try:
            file_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0.0
        except Exception:
            file_mtime = 0.0
        rules_config = GPAConverter.load_gpa_conversion_rules(config_path, file_mtime)
        if not rules_config:
            return None

        if background_university and background_university in rules_config.get(
            "conversion_rules", {}
        ):
            rule = rules_config["conversion_rules"][background_university]
            if rule.get("trigger_scale") == scale_key:
                try:
                    return GPAConverter._apply_conversion_rule(raw_gpa, rule)
                except Exception as e:
                    logger.error(f"错误：应用大学转换规则失败 ({background_university}): {e}")

        if country and country in rules_config.get("country_rules", {}):
            rule = rules_config["country_rules"][country]
            if rule.get("trigger_scale") == scale_key:
                try:
                    return GPAConverter._apply_conversion_rule(raw_gpa, rule)
                except Exception as e:
                    logger.error(f"错误：应用国家转换规则失败 ({country}): {e}")

        return None

    @staticmethod
    def _apply_conversion_rule(raw_value: float, rule: dict[str, Any]) -> float:
        if not isinstance(rule, dict):
            raise ValueError("转换规则必须是字典格式")

        raw_value = float(raw_value)

        ranges = rule.get("ranges", [])
        if not isinstance(ranges, list):
            raise ValueError("ranges字段必须是列表格式")

        for range_rule in ranges:
            if not isinstance(range_rule, dict):
                continue

            try:
                min_val = float(range_rule.get("min", 0))
                max_val = float(range_rule.get("max", 0))

                if min_val <= raw_value < max_val:
                    if "target_gpa" in range_rule:
                        return round(float(range_rule["target_gpa"]), 2)
                    elif "target_min" in range_rule and "target_max" in range_rule:
                        target_min = float(range_rule["target_min"])
                        target_max = float(range_rule["target_max"])

                        if max_val == min_val:
                            ratio = 0.0
                        else:
                            ratio = (raw_value - min_val) / (max_val - min_val)

                        target_val = target_min + ratio * (target_max - target_min)
                        return round(target_val, 2)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"警告：跳过无效的范围规则: {e}")
                continue

        try:
            if rule.get("is_percentage"):
                base = (raw_value / 100.0) * 4.0
                multiplier = float(rule.get("fallback_multiplier", 1.0))
                result = base * multiplier
                return round(max(0.0, min(result, 4.0)), 2)

            multiplier = float(rule.get("fallback_multiplier", 1.0))
            result = raw_value * multiplier
            return round(max(0.0, min(result, 4.0)), 2)
        except (ValueError, TypeError) as e:
            logger.warning(f"警告：应用fallback公式失败: {e}")
            return round(raw_value, 2)
