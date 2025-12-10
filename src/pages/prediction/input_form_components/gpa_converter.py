import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

GPA_RULES_CONFIG_PATH = Path("config/gpa_conversion_rules.json")


class GPAConverter:
    def __init__(self, school_base_df):
        if school_base_df is not None and not school_base_df.empty:
            self.school_country_map = dict(zip(school_base_df["学校名称"], school_base_df["国家"]))
        else:
            self.school_country_map = {}

    def get_university_country(self, university_name):
        if not university_name:
            return None
        return self.school_country_map.get(university_name)

    @staticmethod
    @st.cache_data
    def load_gpa_conversion_rules(config_path: str, file_mtime: float):
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"GPA转换规则文件不存在: {config_path}")
            return None

        with path.open(encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            logger.warning("GPA转换规则文件格式错误，应为JSON对象")
            return None

        required_keys = ["conversion_rules", "country_rules"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            logger.warning(f"GPA转换规则文件缺少必要字段: {missing}")
            return None

        return config

    @staticmethod
    def _get_rules_config() -> dict | None:
        file_mtime = (
            GPA_RULES_CONFIG_PATH.stat().st_mtime if GPA_RULES_CONFIG_PATH.exists() else 0.0
        )
        return GPAConverter.load_gpa_conversion_rules(str(GPA_RULES_CONFIG_PATH), file_mtime)

    @staticmethod
    def _find_matching_rule(
        rules_config: dict, scale_key: str, background_university: str = None, country: str = None
    ) -> tuple[dict | None, str]:
        if background_university:
            rule = rules_config.get("conversion_rules", {}).get(background_university)
            if rule and rule.get("trigger_scale") == scale_key:
                return rule, f"大学规则({background_university})"

        if country:
            rule = rules_config.get("country_rules", {}).get(country)
            if rule and rule.get("trigger_scale") == scale_key:
                return rule, f"国家规则({country})"

        return None, ""

    @staticmethod
    def convert_gpa_by_rules(
        raw_gpa: float,
        scale_key: str,
        background_university: str = None,
        country: str = None,
    ) -> float | None:
        if raw_gpa is None or raw_gpa == "":
            return None

        try:
            raw_gpa = float(raw_gpa)
        except (ValueError, TypeError):
            logger.warning(f"无法转换GPA值为数字: {raw_gpa}")
            return None

        rules_config = GPAConverter._get_rules_config()
        if not rules_config:
            return None

        rule, rule_name = GPAConverter._find_matching_rule(
            rules_config, scale_key, background_university, country
        )
        if not rule:
            return None

        return GPAConverter._apply_conversion_rule(raw_gpa, rule)

    @staticmethod
    def _interpolate(
        raw_value: float, min_val: float, max_val: float, target_min: float, target_max: float
    ) -> float:
        if abs(max_val - min_val) < 1e-9:
            return target_min
        ratio = (raw_value - min_val) / (max_val - min_val)
        return target_min + ratio * (target_max - target_min)

    @staticmethod
    def _apply_range_rule(raw_value: float, range_rule: dict) -> float | None:
        min_val = float(range_rule.get("min", 0))
        max_val = float(range_rule.get("max", 0))

        if not (min_val <= raw_value < max_val):
            return None

        if "target_gpa" in range_rule:
            return round(float(range_rule["target_gpa"]), 2)

        if "target_min" in range_rule and "target_max" in range_rule:
            result = GPAConverter._interpolate(
                raw_value,
                min_val,
                max_val,
                float(range_rule["target_min"]),
                float(range_rule["target_max"]),
            )
            return round(result, 2)

        return None

    @staticmethod
    def _apply_fallback(raw_value: float, rule: dict) -> float:
        multiplier = float(rule.get("fallback_multiplier", 1.0))
        if rule.get("is_percentage"):
            result = (raw_value / 100.0) * 4.0 * multiplier
        else:
            result = raw_value * multiplier
        return round(max(0.0, min(result, 4.0)), 2)

    @staticmethod
    def _apply_conversion_rule(raw_value: float, rule: dict[str, Any]) -> float:
        raw_value = float(raw_value)
        ranges = rule.get("ranges", [])

        for range_rule in ranges:
            if not isinstance(range_rule, dict):
                continue
            result = GPAConverter._apply_range_rule(raw_value, range_rule)
            if result is not None:
                return result

        return GPAConverter._apply_fallback(raw_value, rule)
