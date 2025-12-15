import hashlib
import json
import time
from typing import Any

import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_GPA_SCALE,
    GPA_SCALES,
    TARGET_COUNTRY_UNIVERSITY_MAP,
)
from src.pages.prediction.input_form_components.language_score_converter import (
    LanguageScoreConverter,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

form_state_logger = setup_logger("page3", "prediction")

AUTO_SAVE_THROTTLE_TEXT = 4.0
AUTO_SAVE_THROTTLE_SELECT = 1.5
AUTO_SAVE_THROTTLE_DEFAULT = 2.0


class FormStateManager:
    @staticmethod
    def _clear_widget_state(widget_key: str) -> None:
        if widget_key in st.session_state:
            del st.session_state[widget_key]

    @staticmethod
    def initialize_session_state(session_manager: SessionManager) -> None:
        current_user = session_manager.get_current_user_info()
        user_id = current_user.get("username") if current_user else None

        default_states: dict[str, Any] = {
            "selected_target_universities": [],
            "selected_target_majors": [],
            "selected_target_countries": [],
            "selected_major_categories": [],
            "submitted": False,
            "form_data_changed": False,
            "last_gpa_warning_key": None,
            "last_lang_warning_key": None,
            "prediction_submit_lock": False,
            "school_base_df": None,
            "gpa_scale": DEFAULT_GPA_SCALE,
            "gpa_raw_input": None,
            "language_type": "雅思",
            "language_score_input": None,
            "background_university_initial": None,
            "background_major_original_initial": None,
            "research_count_initial": 0,
            "award_count_initial": 0,
            "internship_count_initial": 0,
            "paper_count_initial": 0,
            "research_details_initial": "",
            "award_details_initial": "",
            "internship_details_initial": "",
            "paper_details_initial": "",
            "current_user_id": user_id,
        }

        for key, default_value in default_states.items():
            if session_manager.get(key) is None:
                session_manager.set(**{key: default_value})

        if not session_manager.get("current_user_id") and user_id:
            session_manager.set(current_user_id=user_id)

    @staticmethod
    def _snapshot_hash(snapshot: dict) -> str:
        payload = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _get_current_form_snapshot(session_manager: SessionManager) -> dict:
        return {
            "selected_target_countries": session_manager.get("selected_target_countries", []),
            "selected_major_categories": session_manager.get("selected_major_categories", []),
            "selected_target_universities": session_manager.get("selected_target_universities", []),
            "selected_target_majors": session_manager.get("selected_target_majors", []),
            "gpa_raw": session_manager.get("gpa_raw_input"),
            "gpa_scale": session_manager.get("gpa_scale"),
            "exam_type": session_manager.get("standardized_test_type"),
            "exam_score": session_manager.get("current_exam_score"),
            "language_type": session_manager.get("language_type"),
            "language_score_raw": session_manager.get("language_score_input"),
            "background_university": session_manager.get_widget_value(
                "background_university_selectbox"
            ),
            "background_major_original": session_manager.get_widget_value(
                "background_major_selectbox"
            ),
            "research_count": session_manager.get_widget_value("research_count_input", 0),
            "award_count": session_manager.get_widget_value("award_count_input", 0),
            "internship_count": session_manager.get_widget_value("internship_count_input", 0),
            "paper_count": session_manager.get_widget_value("paper_count_input", 0),
            "experience_details": {
                "research_details": session_manager.get_widget_value("research_details_input", ""),
                "award_details": session_manager.get_widget_value("award_details_input", ""),
                "internship_details": session_manager.get_widget_value(
                    "internship_details_input", ""
                ),
                "paper_details": session_manager.get_widget_value("paper_details_input", ""),
            },
        }

    @staticmethod
    def _auto_save_form_data(
        session_manager: SessionManager, throttle_seconds: float = None
    ) -> None:
        current_form_data = FormStateManager._get_current_form_snapshot(session_manager)

        now = time.time()
        last_save_ts = session_manager.get("last_auto_save_ts", 0)
        last_hash = session_manager.get("last_saved_form_snapshot_hash")

        throttle_seconds = (
            AUTO_SAVE_THROTTLE_DEFAULT if throttle_seconds is None else float(throttle_seconds)
        )
        cur_hash = FormStateManager._snapshot_hash(current_form_data)

        if (now - last_save_ts) >= throttle_seconds and cur_hash != last_hash:
            session_manager.set(last_auto_save_ts=now, last_saved_form_snapshot_hash=cur_hash)

    @staticmethod
    def update_form_snapshot_hash_after_prediction(session_manager: SessionManager) -> None:
        current_form_data = FormStateManager._get_current_form_snapshot(session_manager)
        cur_hash = FormStateManager._snapshot_hash(current_form_data)
        now = time.time()
        session_manager.set(last_auto_save_ts=now, last_saved_form_snapshot_hash=cur_hash)

    @staticmethod
    def on_form_change(session_manager: SessionManager, change_type: str = None) -> None:
        session_manager.batch_set(
            {
                "submitted": False,
                "form_data_changed": True,
                "last_submission_logged": False,
                "prediction_submit_lock": False,
                "last_gpa_warning_key": None,
                "last_lang_warning_key": None,
                "cross_faculty_confirmed": False,
                "cross_faculty_cancelled": False,
                "pending_cross_faculty_prediction": False,
                "pending_prediction_data": None,
            }
        )

        throttle = AUTO_SAVE_THROTTLE_TEXT if change_type == "text" else AUTO_SAVE_THROTTLE_SELECT
        FormStateManager._auto_save_form_data(session_manager, throttle_seconds=throttle)

    @staticmethod
    def _on_target_selection_change(
        session_manager: SessionManager,
        session_state_key: str,
        widget_key: str,
        log_message_template: str,
    ) -> None:
        old_values = session_manager.get(session_state_key, [])
        new_values = session_manager.get_widget_value(widget_key, [])

        if old_values != new_values:
            form_state_logger.info(log_message_template.format(old=old_values, new=new_values))

        session_manager.set(**{session_state_key: new_values})

        if session_state_key == "selected_target_countries":
            current_selected_unis = session_manager.get("selected_target_universities", [])
            if new_values:
                country_uni_map = TARGET_COUNTRY_UNIVERSITY_MAP
                unis_in_selected_countries = [
                    uni for country in new_values for uni in country_uni_map.get(country, [])
                ]
                filtered_unis = [
                    uni for uni in current_selected_unis if uni in unis_in_selected_countries
                ]

                if current_selected_unis != filtered_unis:
                    form_state_logger.info(
                        f"由于国家变更，目标院校自动筛选 - 从 {current_selected_unis} 筛选为 {filtered_unis}"
                    )
                session_manager.set(selected_target_universities=filtered_unis)
                if "target_universities_multiselect" in st.session_state:
                    widget_val = st.session_state.get("target_universities_multiselect")
                    widget_list = (
                        list(widget_val) if isinstance(widget_val, (list, tuple, set)) else []
                    )
                    if widget_list != filtered_unis:
                        st.session_state["target_universities_multiselect"] = filtered_unis

        session_manager.set(target_options_cache={})
        FormStateManager.on_form_change(session_manager)

    @staticmethod
    def on_target_country_change(session_manager: SessionManager) -> None:
        FormStateManager._on_target_selection_change(
            session_manager,
            "selected_target_countries",
            "target_countries_multiselect",
            "用户更改目标国家 - 从 {old} 变更为 {new}",
        )

    @staticmethod
    def on_major_category_change(session_manager: SessionManager) -> None:
        FormStateManager._on_target_selection_change(
            session_manager,
            "selected_major_categories",
            "target_major_categories_multiselect",
            "用户更改专业大类 - 从 {old} 变更为 {new}",
        )

    @staticmethod
    def on_target_university_change(session_manager: SessionManager) -> None:
        FormStateManager._on_target_selection_change(
            session_manager,
            "selected_target_universities",
            "target_universities_multiselect",
            "用户更改目标院校 - 从 {old} 变更为 {new}",
        )

    @staticmethod
    def on_target_major_change(session_manager: SessionManager) -> None:
        FormStateManager._on_target_selection_change(
            session_manager,
            "selected_target_majors",
            "target_majors_multiselect",
            "用户更改目标专业 - 从 {old} 变更为 {new}",
        )

    @staticmethod
    def on_submit_click(session_manager: SessionManager) -> None:
        session_manager.set(submitted=True, form_data_changed=False, last_submission_logged=False)
        FormStateManager._auto_save_form_data(session_manager)

    @staticmethod
    def _convert_language_score(old_type: str, new_type: str, score: float) -> float | None:
        if old_type == "托福" and new_type == "雅思":
            return LanguageScoreConverter.toefl_to_ielts(score)
        if old_type == "雅思" and new_type == "托福":
            return LanguageScoreConverter.ielts_to_toefl(score)
        return None

    @staticmethod
    def on_language_type_change(session_manager: SessionManager) -> None:
        old_lang_type = session_manager.get("language_type")
        new_lang_type = session_manager.get_widget_value("language_type_widget_key")

        if old_lang_type == new_lang_type:
            return

        form_state_logger.info(f"用户更改语言类型 - 从 {old_lang_type} 变更为 {new_lang_type}")

        current_score = session_manager.get("language_score_input")
        converted_score = None

        if current_score is not None:
            cache_key = f"{old_lang_type}_{new_lang_type}_{current_score}"
            lang_cache = session_manager.get("lang_conversion_cache", {})

            if cache_key in lang_cache:
                converted_score = lang_cache[cache_key]
            else:
                converted_score = FormStateManager._convert_language_score(
                    old_lang_type, new_lang_type, float(current_score)
                )
                lang_cache[cache_key] = converted_score
                session_manager.set(lang_conversion_cache=lang_cache)

            if converted_score is not None:
                form_state_logger.info(
                    f"语言成绩自动转换 - {old_lang_type}: {current_score} -> {new_lang_type}: {converted_score}"
                )
            session_manager.set(language_score_input=converted_score)

        FormStateManager._clear_widget_state("language_score_input_widget")
        session_manager.set(language_type=new_lang_type)

    @staticmethod
    def _convert_gpa(old_scale_key: str, new_scale_key: str, gpa: float) -> float | None:
        old_scale = GPA_SCALES.get(old_scale_key)
        new_scale = GPA_SCALES.get(new_scale_key)
        if not old_scale or not new_scale:
            return None

        old_max = old_scale.get("max")
        new_max = new_scale.get("max")
        if not isinstance(old_max, (int, float)) or not isinstance(new_max, (int, float)):
            return None
        if old_max <= 0:
            return None

        raw_val = (gpa / old_max) * new_max
        return int(raw_val * 100 + 0.5 + 1e-9) / 100.0

    @staticmethod
    def gpa_scale_changed(session_manager: SessionManager) -> None:
        old_scale_key = session_manager.get("gpa_scale")
        new_scale_key = session_manager.get_widget_value("gpa_scale_widget_key")

        if old_scale_key == new_scale_key:
            return

        form_state_logger.info(f"用户更改GPA分制 - 从 {old_scale_key} 变更为 {new_scale_key}")

        current_gpa = session_manager.get("gpa_raw_input")
        if current_gpa is not None and isinstance(current_gpa, (int, float)):
            cache_key = f"{old_scale_key}_{new_scale_key}_{current_gpa}"
            gpa_cache = session_manager.get("gpa_conversion_cache", {})

            if cache_key in gpa_cache:
                converted_gpa = gpa_cache[cache_key]
            else:
                converted_gpa = FormStateManager._convert_gpa(
                    old_scale_key, new_scale_key, current_gpa
                )
                if converted_gpa is not None:
                    gpa_cache[cache_key] = converted_gpa
                    session_manager.set(gpa_conversion_cache=gpa_cache)
                else:
                    converted_gpa = current_gpa

            if converted_gpa != current_gpa:
                form_state_logger.info(
                    f"GPA自动转换 - {old_scale_key}: {current_gpa:.2f} -> {new_scale_key}: {converted_gpa:.2f}"
                )

            session_manager.set(gpa_raw_input=converted_gpa)
            st.session_state["gpa_raw_input_widget"] = converted_gpa

        session_manager.set(gpa_scale=new_scale_key)
