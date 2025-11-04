import pandas as pd
import streamlit as st

from src.pages.prediction.page_data_loader import cached_load_bg_target_similarity_cache
from src.pages.prediction.result_modifier.config import DEFAULT_TEXT_BOOST_CONFIG
from src.pages.prediction.result_modifier.probability_adjuster import (
    ProbabilityAdjuster,
    penalize_cross_major_without_cases,
)
from src.pages.prediction.result_modifier.professional_adjustment import (
    adjust_for_professional_majors,
)
from src.pages.prediction.result_modifier.text_boost_provider import (
    get_text_boost_provider,
)
from src.pages.prediction.result_modifier.utils import has_meaningful_experience_text
from src.pages.prediction.results_handler import (
    combine_and_deduplicate_results,
)
from src.pages.prediction.run_prediction import run_single_prediction
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel

prediction_handler_logger = setup_logger("page3", "prediction")


def validate_model_and_features(prediction_model):
    if prediction_model is None:
        st.error(
            "关键配置错误：无法加载预测模型或其依赖的全局类别数据。预测功能无法启动。请检查应用日志并联系管理员。"
        )
        prediction_handler_logger.critical("预测函数无法继续：prediction_model 为 None。")
        return None

    if hasattr(prediction_model, "feature_names") and prediction_model.feature_names is not None:
        return prediction_model.feature_names
    else:
        prediction_handler_logger.error("模型特征列表为空，但模型已加载。这通常表示模型配置问题。")
        st.error("模型配置错误：特征列表为空。")
        return None


def prepare_input_data(input_data_from_form):
    from src.utils.app_data_loader import load_raw_cases_data
    from src.utils.school_level_service import get_school_level_service

    background_uni_name = input_data_from_form.get("background_university", "")
    service = get_school_level_service()
    school_level = service.get_school_level(background_uni_name)

    input_data = input_data_from_form.copy()
    input_data["school_level"] = school_level

    background_major = input_data.get("background_major")
    if background_major:
        from src.pages.prediction.prediction_utils import get_background_faculty

        cases_df = load_raw_cases_data()
        faculty = get_background_faculty(background_major, cases_df)
        if faculty:
            input_data["faculty"] = faculty

    return input_data


def _pipeline_adjust_results(
    results: list,
    probability_adjuster,
    text_boost_provider,
    experience_details: dict,
    gpa,
    language_score,
    background_university,
    is_new_major_cache: dict[tuple[str, str], bool] | None = None,
):
    if not results or not isinstance(results, list):
        return results

    dict_indices = [idx for idx, r in enumerate(results) if isinstance(r, dict)]
    if not dict_indices:
        return results

    base_probs = [results[i].get("probability", 0.0) for i in dict_indices]

    if probability_adjuster and gpa is not None and language_score is not None:
        adjusted_probs = []
        for p in base_probs:
            try:
                ap = probability_adjuster.adjust_probability(
                    p,
                    gpa,
                    language_score,
                    background_university_name=background_university,
                )
                adjusted_probs.append(max(0.0, min(1.0, float(ap))))
            except Exception as e:
                prediction_handler_logger.warning(f"概率调整失败: {e}")
                adjusted_probs.append(p)
    else:
        adjusted_probs = base_probs

    if text_boost_provider is not None and isinstance(experience_details, dict):
        try:
            boosted_probs, _ = text_boost_provider.apply(adjusted_probs, experience_details)
        except Exception as e:
            prediction_handler_logger.warning(f"文本增强失败: {e}")
            boosted_probs = adjusted_probs
    else:
        boosted_probs = adjusted_probs

    for pos, idx in enumerate(dict_indices):
        if pos < len(boosted_probs):
            try:
                results[idx]["probability"] = max(0.0, min(1.0, float(boosted_probs[pos])))
            except Exception as e:
                prediction_handler_logger.warning(f"概率赋值失败: {e}")

    if is_new_major_cache is not None:
        for idx in dict_indices:
            r = results[idx]
            uni = r.get("university")
            major = r.get("major")
            if uni and major:
                key = (uni, major)
                r["is_new_major"] = is_new_major_cache.get(key, False)
            else:
                r["is_new_major"] = False

    return results


def _batch_adjust_results(
    results_list: list[list],
    probability_adjuster,
    text_boost_provider,
    experience_details: dict,
    gpa,
    language_score,
    background_university,
):
    if not results_list or not any(results_list):
        return results_list

    from src.pages.prediction.prediction_utils import is_new_major

    all_combinations = set()
    for results in results_list:
        if results and isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    uni = r.get("university")
                    major = r.get("major")
                    if uni and major:
                        all_combinations.add((uni, major))

    is_new_major_cache: dict[tuple[str, str], bool] = {}
    if all_combinations:
        try:
            for uni, major in all_combinations:
                is_new_major_cache[(uni, major)] = is_new_major(uni, major)
        except Exception as e:
            prediction_handler_logger.warning(f"批量查询新专业失败: {e}")

    adjusted_results_list = []
    for results in results_list:
        adjusted = _pipeline_adjust_results(
            results,
            probability_adjuster,
            text_boost_provider,
            experience_details,
            gpa,
            language_score,
            background_university,
            is_new_major_cache,
        )
        adjusted_results_list.append(adjusted)

    return adjusted_results_list


def compute_list_fingerprint(lst: list) -> tuple[int, int]:
    if not lst:
        return (0, 0)
    try:
        import hashlib

        sorted_list = sorted(lst)
        list_str = ",".join(str(item) for item in sorted_list)
        hash_value = int(hashlib.md5(list_str.encode()).hexdigest()[:16], 16)
        return (len(lst), hash_value)
    except Exception as e:
        prediction_handler_logger.warning(f"计算列表指纹失败: {e}")
        return (len(lst), 0)


def compute_df_fingerprint(df) -> int:
    if df is None or df.empty:
        return 0
    try:
        from pandas.util import hash_pandas_object

        key_cols = [
            c
            for c in ["background_university", "target_university", "target_major"]
            if c in df.columns
        ]
        if key_cols:
            return int(hash_pandas_object(df[key_cols]).sum())
        return len(df)
    except Exception as e:
        prediction_handler_logger.warning(f"计算DataFrame指纹失败: {e}")
        return len(df)


@st.cache_data(ttl=3600)
def _get_cached_cases_df(fingerprint: int):
    from src.utils.app_data_loader import load_raw_cases_data

    return load_raw_cases_data()


@st.cache_data(ttl=600, show_spinner=True)
def run_prediction_pipeline(
    input_data,
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names,
    all_universities_fingerprint: tuple[int, int],
    all_majors_fingerprint: tuple[int, int],
):
    from src.pages.prediction.page_data_loader import cached_get_prediction_model

    prediction_model = cached_get_prediction_model(model_name)

    if prediction_model is None:
        return PredictionResultModel()

    cases_df = _get_cached_cases_df(cases_df_fingerprint)

    all_universities_target = input_data.get("_all_universities_target", [])
    all_majors_target = input_data.get("_all_majors_target", [])

    bg_target_similarity_cache = cached_load_bg_target_similarity_cache()

    target_universities = input_data.get("target_universities", [])
    num_target_universities = len(target_universities) if target_universities else 0

    sim_results, cross_results, user_specified_results, _ = run_single_prediction(
        current_input_data=input_data,
        prediction_model=prediction_model,
        cases_df=cases_df,
        bg_target_similarity_cache=bg_target_similarity_cache,
        expected_features=loaded_feature_names,
        all_universities_target=all_universities_target,
        all_majors_target=all_majors_target,
        num_target_universities=num_target_universities,
    )

    if all(x is None for x in [sim_results, cross_results, user_specified_results]):
        prediction_handler_logger.error("预测失败：所有结果为None")
        return PredictionResultModel()

    internship_count = input_data.get("internship_count", 0)
    user_specified_majors = input_data.get("target_majors", [])

    sim_results = adjust_for_professional_majors(
        sim_results, internship_count, user_specified_majors
    )
    cross_results = adjust_for_professional_majors(
        cross_results, internship_count, user_specified_majors
    )
    user_specified_results = adjust_for_professional_majors(
        user_specified_results if isinstance(user_specified_results, list) else [],
        internship_count,
        user_specified_majors,
    )

    background_major = input_data.get("background_major")
    user_specified_results = penalize_cross_major_without_cases(
        user_specified_results=user_specified_results,
        background_major=background_major,
        cases_df=cases_df,
    )

    gpa = input_data.get("gpa")
    language_score = input_data.get("language_score")
    background_university = input_data.get("background_university")

    probability_adjuster = None
    if gpa is not None and language_score is not None:
        probability_adjuster = ProbabilityAdjuster(
            cases_df if cases_df is not None else pd.DataFrame()
        )

    experience_details = input_data.get("experience_details", {})
    if isinstance(experience_details, dict):
        for k in ("research_count", "award_count", "internship_count", "paper_count"):
            if k in input_data:
                try:
                    experience_details[k] = int(input_data.get(k, 0) or 0)
                except Exception:
                    experience_details[k] = 0
    if has_meaningful_experience_text(experience_details):
        text_provider = get_text_boost_provider(DEFAULT_TEXT_BOOST_CONFIG)
    else:
        text_provider = None

    sim_results, cross_results, user_specified_results = _batch_adjust_results(
        [sim_results, cross_results, user_specified_results],
        probability_adjuster,
        text_provider,
        experience_details,
        gpa,
        language_score,
        background_university,
    )

    unique_results = combine_and_deduplicate_results(
        sim_results, cross_results, user_specified_results
    )

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        unified_results=unique_results,
    )


def persist_input_state(
    session_manager,
    current_input_data: dict,
    session_key_input_data: str,
    session_key_is_school_selection_submit: str,
) -> None:
    session_manager.set(
        **{
            session_key_input_data: current_input_data,
            session_key_is_school_selection_submit: False,
        }
    )


def run_prediction_with_guard(
    session_manager,
    page_state,
    current_input_data: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    session_key_has_predicted: str,
    session_key_predict_lock: str,
) -> bool:
    try:
        input_data_with_lists = current_input_data.copy()
        input_data_with_lists["_all_universities_target"] = all_universities_target
        input_data_with_lists["_all_majors_target"] = all_majors_target

        cases_df_fingerprint = compute_df_fingerprint(page_state.cases_df)
        all_universities_fingerprint = compute_list_fingerprint(all_universities_target)
        all_majors_fingerprint = compute_list_fingerprint(all_majors_target)

        prediction_result_model = run_prediction_pipeline(
            input_data_with_lists,
            "xgboost",
            cases_df_fingerprint,
            page_state.loaded_feature_names,
            all_universities_fingerprint,
            all_majors_fingerprint,
        )
        if prediction_result_model and prediction_result_model.unified_results is not None:
            session_manager.set(
                prediction_results=prediction_result_model,
                **{session_key_has_predicted: True, session_key_predict_lock: False},
                fresh_prediction_result=True,
            )
            return True
        from src.pages.prediction.results_handler import reset_prediction_results

        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_predict_lock: False})
        return False
    except Exception as e:
        error_message = f"预测过程中发生意外错误: {str(e)}"
        prediction_handler_logger.error(error_message, exc_info=True)
        st.error(f"预测过程中发生意外错误: {e}")
        from src.pages.prediction.results_handler import reset_prediction_results

        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_predict_lock: False})
        return False


def handle_form_submission(
    session_manager,
    page_state,
    input_data_from_form: dict,
    all_universities_target: list[str],
    all_majors_target: list[str],
    original_form_data: dict | None,
    session_key_form_data_changed: str,
    session_key_input_data: str,
    session_key_predict_lock: str,
    session_key_has_predicted: str,
    session_key_is_school_selection_submit: str,
    session_key_last_submission_logged: str,
) -> None:
    from src.pages.prediction.input_form_components.cross_faculty_guard import (
        cross_faculty_confirm_dialog,
        quick_cross_faculty_check,
    )
    from src.pages.prediction.page_components.submission_logger import (
        log_first_submission_if_needed,
    )
    from src.pages.prediction.results_handler import reset_prediction_results

    session_manager.set(**{session_key_form_data_changed: False})

    if not all(
        [
            input_data_from_form.get("background_university"),
            input_data_from_form.get("background_major"),
        ]
    ):
        reset_prediction_results(session_manager)
        session_manager.delete(session_key_input_data)
        return

    background_major = input_data_from_form.get("background_major")

    user_selected_categories = session_manager.get("selected_major_categories", []) or []
    user_selected_majors = session_manager.get("selected_target_majors", []) or []

    if background_major and (user_selected_categories or user_selected_majors):
        is_cross_faculty, bg_faculty, target_faculties = quick_cross_faculty_check(
            background_major,
            user_selected_categories,
            user_selected_majors,
            page_state.cases_df,
        )

        if is_cross_faculty:
            if session_manager.get("cross_faculty_cancelled", False):
                session_manager.set(
                    cross_faculty_cancelled=False,
                    cross_faculty_confirmed=False,
                    pending_prediction_data=None,
                    pending_cross_faculty_prediction=False,
                    prediction_submit_lock=False,
                    submitted=False,
                )
                st.info("已取消预测操作")
                return

            if not session_manager.get("cross_faculty_confirmed", False):
                session_manager.set(
                    pending_prediction_data={
                        "input_data": input_data_from_form,
                        "all_universities": all_universities_target,
                        "all_majors": all_majors_target,
                        "original_form": original_form_data,
                    }
                )
                cross_faculty_confirm_dialog(session_manager, bg_faculty, target_faculties)
                return

    session_manager.set(**{session_key_predict_lock: True})
    current_input_data = prepare_input_data(input_data_from_form)
    persist_input_state(
        session_manager,
        current_input_data,
        session_key_input_data,
        session_key_is_school_selection_submit,
    )
    log_first_submission_if_needed(
        session_manager,
        original_form_data,
        input_data_from_form,
        session_key_last_submission_logged,
    )
    run_prediction_with_guard(
        session_manager,
        page_state,
        current_input_data,
        all_universities_target,
        all_majors_target,
        session_key_has_predicted,
        session_key_predict_lock,
    )
