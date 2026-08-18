"""
HK 预测页面的输入表单编排层。

职责：组装表单 UI 组件，处理提交/校验/归一化流程。
不包含具体 widget 渲染逻辑（见 input_form_components/）。

表单提交流程：
  1. FormStateManager 初始化 session_state
  2. 渲染双层布局（左：背景+GPA+目标，右：语言+经历）
  3. 用户点击提交 → FormValidator 校验
  4. 校验通过 → normalize_form_data_for_prediction 归一化
  5. 返回 (is_new_submission, input_data, all_unis, all_majors, original_form)

返回值语义：
  - is_new_submission=True：新提交，调用方应触发预测
  - is_new_submission=False：返回当前表单快照（用于 lead-in auto-submit）
"""

from contextlib import nullcontext

import streamlit as st

from src.pages.prediction.flow.form_normalizer import (
    calculate_processed_gpa,  # GPA 原始值 → 归一化值（考虑院校、标化考试）
    calculate_processed_language_score,  # 语言原始值 → 归一化值（考虑海本豁免）
    get_background_university_for_model,  # 院校名 → 模型可识别的院校名（模糊匹配）
    normalize_form_data_for_prediction,  # 表单数据 → 模型输入格式（完整归一化）
)
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_WIDGET_KEYS
from src.pages.prediction.input_form_components import (
    FormStateManager,  # session_state 初始化 + 表单状态管理
    FormUIComponents,  # 表单 UI 组件渲染器（背景/GPA/语言/目标/经历/提交按钮）
    FormValidator,  # 表单数据校验规则
    GPAConverter,  # GPA 分制转换（4.0/5.0/100 → 归一化）
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger
from src.utils.schools.data import load_school_base_data  # 院校基础数据（排名、分类等）
from src.utils.schools.level_service import (
    get_school_level_service,  # 院校等级服务（985/211/双非/海本）
)
from src.utils.session_manager import SessionManager

form_logger = setup_logger("page3", "prediction")


def create_input_form(
    session_manager: SessionManager,
    cases_df,
    *,
    parent_container=None,
    wrap_container: bool = True,
    border: bool = True,
):
    """渲染完整的输入表单并处理提交逻辑。

    Args:
        session_manager: SessionManager 实例
        cases_df: 历史案例 DataFrame（用于院校/专业下拉选项）
        parent_container: 外部容器（如 st.expander），None 则直接渲染
        wrap_container: 是否在表单外包裹 st.container(border=True)
        border: 容器是否带边框

    Returns:
        tuple: (is_new_submission, input_data, all_unis, all_majors, original_form)
        - is_new_submission=True → 用户刚提交，input_data 是归一化后的模型输入
        - is_new_submission=False → 返回当前表单快照（用于 lead-in auto-submit）
    """
    # 0. 初始化表单 session_state（幂等：首次加载初始化默认值，后续重跑跳过）
    FormStateManager.initialize_session_state(session_manager)

    # 2. 提交互斥锁：预测进行中时禁用表单，防止重复提交
    disabled_status = session_manager.get("prediction_submit_lock", False)

    # 3. 延迟加载院校基础数据（首次加载后缓存到 session_state）
    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    # 4. GPA 转换器（依赖 school_base_df，因此需要在加载后实例化）
    if session_manager.get("gpa_converter") is None:
        session_manager.set(gpa_converter=GPAConverter(session_manager.get("school_base_df")))
    gpa_converter = session_manager.get("gpa_converter")

    ui_components = FormUIComponents(session_manager)

    # 5. 容器层级：parent_container（expander/fragment）→ input_ctx（带边框容器）→ 双列
    outer_ctx = parent_container if parent_container is not None else nullcontext()

    with outer_ctx:
        input_ctx = st.container(border=border) if wrap_container else nullcontext()
        with input_ctx:
            # 双列布局：左列（背景+GPA+目标），右列（语言+经历+提交）
            col1, col2 = st.columns([1, 1], gap="small")

            with col1:
                (
                    background_university,
                    selected_background_major_original,
                    background_major,
                    background_major_2_original,
                    background_major_2,
                    is_dual_degree,
                    dual_alpha,
                ) = ui_components.render_background_section(cases_df)

                gpa_col, test_col = st.columns([2, 1], gap="medium")
                with gpa_col:
                    ui_components.render_gpa_section()
                with test_col:
                    exam_type, exam_score = ui_components.render_standardized_test_section()

                (
                    final_target_universities,
                    final_target_majors,
                    all_universities_target,
                    all_majors_target,
                ) = ui_components.render_target_section(cases_df)

            with col2:
                language_type, raw_language_score_value = ui_components.render_language_section()
                (
                    research_count,
                    award_count,
                    internship_count,
                    paper_count,
                    experience_details,
                ) = ui_components.render_experience_section()

            submit_button = ui_components.render_submit_button(disabled_status)

    # 6. 提交处理：校验 → 归一化 → 返回
    # Streamlit >=1.61 服务端强制 disabled：on_submit_click 回调在本帧把按钮置为
    # disabled（is_currently_submitting）后，浏览器发来的点击值会被丢弃，
    # st.button 返回 False。因此以 submitted 标记（回调已置位）作为补充触发条件。
    if submit_button or session_manager.get("submitted", False):
        widget_lang = st.session_state.get(DEFAULT_WIDGET_KEYS.language_score)
        if widget_lang is not None:
            try:
                parsed_lang = float(widget_lang)
            except (TypeError, ValueError):
                parsed_lang = None
            if parsed_lang is not None and parsed_lang > 0:
                raw_language_score_value = parsed_lang
                session_manager.set(
                    language_score_input=parsed_lang,
                    **{DEFAULT_FORM_KEYS.language_score_user_provided: True},
                )

        form_data = {
            "target_majors": final_target_majors,
            "target_universities": final_target_universities,
            "background_university": background_university,
            "background_major_original": selected_background_major_original,
            "background_major": background_major,
            "background_major_2_original": background_major_2_original,
            "background_major_2": background_major_2,
            "is_dual_degree": is_dual_degree,
            "dual_alpha": dual_alpha,
            "degree_type": st.session_state.get(DEFAULT_WIDGET_KEYS.dual_degree_type, "辅修"),
            "gpa_raw": session_manager.get("gpa_raw_input"),
            "gpa_scale": session_manager.get("gpa_scale"),
            "exam_type": exam_type,
            "exam_score": exam_score,
            "language_type": language_type,
            "language_score_raw": raw_language_score_value,
            "language_score_user_provided": session_manager.get(
                DEFAULT_FORM_KEYS.language_score_user_provided, False
            ),
            "language_score_input_error": session_manager.get("language_score_input_error", False),
            "research_count": research_count,
            "award_count": award_count,
            "internship_count": internship_count,
            "paper_count": paper_count,
            "experience_details": experience_details,
        }

        validation_errors = FormValidator.validate_form_data(form_data, gpa_converter)

        if validation_errors:
            # 校验失败：逐个展示 toast 后 rerun 重置提交状态
            error_messages = [str(err) for err in validation_errors]
            form_logger.warning(f"表单验证失败 - 错误信息: {error_messages}")
            for err in validation_errors:
                st.toast(str(err))
            session_manager.set(
                submitted=False, form_data_changed=False, prediction_submit_lock=False
            )
            reset_prediction_results(session_manager)
            st.rerun()
        else:
            # 校验通过：加锁 → 归一化 → 返回
            session_manager.set(prediction_submit_lock=True)
            success, processed_input_data, all_unis, all_majors, original_form_data = (
                _process_successful_submission(
                    session_manager,
                    form_data,
                    cases_df,
                    all_universities_target,
                    all_majors_target,
                    gpa_converter,
                )
            )
            return (
                True,
                processed_input_data,
                all_unis,
                all_majors,
                original_form_data,
            )

    # 7. 非提交路径：返回当前表单快照（用于 lead-in auto-submit 读取当前值）
    return _get_current_form_state(
        session_manager,
        background_university,
        background_major,
        selected_background_major_original,
        final_target_universities,
        final_target_majors,
        language_type,
        raw_language_score_value,
        research_count,
        award_count,
        internship_count,
        paper_count,
        experience_details,
        cases_df,
        all_universities_target,
        all_majors_target,
        gpa_converter,
        exam_type,
        exam_score,
        background_major_2=background_major_2,
        background_major_2_original=background_major_2_original,
        is_dual_degree=is_dual_degree,
        dual_alpha=dual_alpha,
        degree_type=st.session_state.get(DEFAULT_WIDGET_KEYS.dual_degree_type, "辅修"),
    )


def _process_successful_submission(
    session_manager,
    form_data,
    cases_df,
    all_universities_target,
    all_majors_target,
    gpa_converter,
):
    """校验通过后的数据处理：归一化表单数据为模型输入格式。

    关键步骤：
    1. 重新加载 page_state（获取 background_universities 白名单）
    2. normalize_form_data_for_prediction 执行完整归一化：
       - GPA 分制转换（4.0/5.0/100 → 归一化值）
       - 语言成绩标准化
       - 院校名模糊匹配 → 模型可识别的标准名
       - 专业名映射
       - 四段经历文本向量化准备
    """
    from src.pages.prediction.page_data_loader import machine_learning_model

    page_state = machine_learning_model.resource_loader()

    input_data = normalize_form_data_for_prediction(
        form_data,
        cases_df,
        gpa_converter,
        background_university_set=page_state.background_universities,
    )

    session_manager.set(submitted=True, form_data_changed=False)
    return True, input_data, all_universities_target, all_majors_target, form_data


def _get_current_form_state(
    session_manager,
    background_university,
    background_major,
    background_major_original=None,
    final_target_universities=None,
    final_target_majors=None,
    language_type=None,
    raw_language_score_value=None,
    research_count=None,
    award_count=None,
    internship_count=None,
    paper_count=None,
    experience_details=None,
    cases_df=None,
    all_universities_target=None,
    all_majors_target=None,
    gpa_converter=None,
    exam_type=None,
    exam_score=None,
    background_major_2=None,
    background_major_2_original=None,
    is_dual_degree=False,
    dual_alpha=0.85,
    degree_type="辅修",
):
    """构建当前表单状态的快照（非提交路径）。

    用于 lead-in auto-submit：当 AI 提取完成并自动展开表单后，
    _form_fragment 调用此函数获取当前 widget 值构建 input_data，
    而不需要用户手动点击提交按钮。

    与 _process_successful_submission 的区别：
    - 此函数做"尽力归一化"（calculate_processed_gpa），不完全校验
    - 不设置 submitted=True，不触发锁
    - is_new_submission=False，但返回 form_data 快照供 LeadIn 自动提交日志使用
    """
    school_service = get_school_level_service()
    is_overseas = (
        school_service.is_overseas_school(background_university) if background_university else False
    )

    _, current_normalized_score = calculate_processed_language_score(
        raw_language_score_value,
        language_type,
        background_university,
        is_overseas,
        user_provided=bool(
            session_manager.get(DEFAULT_FORM_KEYS.language_score_user_provided, False)
        ),
    )

    current_normalized_gpa = calculate_processed_gpa(
        session_manager.get("gpa_raw_input"),
        session_manager.get("gpa_scale"),
        background_university,
        gpa_converter,
        exam_type,
        exam_score,
    )

    background_uni_for_model = get_background_university_for_model(background_university)

    input_data = {
        "background_university": background_uni_for_model,
        "background_major": background_major,
        "background_major_original": background_major_original or "",
        "background_major_2": background_major_2,
        "background_major_2_original": background_major_2_original,
        "is_dual_degree": is_dual_degree,
        "dual_alpha": dual_alpha,
        "degree_type": degree_type,
        "target_universities": final_target_universities,
        "target_majors": final_target_majors,
        "gpa": current_normalized_gpa,
        "gpa_raw": session_manager.get("gpa_raw_input"),
        "gpa_scale": session_manager.get("gpa_scale"),
        "language_score": current_normalized_score,
        "language_score_raw": raw_language_score_value,
        "language_type": language_type,
        "exam_type": exam_type,
        "exam_score": exam_score,
        "research_count": research_count,
        "award_count": award_count,
        "internship_count": internship_count,
        "paper_count": paper_count,
        "experience_details": experience_details,
    }

    form_data = {
        "target_majors": final_target_majors,
        "target_universities": final_target_universities,
        "background_university": background_university,
        "background_major_original": background_major_original,
        "background_major": background_major,
        "background_major_2_original": background_major_2_original,
        "background_major_2": background_major_2,
        "is_dual_degree": is_dual_degree,
        "dual_alpha": dual_alpha,
        "gpa_raw": session_manager.get("gpa_raw_input"),
        "gpa_scale": session_manager.get("gpa_scale"),
        "exam_type": exam_type,
        "exam_score": exam_score,
        "language_type": language_type,
        "language_score_raw": raw_language_score_value,
        "language_score_user_provided": session_manager.get(
            DEFAULT_FORM_KEYS.language_score_user_provided, False
        ),
        "language_score_input_error": session_manager.get("language_score_input_error", False),
        "research_count": research_count,
        "award_count": award_count,
        "internship_count": internship_count,
        "paper_count": paper_count,
        "experience_details": experience_details,
    }

    return (
        False,  # is_new_submission=False
        input_data,  # 尽力归一化的快照数据
        all_universities_target,
        all_majors_target,
        form_data,
    )
