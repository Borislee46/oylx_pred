"""
Session State Key 注册表 + 表单提交上下文 dataclass。

设计意图：
  Streamlit 的 session_state 是一个扁平的全局字典。随着功能增长，
  各模块各自往里面写 key 会导致命名冲突、拼写错误、难以追踪"谁在什么时候写了什么"。

解决方案：
  - 所有 session_state key 集中定义为不可变 dataclass 实例（DEFAULT_*_KEYS）
  - 单点引用：所有模块 import 同一个 key 常量，而非各自写字符串字面量
  - FormSubmissionContext 作为 handler 的输入契约，显式化数据依赖

Key 分层：
  SessionKeys    — 会话生命周期（has_predicted、predict_lock）
  UIStateKeys    — UI 状态机（hk_ui_phase、cross_faculty flags）
  FormStateKeys  — 表单输入数据（GPA、语言、目标院校、lead-in）
  FormWidgetKeys — Streamlit widget key（与 widget 实例绑定）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager


# ── SessionKeys：会话级生命周期 key ─────────────────────────
# 这些 key 追踪"当前会话是否完成过预测"、"提交锁"等跨请求状态。
# 与具体的业务数据（GPA、院校名等）无关。
@dataclass
class SessionKeys:
    form_data_changed: str = "form_data_changed"       # 表单是否有未保存的修改
    input_data: str = "input_data"                     # 当前表单输入数据（dict）
    predict_lock: str = "prediction_submit_lock"       # 防止重复提交的互斥锁
    has_predicted: str = "has_predicted"               # 当前会话是否已执行过预测
    is_school_selection_submit: str = "is_school_selection_submit"  # 提交类型区分
    last_submission_logged: str = "last_submission_logged"  # 上次提交的日志标记（防重复日志）


# ── UIStateKeys：UI 状态机 key ──────────────────────────────
# 这些 key 由 hk.py、handler.py、cross_faculty_guard、results_handler 共享，
# 共同维护 HK 页面的 phase 状态机。
# ────────────────────────────────────────────────────────────
# 状态流转：
#   idle → running → done / awaiting_confirm / error
#
# 跨学部确认子状态：
#   awaiting_confirm → (用户确认) pending_cross_faculty_prediction=True
#                    → (用户取消) pending_cross_faculty_prediction=False
# ────────────────────────────────────────────────────────────
@dataclass
class UIStateKeys:
    """页面 UI 生命周期 key — hk.py / handler.py / cross_faculty_guard / results_handler 共享"""

    hk_ui_phase: str = "hk_ui_phase"                       # 状态机当前阶段 [idle|running|awaiting_confirm|done|error]
    hk_run_id: str = "hk_run_id"                           # 当前预测运行的唯一 ID（用于日志关联）
    hk_last_error: str = "hk_last_error"                   # 最近一次错误的描述文本
    pending_cross_faculty_prediction: str = "pending_cross_faculty_prediction"  # 跨学部确认后等待重试
    pending_prediction_data: str = "pending_prediction_data"  # 重试时保存的原始提交数据
    cross_faculty_confirmed: str = "cross_faculty_confirmed"  # 用户已确认跨学部申请
    cross_faculty_cancelled: str = "cross_faculty_cancelled"  # 用户已取消跨学部申请
    form_expanded: str = "form_expanded"                   # 表单 expander 是否展开
    processing_lock: str = "processing_lock"               # 预测进行中的互斥锁
    lock_start_time: str = "lock_start_time"               # 锁获取时间戳（用于超时检测）
    app_initialized: str = "app_initialized"               # 首次加载初始化完成标记（仅执行一次）
    fresh_prediction_result: str = "fresh_prediction_result"  # 新预测结果就绪标记
    student_background_chart_visible: str = "student_background_chart_visible"  # 学生背景雷达图是否可见
    prediction_results: str = "prediction_results"         # 当前预测结果缓存
    last_saved_results_hash: str = "last_saved_results_hash"  # 上次持久化结果的哈希（变更检测）
    previous_prediction_results: str = "previous_prediction_results"  # 上一次预测结果（用于对比）
    previous_input_data: str = "previous_input_data"       # 上一次输入数据（用于对比）


# ── FormStateKeys：表单数据 key ─────────────────────────────
# 每个表单字段在 session_state 中对应一个 key。分为：
#   - 数据 key（DEFAULT_FORM_KEYS）：字段值本身
#   - widget key（DEFAULT_WIDGET_KEYS）：Streamlit widget 实例的 key
# 两套 key 分离是为了让数据层和 UI 层独立演化。
# ────────────────────────────────────────────────────────────
@dataclass
class FormStateKeys:
    """表单输入状态 key — input_form.py / form_state.py / form_bridge.py / 各 UI 组件共享"""

    # GPA 子组
    gpa_raw_input: str = "gpa_raw_input"                   # 原始 GPA 输入值（字符串，未转换）
    gpa_scale: str = "gpa_scale"                           # GPA 分制（4.0 / 5.0 / 100）
    gpa_conversion_cache: str = "gpa_conversion_cache"     # GPA 转换结果缓存（避免重复计算）
    gpa_converter: str = "gpa_converter"                   # GPA 转换器实例（缓存）
    last_gpa_warning_key: str = "last_gpa_warning_key"     # 上次 GPA 警告内容（防重复弹窗）
    # 语言子组
    language_type: str = "language_type"                   # 语言考试类型（IELTS/TOEFL/CET-6/None）
    language_score_input: str = "language_score_input"     # 语言成绩原始输入
    language_score_input_error: str = "language_score_input_error"  # 语言成绩校验错误信息
    lang_conversion_cache: str = "lang_conversion_cache"   # 语言成绩转换缓存
    last_lang_warning_key: str = "last_lang_warning_key"   # 上次语言警告内容
    last_ielts_step_warning_key: str = "last_ielts_step_warning_key"  # 上次 IELTS 步进警告
    # 标准化考试
    standardized_test_type: str = "standardized_test_type"  # 标化考试类型（GRE/GMAT/None）
    current_exam_score: str = "current_exam_score"          # 标化考试成绩
    # 目标院校/专业
    selected_target_countries: str = "selected_target_countries"      # 选择的目标国家/地区
    selected_target_universities: str = "selected_target_universities"  # 选择的目标院校
    selected_target_majors: str = "selected_target_majors"            # 选择的目标专业
    selected_major_categories: str = "selected_major_categories"      # 选择的目标专业大类
    target_options_cache: str = "target_options_cache"                # 目标选项缓存（院校+专业列表）
    # 背景院校/专业
    school_base_df: str = "school_base_df"                 # 院校基础数据 DataFrame（缓存）
    background_university: str = "background_university"   # 用户本科院校
    background_universities_cache: str = "background_universities_cache"  # 院校搜索缓存
    background_majors_cache: str = "background_majors_cache"          # 专业搜索缓存
    # LeadIn / Agent 桥接
    lead_in_form_summary: str = "lead_in_form_summary"     # LeadInAgent 提取的结构化摘要
    lead_in_form_filled: str = "lead_in_form_filled"       # AI 提取完成标记
    user_history_data: str = "user_history_data"           # 用户历史使用数据
    user_nickname: str = "user_nickname"                   # 用户显示昵称
    user_message: str = "user_message"                     # 系统消息（提示/警告/通知）
    # 提交 / 表单生命周期
    submitted: str = "submitted"                           # 表单已提交标记
    current_user_id: str = "current_user_id"               # 当前用户 ID
    last_auto_save_ts: str = "last_auto_save_ts"           # 上次自动保存时间戳
    last_saved_form_snapshot_hash: str = "last_saved_form_snapshot_hash"  # 上次保存的表单快照哈希
    _input_form_pending_submission: str = "_input_form_pending_submission"  # 表单待提交数据（内部用）
    # 经历计数初始值（用于 reset 时恢复）
    research_count_initial: str = "research_count_initial"
    award_count_initial: str = "award_count_initial"
    internship_count_initial: str = "internship_count_initial"
    paper_count_initial: str = "paper_count_initial"
    # 经历详情初始值
    research_details_initial: str = "research_details_initial"
    award_details_initial: str = "award_details_initial"
    internship_details_initial: str = "internship_details_initial"
    paper_details_initial: str = "paper_details_initial"
    # 背景初始值
    background_university_initial: str = "background_university_initial"
    background_major_original_initial: str = "background_major_original_initial"


# ── FormWidgetKeys：Streamlit Widget Key 注册表 ────────────
# 这些 key 直接赋给 Streamlit widget 的 key 参数。
# 与 FormStateKeys（数据 key）分离的原因：
#   - Widget key 必须在页面脚本中一致（否则 Streamlit 会重建 widget 丢失状态）
#   - 数据 key 可以独立于 widget 生命周期存在
#   - 同一个数据可能由不同 widget 编辑（如 lead-in 和手动表单共用一个数据 key）
# ────────────────────────────────────────────────────────────
@dataclass
class FormWidgetKeys:
    """Streamlit widget key 注册表 — form_bridge.py / form_state.py / 各 UI 组件共享"""

    background_university: str = "background_university_selectbox"
    background_major: str = "background_major_selectbox"
    gpa_scale: str = "gpa_scale_widget_key"
    gpa_raw_input: str = "gpa_raw_input_widget"
    language_type: str = "language_type_widget_key"
    language_score: str = "language_score_input_widget"
    target_countries: str = "target_countries_multiselect"
    target_universities: str = "target_universities_multiselect"
    target_majors: str = "target_majors_multiselect"
    standardized_test_type: str = "standardized_test_type_widget"
    research_count: str = "research_count_input"
    award_count: str = "award_count_input"
    internship_count: str = "internship_count_input"
    paper_count: str = "paper_count_input"
    research_details: str = "research_details_input"
    award_details: str = "award_details_input"
    internship_details: str = "internship_details_input"
    paper_details: str = "paper_details_input"


# ── FormSubmissionContext：Handler 输入契约 ─────────────────
# handler.py 的 handle_form_submission 需要 8+ 个参数。
# 与其传散列参（容易漏、顺序错），打包为一个 dataclass。
# 新增依赖（如 background_faculty）只需加字段 + 默认值，不破坏现有调用方。
# ────────────────────────────────────────────────────────────
@dataclass
class FormSubmissionContext:
    """表单提交的完整上下文，作为 handle_form_submission 的单一输入参数。

    Fields:
        session_manager: SessionManager 实例（读写 session_state）
        page_state:     模型资源加载器返回的 page_state（cases_df、相似度矩阵等）
        input_data_from_form: 表单输出的结构化输入数据（GPA、语言、院校、专业）
        all_universities_target: 用户选择的所有目标院校列表
        all_majors_target:       用户选择的所有目标专业列表
        original_form_data:      表单原始数据（用于对比和回填）
        session_keys:            SessionKeys 实例（key 常量引用）
        background_faculty:      用户背景专业的学部（可选，pipeline 内部也可计算）
        admitted_combinations:   已知录取的组合集合（可选，用于历史数据标记）
    """
    session_manager: "SessionManager"
    page_state: "machine_learning_model"
    input_data_from_form: dict
    all_universities_target: list[str]
    all_majors_target: list[str]
    original_form_data: dict | None
    session_keys: SessionKeys
    background_faculty: str | None = None
    admitted_combinations: set[tuple[str, str]] | None = None

    @classmethod
    def create(
        cls,
        session_manager: "SessionManager",
        page_state: "machine_learning_model",
        input_data_from_form: dict,
        all_universities_target: list[str],
        all_majors_target: list[str],
        original_form_data: dict | None = None,
        session_keys: SessionKeys | None = None,
        background_faculty: str | None = None,
        admitted_combinations: set[tuple[str, str]] | None = None,
    ) -> "FormSubmissionContext":
        """工厂方法：提供默认值，避免调用方需要手动实例化 SessionKeys。"""
        return cls(
            session_manager=session_manager,
            page_state=page_state,
            input_data_from_form=input_data_from_form,
            all_universities_target=all_universities_target,
            all_majors_target=all_majors_target,
            original_form_data=original_form_data,
            session_keys=session_keys or SessionKeys(),
            background_faculty=background_faculty,
            admitted_combinations=admitted_combinations,
        )


# ── 全局单例 ───────────────────────────────────────────────
# 每个 key 注册表只实例化一次。所有模块 import 同一个实例。
# 为什么用 dataclass 而不是 Enum/Constants 模块？
#   - 支持 IDE 自动补全 + 跳转定义
#   - 添加新 key 只需加一行字段
#   - 不可变（frozen 未启用，但约定上不修改）
DEFAULT_SESSION_KEYS = SessionKeys()
DEFAULT_UI_KEYS = UIStateKeys()
DEFAULT_FORM_KEYS = FormStateKeys()
DEFAULT_WIDGET_KEYS = FormWidgetKeys()
