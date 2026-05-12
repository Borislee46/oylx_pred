"""
trace_display 的 CSS 与文案资产。

拆分理由：CSS 和 STEP_INTENT 文案体量大且为纯静态常量，
独立成文件让 trace_display.py 主体保持在可读范围内。
CSS 本体在 assets/hk_style/51_trace.css，运行时加载。
"""

from pathlib import Path

_TRACE_CSS_PATH = Path(__file__).resolve().parents[4] / "assets" / "hk_style" / "51_trace.css"

_STYLE_HEAD = "<style>"
_STYLE_TAIL = "</style>"

STEP_INTENT: dict[str, str] = {
    "GPA Penalty": (
        "GPA 低于历史均值时触发，z-score² 缩放。"
        "二次曲线使低分段每 0.1 分的边际影响递增，高分段递减。"
    ),
    "Language Penalty": (
        "阶梯式惩罚。pass_line（均值 − 0.5σ）以上不触发；"
        "以下分 85% / 70% / 40% 三档；低于 minimum → ×95%。"
    ),
    "Cross Major Penalty": (
        "相似度 < 0.89 触发，最大 ×0.5。"
        "比跨学部惩罚（×0.3）轻——同学部内转专业可行性更高。历史录取案例自动豁免。"
    ),
    "Faculty Out of Scope Penalty": (
        "学部跨度过大触发，固定 ×0.3。"
        "基于硬编码学部规则——跨学部录取案例 < 1%，数据量不足以可靠建模。"
    ),
    "Professional Major Penalty": (
        "职业导向学位（MBA 等）+ 无实习经历触发。"
        "用户主动选 ×0.5，系统推荐 ×0.7。仅检查实习——商科申请最直接的竞争力信号。"
    ),
    "NLP Text Boost": (
        "TF-IDF 评估背提文本质量，上限 +15%。"
        "经 sim_gate 过滤、平滑、cap 三重约束。在所有惩罚之后应用。"
    ),
    "XGBoost 原始": (
        "经 CalibratedClassifierCV(sigmoid, prefit) 校准的 XGBoost 输出，可直接作为概率解释。"
    ),
    "最终概率": ("5 层调整链聚合结果。总惩罚 ≤ 70%，总提升 ≤ 30%，floor 0.5%。"),
}


def _load_css() -> str:
    import logging

    _log = logging.getLogger("trace_assets")
    try:
        return _STYLE_HEAD + _TRACE_CSS_PATH.read_text(encoding="utf-8") + _STYLE_TAIL
    except (OSError, UnicodeDecodeError) as e:
        _log.warning("Failed to load trace CSS from %s: %s", _TRACE_CSS_PATH, e)
        return ""


CSS_TEMPLATE = _load_css()
