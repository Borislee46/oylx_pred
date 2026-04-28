import json
import time
from typing import Any

from src.agent.base_agent import BaseAgent

FORM_VALIDATION_SYSTEM_PROMPT = """\
你是留学申请表单数据校验专家。检查表单数据的一致性和合理性。

校验规则：
1. GPA量纲：4.0分制GPA应在0-4.0范围，5.0分制GPA应在0-5.0范围。检查GPA与量纲是否匹配
2. 语言成绩：雅思0-9，托福0-120。检查分数是否在合理范围
3. 背景-目标匹配：顶尖院校（港大/港中文/港科大/NUS/NTU/SMU）申请者GPA通常≥2.8/4.0或≥3.5/5.0
4. 字段完整性：院校、专业、GPA为必填核心字段
5. 经历一致性：某类经历计数>0但缺少描述文本时报warn

返回JSON：
{
  "issues": [
    {"field": "字段名", "severity": "error|warn|info", "message": "中文提示"}
  ],
  "overall_assessment": "整体数据质量评价(30字内)"
}
若无问题，issues为空数组。
"""


def _build_validation_prompt(form_data: dict[str, Any]) -> str:
    lines = ["## 表单数据"]

    gpa = form_data.get("gpa", 0)
    gpa_scale = form_data.get("gpa_scale", "4.0")
    language_type = form_data.get("language_type", "")
    language_score = form_data.get("language_score", 0)
    bg_university = form_data.get("background_university", "")
    bg_major = form_data.get("background_major", "")
    target_universities = form_data.get("selected_target_universities", []) or []
    target_majors = form_data.get("selected_target_majors", []) or []
    target_countries = form_data.get("selected_target_countries", []) or []

    lines.append(f"- 院校：{bg_university or '未填写'}")
    lines.append(f"- 专业：{bg_major or '未填写'}")
    lines.append(f"- GPA：{gpa}（{gpa_scale}分制）")
    lines.append(f"- 语言：{language_type} {language_score}" if language_type else "- 语言：未填写")
    lines.append(f"- 目标地区：{', '.join(target_countries) if target_countries else '未选择'}")
    lines.append(f"- 目标院校：{', '.join(target_universities[:5]) if target_universities else '未选择'}")
    lines.append(f"- 目标专业：{', '.join(target_majors[:5]) if target_majors else '未选择'}")

    experience = form_data.get("experience_details", {}) or {}
    fields = {"research": "科研", "internship": "实习", "paper": "论文", "award": "获奖"}
    for k, name in fields.items():
        count = form_data.get(f"{k}_count", 0)
        has_text = bool(experience.get(k))
        if count > 0:
            lines.append(f"- {name}：{count}段" + ("" if has_text else "（⚠ 缺少描述）"))

    return "\n".join(lines)


class FormValidationAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=8, agent_name="表单校验Agent")

<<<<<<< HEAD
    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Orchestrator-compatible entry point.

        Expects ``form_data`` in kwargs, or reads from StudentContext.
        """
        form_data: dict[str, Any] = kwargs.get("form_data", {})
        if not form_data and hasattr(context, "extracted_background"):
            form_data = getattr(context, "extracted_background", {})
        return self.validate(form_data)

=======
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc
    def validate(self, form_data: dict[str, Any]) -> dict[str, Any]:
        t_start = time.perf_counter()
        self.logger.info(
            f"[{self.agent_name}] VALIDATE START | "
            f"uni={form_data.get('background_university', '?')[:10]} | "
            f"gpa={form_data.get('gpa', '?')}"
        )

        prompt = _build_validation_prompt(form_data)
        full_prompt = f"{FORM_VALIDATION_SYSTEM_PROMPT}\n\n{prompt}"
        raw = self._call_api(
            full_prompt, cache_prefix="form_validation", max_tokens=200,
        )

        result = self._parse_response(raw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if result is None:
            self.logger.warning(
                f"[{self.agent_name}] VALIDATE FAILED | total={elapsed_ms:.0f}ms"
            )
            return {"issues": [], "overall_assessment": "校验暂不可用", "_error": "api_failed"}

        issues = result.get("issues", [])
        self.logger.info(
            f"[{self.agent_name}] VALIDATE OK | total={elapsed_ms:.0f}ms | "
            f"issues={len(issues)} | "
            f"assessment={result.get('overall_assessment', '')[:60]}"
        )
        return result

    @staticmethod
    def quick_check(form_data: dict[str, Any]) -> list[dict]:
        """Fast rule-based checks (no API call). Returns list of issues."""
        issues = []

        gpa = form_data.get("gpa", 0) or 0
        gpa_scale = form_data.get("gpa_scale", "4.0") or "4.0"

        if gpa > 0:
            if gpa_scale == "4.0" and gpa > 4.0:
                issues.append({
                    "field": "gpa", "severity": "error",
                    "message": f"GPA {gpa} 超出4.0分制范围，请检查量纲或重新输入"
                })
            elif gpa_scale == "5.0" and gpa > 5.0:
                issues.append({
                    "field": "gpa", "severity": "error",
                    "message": f"GPA {gpa} 超出5.0分制范围"
                })

        lang_score = form_data.get("language_score", 0) or 0
        lang_type = form_data.get("language_type", "") or ""
        if lang_score > 0 and lang_type:
            # Skip range check for normalized scores (0-1 range, already processed)
            is_normalized = isinstance(lang_score, (int, float)) and 0 < lang_score <= 1
            if not is_normalized:
                if lang_type == "雅思" and (lang_score < 3 or lang_score > 9):
                    issues.append({
                        "field": "language_score", "severity": "error",
                        "message": f"雅思成绩 {lang_score} 不在合理范围(3-9)"
                    })
                elif lang_type == "托福" and (lang_score < 30 or lang_score > 120):
                    issues.append({
                        "field": "language_score", "severity": "error",
                        "message": f"托福成绩 {lang_score} 不在合理范围(30-120)"
                    })

        bg_uni = (form_data.get("background_university") or "").strip()
        bg_major = (form_data.get("background_major") or "").strip()
        if not bg_uni:
            issues.append({
                "field": "background_university", "severity": "error",
                "message": "院校未填写，此为必填项"
            })
        if not bg_major:
            issues.append({
                "field": "background_major", "severity": "error",
                "message": "专业未填写，此为必填项"
            })

        experience = form_data.get("experience_details", {}) or {}
        for k, name in [("research", "科研"), ("internship", "实习"), ("paper", "论文")]:
            count = form_data.get(f"{k}_count", 0) or 0
            has_text = bool(experience.get(k))
            if count > 0 and not has_text:
                issues.append({
                    "field": f"{k}_details",
                    "severity": "warn",
                    "message": f"{name}有{count}段但缺少描述，建议补充以提升评估准确度"
                })

        return issues

    def _parse_response(self, raw: str | None) -> dict[str, Any] | None:
        return self._parse_json_response(
            raw,
            schema_hint='{"issues": [{"field":"","severity":"error|warn|info","message":""}], "overall_assessment": "..."}',
            cache_prefix="form_validation_json_repair",
            max_tokens=200,
        )
