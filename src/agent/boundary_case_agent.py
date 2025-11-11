import json
from typing import Any

import pandas as pd
import requests

from src.agent.prompts import build_boundary_evaluation_prompt
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

boundary_case_logger = setup_logger("page3", "prediction")


class BoundaryCaseAgent:
    def __init__(self, cases_df: pd.DataFrame, config: dict[str, Any] | None = None):
        self.cases_df = cases_df

        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")

        if not self.api_url or not self.api_key:
            boundary_case_logger.error(
                "[边界CaseAgent] API URL 或 API Key 未在配置中找到。"
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def evaluate_boundary_cases(
        self, background_major: str, boundary_cases: list[dict[str, Any]], mode: str
    ) -> dict[str, Any]:
        if not boundary_cases:
            boundary_case_logger.warning(
                "[边界CaseAgent] 边界案例列表为空，跳过评估"
            )
            return {"decisions": [], "needs_adjustment": False, "reason": "无待评估案例"}

        boundary_case_logger.info(
            f"[边界CaseAgent] 开始评估 - 模式: {mode}, 背景专业: {background_major}, "
            f"案例数量: {len(boundary_cases)}"
        )

        prompt = build_boundary_evaluation_prompt(background_major, boundary_cases, mode)

        data = {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
        }

        try:
            boundary_case_logger.debug(
                f"[边界CaseAgent] 发送请求到 {self.api_url}, 模型: {self.model}, "
                f"prompt长度: {len(prompt)}"
            )
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=5
            )
            response.raise_for_status()

            response_json = response.json()
            boundary_case_logger.debug(
                f"[边界CaseAgent] 收到响应，状态码: {response.status_code}"
            )

            if response_json.get("choices"):
                content = response_json["choices"][0].get("message", {}).get("content", "")
                content = content.strip()

                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                result = json.loads(content)

                decisions = result.get("decisions", [])
                needs_adjustment = result.get("needs_adjustment", True)
                reason = result.get("reason", "")

                if len(decisions) != len(boundary_cases):
                    boundary_case_logger.warning(
                        f"[边界CaseAgent] decisions 长度 ({len(decisions)}) 与 boundary_cases 长度 "
                        f"({len(boundary_cases)}) 不匹配"
                    )
                    decisions = [False] * len(boundary_cases)
                    needs_adjustment = False

                result = {
                    "decisions": decisions,
                    "needs_adjustment": needs_adjustment,
                    "reason": reason,
                }
                boundary_case_logger.info(
                    f"[边界CaseAgent] 评估完成 - 需要调整: {needs_adjustment}, "
                    f"决策数: {len(decisions)}, 通过数: {sum(decisions) if decisions else 0}"
                )
                return result
            else:
                error_info = response_json.get("error", {})
                error_message = error_info.get("message", "未知错误")
                boundary_case_logger.error(
                    f"[边界CaseAgent] API返回错误: {error_message}"
                )
                return {
                    "decisions": [False] * len(boundary_cases),
                    "needs_adjustment": False,
                    "reason": "",
                }

        except requests.exceptions.Timeout:
            boundary_case_logger.warning(
                f"[边界CaseAgent] 请求超时（5秒），使用fallback输出（保持原始结果）"
            )
            return {
                "decisions": [False] * len(boundary_cases),
                "needs_adjustment": False,
                "reason": "",
            }
        except json.JSONDecodeError as e:
            boundary_case_logger.error(
                f"[边界CaseAgent] JSON解析失败 - 错误: {e}, "
                f"响应内容长度: {len(content) if 'content' in locals() else 0}"
            )
            if "content" in locals():
                boundary_case_logger.debug(
                    f"[边界CaseAgent] 响应内容预览: {content[:200]}"
                )
            return {
                "decisions": [False] * len(boundary_cases),
                "needs_adjustment": False,
                "reason": "",
            }
        except requests.exceptions.RequestException as e:
            boundary_case_logger.error(f"[边界CaseAgent] 网络请求失败 - 错误: {e}")
            return {
                "decisions": [False] * len(boundary_cases),
                "needs_adjustment": False,
                "reason": "",
            }
        except Exception as e:
            boundary_case_logger.error(
                f"[边界CaseAgent] 未知错误 - 错误类型: {type(e).__name__}, 错误信息: {e}",
                exc_info=True,
            )
            return {
                "decisions": [False] * len(boundary_cases),
                "needs_adjustment": False,
                "reason": "",
            }

