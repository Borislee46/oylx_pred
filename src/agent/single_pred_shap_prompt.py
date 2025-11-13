from typing import Dict, List, Optional

import numpy as np


def build_shap_explanation_prompt(
    target_university: str,
    target_major: str,
    background_major: str,
    feature_names: List[str],
    shap_values: np.ndarray,
    raw_feature_values: np.ndarray,
    max_features: int = 8,
    feature_display_map: Optional[Dict[str, str]] = None,
    min_shap_abs: float = 0.005,
) -> str:
    if feature_display_map is None:
        feature_display_map = {}

    shap_values_rounded = np.round(shap_values, decimals=4)

    def normalize_raw_value(raw):
        if isinstance(raw, (int, float)):
            if abs(raw) < 1000:
                return f"{raw:.2f}"
            else:
                return f"{raw:.0f}"
        elif isinstance(raw, np.integer):
            return str(int(raw))
        elif isinstance(raw, np.floating):
            if abs(float(raw)) < 1000:
                return f"{float(raw):.2f}"
            else:
                return f"{float(raw):.0f}"
        else:
            raw_str = str(raw).strip()
            if raw_str.lower() in ("nan", "none", ""):
                return ""
            return raw_str

    feature_list = [
        (name, shap, normalize_raw_value(raw))
        for name, shap, raw in zip(feature_names, shap_values_rounded, raw_feature_values)
        if abs(shap) >= min_shap_abs
    ]

    feature_list.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_list[:max_features]

    feature_lines = []
    for name, shap, raw_str in top_features:
        display_name = feature_display_map.get(name, name.replace("_", " ").title())
        strength = "强" if abs(shap) > 0.03 else "一般"
        direction = "加分" if shap > 0 else "扣分"
        feature_lines.append(f"{display_name}: {direction}{strength}，值={raw_str}")

    prompt = f"""分析申请情况：

目标：{target_university} {target_major}
本科专业：{background_major}

主要影响因素：
{chr(10).join(feature_lines)}

要求：
1. 加分项（>0.01）：说明优势及原因
2. 扣分项（<-0.01）：指出不足及改进建议
3. 建议：针对1-2个主要扣分项提供具体改进措施

注意：
- 避免技术术语
- 语言正式简洁
- 提供建设性建议
- 跨专业申请需结合特征判断
- 用最少的字数解释，尽量不要超过200字
"""

    return prompt
