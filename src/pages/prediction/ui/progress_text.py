_SALES_PROGRESS_REPLACES: tuple[tuple[str, str], ...] = (
    ("校准", "校验方案可靠度"),
    ("推理", "智能匹配院校"),
    ("检索", "比对相似学生"),
    ("特征", "背景画像"),
    ("调整链", "方案"),
    ("样本", "历史案例"),
    ("外推", "估算"),
    ("学部权重", "专业方向"),
    ("跨学部", "专业方向"),
    ("跨学科", "专业方向"),
    ("风险", "差异"),
    ("ECE", ""),
    ("稀疏", "有限"),
)


def soften_progress_text(text: str) -> str:
    out = str(text)
    for old, new in _SALES_PROGRESS_REPLACES:
        if new:
            out = out.replace(old, new)
        else:
            out = out.replace(old, "")
    return out.strip() or "正在匹配院校方案..."


def status_label(phase: str) -> str:
    if phase == "running":
        return "需要确认专业方向"
    if phase == "complete":
        return "方案已生成"
    return "正在匹配院校方案"
