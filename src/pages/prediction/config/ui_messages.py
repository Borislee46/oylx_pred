PIPELINE_MESSAGES = {
    "init_engine": "初始化智算引擎",
    "verify_data": "核验背景信号",
    "wake_model": "{model_name} 模型预热",
    "search_cases": "对标核心案例库",
    "check_consistency": "核验 {count} 组核心信号",
    "build_features": "生成 {dim} 维特征向量",
    "prepare_pool": "匹配院校候选矩阵",
    "load_similarity": "加载 {count} 条相似信号",
    "extract_profile": "画像提取: {gpa} | {lang} | {uni} | {status}",
    "running_calc": "拟合 {total} 条申请路径",
    "initial_filter": "权重修正: 候选 {count} 项",
    "analyze_text": "剖析软背景潜力",
    "cross_check": "修正跨学科权重",
    "merging": "优化推荐方案",
    "empty_results": "无匹配信号，请调整目标",
    "done": "预测完成: {count} 条结论",
}

PROGRESS_ANIMATION_FALLBACKS = [
    (20, "解析背景画像"),
    (50, "对标核心案例"),
    (80, "模拟录取概率"),
    (100, "优化推荐方案"),
]

EXPERIENCE_ITEM_NAMES = {
    "research_details": "科研积淀",
    "internship_details": "实习经历",
    "award_details": "荣誉奖项",
    "paper_details": "学术产出",
}

EXPERIENCE_BOOST_TEMPLATE = "正在核验软背景：{items}（信息抽取与有效性检查）"
EXPERIENCE_DEFAULT_MSG = "正在核验软背景信息有效性"

FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

EXPERIENCE_ANALYSIS_MESSAGES = [
    "解析 {field} 亮点信号",
    "拟合 {field} 经历权重",
    "挖掘 {field} 核心增益",
    "核验软背景信号密度",
]

EXPERIENCE_VALIDATION_TEMPLATE = "校准 [{idx}/{total}]: {field_name} ({length}字 | {method})"

RANKER_MESSAGES = {
    "basic": [
        "[{tone}] 测算 {majors} 概率",
        "[{tone}] 校验 {majors} 契合度",
        "[{tone}] 对标 {majors} 竞争压",
    ],
    "cross_major": [
        "[{tone}] 跨申 {majors}: 分析 {bg_major} 路径",
        "[{tone}] 跨申 {majors}: 校准相关性信号",
    ],
    "faculty": [
        "[{tone}] {faculty} 维度加权评估",
        "[{tone}] {faculty} 跨学科信号增强",
    ],
    "relax": [
        "[{tone}] 放宽参数边界",
    ],
    "tighten": [
        "[{tone}] 收紧模型阈值",
    ],
    "fallback": [
        "[{tone}] 聚合特征矩阵",
        "[{tone}] 优化结果集格式",
    ],
}
