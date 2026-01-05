PIPELINE_MESSAGES = {
    "init_engine": "[阶段]系统环境初始化...",
    "verify_data": "核验背景信息...",
    "wake_model": "加载预测模型...",
    "search_cases": "对标案例库...",
    "check_consistency": "匹配核心信号...",
    "build_features": "提取特征向量...",
    "prepare_pool": "筛选院校矩阵...",
    "load_similarity": "分析历史数据...",
    "extract_profile": "[阶段]生成背景画像...",
    "running_calc": "[阶段]拟合申请路径...",
    "initial_filter": "优化方案权重...",
    "analyze_text": "[阶段]评估软背景增益...",
    "cross_check": "修正跨学科权重...",
    "merging": "[阶段]生成最优方案...",
    "empty_results": "暂无匹配项目",
    "done": "[阶段]方案生成成功",
}

EXPERIENCE_ITEM_NAMES = {
    "research_details": "科研积淀",
    "internship_details": "实习经历",
    "award_details": "荣誉奖项",
    "paper_details": "学术产出",
}

EXPERIENCE_BOOST_TEMPLATE = "正在评估：{items}"
EXPERIENCE_DEFAULT_MSG = "核验软背景信息"

FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

EXPERIENCE_ANALYSIS_MESSAGES = [
    "解析 {field} 亮点...",
    "评估 {field} 权重...",
    "提取核心增益...",
    "核验信息密度...",
]

EXPERIENCE_VALIDATION_TEMPLATE = "正在核验：{field_name} ({idx}/{total})..."

RANKER_MESSAGES = {
    "basic": [
        "[{tone}] 测算录取概率...",
        "[{tone}] 校验项目契合度...",
        "[{tone}] 评估竞争压力...",
    ],
    "cross_major": [
        "[{tone}] 跨申路径分析...",
        "[{tone}] 校准相关性信号...",
    ],
    "faculty": [
        "[{tone}] 学院维度加权...",
        "[{tone}] 跨学科信号增强...",
    ],
    "relax": [
        "[{tone}] 调整参数边界...",
    ],
    "tighten": [
        "[{tone}] 收紧模型阈值...",
    ],
    "fallback": [
        "[{tone}] 聚合特征矩阵...",
        "[{tone}] 优化结果集...",
    ],
}
