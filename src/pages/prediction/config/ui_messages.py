PIPELINE_MESSAGES = {
    "init_engine": "正在初始化智算引擎",
    "verify_data": "核验背景信息完整性",
    "wake_model": "启动{model_name}模型，参数预热中",
    "search_cases": "检索案例库，对标录取基准",
    "check_consistency": "已检索{count}条核心案例，核验模型一致性",
    "build_features": "生成{dim}维申请特征向量",
    "prepare_pool": "构建院校候选池，预匹配申请路径",
    "load_similarity": "预加载{count}条相似性信号",
    "extract_profile": "提取学生画像：GPA {gpa}，语言{lang}，院校{uni}，校准{status}",
    "running_calc": "测算{total}条申请路径成功率",
    "initial_filter": "初筛完成，候选{count}项，启动多因子权重修正",
    "analyze_text": "剖析软背景，量化科研与实习的潜力",
    "cross_check": "执行跨学科风险对标，调整申请路径权重",
    "merging": "聚合多源预测信号，优化最终方案",
    "empty_results": "未找到匹配结果，建议调整背景或拓宽目标",
    "done": "分析完成，已生成{count}条结果",
}

PROGRESS_ANIMATION_FALLBACKS = [
    (20, "初始化引擎，解析个人背景"),
    (50, "检索历史数据库，寻找相似案例"),
    (80, "权衡各项指标，模拟录取概率"),
    (100, "最终校准，确保预测严谨性"),
]

EXPERIENCE_ITEM_NAMES = {
    "research_details": "科研积淀",
    "internship_details": "实习经历",
    "award_details": "荣誉奖项",
    "paper_details": "学术产出",
}

EXPERIENCE_BOOST_TEMPLATE = "分析您的{items}对录取概率的提升效果"
EXPERIENCE_DEFAULT_MSG = "剖析您的申请亮点"

RANKER_MESSAGES = {
    "basic": [
        "{tone}，专业{majors}，评估与您背景的契合度",
        "{tone}，专业{majors}，审阅课程匹配细节",
        "{tone}，专业{majors}，解析专业对您的偏好程度",
    ],
    "cross_major": [
        "{tone}，跨申{majors}，分析从{bg_major}转申的可行性",
        "{tone}，跨申{majors}，量化经历与目标专业的相关性",
    ],
    "faculty": [
        "{tone}，{faculty}方向，结合学院偏好与行业趋势加权评估",
        "{tone}，{faculty}方向，参考往届案例，优化策略",
    ],
    "relax": [
        "{tone}，放宽条件发掘潜在选择",
    ],
    "tighten": [
        "{tone}，聚焦录取把握最高的目标",
    ],
    "fallback": [
        "{tone}，综合所有参数，生成最终排序",
        "{tone}，完成报告整合与格式优化",
    ],
}
