PIPELINE_MESSAGES = {
    "init_engine": ["·系统环境初始化...", "·正在配置预测模型...", "·底层算力调度中..."],
    "verify_data": ["·核验背景信息...", "·核验申请人画像...", "·深度解析背景属性..."],
    "wake_model": ["·加载模型...", "·正在激活机器学习模型...", "·XGBoost模型权重加载中..."],
    "search_cases": ["·对标案例库...", "·检索历史录取数据...", "·匹配海量历史样本..."],
    "check_consistency": ["·匹配核心信号...", "·验证数据一致性...", "·信号校准中..."],
    "build_features": ["·提取特征向量...", "·特征工程并行计算...", "·构建多维画像空间..."],
    "prepare_pool": ["·筛选院校矩阵...", "·扫描院校准入阈值...", "·锁定目标院校池..."],
    "load_similarity": ["·分析历史数据...", "·计算轨迹相关性...", "·检索相似路径案例..."],
    "extract_profile": ["·生成背景画像...", "·构建申请人竞争力模型...", "·评估个体特征显著度..."],
    "running_calc": ["·拟合申请路径...", "·执行蒙特卡洛模拟...", "·多路径概率收敛计算..."],
    "initial_filter": ["·优化方案权重...", "·精炼初筛结果...", "·动态权重衰减调整..."],
    "analyze_text": ["·评估软背景增益...", "·NLP提取文书信号...", "·解析非结构化软背景..."],
    "cross_check": ["·修正跨学科权重...", "·执行跨专业系数补偏...", "·学院偏好一致性校验..."],
    "merging": ["·生成最优方案...", "·聚合高可信度路径...", "·排序并导出推荐方案..."],
    "empty_results": ["·暂无匹配项目", "·未发现高契合度路径", "·当前参数下无匹配案例"],
    "done": ["·预测结果生成成功", "·预测任务已完成", "·已罗列出推荐学校专业"],
}

EXPERIENCE_ITEM_NAMES = {
    "research_details": "科研积淀",
    "internship_details": "实习经历",
    "award_details": "荣誉奖项",
    "paper_details": "学术产出",
}

EXPERIENCE_BOOST_TEMPLATE = "·正在评估：{items}"
EXPERIENCE_DEFAULT_MSG = "·核验软背景信息"

FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

EXPERIENCE_ANALYSIS_MESSAGES = [
    "·解析 {field} 亮点...",
    "·评估 {field} 权重...",
    "·提取核心背提文本增益...",
    "·核验背提文本信息密度...",
    "·检验用户输入文本是否有效...",
    "·正在计算文本香浓熵值，确认文本是否有无效信息...",
]

EXPERIENCE_VALIDATION_TEMPLATE = [
    "·正在Agent校验文本：{field_name} ({idx}/{total})...",
    "·正在调用大模型校验用户输入的{field_name}文本，进度：{idx}/{total}...",
    "·正在查看学生的{field_name}...",
]

RANKER_MESSAGES = {
    "basic": [
        "·{tone}, 测算{target_major}录取概率...",
        "·{tone}, 校验{target_major}项目契合度...",
        "·{tone}, 评估{target_major}竞争压力...",
    ],
    "cross_major": [
        "·{tone}, 跨申路径分析：{background_major_ori} → {target_major}...",
        "·{tone}, 校准{background_major_ori}和{target_major}的相关性信号...",
        "·{tone}, 研判{background_major_ori}和{target_major}的跨专业程度...",
    ],
    "faculty": [
        "·{tone}, {target_major}学院维度加权...",
        "·{tone}, {target_major}跨学科信号增强...",
        "·{tone}, 评估{target_major}学院维度权重...",
    ],
    "relax": [
        "·{tone}, 调整{target_major}参数边界...",
        "·{tone}, 评估{target_major}参数边界权重...",
        "·{tone}, 评估{target_major}是否要放入推荐专业池中...",
    ],
    "tighten": [
        "·{tone}, 收紧{target_major}模型阈值...",
        "·{tone}, 评估{target_major}是否要从推荐专业池中剔除...",
    ],
    "fallback": [
        "·{tone}, 聚合{target_major}特征矩阵...",
        "·{tone}, 优化{target_major}结果集...",
    ],
}
