# School View 模块技术文档

## 1. 模块概述

`school_view` 是学校视角分析页面（2026-05 新增），提供三个维度的院校洞察：学校录取画像（P50/P25/P75 分位数）、What-If 反事实模拟（背景提升后概率变化）、多校横向对比。不作为预测主流程的一部分，而是独立的数据探索工具。

## 2. 目录结构

```
school_view/
├── __init__.py           # 空（入口通过 page.py）
├── page.py               # 页面主入口：3 Tab 渲染（画像 / What-If / 对比）
├── school_form.py        # 独立表单（背景 + 目标院校选择，复用 GPAConverter）
├── school_profiles.py    # SchoolProfileCalculator：分位数画像 + 差距分析
├── what_if_simulator.py  # WhatIfSimulator：反事实模拟 + ROI 排序
└── data_loader.py        # @st.cache_data 数据加载 + FEATURE_LABELS
```

## 3. 端到端流程

```
用户访问 School View 页面
    │
    ├── 1. load_full_cases() → cases_df（@st.cache_data）
    ├── 2. SchoolProfileCalculator(cases_df)
    ├── 3. render_school_form(cases_df) → form_data
    │       ├── 背景：院校 / 专业 / GPA（复用 GPAConverter）/ 语言 / 经历
    │       └── 目标：multiselect 院校
    │
    ├── Tab 1: 学校画像
    │   ├── compute_school_profile(university) → P50/P25/P75（6 维特征）
    │   ├── get_student_percentile() → 学生在录取者中的百分位
    │   └── get_gap_analysis() → 差距排序（最大负差距优先）
    │
    ├── Tab 2: What-If 模拟
    │   ├── WhatIfSimulator.simulate(form_data, schools)
    │   │   ├── 取 top 5 热门 (uni, major) 组合
    │   │   ├── 6 个场景（baseline / GPA+0.2 / GPA+0.4 / 语言+0.5 / 科研+1 / 实习+1）
    │   │   └── XGBoost 批量推理 → 按学校聚合
    │   ├── compute_roi_table() → 概率变化表
    │   └── ROI 排序：哪种干预提升最大？
    │
    └── Tab 3: 学校对比
        └── 多校横向对比（6 维特征），st.metric + 百分位 delta
```

## 4. 核心组件

### 4.1 SchoolProfileCalculator

- `compute_school_profile(university)`：筛选该校 admitted==1 的样本，计算 6 维特征的 P50/P25/P75。n < 10 返回 insufficient
- `get_student_percentile(university, feature, value)`：学生在录取者中的百分位（≤ 该值的比例）。n < 5 返回 None
- `get_gap_analysis(university, student_values)`：逐维度计算 student - P50，按差距升序排列
- `get_top_majors(university, top_n=10)`：该校最热门专业排名

### 4.2 WhatIfSimulator

- 6 个预定义场景：baseline / GPA+0.2 / GPA+0.4 / 语言+0.05 / 科研+1段 / 实习+1段
- 每所学校取 top 5 热门 (university, major) 组合做批量推理
- 按学校聚合为平均概率
- `compute_roi_table()` 生成对比表
- ROI 排序：边际增益最大的干预

### 4.3 school_form.py

独立于预测主流程的轻量表单，复用 `GPAConverter` 和 `normalize_language_score`。

## 5. 设计取舍

- **为什么不用完整的 prediction pipeline 而只调 XGBoost？** What-If 关注的是特征扰动对模型输出的方向性影响，不是精确的录取概率。跳过 5 层调整链避免了调整链的乘法衰减掩盖模型本身的灵敏度。
- **为什么取 top 5 专业而非全部？** 全量 (uni, major) 组合过多，且用户关心的是该校"代表性专业"的录取难度。top 5 高频专业足以反映学校的整体水平。
- **n < 10 不显示画像**：历史录取样本过少时，分位数无统计意义。显示一个 P50 基于 3 个样本的"画像"比不显示更有害。

## 6. 依赖

- `streamlit`：UI
- `pandas`、`numpy`：数据处理
- `prediction` 模块：`PredictionModel`、`GPAConverter`、`normalize_language_score`、`page_data_loader`
- `src/machine_learning_models/data/cases.feather`：训练数据
