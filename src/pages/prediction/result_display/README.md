# Result Display — 预测结果展示

## 1. 模块概述

`result_display` 负责预测结果的视觉呈现：Hero 摘要横幅、三类推荐表格（含 Δ 对比）、算法 Trace 瀑布图。所有组件通过 `__init__.py` 暴露 4 项公共 API。

## 2. 目录结构

```
result_display/
├── __init__.py            # 公共 API（DeltaCalculator, ResultsDisplay, render_hero_summary）
├── results_display.py     # 结果表格渲染（相似/跨专业/用户指定 + 列配置 + 排序）
├── delta_calculator.py    # 跨提交概率 Δ 计算（(uni, major) 键匹配）
├── hero_summary.py        # 顶部摘要横幅：校徽墙 + 梯度分布条
├── trace_display.py       # 算法 Trace：概率调整瀑布图 + 反事实 + 校准指标
└── trace_assets.py        # Trace CSS + STEP_INTENT 调整因子说明文案
```

## 3. 端到端流程

```
预测完成（unified_results）
    │
    ├── hero_summary.render_hero_summary()
    │   ├── 校徽墙（top 5, Linear-style 重叠排列）
    │   └── 梯度分布条（冲刺/适中/保底 数量 + 百分比）
    │
    ├── results_display.ResultsDisplay
    │   ├── 相似专业表格（TOP_SIM_RESULT_UI_CONFIG 列配置）
    │   ├── 跨专业表格（TOP_CROSS_RESULT_UI_CONFIG 列配置）
    │   ├── 用户指定表格（如有）
    │   ├── Δ 列（DeltaCalculator: +3.2% / -1.5% / NEW / —）
    │   └── 排序：概率降序，港校优先（UNIVERSITY_ORDER_MAP）
    │
    └── trace_display.render_trace_for_results()
        ├── 🥇 Top1 / 🥈 Top2 / 🥉 Top3 radio 切换
        ├── 瀑布图（_adjustment_steps 逐级展示）
        ├── 反事实扰动（GPA±0.2, 语言±0.05, 实习+1）
        └── 校准指标（Brier / AUC / Threshold / 阳性率偏差）
```

## 4. 核心组件

### 4.1 results_display.py — ResultsDisplay

主结果表格渲染器：
- 三类推荐分表展示（相似专业 / 跨专业 / 用户指定）
- 列配置来自 `data_sort_config`（`TOP_SIM_RESULT_UI_CONFIG`、`TOP_CROSS_RESULT_UI_CONFIG`）
- Δ 列对比前后两次提交的概率变化（绿涨红跌蓝新增）
- 排序规则：概率降序，同概率按 `UNIVERSITY_ORDER_MAP` 港校优先

### 4.2 delta_calculator.py — DeltaCalculator

跨提交概率对比：
- `build_prob_map()` — 将结果列表转为 `{(uni, major): prob}` 查找表
- `compute_delta()` — 以 (university, major) 为复合键对比新旧概率
- 四种输出：`+3.2%`（涨）、`-1.5%`（跌）、`NEW`（新增）、`—`（无变化）

### 4.3 hero_summary.py — Hero 摘要横幅

预测结果第一眼——不依赖 ExplainAgent，即时渲染：
- **校徽墙**：top 5 院校 logo 重叠排列（Linear/Stripe 风格）
- **梯度分布条**：冲刺(<30%) / 适中(30-55%) / 保底(>=55%) 三段比例条
- 静态展示，ExplainAgent 后续流式解读不影响此区域

### 4.4 trace_display.py — 算法 Trace

开发者/算法人员调试工具，展示概率调整链路：
- **case 选择器**：🥇🥈🥉 Top 1-3 radio 切换
- **瀑布图**：真实瀑布柱，baseline 虚线作锚点，每步带设计意图 tooltip
- **反事实**：4 个扰动场景（GPA±0.2, 语言+0.05, 实习+1）下概率变化
- **校准指标**：Brier / AUC / 阈值 / 阳性率偏差，证明 pipeline 必要性
- 设计目标：30 秒讲完一个 case 的概率调整故事

### 4.5 trace_assets.py — Trace 静态资产

- `STEP_INTENT`：9 个调整因子的中文说明文案（GPA Penalty / Language Penalty / Cross Major...）
- CSS 模板：外部引用 `assets/hk_style/51_trace.css`

## 5. 依赖

- [prediction 模块](../README.md) — `data_sort_config`、`result_modifier`、`handler_config`
- `src/utils/session_manager.py` — SessionManager
- `assets/hk_style/51_trace.css` — Trace 专属样式
- `assets/school_logos/` — 校徽 PNG（hero_summary）
