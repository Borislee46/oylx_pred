# HK 运营数据看板

Streamlit 多 Tab 看板页面，为香港国际教育（XDF HK）提供运营、营收、招生、教务、续费五大维度的数据可视化。

## 目录结构

```
src/pages/hk_dashboard/
├── __init__.py              # render() 公共入口
├── render.py                # 主渲染：CSS + 数据加载 + st.tabs 分发
├── config.py                # 列名常量、文件路径、奖金阶梯表
├── data_loader.py            # 7 个 @st.cache_data(ttl=600) CSV 加载器
├── styles.py                # CSS 主题注入（#008a6c 主色）
├── tabs/
│   ├── overview.py          # Tab 1: 综合概览 — KPI 卡片 + 趋势 + 分布
│   ├── revenue.py           # Tab 2: 营收分析 — 现金收入 + 结转收入
│   ├── enrollment.py        # Tab 3: 招生转化 — 漏斗 + 顾问 + 渠道
│   ├── academics.py         # Tab 4: 教务教学 — 名册 + 考勤 + 课量
│   └── renewal.py           # Tab 5: 续费看板 — 续费率 + 奖金
├── charts/
│   ├── line_chart.py        # Altair 月度趋势折线
│   ├── bar_chart.py         # Altair 柱状图（简单/分组/水平）
│   └── pie_chart.py         # Altair 环形图（自动归并"其他"）
├── components/
│   ├── kpi_cards.py         # st.metric 指标卡片网格
│   ├── data_table.py        # 带搜索的筛选表格
│   └── filters.py           # 月份/分类筛选器
└── metrics/
    ├── revenue_metrics.py   # 现金收入、结转收入聚合计算
    ├── funnel_metrics.py    # 转化漏斗、顾问排名、班容指标
    └── renewal_metrics.py   # 续费率（队列法）+ 奖金阶梯计算
```

## 端到端流程

```
pages/hk_dashboard.py (薄路由)
  │
  ├── init_page(module_name="hk_dashboard")
  │     └── E2 认证守卫 → CSS 注入 → 水印
  │
  ├── data_loader.load_all_data()
  │     ├── 7 个独立 @st.cache_data(ttl=600)
  │     └── 返回 dict[str, pd.DataFrame]
  │
  └── st.tabs(5)
        ├── overview.render(data)   → KPI + 趋势 + 分布
        ├── revenue.render(data)    → 现金收入 + 结转收入
        ├── enrollment.render(data) → 漏斗 + 顾问 + 渠道
        ├── academics.render(data)  → 名册 + 考勤 + 课量
        └── renewal.render(data)    → 续费率 + 奖金
```

## 核心组件

| 组件 | 说明 |
|------|------|
| `data_loader.py` | 7 个 `@st.cache_data(ttl=600)` 加载器，读取 `data/hk/*.csv`，容错返回空 DataFrame |
| `config.py` | 列名常量、文件路径常量、奖金阶梯表 `lookup_bonus_rate(rate, count)` |
| `styles.py` | CSS 主题注入，主色 `#008a6c`，绿色左边框 h2，圆角容器，绿色 metric 值 |
| `renewal_metrics.py` | 续费率队列法：当月学员 → 次月在班 → 续费率；奖金 = 阶梯单价 × 当月课时 |
| `funnel_metrics.py` | 4 阶段转化漏斗：总资源 → 已外呼 → 有工单 → 已签约 |
| `charts/` | Altair 图表工具（折线/柱状/环形），自动处理空数据 |

## 数据流

```
客服资源 ─┐
TMK      ─┼─→ enrollment tab (资源id join)
签约列表 ─┘

班级维表 ─┐
花名册   ─┼─→ academics tab + renewal tab (班级编码 join)
收入人次 ─┘

结转收入 ───→ revenue tab (班级编号 join 班级维表)
```

各 Tab 按需 join，不做预关联，避免空值爆炸。

## 续费率算法

采用队列法（与 `data/hk/旺角2月教学端绩效核对.xlsx` 一致）：

1. 选定月份 M（如 2026-02）
2. 确定 M 月在班的学员：`进班日期 <= M月底 AND (离班日期 >= M月初 OR 离班日期 IS NULL)`
3. 确定 M+1 月在班的学员（同上逻辑）
4. 按教师聚合：`续费率 = M+1月在班数 ÷ M月学员数`
5. 查阶梯表：续费率区间 × 班型人数区间 → 港币/课时单价
6. 奖金 = 单价 × 当月课时

## 依赖

- `streamlit==1.56.0`
- `pandas==2.3.1`
- `altair`（随 Streamlit 内置）
- 无新增外部依赖

## 路由

`pages/hk_dashboard.py` — 薄路由，调用 `init_page(module_name="hk_dashboard")` + `render()`

权限注册：`src/utils/auth/permission_checker.py` `MODULE_IDS` 已添加 `"hk_dashboard"`

## 数据模型

完整的表关联 ER 图、键值交集矩阵、列名修正详见 **[DATA_MODEL.md](DATA_MODEL.md)**。
