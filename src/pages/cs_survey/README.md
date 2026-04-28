# CS Survey 模块技术文档

## 1. 模块概述

`cs_survey` 是客服问卷数据可视化模块，采用 YAML 配置驱动的插件架构。通过 `config/cs_survey/*.yaml` 定义调研问卷的数据源、评分规则、视图类型、主题归类、产品分组等，无需修改代码即可新增调研。支持多视图切换（概览 / 按产品矩阵 / 按维度矩阵）、KPI 卡片、评分分布图、反馈详情表、主题分析等功能。

## 2. 目录结构

```
cs_survey/
├── __init__.py            # 空（入口通过 page.py）
├── page.py                # 页面入口：render()，调研/视图选择
├── schema.py              # 数据模型：SurveyConfig, DataSource, ScoringSpec 等 12 个 dataclass
├── registry.py            # YAML 配置扫描、调研/视图注册
├── filters.py             # 交叉筛选器逻辑
├── loader.py              # 数据加载（@st.cache_data + feather 缓存）
├── text_utils.py          # 文本工具
├── layout.py              # 页面布局辅助
├── engine/                # 计算引擎
│   ├── kpis.py            # KPI 指标计算
│   ├── scoring.py         # 评分计算（NPS / Likert / 百分比）
│   ├── themes.py          # 主题归类与情感分析
│   └── aggregations.py    # 聚合计算
├── ui/                    # UI 组件
│   ├── blocks.py          # 通用 UI 块
│   ├── charts.py          # Plotly 图表
│   ├── filter_bar.py      # 筛选器栏
│   ├── navigation.py      # 视图导航
│   ├── tables.py          # 数据表格
│   ├── footer.py          # 页脚（返回概览按钮）
│   └── theme_css.py       # CSS 主题注入
└── views/                 # 视图渲染器
    ├── base.py            # 视图基类
    ├── overview.py        # 概览视图
    ├── matrix_by_product.py   # 按产品矩阵
    ├── matrix_by_pillar.py    # 按维度矩阵
    └── detail_shared.py   # 详情页共享组件
```

## 3. 端到端流程

```
用户访问 pages/cs_survey.py
        │
        ├── inject_base_css()
        └── render()
            │
            ├── registry.list_surveys() 扫描 config/cs_survey/*.yaml
            │       └── parse_survey_config() → SurveyConfig dataclass
            │
            ├── _pick_survey()         ← query_params["survey"]
            │       └── st.selectbox（多调研时显示选择器）
            │
            ├── _pick_view()           ← query_params["view"]
            │       └── VIEW_REGISTRY → 视图渲染器
            │
            ├── loader 加载数据源（CSV/Excel → @st.cache_data）
            ├── filters 交叉筛选（部门/产品/时间等）
            │
            └── renderer(cfg)
                │
                ├── engine/kpis → KPI 卡片
                ├── engine/scoring → 评分计算
                ├── engine/themes → 主题归类
                ├── ui/charts → Plotly 图表
                ├── ui/tables → 详情表格
                └── ui/footer → 返回按钮
```

## 4. 核心组件

### 4.1 schema.py

定义 12 个 dataclass，是整个模块的类型系统：
- **SurveyConfig**：问卷根配置（id, title, sources, pillars, products, views, themes...）
- **DataSource**：数据源定义（id, label, path）
- **ScoringSpec**：评分规则（type + params，支持 likert/nps/percentage/reverse）
- **PillarSpec** / **ProductGroupSpec** / **ViewSpec**：维度/产品/视图定义
- **ThemeRuleset** / **ThemeRule**：主题归类规则

### 4.2 registry.py

`list_surveys()` 扫描 `config/cs_survey/*.yaml`，解析为 `SurveyConfig` 对象，带 `@lru_cache` 缓存。也提供 `repo_root()`、`resolve_data_path()` 等路径工具。

### 4.3 page.py

`render()` 是唯一入口。职责：
1. 调研选择（URL query param 或 selectbox）
2. 视图切换（URL query param 或 session state）
3. 调用对应视图渲染器
4. 渲染页脚

### 4.4 engine/ 子包

- **scoring.py**：`ScoringSpec` → 实际计算（NPS = promoters% - detractors%、Likert 均值、百分比）
- **kpis.py**：调研级 KPI（总样本数、整体满意度、NPS 等）
- **themes.py**：关键词匹配 → 主题归类（好评/差评原因分类）
- **aggregations.py**：跨数据源聚合

### 4.5 views/ 子包

`VIEW_REGISTRY` 注册视图类型 → 渲染器映射：
- **overview**：概览（KPI + 评分分布 + 反馈统计）
- **matrix_by_product**：按产品分组的矩阵视图
- **matrix_by_pillar**：按维度分组的矩阵视图

### 4.6 ui/ 子包

- **charts.py**：Plotly 图表（柱状图、饼图、热力图、分布图）
- **tables.py**：详情表格、反馈明细表
- **filter_bar.py**：交叉筛选器（根据 `CrossFilterSpec` 动态生成）
- **theme_css.py**：模块专属 CSS（~680行 CSS 模板）

## 5. 数据流

```
config/cs_survey/*.yaml
    │
    ▼
registry.list_surveys() → SurveyConfig (dataclass)
    │
    ├── DataSource.path → loader → pandas DataFrame
    │       └── @st.cache_data(.feather 缓存)
    │
    ├── CrossFilterSpec → filter_bar → 筛选后 DataFrame
    │
    └── 视图渲染器
        ├── ScoringSpec → engine/scoring → 分数
        ├── ThemeRuleset → engine/themes → 主题标签
        ├── ProductGroupSpec → 产品分组聚合
        └── ui/charts + ui/tables → 渲染
```

## 6. 扩展新调研

1. 在 `config/cs_survey/` 下添加 `新调研.yaml`
2. 定义 `id`、`title`、`sources`（数据源路径）、`pillars`（维度 + 评分规则）、`views`
3. 无需修改任何 Python 代码

## 7. 依赖

- `streamlit`：UI + `@st.cache_data` + `query_params`
- `pandas`、`numpy`：数据处理
- `plotly`：图表
- `pyyaml`：YAML 配置解析
- [Utils](src/utils/README.md) — `init_page()`、`SessionManager`
- [Config](config/README.md) — `cs_survey/*.yaml` 问卷配置
