<!-- !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files. -->
# PDF 报告生成与下载 (PDF Report Generator)

基于 **reportlab** 的个性化留学申请方案 PDF 报告引擎。将预测结果 + 智能优化方案渲染为多段式 A4 PDF，含中文水印、学校 Logo、雷达图、AI 分析等，通过 Streamlit download_button 一键下载。

## 目录结构

```
exported_pdf/
├── page_components/pdf_generation/          # PDF 生成主模块
│   ├── __init__.py           # PDFSectionBuilder 导出
│   ├── config.py             # 布局参数（页边距/字体/颜色/水印）
│   ├── utils.py              # DataNormalizer, ContentFormatter, pdf_cache
│   ├── pdf_download_section.py  # Streamlit 下载按钮 + 缓存管理
│   ├── content/              # 数据可视化内容
│   │   ├── __init__.py
│   │   ├── radar.py          # matplotlib 雷达图（学生特征 vs 全校分布）
│   │   └── school_specific_content.py  # 学校专属定制文案
│   └── generators/           # PDF 生成核心
│       ├── __init__.py
│       ├── pdf_report_generator.py  # WatermarkedDocTemplate + PDFReportGenerator
│       ├── pdf_data_extractor.py    # 从 session_state 提取所有数据
│       ├── pdf_ai_agent.py          # DeepSeek AI 驱动分析建议
│       ├── pdf_styler.py            # 中文字体 + ParagraphStyle 定义
│       └── section_builder/         # 各段内容构建器
│           ├── __init__.py          # PDFSectionBuilder 编排器
│           ├── cover_page.py        # 封面（标题/logo/日期/水印）
│           ├── background_section.py # 学生背景分析页（表格+雷达图）
│           ├── analyst_notes.py     # AI 分析建议段
│           ├── school_detail.py     # 推荐学校详情段
│           └── optimization_strategies_section.py # 各策略方案对比
├── admission_probability_calculator_components/optimization/
│   └── pdf_generator.py     # 优化完成 → PDF 生成触发
├── agent/
│   ├── pdf_agent.py         # LLM Agent：生成 PDF 中的分析文案
│   └── pdf_prompts.py       # System/user prompt 模板
├── school_specific_content.json  # 学校专属内容配置
└── school_combination_optimizer_api.md  # API 文档
```

## 端到端流程

```
[用户点击"智能优化" → 优化完成]
    │
    ├── 1. optimization_executor.py 调用 pdf_generator.py
    │   └── 触发 PDF 生成
    │
    ├── 2. PDFDataExtractor.extract_user_data()
    │   ├── 从 session_state 提取: input_data, original_form_data
    │   ├── 提取: prediction_results, optimization_recommendations
    │   ├── 提取: cases_df (历史案例)
    │   └── 提取: user_nickname, user_email (水印用)
    │
    ├── 3. PDFReportGenerator.generate_report()
    │   │
    │   ├── 3a. WatermarkedDocTemplate 初始化
    │   │   ├── A4 页面, 自定义页边距
    │   │   ├── CoverPage 模板（无页眉页脚）
    │   │   ├── LaterPages 模板（含页眉线 + 页码 + 水印）
    │   │   └── 水印: "EasyApply 2025-XX-XX {昵称}" 平铺 30° 旋转
    │   │
    │   ├── 3b. PDFSectionBuilder 按顺序构建 story:
    │   │   ├── create_cover_page()          → 封面页
    │   │   │   ├── EasyApply Logo + 标题
    │   │   │   ├── 副标题 "个性化申请方案报告"
    │   │   │   ├── 生成日期 + 用户昵称
    │   │   │   └── 半透明背景装饰
    │   │   │
    │   │   ├── [PageBreak → LaterPages 模板启用]
    │   │   │
    │   │   ├── create_background_section()  → 背景分析
    │   │   │   ├── 学生背景表（院校/专业/GPA/语言/经历）
    │   │   │   ├── matplotlib 雷达图（6维 vs 全校分位数）
    │   │   │   └── AI 分析建议（通过 PDFAIAgent 调用 DeepSeek）
    │   │   │
    │   │   └── create_optimization_strategies_section() → 策略方案
    │   │       ├── for each 策略 (策略1/2/3):
    │   │       │   ├── 策略名称 + 学校数量
    │   │       │   ├── 指标卡片（全拒率/录取信心/多样性）
    │   │       │   └── 学校详情表（logo/名称/专业/概率/难度/分析）
    │   │       └── 策略对比总结
    │   │
    │   └── 3c. doc.build(story) → BytesIO buffer → bytes
    │
    ├── 4. 结果回写 session_state
    │   ├── pdf_data: bytes
    │   ├── pdf_filename: "EasyApply申请方案_{昵称}_{日期}.pdf"
    │   ├── pdf_generated: True
    │   └── pdf_generation_time: timestamp
    │
    └── 5. pdf_download_section.display_pdf_download_section()
        └── st.download_button(label="下载完整报告", data=pdf_data,
                               mime="application/pdf", type="secondary")
```

## 核心组件详解

### WatermarkedDocTemplate

继承 `BaseDocTemplate`，双页模板架构：
- **CoverPage**: 纯封面内容，无页眉页脚
- **LaterPages**: 自动添加页眉线 + 页码 + 水印

水印特性：
- 文本格式: `EasyApply {date} {nickname}`
- 30° 旋转，低透明度，棋盘格交错排列
- 防截图泄露追溯

### PDFStyler（中文字体）

- 字体: `STSong-Light`（华文宋体，UnicodeCID）
- 定义 7 级 ParagraphStyle: CustomTitle / SectionTitle / SubTitle / CustomBody / SmallText / Highlight / LeftSmallText
- 颜色主题: 主色 `#1E90FF`，正文 `#333333`，次要 `#777777`

### PDFDataExtractor

从 Streamlit session_state 提取所有 PDF 所需数据：
- `input_data` + `original_form_data` → 用户背景
- `prediction_results` → 预测结果
- `optimization_recommendations` → 优化方案
- `cached_load_cases_data()` → 历史案例
- `e2_user_nickname` / `user_email` → 用户标识

`validate_data_for_pdf_generation()` 一次性返回完整数据包。

### PDFAIAgent

调用 DeepSeek API 生成个性化分析文案：
- 输入: 学生背景 + 推荐方案摘要
- 输出: 策略分析/优劣势/建议（用于 analyst_notes 段）
- 缓存: MD5 hash → 24h TTL

### Section Builders

| Builder | 职责 | 输入 | 输出 |
|---------|------|------|------|
| CoverPageBuilder | 封面页 | 昵称/标题 | Paragraph+Spacer+Table |
| BackgroundSectionBuilder | 背景分析 | user_data/cases | 表格+雷达图+AI建议 |
| SchoolDetailBuilder | 学校详情 | school/major/cases | 表格(logo/详情/分析) |
| OptimizationStrategiesSectionBuilder | 策略方案 | optimization_results | 多段策略对比 |
| AnalystNotesGenerator | AI分析 | 背景+策略摘要 | AI生成文案 |

### PDF 下载按钮

`pdf_download_section.py`:
- 仅在 `optimization_performed=True` 时显示
- PDF 生成成功 → `st.download_button(label="下载完整报告", mime="application/pdf")`
- 失败 → disabled button + 错误提示
- `clear_pdf_cache()` 清除 6 个相关 session key

## 关键依赖

- `reportlab` — PDF 生成 (BaseDocTemplate, platypus, pdfmetrics)
- `matplotlib` — 雷达图渲染
- `streamlit` — 下载按钮 + session_state
- `requests` — DeepSeek API 调用
- `pandas` — 历史数据处理
- 内部模块: `pdf_agent`, `session_manager`, `env_config_loader`, `app_data_loader`
