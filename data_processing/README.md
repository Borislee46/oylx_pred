# 外部数据清洗

> 配套文档：`../docs/external_data_integration.md`（总纲，优先读那个）
>
> 本目录是 Phase 0 数据清洗的工作区。

## 数据清单

| 文件 | 条数 | 来源 | 状态 |
|------|------|------|------|
| `指南者case总汇.xlsx` | 28,506 | 竞品案例库 | 待复制（在 Desktop/0708/） |
| `cases_newst_result.xlsx` | 9,175 | 指南者新爬（2026-05-11） | ✅ 已就位 |
| `applysquare案例总汇.xlsx` | 6,883 | 全球论坛 | ✅ 已就位 |
| `2025-硕士-外部案例.xlsx` | 13,698 | 早期已洗 | ✅ 已就位 |

## 目标 Schema

以 `../src/machine_learning_models/data/cases.feather`（61716行×20列）为基准。

**核心9列（必须尽力填满）**：

| # | 列名 | 指南者 | ApplySquare |
|---|------|--------|-------------|
| 1 | `admitted` | 全1 | OFFER→1, 被拒→0, 其余过滤 |
| 2 | `gpa` | 正则提取 | 归一化到4.0 |
| 3-4 | `toefl`/`ielts` | 正则提取 | 直接映射 |
| 5-8 | `research/internship/award/paper_count` | 经历拆分+分类 | 心得文本提取 |
| 9-10 | `target_university`/`target_major` | 名称对齐 | 名称对齐 |

## 可用维表

| 维表 | 内容 | 用途 |
|------|------|------|
| `cases.feather` | 94类 `background_major` + 12类 `faculty` | 专业标准化参照 |
| `school_major_details.feather` | 1547行：学校/专业中英文/专业大类 | 专业中→英翻译 + faculty 查表 |
| `school_base.feather` | 8712行：学校名/英文名/985·211/QS等级 | 背景学校标准化 |

## 清洗策略

**原则**：能用维表+规则搞定的不用 LLM，只有经历文本分类确实需要 LLM。

| 任务 | 方法 | 工具 |
|------|------|------|
| 学校名对齐 | fuzzy match `school_base` / `school_major_details` | rapidfuzz |
| 专业名中→英翻译 | fuzzy match `school_major_details.专业中文名称` | rapidfuzz |
| 本科专业→94类标准化 | fuzzy match `cases.background_major` | rapidfuzz |
| 本科专业→12学院 | 查表 `school_major_details.专业大类` | pandas |
| 经历文本→5类计数 | **关键词规则 + LLM兜底** | rapidfuzz + deepseek |
| GPA归一化 | 分制转换 `GPA/GPA分制×4.0` | pandas |
| 基本背景→GPA/语言 | 正则 | re |

### 经历分类规则优先

> 文档 §5.3：指南者 `主要经历` 拆分+分类。**规则覆盖率预计 >80%**，LLM 仅兜底。

关键词信号：
- 含"实习"/"intern"/"公司" → `internship`
- 含"项目"/"课题"/"论文"/"研究"/"实验室" → `research`
- 含"发表"/"专利"/"Accepted" → `paper`
- 含"奖学金"/"竞赛"/"获奖"/"一等奖" → `award`
- 含"社团"/"志愿者"/"学生会" → `activity`
- 无明确信号的 → LLM 分类

### 学校名对齐流程

```
优先级1：精确匹配 XDF 内部取值集合
优先级2：rapidfuzz 模糊匹配（阈值 85%），人工抽查 top-100
优先级3：手动映射表（config/school_name_mapping.json）
优先级4：无法对齐 → 标记 external_only
```

## 进度追踪

### Phase 0 检查清单

- [ ] 指南者 `基本背景` 正则提取：GPA、语言、应届/往届
- [ ] 指南者 `主要经历` 拆分+分类 → count 字段
- [ ] ApplySquare GPA 归一化到 4.0
- [ ] ApplySquare `录取结果` → `admitted`，过滤非终态
- [ ] ApplySquare `本科` 缺失回填（从心得文本正则提取）
- [ ] 学校名映射表（rapidfuzz + 人工 top-100）
- [ ] 专业名映射表（中→英，词序不敏感 fuzzy）
- [ ] 产出 `external/zhinanzhe_clean.feather`
- [ ] 产出 `external/applysquare_clean.feather`
- [ ] 产出 `external/external_merged.feather`

### 脚本对应

| 脚本 | 对应任务 |
|------|---------|
| `crawler.py` | 指南者增量爬取（已完成一次全量） |
| `convert_to_cases.py` | 指南者+ApplySquare 清洗转换（需改成 fuzzy+规则版） |
| `convert_external_to_cases.py` | 外部案例清洗（2025-硕士-外部案例.xlsx，需改成 fuzzy 版） |

## 依赖

```bash
pip install pandas pyarrow openpyxl rapidfuzz
# 如需 LLM 兜底：
pip install openai
```

## 输出目录（对齐 external_data_integration.md §7）

```
src/machine_learning_models/data/external/
├── applysquare_raw/              # 原始文件（保留不删）
├── zhinanzhe_raw/                # 原始文件（保留不删）
├── applysquare_clean.feather     # 清洗后（schema = cases.feather）
├── zhinanzhe_clean.feather       # 清洗后（schema = cases.feather）
└── external_merged.feather       # 合并后，供展示层加载
```

## 关键约束（external_data_integration.md §2）

- **XGBoost 训练数据不变**：外部数据不进 `cases_min.feather`
- 外部数据只用于展示层：反推卡片画像、差异分析、相似检索
- 每个统计数字需标注数据来源
- 外部数据不参与 `admission_cache`
