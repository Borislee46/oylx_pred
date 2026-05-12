# SKILL.md — Fortune AI-Native Interface

> **给 Claude 读的**。本文件定义 fortune 系统内的所有操作方式、数据结构和行为规则。
> 用户通过自然语言交互，Claude 直接读写文件完成操作。脚本 (`record.py`, `stats.py`) 是可选后端。

---

## 1. 系统身份

命主：李佳鹏 | 癸酉 1993.04.06 子时 | 安星码 C5VUC
体系：紫微斗数（三合+飞星+河洛+钦天）+ 八字四柱 + 八宅风水 + 人物图谱 + 预测追踪
认识论：贝叶斯推断 — 命盘=先验 P(H)，事实日志=似然 P(D|H)，验证结果=后验 P(H|D)

---

## 2. 文件清单

| 文件 | 用途 | 何时读 |
|------|------|--------|
| `README.md` | 导航枢纽，多维度索引（天/地/人/验/时） | 首次进入系统，或需要快速定位 |
| `记录规则.md` | 方法论+行为铁律+贝叶斯框架 | 需要理解"为什么这样记录"或铁律全文 |
| `predictions/predictions.json` | **预测数据库**（唯一读写目标） | 任何预测操作都必须读 |
| `predictions/record.py` | CLI 后端（批量/自动化用，日常不用） | 批量导入或脚本化时 |
| `predictions/stats.py` | 统计报告生成 | 需要完整 JSON 统计输出时 |
| `data/命盘数据_1993年4月6日.json` | 紫微斗数原始星盘数据 | 验证 json_ref 引用、查询星曜分布 |
| `data/来广营户型空间数据.json` | 户型方位+居住者位置 | 风水相关查询 |
| `analyses/紫微斗数命盘综合分析_完整版.md` | 最新完整命盘分析（含限流叠宫+逐年+专题） | 命盘推理需要引用时 |
| `analyses/来广营风水×命盘交叉分析.md` | 八宅方位×命盘映射+流年飞星（2026.05.11校准） | 风水相关查询 |
| `analyses/紫微×风水交互分析_2026.md` | **紫微×风水乘法交互**（空间K值调节命盘信号强度） | 需要理解"课题为什么难/易、怎么来"时必读 |
| `living/事实日志.md` | 可观测事件时间线（真相源） | 验证预测或对照命盘时需要 |
| `living/命理对照本.md` | 命盘信号×现实事件对照验证 | 需要查历史验证记录 |
| `living/人物知识图谱.md` | 六人 YAML 档案+交互矩阵 | 提到任何人时 |
| `living/2026年整体策略_天地人.md` | **年度行动策略**（天·已定/地·可改/人·应对），逐月操作清单 | 需要知道"这个月做什么"时 |
| `monthly/actions/` | 月度行动清单（YYYY-MM.md + archive/） | 月初创建或月底回顾 |

---

## 3. 数据结构

### 3.1 predictions.json 顶层

```json
{
  "meta": {
    "created": "2026-05-09",
    "owner": "李佳鹏",
    "birth": "1993-04-06 子时",
    "anxing_code": "C5VUC",
    "description": "命理预测追踪数据库..."
  },
  "predictions": [ /* 预测对象数组 */ ]
}
```

### 3.2 预测对象完整字段

```
{
  "id": "pYYYY_NNN",              // 格式: p{年}_{序号3位}，如 p2026_005
  "created": "YYYY-MM-DD",        // 创建日期
  "source": {
    "type": "事前预测",            // "事前预测" | "事后匹配"
    "chart_signal": "流年丙午天同禄入命宫（辰）",  // 必填: 命盘信号描述
    "json_ref": "si_hua_kou_jue.丙.lu=天同",      // 必填: JSON 字段引用
    "document_ref": "完整版 §全年总览"              // 必填: 文档引用
  },
  "prediction": "具体预测文本...",  // 必填: 必须具体、可证伪、有应期、有通关条件
  "category": "career",           // career|relationship|health|finance|housing|other
  "confidence": "中",             // 高|中|低 — 默认 "中"
  "tags": ["天同","禄","命宫"],    // 星曜/宫位/四化/流年标签，至少填1个
  "window": {
    "start": "2026-01-01",        // 应期开始
    "end": "2026-12-31"           // 应期结束（必填，不能为空）
  },
  "status": "pending",            // pending|ongoing|verified|missed
  "verification": null            // null 或 验证对象(见3.3)
}
```

**ID 生成规则**：读取 predictions 数组 → 找到最大序号 → `p{当前年份}_{max+1:03d}`。例：现有 p2026_018，下一个是 p2026_019。

**状态含义**：
- `pending`: 未到应期，等待中
- `ongoing`: 应期内，正在观察
- `verified`: 已验证（吻合或部分吻合）
- `missed`: 未应验（不吻合）

### 3.3 验证对象完整字段

```
{
  "date": "YYYY-MM-DD",               // 验证日期
  "outcome": "实际发生了什么",          // 必填: 具体可观测的结果
  "match_level": "部分吻合",           // 吻合|部分吻合|不吻合|暂无证据
  "adverse_evidence": "",             // 命盘预测了但没发生的，没有则留空
  "is_post_hoc": false,               // true = 事后匹配（不算验证力）
  "intervention_applied": false,      // true = 预测本身改变了行为
  "temporal_precision": "窗口内"       // 窗口内|相邻月|隔季|跨年
}
```

**状态转换逻辑**（必须严格遵守）：

| match_level | 新 status |
|-------------|-----------|
| "吻合" | verified |
| "部分吻合" | verified |
| "不吻合" | missed |
| "暂无证据" | **不变**（保持 pending/ongoing） |

---

## 4. 操作模板

### 4.1 新增预测

**触发词**：`记一条预测` `add prediction` `新预测` `记录一下`

**流程**：
1. 读 `predictions/predictions.json` 获取当前状态和最大ID
2. 向用户确认预测内容（至少：预测文本、类别、应期、命盘信号）
3. 生成新ID，构造完整预测对象
4. 追加到 predictions 数组，写回 JSON
5. 回复确认：`已记录 {id}。应期 {start}~{end}，置信度 {confidence}。`

**模板**（Claude 填充后写入）：
```json
{
  "id": "<auto>",
  "created": "<today>",
  "source": {
    "type": "事前预测",
    "chart_signal": "<REQUIRED: 具体星曜+宫位+四化>",
    "json_ref": "<REQUIRED: 命盘JSON中的具体路径>",
    "document_ref": "<REQUIRED: 分析文档中的具体章节>"
  },
  "prediction": "<REQUIRED: 具体、可证伪的预测文本>",
  "category": "<career|relationship|health|finance|housing|other>",
  "confidence": "中",
  "tags": ["<至少1个>"],
  "window": {"start": "<YYYY-MM-DD>", "end": "<YYYY-MM-DD>"},
  "status": "pending",
  "verification": null
}
```

**写入前检查**：
- prediction 是否具体可证伪？（不能是"注意人际关系"这种）
- window 是否有明确结束日期？
- json_ref 是否指向命盘JSON中真实存在的路径？
- chart_signal 和 document_ref 是否填写？
- tags 是否包含相关星曜/宫位/四化名称？

### 4.2 验证预测

**触发词**：`验证 p2026_XXX` `verify` `这条应验了` `这条没应验` `结果出来了`

**流程**：
1. 读 `predictions/predictions.json`
2. 找到目标预测（by id）
3. 收集验证信息：outcome, match_level, adverse_evidence, temporal_precision, intervention_applied
4. 按状态转换逻辑更新 status
5. 写回 JSON
6. 回复确认

**模板**（Claude 填充后写入）：
```json
{
  "date": "<today>",
  "outcome": "<REQUIRED: 实际发生什么>",
  "match_level": "部分吻合",
  "adverse_evidence": "<命盘预测了但没发生的>",
  "is_post_hoc": false,
  "intervention_applied": false,
  "temporal_precision": "窗口内"
}
```

### 4.3 查看预测

**触发词**：`看 p2026_XXX` `这条预测详情` `show prediction`

**流程**：读 JSON → 找目标 → 格式化展示全部字段

### 4.4 列出/筛选预测

**触发词**：
- `有哪些等着验证的` / `due predictions` → filter: window.end ≤ today AND status ∈ {pending, ongoing}
- `感情类预测` → filter: category = "relationship"
- `事业类预测` → filter: category = "career"
- `今年的预测` → filter by year in window
- `全部预测` → show all

**流程**：读 JSON → 按条件过滤 → 格式化列表展示

### 4.5 统计报告

**触发词**：`统计报告` `准确率` `stats` `应验率`

**流程**：
- 简单统计（总数/按类别/按状态/到期未验证）：Claude 直接读 JSON 计算
- 复杂统计（宫位覆盖盲区/校准分层/不利证据汇总）：运行 `python predictions/stats.py --json` 解析输出

### 4.6 月度行动清单

**触发词**：`创建本月行动清单` `{N}月行动清单` `月度回顾`

**流程**：
- 创建：按 `记录规则.md §五` 格式，写入 `monthly/actions/YYYY-MM.md`
- 回顾：读当月文件 → 逐条标注完成状态 → 决定迁移或放弃 → 移至 archive/

---

## 5. 命理分析五条铁律

> 完整版见 `记录规则.md §六`。此处为操作摘要。

### 5.1 推理标注出处
每条结论引用命盘JSON字段或分析文档章节。格式：`JSON: palaces[X].stars.main` 或 `完整版 §十二·迁移宫`。
禁止：`你的夫妻宫不好`（没出处）。正确：`夫妻宫（寅）空宫+大耗+劫煞 (JSON: palaces[2])，体质偏弱`。

### 5.2 列出多种可能
同一星曜组合 → 至少2种现实对应 → 标注哪种在当前背景下更可能。不能一对一映射。

### 5.3 不逢迎，有一说一
命盘有就说，没有就说没有。禁止选择性汇报。信息不足说"信息不足，不强行解读"。

### 5.4 风险+缓解+通关条件
推到行动建议时：风险按概率排序 → 每个风险附缓解措施 → 指明通过条件。
禁止"命中注定""躲不过"。核心公式：**真实风险 + 可执行缓解 + 通过条件 = 赋能**。

### 5.5 What-If 反事实
每条行动建议带三行对比：
- **不做的最坏情况**：[N个月后具体会怎样，引命盘时间窗口]
- **失败的最坏情况**：[尝试后最坏结果 + 可逆/不可逆]
- **为什么失败更优（或更差）**：[一句话诚实比较，可以是"不做更好"]

---

## 6. 写入前检查清单

每次修改 `predictions.json` 前自问：

1. ☐ 预测内容是否具体、可证伪？（不是模糊陈述）
2. ☐ 必填字段是否齐全？（source 三个字段、prediction、category、window.end）
3. ☐ json_ref 是否指向命盘JSON中的真实路径？
4. ☐ confidence 是否为 高/中/低 之一？
5. ☐ 如果是验证：match_level 对应的状态转换是否正确？
6. ☐ 是否记录了不利证据？（如果命盘预测X但实际非X）
7. ☐ 如果是事后匹配，是否标记了 `is_post_hoc: true`？
8. ☐ 写入前是否重新读取了 JSON？（防止覆盖他人变更）

---

## 7. 快速参考

| 用户说 | Claude 做什么 |
|--------|-------------|
| `记一条预测` | 按 §4.1 模板，追加到 predictions.json |
| `验证 p2026_005` | 按 §4.2 模板，更新 verification + status |
| `看 p2026_012` | 读 JSON，格式化展示 |
| `有哪些等着验证的` | 读 JSON，筛选到期且 pending/ongoing |
| `感情类预测应验了多少` | 读 JSON，按 category+status 统计 |
| `统计报告` | 计算或运行 stats.py，展示覆盖+校准 |
| `这个月要注意什么` | 读 predictions.json 当月窗口条目 + 完整版 §十二逐月 |
| `{名字}最近...` | 读 `人物知识图谱.md`，用代号查询 |
| `我的卧室风水` | 读 `来广营风水×命盘交叉分析.md` 对应章节 |
| `命宫怎么说` | 读完整版 §6.1 + `命理对照本.md` #命宫 |
| `创建本月行动清单` | 按 `记录规则.md §五` 格式创建 monthly/actions/YYYY-MM.md |
