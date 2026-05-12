# Pathfinder — AI 选校解读

> 按钮标签 "Pathfinder AI解读（beta）" 的功能模块。点击后生成五维雷达图 + 选校梯度 + 产品匹配 + LLM 流式深度解读 + 院校历史数据对比。

---

## 1. 架构概览

```
用户点击按钮
    │
    ├── 缓存命中? → 直接渲染（跳过 LLM）
    │
    ├── render_static_frame()    静态框架：雷达图、梯度条、产品匹配
    │
    ├── classify_profile()       选择 profile → 决定 System Prompt 语调
    │     ├── strong_elite       强背景，冲顶尖
    │     ├── medium_mixed       中等背景，梯度分布
    │     ├── weak_gaps          短板较多，需要提升路径
    │     └── cross_major        跨专业申请
    │
    ├── ExplainAgent.stream()    DeepSeek 流式生成 JSON
    │     └── 逐 chunk 提取部分字段 → 每 25ms/6字渲染一次
    │
    ├── render_ai_section()      最终静态渲染（概述/优势/需关注/总结/推荐说明）
    │
    └── render_school_cards()    院校卡片（概率条 + AI 备注 + 百分位数据）
```

---

## 2. 文件清单

| 文件 | 职责 |
|------|------|
| `page_components/content_display.py` | 主编排器：按钮门控、缓存、流式调度、异常恢复 |
| `ai_report.py` | 静态框架（雷达图、梯度条、产品匹配）+ 最终渲染 |
| `ai_report_sections.py` | 流式渲染组件、学校卡片、推荐说明 |
| `ai_report_styles.py` | 全部 CSS（卡片、动画、响应式） |
| `ai_report_catalog.py` | 产品目录（6 个可推荐服务） |
| `ai_school_stats.py` | 历史数据分位数统计 |
| `agent/explain_agent.py` | ExplainAgent：prompt 构建 + 流式/同步 API 调用 |
| `agent/explain_profiles.py` | 四档 profile 分类器 + 每档专属 System Prompt |
| `agent/base_agent.py` | 基类：API 调用、四级 JSON 修复、内存缓存 |

---

## 3. 数据流

### 3.1 Prompt 构建

`ExplainAgent._prepare()` 将学生背景 + 预测结果 + 已推荐产品拼接为一个 prompt：

```
## 学生背景
院校：北京大学
专业：计算机科学
GPA：3.6
语言：雅思 7.0
经历：科研：图像识别项目；实习：字节跳动算法岗

## 预测结果（共 12 条推荐）
相似专业推荐：
  1. 香港大学 Computer Science 85%
  2. 新加坡国立大学 Computer Science 72%
  ...

### 已推荐服务产品
- 标准申请服务（港-A-4 · ¥12,000-20,000）
- 学术辅导（英领/博睿 · ¥12,000-30,000）
```

### 3.2 LLM 输出 Schema

```json
{
  "overview": "概述 80-120 字",
  "strengths": ["**GPA 3.6** 远超...", "..."],
  "concerns": ["**雅思 7.0** 距顶尖项目...", "..."],
  "summary": "总结 40-60 字，含下一步建议",
  "school_notes": [
    {"university": "香港大学", "major": "Computer Science", "note": "**85%概率** 主要靠背景匹配..."}
  ],
  "products": [
    {"name": "标准申请服务", "reason": "提供港新名校..."}
  ]
}
```

### 3.3 JSON 修复链（四级降级）

```
原始响应 → direct json.loads
  ↓ 失败
轻量正则修复（补逗号、去尾逗号）→ json.loads
  ↓ 失败
json_repair 库 → json.loads
  ↓ 失败
API repair：让 LLM 自己修复自己 → json.loads
```

### 3.4 学校名模糊匹配

LLM 输出的校名/专业名可能与预测数据有微小差异。`render_school_cards` 中的匹配策略：

1. 精确匹配
2. 空白字符规范化（合并空格）
3. `rapidfuzz.fuzz.ratio`（双阈值 0.80，校名和专业名分别匹配）

确保概率条和百分位数据不因命名差异丢失。

---

## 4. Profile 系统

`classify_profile()` 根据预测结果自动分档，每档有专属 System Prompt 语调：

| Profile | 条件 | 语调 |
|---------|------|------|
| `strong_elite` | avg ≥ 0.55, penalty ≤ 1 | 先肯定优势，再冷静指出差异化才是关键 |
| `medium_mixed` | avg ≥ 0.30, penalty ≤ 3 | 平和务实，帮客户建立合理预期 |
| `weak_gaps` | 其余 | 正面积极，每条 concern 后紧跟改进建议 |
| `cross_major` | 跨专业 ≥ 40% | 专业客观，区分"相似跨"和"大跨度跨" |

---

## 5. 流式渲染

`_stream_explain_content()` 中的渲染策略：

- **节流**：25ms 或累积 6 个新字符，任一满足即刷新
- **局部 JSON 提取**：`_try_extract_partial()` 用括号计数从半成品 JSON 中提取已完成字段
- **渐进揭示**：overview → strengths/concerns → summary → school_notes → products，每个字段首次出现时带 `ar-section-enter` 入场动画
- **降级**：流式失败/无输出时自动切换到同步 `agent.run()`

---

## 6. 缓存策略

### 会话缓存（session_state）
- 以 MD5(cache_key) 索引，cache_key 包含：profile、top5 结果、GPA、语言、经历摘要
- 命中时跳过 LLM 调用，直接渲染

### 磁盘持久化
- JSON 文件存储在 `.explain_cache/explain.json`
- LRU 上限 50 条，TTL 30 分钟
- 页面刷新或重启 Streamlit 后缓存仍有效

---

## 7. 按钮状态机

```
idle → 显示 "Pathfinder AI解读（beta）" 可点击
  ↓ 点击
generating → 按钮灰掉，文案变 "AI解读中..."
  ↓ 完成/异常
finally 清 flag → st.rerun() → 缓存命中直接出结果 + 按钮恢复
```

异常安全：`try/except/finally` 保证 `finally` 一定清 flag，按钮不会永久锁定。

---

## 8. 静态框架

### 五维雷达图
- 学术绩点（GPA/4.0 * 100，GMAT≥700 或 GRE≥320 +10%）
- 语言能力（raw/max * 100）
- 科研论文（research×0.6 + paper×0.4）/ 3 * 100
- 实习获奖（internship×0.5 + award×0.5）/ 3 * 100
- 学校水平（SCHOOL_LEVEL_SCORES 查表）

### 选校梯度
- 混合策略：绝对值阈值（≥0.70 较稳 / ≥0.40 适中 / <0.40 冲刺）+ Jenks 自然断点
- 两种方法不一致时用 Jenks，附加分布说明

### 产品匹配
- 根据 GPA、语言、经历、跨专业情况自动推荐 2-5 个产品
- 每个产品带名称、变体、价格区间、推荐原因

---

## 9. 院校对比（Phase 2）

`SchoolFeatureStats` 从历史 cases 中按 target_university 计算分位数分布：
- GPA、语言成绩、科研/实习/获奖/论文数量
- 每个学生的特征在该校录取者中的分位数 + 标签（"偏低"/"中等"/"较高"）
- 渲染在学校卡片底部，带颜色编码的进度条

---

## 10. 关键设计决策

- **按钮用 disabled 而非消失**：给用户明确的状态反馈，`finally` 保证恢复
- **学校名模糊匹配而非严格匹配**：LLM 输出的校名/专业名允许轻微偏差
- **流式节流双阈值**：时间 + 字符数，兼顾流畅和响应
- **四级 JSON 修复**：先快速本地修复，最后才求助 API
- **磁盘持久化缓存**：刷新不丢，30 分钟 TTL 防膨胀
