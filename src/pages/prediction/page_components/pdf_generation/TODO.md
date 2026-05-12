<!-- !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files. -->
# PDF TODO — 渲染层替换、产品化路线

> 当前状态：代码已从 GitLab 历史中恢复（`faa1d83^`），核心架构完整，但 reportlab 渲染层维护成本过高。
> 核心理念：架构三层不变，只换渲染引擎。

---

## 一、为什么删掉的

| 问题 | 具体表现 |
|------|---------|
| 中文字体地狱 | `STSong-Light` + `UnicodeCIDFont` 在 Linux 服务器直接乱码，不同 OS 行为不一致 |
| 布局调试成本 | reportlab platypus 的 Table/Paragraph/Spacer 组合，调一个 margin 要改 3 处 |
| 分页不可控 | `CondPageBreak(3.0*inch)` 是猜出来的，A4 实际渲染和预览不一致 |
| 调整成本 >> 用户感知价值 | 花半天调 watermark 旋转角度，顾问只看到"哦有 PDF 下载" |

**但删掉的是渲染层，不是架构。** 下面的三层是干净的、可复用的：

---

## 二、当前架构：三层分离（删之前已经在做的）

```
┌─────────────────────────────────────────┐
│  UI 层                                   │
│  pdf_download_section.py                │
│  st.download_button(label, data, mime)  │
├─────────────────────────────────────────┤
│  数据提取层                              │
│  pdf_data_extractor.py                  │
│  session_state → 结构化 dict            │
├─────────────────────────────────────────┤
│  内容编排层                              │
│  PDFSectionBuilder (section_builder/)   │
│  封面 → 背景 → 策略 → 学校详情          │
│  PDFAIAgent / PDFAgent (DeepSeek)       │
│  雷达图 matplotlib                      │
├─────────────────────────────────────────┤
│  渲染层 ← 问题在这一层                    │
│  reportlab: pdf_report_generator.py     │
│  reportlab: pdf_styler.py               │
│  UnicodeCIDFont, platypus, TableStyle   │
└─────────────────────────────────────────┘
```

数据提取层和内容编排层不依赖 reportlab——它们产出的是 story list（platypus 对象），改成产出 HTML string 就能切换到 weasyprint。

---

## 三、渲染层替代方案对比

| 方案 | 中文 | 样式调试 | 部署 | 适合你的场景 |
|------|------|---------|------|------------|
| **weasyprint** | 零配置，系统字体 | CSS，浏览器 DevTools 调 | 需要 Cairo 系统库 | ⭐⭐⭐ 最推荐 |
| **Playwright print** | 零配置 | CSS，完全和页面一致 | 需要 Chromium | ⭐⭐ 太重但效果最好 |
| **xhtml2pdf** | 一般 | CSS 子集，部分不支持 | pip install | ⭐ 比 reportlab 好但不够 |
| reportlab (保留) | 看 OS 脸色 | platypus 硬调 | pip install | ❌ 当前方案 |

**建议**：weasyprint。HTML→PDF，CSS `@page` 控制分页，中文用 `font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif` 就行。

---

## 四、迁移任务清单

### 任务 1：渲染层 POC — weasyprint 最小可行

**目标**：一个 py 文件，输入用户数据 dict，输出 PDF bytes，能跑通封面+背景页。

**步骤**：
1. 写一个 HTML 模板字符串，包含封面 + 背景分析
2. CSS `@page` 控制 A4 尺寸 + 页边距
3. CSS `position: fixed` 做页眉页脚 + 水印
4. `weasyprint.HTML(string=html).write_pdf()` 输出

**验证**：中文不乱码，封面和背景在不同页。

**预估**：半天。

**建议文件**：`experiments/pdf/exp1_weasyprint_poc.py`

---

### 任务 2：迁移现有内容编排层

**目标**：把 section_builder 的 5 个 builder 从 reportlab platypus → HTML string。

**改动范围**：

| Builder | 当前输出 | 目标输出 |
|---------|---------|---------|
| CoverPageBuilder | platypus Table+Paragraph | HTML div + CSS |
| BackgroundSectionBuilder | platypus Table + matplotlib Image | HTML table + img(base64) |
| SchoolDetailBuilder | 同上 | 同上 |
| OptimizationStrategiesSectionBuilder | 同上 | 同上 |
| AnalystNotesGenerator | AI text → Paragraph | AI text → HTML div |

**注意事项**：
- matplotlib 雷达图用 `base64` 嵌入 `<img src="data:image/png;base64,...">`
- 学校 logo 同理
- 表格用 HTML `<table>` + CSS，比 reportlab TableStyle 好调 10 倍

**预估**：1-2 天。

---

### 任务 3：水印 + 页眉页脚

**目标**：CSS 实现原 reportlab 水印效果。

```css
.watermark {
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  background: repeating-linear-gradient(
    35deg,
    transparent,
    transparent 180px,
    rgba(0,0,0,0.06) 180px,
    rgba(0,0,0,0.06) 220px
  );
  pointer-events: none;
}
.watermark::after {
  content: "EasyApply " attr(data-date) " " attr(data-user);
  /* ... 平铺逻辑用 CSS grid 或 JS 生成 */
}
```

纯 CSS 水印可能需要 JS 辅助生成平铺文本，但比 reportlab Canvas.translate + rotate 好调一个数量级。

**预估**：1-2 小时。

---

### 任务 4：分页控制

**目标**：策略段不分页断裂，封面独立一页。

```css
@page {
  size: A4;
  margin: 2cm 1.8cm 2cm 1.8cm;
  @bottom-center {
    content: "Page " counter(page);
    font-size: 8pt;
    color: #999;
  }
}

.cover-page {
  page-break-after: always;
}

.keep-together {
  page-break-inside: avoid;
}
```

**预估**：1 小时。

---

### 任务 5：下载按钮 + 缓存逻辑（基本不动）

`pdf_download_section.py` 和 `PDFCacheManager` 不依赖渲染层——它们只管 byte 流的存取和下载触发。迁移后可以直接复用。

---

## 五、要不要保留的东西

| 保留 | 原因 |
|------|------|
| `PDFDataExtractor` | 数据提取层，和渲染无关，干净 |
| `PDFSectionBuilder` 编排逻辑 | 改输出格式即可 |
| `PDFAIAgent` / `PDFAgent` | DeepSeek 生成分析文案，独立模块 |
| `PDFCacheManager` | 通用缓存，不依赖 reportlab |
| `pdf_config.py` (PDFConfig dataclass) | 配置集中管理，改 CSS 也能用 |
| `radar.py` (matplotlib) | 改成输出 base64 PNG 即可 |

| 砍掉 | 原因 |
|------|------|
| `pdf_report_generator.py` (WatermarkedDocTemplate) | reportlab 核心，全部替换 |
| `pdf_styler.py` (PDFStyler) | ParagraphStyle 替换为 CSS class |
| `pdf_download_section.py` 里的 reportlab import | 移除依赖 |

---

## 六、面试叙事角度

### 产品经理面试

> "我们的系统预测了录取概率，但顾问需要一个能拿给家长看的东西。我设计了一套自动生成 PDF 报告的系统——有封面、有背景分析雷达图、有个性化 AI 建议、有水印防泄露。从用户反馈来看，有这个 PDF 之后顾问对系统的信任度明显提升，因为他不需要自己再写报告了。"

加分点：用户洞察（顾问要拿给家长看）→ 功能设计（报告分段）→ 技术实现（AI + 数据可视化）。

### DS 面试

不强推这个模块。但如果被问到"工程能力"，可以提：

> "我搭了一个可扩展的 PDF 报告生成框架，数据提取、内容编排、渲染三层分离。后来因为 reportlab 中文字体维护成本高，我把渲染层从 reportlab 迁移到了 weasyprint，上面两层代码没动。"

加分点：架构设计 + 技术选型权衡。

---

## 七、优先级

| 优先级 | 任务 | 工时 | 原因 |
|--------|------|------|------|
| P0 | 任务 1：weasyprint POC | 0.5d | 先跑通再决定是否全面迁移 |
| P1 | 任务 2：迁移 section builders | 1-2d | 内容编排层是核心 |
| P1 | 任务 3+4：水印+分页 | 0.5d | CSS 搞定 |
| P2 | 任务 5：集成测试 | 0.5d | 端到端验证 |

总计约 3 天，从 reportlab 完全迁移到 weasyprint。如果只做 P0 验证可行性，半天。
