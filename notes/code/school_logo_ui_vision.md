# School Logo UI Vision — Hero + Path Finder + Per-Card Logo

> 设计目标：把预测结果从冷冰冰的表格升级为 AI 人格化叙事，Linear/Vercel 极客风 + 留学产品属性。

---

## 用法 ①：Hero 学校组叠层（强推，对应 Linear/Stripe 的"客户 logo wall"美学）

**形态**：预测完成后，结果区顶部出一行 hero summary：

```
┌────────────────────────────────────────────────────┐
│  [○][○][○][○][○]+2     AI 为你筛选了 7 所院校      │
│  ↑ 5 logo 圆形叠层      覆盖 3 个梯度，1 所保底    │
│  -8px 重叠              4 所目标 2 所冲刺          │
└────────────────────────────────────────────────────┘
```

### 视觉规则

- logo 圆形蒙版 32px + 2px 白色 ring + 8px 负 margin 叠层
- 默认全彩（不要灰度，留学产品灰色不吉利）
- 超过 5 个用 "+N" 的圆角小标补
- 右侧一句 AI 整合文案（用现有 ExplainAgent 输出）
- 整个 banner 极简：1px slate-200 描边 + 8px 圆角 + 浅底
- hover logo 微微上浮 2px

### 为什么惊艳

用户预测完看到的第一眼，是"AI 整理过了"的视觉总结，不是一张冷冰冰的表。这是 ChatGPT 那种"我看完所有信息后跟你说"的人格化感。

### 实现位置

- Python: `src/pages/prediction/result_display/hero_summary.py` — `render_hero_summary()`
- CSS: `assets/hk_style/50_ux.css` — `.hk-hero-summary`, `.hk-hero-logo`, `.hk-hero-headline` 等
- 调用点: `ResultsDisplay.display()` 第 203 行，在所有结果展示前调用

---

## 用法 ②：Path Finder 路径节点叙事（对"组合优化"卖点最强化）

**形态**：把 school_combination_optimizer_algorithm 输出从 dataframe 升级为水平路径图：

```
保底 ─────────●───────────●──────────●─────── 冲刺
              [○]          [○]          [○]
              68%          43%          21%
              港城 CS      港中文 DS    港大 MSAI
```

### 视觉规则

- 一条水平 track（slate-100 底 + cyan 渐变 fill 表示概率梯度）
- 每个学校 logo 圆形 36px 钉在 track 上
- logo 下方紧贴："概率%" + "学校名" + "专业" 三行
- logo 之间用细虚线连接，提示"组合"关系
- hover logo 弹出该学校的 trace waterfall（已有 `51_trace.css`）

### 为什么惊艳

把"组合方案"从表格变成一条叙事时间线——保底→目标→冲刺一目了然，且每个 logo 都是"AI 帮你定位的坐标点"。对应 resume 里"path finder 算法"，面试就能开口讲。

### 待实现

- [ ] Path Finder 组件（新文件 `result_display/path_finder.py`）
- [ ] CSS: track、节点、虚线连接线
- [ ] 集成到 `ResultsDisplay.display()` 流程（hero → path finder → tables → trace）

---

## 用法 ③：概率条尾部 logo 终点旗（最便宜，单卡片层面）

**形态**：每张学校结果卡片的概率条尾部贴一个迷你 logo：

```
[○] 港中文 · 数据科学                          43%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━[○]━━━━━━━━━━
    （cyan fill 0~43%）          ↑ logo 标记终点
```

### 视觉规则

- logo 24px 圆形 + 1px slate-200 描边
- 概率条 6px 高（保留现有 `.hk-result-prob-bar`）
- logo 用 `position: absolute; left: {prob}%`
- 进入时 0 → P% 缓动动画（cubic-bezier 0.6s），logo 跟着滑

### 为什么惊艳

每条概率从抽象数字变成"距离这所学校还有多远"的具象旅程。但比 ① ② 工作量小很多。

---

## 推荐组合

考虑 Linear/Vercel 极客风 + 求职作品定位：

**强推 ① + ②**：

- ① 是"开场总结"——用户看到结果第一秒就有"AI 在帮我做事"的视觉冲击 ✅ 已实现
- ② 是"深度展开"——组合优化算法可视化，对应简历里的"path finder 算法"
- ① 改动最小（一个新 component + 几条 CSS），② 工作量大但是项目级的差异化亮点

③ 作为补充，每个学校卡片层面用，让"分散的卡片们"也有 logo 视觉一致性。

忽略：DataFrame 内嵌 logo（streamlit 默认 ImageColumn 不够高级）。

---

## 当前进度

| 用法 | 状态 | 文件 |
|------|------|------|
| ① Hero logo wall | ✅ 已实现 | `hero_summary.py` + `50_ux.css` |
| ② Path Finder | ⬜ 待实现 | — |
| ③ Per-card logo | ⬜ 待实现 | — |
