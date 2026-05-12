# V3: 阈值敏感性分析 (Cost-based Threshold)

## 方法

定义 cost ratio `r = cost_FP / cost_FN`，对 201 个 threshold 扫描，找到每个 r 下总代价最小的 threshold。

## 结果

| r | 业务含义 | 最优 threshold | Precision | Recall | 正例率 |
|---|---------|---------------|-----------|--------|--------|
| 0.2 | 漏掉机会代价是推错的 5 倍 | **0.16** | 0.404 | 0.918 | 76.7% |
| 0.5 | 漏掉机会代价是推错的 2 倍 | **0.31** | 0.523 | 0.694 | 44.8% |
| 1.0 | 代价相等 (≈F1最优) | 0.60 | 0.682 | 0.359 | 17.8% |
| 2.0 | 推错代价是漏掉的 2 倍 | **0.68** | 0.763 | 0.183 | 8.1% |
| 5.0 | 推错代价是漏掉的 5 倍 | **0.71** | 0.875 | 0.044 | 1.7% |

## 关键发现

### 1. Production threshold=0.24，当前测试集真正 F1 最优≈0.28

训练时的 F1 最优是 0.24，当前测试集上偏移到了 0.28——这是正常的轻微 calibration drift。F1 隐含假设 FP 和 FN 代价相等，但实际业务中很少如此。

### 2. Threshold 的选择区间是 0.16 → 0.71 (4.4x)

取决于业务方如何权衡两种错误：
- 如果销售导向（宁可多申）：threshold 应该更低（0.16-0.24）
- 如果信任导向（宁缺毋滥）：threshold 应该更高（0.50-0.71）

### 3. "问业务方更怕哪个" 比 "从 F1 推导" 更正确

**面试叙事**：
> "我的 production threshold 是 0.24，但这不是从 F1 直接推出来的——我做了 cost-based sensitivity analysis，证明 threshold 应该由业务方拍板。如果你们更怕'推荐一个不该申的学校让申请季报销'，threshold 应该升到 0.50+；如果更怕'漏掉一个能申的机会'，保持 0.24 是对的。这不是 ML 问题，是产品决策问题。"

### 4. Precision-Recall 空间中各 ratio 的位置分布

r<1 的点聚集在高 recall 区（右下），r>1 的点聚集在高 precision 区（左上）。F1 最优点（r=1）落在中间偏右的位置——说明在偏态分布下，F1 本身就偏向 recall。

## 产物

- `cost_threshold_curve.png` — 4 面板综合分析
- `precision_recall_tradeoff.png` — PR 空间中的最优阈值
- `threshold_sensitivity.json` — 完整数据
- `run_threshold_analysis.py` — 可复现脚本
