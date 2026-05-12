# Experimental Modules Route Map

> 从已删除 commit (`faa1d83^`, ~2025-11-11) 恢复的实验性模块。
> 所有 .py/.md 文件顶部有 `# !!EXPERIMENTAL:` 或 `<!-- !!EXPERIMENTAL: -->` 标记，grep 可批量定位。

## 目录速览

```
src/pages/prediction/
├── school_combination_optimizer_algorithm/   ← 核心算法 (NSGA-III + Monte Carlo)
├── admission_probability_calculator_components/  ← UI 层 (Streamlit tab)
├── page_components/pdf_generation/           ← PDF 报告生成
├── EXPERIMENTAL_ROUTE.md                     ← 本文件

src/agent/
├── pdf_agent.py                              ← PDF AI Agent
└── pdf_prompts.py                            ← PDF Agent prompts

scripts/
└── generate_correlation_matrix.py            ← 从 cases.feather 生成 phi 相关系数矩阵

cache/
├── correlation_matrix.feather                ← 851×851 程序间录取相关性矩阵
└── pair_weight_matrix.feather                ← 每对程序的共现学生数 (n_ij)
```

## 关键入口

| 入口 | 路径 | 说明 |
|------|------|------|
| `run_monte_carlo_simulation` | `school_combination_optimizer_algorithm/monte_carlo.py` | Sobol QMC + Cholesky 相关性模拟，可独立使用 |
| `SchoolSelectionOptimizer` | `school_combination_optimizer_algorithm/optimizer/optimizer.py` | NSGA-III 主类，453行 |
| `filter_schools_by_faculty_rules` | `school_combination_optimizer_algorithm/filters.py` | 候选池预过滤 |
| `calculate_adaptive_thresholds` | `school_combination_optimizer_algorithm/optimizer/threshold_calculator.py` | 自适应冲/稳/保分界 |
| `visualize_recommendations` | `school_combination_optimizer_algorithm/visualizer.py` | Streamlit 可视化 |
| `OptimizationExecutor` | `admission_probability_calculator_components/optimization/optimization_executor.py` | 前端触发 → 优化器调用 |
| `PDFReportGenerator` | `page_components/pdf_generation/generators/pdf_report_generator.py` | PDF 报告主入口 |

## Monte Carlo 优化（面试素材）

在基础 Sobol QMC + Cholesky 之上做了两层优化：

### 1. Shrinkage — 小样本 phi 收缩

phi=0.58 但只有 2 个共同学生 → 不可靠。用 James-Stein 风格收缩：
```
phi_adjusted = phi_raw × n_ij / (n_ij + λ)    (λ=5)
```

`pair_weight_matrix` 记录每对程序的共现学生数 n_ij。低 n_ij 的 phi 被拉向 0（独立假设），高 n_ij 的 phi 几乎不变。

### 2. 分块 Cholesky — 稀疏矩阵降维

真实录取相关性矩阵高度稀疏（中位数 phi ≈ 0）。用 `scipy.sparse.csgraph.connected_components` 按 |phi| > 0.03 拆连通分量，每个分量独立跑低维 Sobol + Cholesky，全拒率 = 各分量乘积。

实际效果：30 个程序 → 26 个分量（24 个孤立点 + 2 个小分量），Sobol 维度从 30 降到 4。每个分量独立缓存（`_simulate_component_cached`，LRU 128），NSGA-III 迭代中重复调用几乎免费。

### 生成脚本

```bash
python scripts/generate_correlation_matrix.py                # 默认参数
python scripts/generate_correlation_matrix.py --min-samples 5  # 更严格过滤
```

从 `cases.feather` 构建 student-program 矩阵，用 10 个背景字段做 student key 去重，计算 pairwise phi 系数 + 共现样本量。输出 `cache/correlation_matrix.feather` 和 `cache/pair_weight_matrix.feather`。

## 依赖链

```
optimization_executor.py (UI 触发)
    ├── DataProcessor.prepare_optimizer_input()
    ├── SchoolSelectionOptimizer.optimize()
    │   ├── filters.py          → 候选池预过滤
    │   ├── threshold_calculator → 自适应阈值
    │   ├── school_adjuster      → 学校难度纠偏
    │   ├── NSGA-III (pymoo)    → 多目标进化
    │   ├── monte_carlo.py      → 全拒率模拟
    │   └── metrics_calculator  → 方案评分
    ├── visualize_recommendations()
    └── PDFGenerator            → PDF 报告
```

## Import 适配记录

从恢复代码到当前项目的 import 变更：

| 旧路径 | 新路径 | 原因 |
|--------|--------|------|
| `prediction_utils` | `core.utils` | 模块已拆分重命名 |
| `UNIVERSITY_DIFFICULTY_ORDER` | `DEFAULT_UNIVERSITY_DIFFICULTY_ORDER` | 常量改名 |
| `data_sort_config.top_result_school_order` | `data_sort_config` | 子模块合并到 `__init__` |

## 补的 Stub 模块

原 commit 中缺失、根据调用方推断重建的文件：

| 文件 | 提供 | 被谁依赖 |
|------|------|---------|
| `admission_probability_calculator_components/data_processor.py` | `DataProcessor.prepare_optimizer_input()` | `optimization_executor.py`, `optimization_ui.py` |
| `admission_probability_calculator_components/school_logo_loader.py` | `get_logo_path()` | `visualizer.py` |

## TODO（来自 TODO.md）

1. **实验 1**: NSGA-III vs 规则方案对比
2. **实验 2**: 自适应阈值 vs 固定阈值
3. **实验 3**: Monte Carlo 相关性模拟偏差量化 ← 最优先
4. **实验 4**: 顾问使用率埋点

详见 `school_combination_optimizer_algorithm/TODO.md`

## 管理命令

```bash
# 查看所有实验文件
grep -rl '!!EXPERIMENTAL' src/ agent/

# 一键删除所有实验文件
grep -rl '!!EXPERIMENTAL' src/ agent/ | xargs rm

# 去标记（正式集成后）
grep -rl '!!EXPERIMENTAL' src/ agent/ | xargs sed -i '/^# !!EXPERIMENTAL:/d'
```
