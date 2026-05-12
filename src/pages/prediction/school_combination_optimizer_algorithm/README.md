<!-- !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files. -->
# 选校组合优化器 (School Combination Optimizer)

基于 **NSGA-III** 多目标进化算法的选校策略自动生成引擎。输入候选学校-专业池 + 学生背景，输出 3 套最优申请方案（6所/9所/10所），每套方案在 5 个目标维度上同时达到 Pareto 最优。

## 目录结构

```
exported_optimizer/
├── school_combination_optimizer_algorithm/   # 核心算法
│   ├── __init__.py              # 公共 API 导出
│   ├── config.py                # 全局配置常量（阈值/权重/规则）
│   ├── problem.py               # NSGA-III 问题定义（5目标 × 多约束）
│   ├── filters.py               # 候选池预过滤（背景/样本量/优先级/保底）
│   ├── school_selector.py       # 平衡选校：reduce + generate（fallback用）
│   ├── metrics_calculator.py    # 方案评分（全拒率/多样性/平衡度等10项）
│   ├── monte_carlo.py           # Sobol QMC + Cholesky 相关性模拟
│   ├── utils.py                 # 工具：LRU缓存/prob裁剪/名称规范化
│   ├── visualizer.py            # Streamlit 可视化（仪表盘+学校表格）
│   └── optimizer/               # 优化器子模块（10文件）
│       ├── __init__.py          # 导出 SchoolSelectionOptimizer
│       ├── context.py           # OptimizationContext 数据类
│       ├── cache_manager.py     # 输入哈希 → 结果缓存
│       ├── filters_handler.py   # 优化前后过滤管道
│       ├── optimization_runner.py   # NSGA-III 参数动态计算 + 执行
│       ├── optimizer.py         # 主类 SchoolSelectionOptimizer（453行）
│       ├── metrics_calculator_wrapper.py  # 指标计算包装
│       ├── recommendation_builder.py      # 结果 → 推荐构建 + fallback
│       ├── school_adjuster.py   # 学校难度概率纠偏
│       ├── solution_selector.py # Pareto 前沿 → 最佳解选择
│       └── threshold_calculator.py  # 自适应阈值（冲/稳/保分界）
└── admission_probability_calculator_components/
    ├── optimization_ui.py       # Streamlit "智能优化" Tab UI
    └── optimization/            # 优化 UI 子组件（6文件）
        ├── __init__.py
        ├── cache_builder.py     # bg_target_similarity 缓存
        ├── optimization_executor.py  # 前端触发 → 优化器调用（365行）
        ├── pdf_generator.py     # 优化结果 → PDF 报告入口
        ├── result_handler.py    # 结果序列化/会话同步
        ├── ui_animation.py      # 加载动画
        └── ui_controls.py       # 参数控件（pop_size/n_gen/plan_config）
```

## 端到端流程

```
用户提交预测
    │
    ├── [Streamlit Tab: 手动选择]
    │   └── checkbox → 选校组合 → 全拒率/录取率/多样性计算
    │
    └── [Streamlit Tab: 智能优化]  ← 本模块核心
        │
        ├── 1. optimization_executor.py
        │   └── 从 session_state 获取候选池（similarity + cross_major results）
        │
        ├── 2. SchoolSelectionOptimizer.optimize()
        │   ├── build_optimization_input_hash → 查缓存
        │   ├── calculate_adaptive_thresholds → 自适应冲/稳/保分界
        │   ├── adjust_probability_by_university_difficulty → 学校难度纠偏
        │   └── for plan_config in [策略1(6所), 策略2(9所), 策略3(10所)]:
        │       │
        │       ├── _optimize_single_plan()
        │       │   ├── apply_all_filters()     → 背景过滤(样本量/优先级/学部/保底)
        │       │   ├── create_problem()        → SchoolSelectionProblem(5obj, 多约束)
        │       │   ├── run_optimization()      → NSGA-III (BinaryRandom+HUX+Bflip)
        │       │   ├── find_best_solution_indices() → Pareto 前沿最佳解
        │       │   └── build_final_recommendation()  → 后过滤 + 指标计算
        │       │
        │       └── 若无解 → get_fallback_recommendation()
        │           └── generate_balanced_selection() → 30/40/30 比例生成
        │
        ├── 3. visualize_recommendations()
        │   └── RecommendationVisualizer → 仪表盘(风险/信心/多样性) + 学校表格(logo/详情)
        │
        └── 4. pdf_generator.py → 触发 PDF 报告生成（见 exported_pdf/）
```

## 五个优化目标

| # | 目标 | 方向 | 权重 | 说明 |
|---|------|------|------|------|
| f1 | 全拒率最小化 | min | 0.9 | 所有学校全拒的概率 |
| f2 | 学校多样性最大化 | max | 5.0 | 不同大学的数量 |
| f3 | 申请策略平衡度最大化 | max | 1.0 | 冲/稳/保比例接近 30/40/30 |
| f4 | 专业相似度最大化 | max | 2.0 | 与背景专业的平均相似度 |
| f5 | 新增专业比例最大化 | max | 0.5 | 系统首次推荐的专业占比 |

## 约束条件（8个）

| 约束 | 说明 |
|------|------|
| max_schools | 不超过计划最大校数 |
| min_schools | 不低于计划最小校数 |
| min_reach | 至少含 N 所冲刺校 |
| min_target | 至少含 N 所目标校 |
| min_safety | 至少含 N 所保底校 |
| hk_violation | 港校数量限制（如适用） |
| min_top3 | 高背景学生至少 N 所 Top3 |
| min_top5 | 高背景学生至少 N 所 Top5 |

## 自适应阈值系统

不硬编码冲/稳/保分界——根据学生背景和候选池分布动态计算：

- **普通背景**: reach = P10, safety = P70
- **985/211/1-100**: reach = P20, safety = P60
- 概率在 safety 以上 → 保底；target_lower~safety → 目标；以下 → 冲刺

## 蒙特卡罗相关性模拟

- **Sobol QMC** 低差异序列替代纯随机采样
- **Cholesky 分解**处理学校间录取相关性（共享申请池）
- **自适应收敛**: 1000~10000 次模拟，convergence threshold = 0.01
- LRU 缓存（maxsize=256）避免重复计算

## 三套申请策略

| 策略 | 校数 | 适用场景 |
|------|------|----------|
| 策略1 | 6所 | 标准化申请，覆盖冲/稳/保 |
| 策略2 | 9所 | 扩展申请，更多选择 |
| 策略3 | 10所 | 最大覆盖，高背景学生 |

## Fallback 机制

NSGA-III 无可行解时自动降级为规则平衡生成：
1. 按概率分三类（冲/稳/保）
2. 按 30/40/30 理想比例从各类选取
3. 不足时从剩余候选中按概率补足

## 关键依赖

- `pymoo` — NSGA-III 算法框架
- `numpy`, `scipy` — 数值计算/QMC/Cholesky
- `streamlit` — UI 渲染
- `pandas` — 数据处理
- 内部模块: `school_level_service`, `app_data_loader`, `prediction_utils`
