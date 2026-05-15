# Algorithm Lab 模块说明

## 1. 模块概述

`algorithm_lab` 是内部算法实验与性能基准测试套件，仅供管理员访问。包含 Quasi-Monte Carlo 效率测试、多目标优化（基于 vendored pymoo）、RapidFuzz 模糊匹配性能、Numba JIT 加速测试、NumPy 深度性能基准等。

### DS 视角

这些实验不是孤立的玩具——它们和 TODO-2（Monte Carlo 全拒率）直接相关。QMC vs MC 效率对比是联合概率模拟的前置研究：Sobol 序列比伪随机采样收敛更快，在做 N 个学校的联合录取概率估计时可以减少模拟次数。多目标优化（NSGA-II）和 TODO-4（Pareto 最优投入分配）直接挂钩——如果要做"GPA 提升 vs 实习 vs 语言"的最优投入建议，这就是基础。

**但要注意**：当前实验只测了性能（速度、收敛性），没测统计准确性。QMC 在录取概率这个特定分布上是否真的比 MC 好、NSGA-II 在只有 3 个目标维度时是否 overkill——这些需要针对具体业务场景验证，不能直接套结论。

## 2. 目录结构

```
algorithm_lab/
├── bench.py                 # 基准测试工具函数
├── fuzzy_test.py            # RapidFuzz 字符串匹配性能测试
├── monte_carlo.py           # QMC vs MC 效率对比
├── multi_obj_optimize.py    # 多目标优化测试（依赖 pymoo）
├── numba_test.py            # Numba JIT 加速对比
├── numba_test_func.py       # Numba JIT 测试函数
├── numpy_bench.py           # NumPy 深度性能基准
└── pymoo/                   # Vendored pymoo 0.6.0.1 fork（多目标优化库）
    ├── algorithms/          # 优化算法实现
    ├── core/                # 核心框架
    ├── problems/            # 标准测试问题集
    ├── operators/           # 交叉/变异/选择算子
    └── vendor/              # pymoo 自身依赖
```

## 3. 端到端流程

```
用户访问 pages/algorithm_lab.py
        │
        ├── init_page(..., admin_only=True, hide_sidebar=True)
        │
        └── st.tabs:
            ├── QMC test: run_qmc_efficiency_test()
            ├── Multi-objective: run_multi_obj_optimization_test()
            ├── RapidFuzz test: run_rapidfuzz_test()
            ├── Numba & NumPy: run_numba_acceleration_test()
            └── NumPy Deep Bench: run_numpy_performance_test()
```

## 4. 核心组件

### 4.1 monte_carlo.py

Quasi-Monte Carlo vs 标准 Monte Carlo 效率对比：
- Sobol/Halton 序列 vs 伪随机采样
- 收敛速度对比图表

### 4.2 multi_obj_optimize.py

多目标优化实验（基于 vendored pymoo）：
- NSGA-II / MOEA/D 等算法
- Pareto 前沿可视化

### 4.3 fuzzy_test.py

RapidFuzz 模糊匹配性能测试：
- 不同算法（ratio/partial_ratio/token_sort_ratio）准确率对比
- 大规模字符串匹配速度

### 4.4 numba_test.py / numpy_bench.py

- **Numba**：JIT 编译加速对比（纯 Python vs Numba）
- **NumPy**：向量化操作、内存布局、广播性能深度基准

### 4.5 pymoo/（第三方代码）

vendored 的多目标优化库 pymoo 0.6.0.1 fork（~318文件，~10,800 LOC），为算法实验提供优化框架。**此目录在治理规则中豁免行数限制。**

## 5. 依赖

- `streamlit`：UI
- `numpy`、`scipy`：数值计算
- `rapidfuzz`：模糊匹配
- `numba`：JIT 加速
- `matplotlib`：图表
- [Utils](src/utils/README.md) — `init_page()`、`SessionManager`
- Vendored `pymoo/`：多目标优化
