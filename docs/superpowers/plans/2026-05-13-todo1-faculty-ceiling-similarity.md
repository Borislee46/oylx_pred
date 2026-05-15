# TODO-1 实施：Faculty 三分级 + 天花板分段 + 相似度阈值

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faculty penalty 从硬编码 ×0.3 升级为三级 severity + 天花板从固定 70% 改为按惩罚层数分段 + 相似度阈值修正

**Architecture:** 三处独立改动，各自有明确的配置入口。Faculty severity 在 `faculty_filters.py` 新增 severity dict + lookup 函数；天花板在 `arbitrator.py` 按活跃惩罚层数查分段 ceiling；相似度阈值在 `config.py` 一行改值。验证统一跑 `test_calibration_report.py`。

**Tech Stack:** Python, pytest

**预期效果:** ECE 0.115 → <0.10，pred-actual gap -9pp → -5pp 以内

---

### Task 1: 定义 Faculty severity 映射 + 配置常量

**Files:**
- Modify: `src/pages/prediction/result_modifier/faculty_filters.py`
- Modify: `src/pages/prediction/result_modifier/config.py`

- [ ] **Step 1: 在 config.py 添加三分级常量**

在 `FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR = 0.3` 下方添加：

```python
# Faculty penalty severity levels (B scheme — three-tier cross-faculty distance)
# Light (×0.7): quantitative/methodological bridges exist
# Medium (×0.5): partial overlap, applied path possible
# Heavy (×0.3, default): fundamental domain switch, extremely rare
FACULTY_PENALTY_LIGHT: float = 0.70
FACULTY_PENALTY_MEDIUM: float = 0.50
FACULTY_PENALTY_HEAVY: float = 0.30
```

- [ ] **Step 2: 在 faculty_filters.py 新增 severity 字典和 lookup 函数**

在 `CROSS_FACULTY_RULES` 定义之后添加：

```python
# Cross-faculty severity for pairs NOT in the whitelist.
# Only light and medium pairs are listed; unlisted out-of-scope pairs default to heavy (×0.3).
# Design rationale: see DECISIONS.md DEC-002 — faculty distance is not binary.
CROSS_FACULTY_SEVERITY: dict[tuple[str, str], str] = {
    # ── Light (×0.7): quantitative/methodological bridges ──
    ("理学院", "社会科学院"): "light",   # quantitative social science
    ("理学院", "医学院"): "light",       # pre-med, biochemistry bridge
    ("理学院", "教育学院"): "light",     # science education
    ("工程学院", "经济金融学院"): "light",  # financial engineering
    ("工程学院", "医学院"): "light",     # biomedical engineering
    ("计算机学院", "经济金融学院"): "light",  # fintech
    ("计算机学院", "社会科学院"): "light",  # computational social science
    ("计算机学院", "设计学院"): "light",  # HCI / UX
    ("计算机学院", "医学院"): "light",   # health informatics
    ("计算机学院", "教育学院"): "light",  # educational technology
    ("计算机学院", "艺术学院"): "light",  # digital art, game design
    ("社会科学院", "商学院"): "light",   # management, organizational behavior
    ("社会科学院", "法学院"): "light",   # socio-legal studies
    ("商学院", "经济金融学院"): "light",  # close disciplinary overlap
    ("商学院", "计算机学院"): "light",   # business analytics, MIS
    ("商学院", "社会科学院"): "light",   # organizational behavior
    ("商学院", "教育学院"): "light",     # education management
    ("艺术学院", "商学院"): "light",     # arts management / creative industries
    ("医学院", "理学院"): "light",       # biomedical research → basic science
    ("医学院", "工程学院"): "light",     # biomedical engineering
    ("医学院", "计算机学院"): "light",   # medical AI, bioinformatics
    ("教育学院", "商学院"): "light",     # education management
    ("教育学院", "计算机学院"): "light",  # educational technology
    ("教育学院", "社会科学院"): "light",  # education policy / sociology of education
    ("设计学院", "计算机学院"): "light",  # HCI / interaction design
    ("设计学院", "商学院"): "light",     # design management
    ("建筑学院", "计算机学院"): "light",  # computational design / smart cities
    # ── Medium (×0.5): partial overlap ──
    ("理学院", "法学院"): "medium",       # IP / patent law
    ("理学院", "经济金融学院"): "medium",  # quant finance (different focus)
    ("理学院", "建筑学院"): "medium",     # structural knowledge
    ("工程学院", "社会科学院"): "medium",  # engineering management
    ("工程学院", "教育学院"): "medium",   # engineering education
    ("工程学院", "艺术学院"): "medium",   # industrial design
    ("工程学院", "设计学院"): "medium",   # product design
    ("计算机学院", "建筑学院"): "medium",  # computational design
    ("社会科学院", "医学院"): "medium",   # public health / medical sociology
    ("社会科学院", "计算机学院"): "medium",  # computational methods (limited)
    ("商学院", "法学院"): "medium",       # business law
    ("商学院", "医学院"): "medium",       # healthcare management
    ("文学院", "法学院"): "medium",       # legal history / language
    ("文学院", "计算机学院"): "medium",   # digital humanities
    ("文学院", "社会科学院"): "medium",   # cultural studies → sociology
    ("文学院", "商学院"): "medium",       # corporate communication
    ("艺术学院", "计算机学院"): "medium",  # creative coding
    ("艺术学院", "社会科学院"): "medium",  # cultural policy
    ("艺术学院", "教育学院"): "medium",   # art education
    ("法学院", "商学院"): "medium",       # corporate law
    ("法学院", "社会科学院"): "medium",   # legal theory
    ("医学院", "社会科学院"): "medium",   # medical sociology / public health
    ("医学院", "教育学院"): "medium",     # medical education
    ("建筑学院", "商学院"): "medium",     # real estate
    ("设计学院", "社会科学院"): "medium",  # design research
    ("设计学院", "教育学院"): "medium",   # design education
}


def get_cross_faculty_penalty_factor(
    background_faculty: str | None,
    target_faculty: str | None,
) -> float:
    """Return the penalty multiplier for a cross-faculty pair.

    Returns 1.0 if the pair is in-scope (no penalty).
    For out-of-scope pairs, returns 0.70 (light) / 0.50 (medium) / 0.30 (heavy)
    based on CROSS_FACULTY_SEVERITY, defaulting to heavy (0.30) if not listed.
    """
    from src.pages.prediction.result_modifier.config import (
        FACULTY_PENALTY_HEAVY,
        FACULTY_PENALTY_LIGHT,
        FACULTY_PENALTY_MEDIUM,
    )

    if not background_faculty or not target_faculty:
        return 1.0

    bg = background_faculty.strip()
    tg = target_faculty.strip()

    if not bg or not tg:
        return 1.0

    # In whitelist → no penalty
    allowed = get_allowed_target_faculties(bg)
    if tg in allowed:
        return 1.0

    # Out of scope → lookup severity
    severity = CROSS_FACULTY_SEVERITY.get((bg, tg))
    if severity == "light":
        return FACULTY_PENALTY_LIGHT
    elif severity == "medium":
        return FACULTY_PENALTY_MEDIUM
    else:
        return FACULTY_PENALTY_HEAVY
```

- [ ] **Step 3: 更新 __init__.py 导出新函数**

检查 `src/pages/prediction/result_modifier/__init__.py` 是否需要导出新函数。

---

### Task 2: 调整 pipeline 使用 severity lookup

**Files:**
- Modify: `src/pages/prediction/result_modifier/adjustment_pipeline.py:158-168`

- [ ] **Step 1: 修改 Layer 3 调用**

将 `adjustment_pipeline.py:158-168` 从：

```python
if ctx.background_faculty:
    target_faculty = result.get("faculty")
    if is_faculty_out_of_scope(ctx.background_faculty, target_faculty):
        arbitrator.add_factor(
            AdjustmentFactor(
                name="Faculty Out of Scope Penalty",
                value=1.0 - FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR,
                factor_type=AdjustmentFactorType.PENALTY,
                description="申请学部跨度过大",
            )
        )
```

改为：

```python
if ctx.background_faculty:
    target_faculty = result.get("faculty")
    if is_faculty_out_of_scope(ctx.background_faculty, target_faculty):
        penalty_factor = get_cross_faculty_penalty_factor(
            ctx.background_faculty, target_faculty
        )
        # Determine severity label for trace description
        severity_label = {
            0.70: "轻度",
            0.50: "中度",
            0.30: "重度",
        }.get(penalty_factor, "重度")
        arbitrator.add_factor(
            AdjustmentFactor(
                name="Faculty Out of Scope Penalty",
                value=1.0 - penalty_factor,
                factor_type=AdjustmentFactorType.PENALTY,
                description=f"申请学部跨度{severity_label}（×{penalty_factor:.2f}）",
            )
        )
```

- [ ] **Step 2: 更新 import**

在 `adjustment_pipeline.py` 顶部 import 中加入 `get_cross_faculty_penalty_factor`：

```python
from src.pages.prediction.result_modifier.faculty_filters import (
    is_faculty_out_of_scope,
    get_cross_faculty_penalty_factor,
)
```

同时可以移除 `FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR` 的 import（如果不再被其他地方使用则保留兼容）。

---

### Task 3: 天花板分段函数

**Files:**
- Modify: `src/pages/prediction/result_modifier/config.py`
- Modify: `src/pages/prediction/result_modifier/arbitrator.py`

- [ ] **Step 1: 在 config.py 添加分段 ceiling 配置**

在 `MAX_TOTAL_PENALTY_RATIO = 0.7` 下方添加：

```python
# 按活跃惩罚层数分段的天花板（替换固定 MAX_TOTAL_PENALTY_RATIO）。
# V5发现: 2-3层惩罚的case ECE最差（>0.2），单层和4+层反而更好。
# 单层惩罚不应打折，多层叠加需更严格的上限。
# key: n_active_penalties (1-indexed), value: max penalty ratio
# 3层以上统一用 0.45。
PENALTY_CEILING_BY_LAYERS: dict[int, float] = {
    1: 0.70,
    2: 0.55,
    3: 0.45,
    # 4+: 0.45 (fallback in code)
}
```

- [ ] **Step 2: 修改 arbitrator.py 使用分段 ceiling**

修改 `arbitrate()` 方法，将 `MAX_TOTAL_PENALTY_RATIO` 替换为动态 ceiling：

在方法开头（penalties 分离之后）添加：

```python
n_penalties = len(penalties)
penalty_ceiling = PENALTY_CEILING_BY_LAYERS.get(n_penalties, 0.45)
```

然后将方法中所有 `MAX_TOTAL_PENALTY_RATIO` 替换为 `penalty_ceiling`（共 3 处）。

更新 import：
```python
from src.pages.prediction.result_modifier.config import (
    ARBITRATION_MIN_PROBABILITY,
    BOOST_DECAY_FACTOR,
    MAX_TOTAL_BOOST_RATIO,
    PENALTY_CEILING_BY_LAYERS,
    PENALTY_DECAY_FACTOR,
)
```

- [ ] **Step 3: 更新 arbitrator 的 docstring 注释**

将第 40 行的 `MAX_TOTAL_PENALTY_RATIO (70%)` 替换为 `PENALTY_CEILING_BY_LAYERS`。

---

### Task 4: 相似度阈值修正

**Files:**
- Modify: `src/pages/prediction/result_modifier/config.py`

- [ ] **Step 1: 修改阈值**

```python
# Before:
MIN_SIMILARITY_THRESHOLD: float = 0.89
# After:
MIN_SIMILARITY_THRESHOLD: float = 0.87
```

V9 审计发现 0.88-0.89 区间的录取率（29.8%）高于 0.89-0.90（29.0%），阈值在这个位置是反向的。

---

### Task 5: 验证 — 跑校准报告

**Files:**
- Test: `tests/data_quality/test_calibration_report.py`

- [ ] **Step 1: 运行校准报告**

```bash
pytest tests/data_quality/test_calibration_report.py -s -v
```

- [ ] **Step 2: 记录关键指标**

关注输出中的：
- ECE: 目标 < 0.10（当前 0.1155）
- 系统性偏差 (pred - actual): 目标 < 5pp（当前 -9pp）
- 分层校准: C9/985/211-双非 各自的 Brier 和偏差
- Reliability diagram: 各 bin 的 gap 是否缩小

---

### Task 6: 回归测试

- [ ] **Step 1: 跑全量 data quality tests**

```bash
pytest tests/data_quality/ -v
```

确保 62 tests 全部通过，没有因参数变更引起的意外 break。

---

### Task 7: Commit

- [ ] **Step 1: Commit 所有改动**

```bash
git add src/pages/prediction/result_modifier/config.py
git add src/pages/prediction/result_modifier/faculty_filters.py
git add src/pages/prediction/result_modifier/adjustment_pipeline.py
git add src/pages/prediction/result_modifier/arbitrator.py
git commit -m "feat: Faculty severity 3-tier + per-layer penalty ceiling + similarity threshold fix

- Faculty penalty: ×0.3 → 3-tier severity (light 0.7 / medium 0.5 / heavy 0.3)
- Ceiling: fixed 70% → per-layer (1L:70%, 2L:55%, 3+L:45%)
- Similarity threshold: 0.89 → 0.87 (V9 found inversion at 0.89)

V5 ablation showed Faculty is #1 calibration killer (ECE 0.15→0.11 when removed).
37% cases hit the 70% ceiling; 2-3 layer cases had worst ECE.
Expected ECE improvement: 0.115 → <0.10."
```
