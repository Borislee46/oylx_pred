# EasyApply 留学择校系统

本数据产品提供一套从“模型训练 → 表单校验与归一 → 预测编排 → 结果调整”的端到端解决方案。

**核心亮点**：
1.  **极致性能**：极低的 I/O 和开销，几乎 0 成本实现秒级推理。
2.  **专家级效果**：预测结果与业务专家对齐，逼近人工选校水平。
3.  **灵活扩展**：部分边界 Case 可用 Agent 解决，或在资源充足时完全替换为 Agent。

## 整站应用架构

- **形态**: Streamlit 多页应用；门户入口为根目录 `main.py`（`streamlit run main.py`）。
- **认证与门禁**: E2 OAuth 回调、`src/utils/page_init.py` 登录与页面初始化、按邮箱模块权限（`src/utils/auth/permission_checker.py`）；维护模式等由配置控制。
- **路由与实现分层**:
    *   `pages/*.py`：Streamlit 识别的页面路由，宜保持薄封装。
    *   `src/pages/`：各业务模块实现（预测在 `src/pages/prediction/`，人力看板等在对应子目录）。
    *   `src/utils/`：认证、配置、数据加载、日志等共享能力。
- **门户可挂载模块**（权限开通后可见，与 `main.py` 中按钮一致）: EasyApply 留学择校 (`pages/hk.py`)、平台使用指南、权限管理 / `algorithm_lab`（管理员）、人力薪资数据看板、人力绩效数据看板、人力结构数据看板；另含外链案例库入口。

下文「架构流程」以 **EasyApply 预测子系统** 为主；其他模块详见各 `pages/*.py` 与 `src/pages/` 下实现。

## 架构流程

1.  **模型训练 (Offline)**: `src/machine_learning_models/`
    *   **核心**: 使用 XGBoost 结合 SMOTE/SMOTENC 处理不平衡数据，并进行概率 Sigmoid 校准。
    *   **产物**: XGBoost 主模型 (`.ubj`)、TF-IDF 文本相似度模型。

2.  **专业相似度预计算 (Offline)**: `scripts/precompute_similarities.py`
    *   **核心**: 使用 E5 Embedding 模型预计算背景专业-目标专业相似度缓存。
    *   **产物**: `cache/background_target_similarity.feather`。

3.  **用户表单 (Online)**: `src/pages/prediction/input_form_components/`
    *   **核心**: `FormStateManager` 管理状态（自动保存/去重），`FormValidator` 进行严格校验与智能转换。
    *   **特色**: 跨学院拦截、目标自动扩展、海外院校语言成绩处理。

4.  **预测编排 (Online)**: `src/pages/prediction/`
    *   **入口**: 页面管线 (`src/pages/prediction/flow/pipeline.py`)；类 JSON 编排见 `src/pages/prediction/api/json_api.py`（进程内、非独立后端，便于后续从 Streamlit 解耦）。
    *   **核心**: 组合生成 → 并行推理 → 推荐生成 → 后处理 → 合并去重。
    *   **结果**: 相似专业推荐、跨专业推荐、用户指定结果。

5.  **结果调整 (Online)**: `src/pages/prediction/result_modifier/`
    *   **核心**: 可控后处理（GPA/语言惩罚、职业型降权、跨专业惩罚、TF-IDF 文本加成）。

---

## 1. 训练模块

**路径**: `src/machine_learning_models`

*   **数据与特征**:
    *   目标列: `admitted` (二分类)
    *   特征: 分类列 (`category` 编码)、计数列 (截尾+log1p)、语言分 (归一化)。
*   **采样**: SMOTE/SMOTENC 动态采样，异常回退。
*   **样本权重**: 文本为空样本降权、最近样本加权、采样权重对齐。
*   **训练与校准**: XGBoost + 单调约束 + `CalibratedClassifierCV` (sigmoid, prefit)。
*   **文本加成训练**: 生成 TF-IDF 向量器、质心和权重文件。

**详细文档**: [机器学习训练管线文档](docs/ml_training_api.md)

## 2. 表单校验与归一

**路径**: `src/pages/prediction/input_form_components`

*   **校验器**: `FormValidator` 提供详细的中文错误提示。
*   **状态管理**: `FormStateManager` 实现自动保存（节流 + 快照 hash）。
*   **跨学院拦截**: `src/pages/prediction/cross_faculty_guard.py` 风险识别与弹窗确认。
*   **GPA 转换**: 优先按院校/国家规则 (`config/gpa_conversion_rules.json`)，否则线性缩放。
*   **语言分数**: 托福/雅思互转与归一化；海外院校选填处理。
*   **组件服务**: 四级联动筛选 (`target_options_service.py`)。

**详细文档**: [表单组件与校验 API 文档](docs/input_form_components_api.md)

## 3. 核心预测流程 (Online)

**路径**: `src/pages/prediction`

*   **流程编排与预警 (Flow Control)**:
    *   入口管线: `flow/pipeline.py`
    *   风险守卫: `cross_faculty_guard.py` (拦截异常跨学院申请)
*   **数据准备与召回 (Preparation)**:
    *   数据归一化: `form_normalizer.py`
    *   混合召回: `preparer.py` 实现 E5 Embedding 向量召回与 Fuzzy 模糊匹配。
*   **核心推理与平准 (Execution)**:
    *   精排推理: `prediction_execution/executor.py` 驱动并行 XGBoost 推理。
    *   结果平准: `flow/processor.py` 结合 `BoundaryCaseAgent` 进行相似推荐与边界探索。
*   **接口方式**:
    *   Streamlit UI 交互: `src/pages/prediction/flow/pipeline.py`
    *   `api/json_api.py`: 与 UI 同进程的预测编排封装（内存任务存储等），非生产级 HTTP 服务；若独立部署需另起服务层与持久化任务状态。

**详细文档**: [预测模块 API 文档](docs/prediction_api.md)

## 4. 概率修正流水线 (Modification)

**路径**: `src/pages/prediction/result_modifier`

*   **调整驱动**: 基于 `adjustment_pipeline.py` 的二级修正流水线。
*   **概率修正因子 (Arbitration)**:
    *   **基础修正**: `ProbabilityAdjuster` 处理 GPA/语言惩罚、院校降权。
    *   **业务逻辑**: `CrossMajorPenalty` 处理跨专业惩罚。
    *   **NLP 提升**: `TextBoostProvider` 实现基于 TF-IDF 的文本 Logit Uplift。
*   **文本加成机制**:
    *   包含门控、平滑、动态封顶机制，防止加成过度。

**详细文档**: [结果修正模块文档](docs/result_modifier_api.md)

## 5. 专业相似度预计算

**脚本**: `scripts/precompute_similarities.py`

*   **模型**: Multilingual E5 Instruct。
*   **功能**: 预计算背景专业-目标专业相似度，加速线上查询。
*   **缓存**: 生成 `cache/background_target_similarity.feather`。

**详细文档**: [专业相似度预计算文档](docs/major_similarity_precompute.md)

---

## 配置清单

*   `config/app_config.json`: 应用级配置。
*   `config/dev_config.json`: 开发环境配置。
*   `config/gpa_conversion_rules.json`: GPA 分制规则。
*   `config/similarity_adjustment_rules.json`: 相似度微调规则。
*   `config/university_difficulty.json`: 院校难度分级。
*   `src/pages/prediction/result_modifier/config.py`: 结果调整配置。

## 数据规范

*   **训练主表** (`cases.feather`): `admitted`, `background_*`, `target_*`, `toefl/ielts`, counts, text details。
*   **学校基础表** (`school_base.feather`): 院校名称、国家、等级。
*   **专业详情表** (`school_major_details.feather`): 专业中英文名、聚合名、大类。
*   **相似度缓存** (`cache/*.feather`): `key` (`major1|major2`), `similarity`。

## 运行与调试

*   **安装依赖**: `pip install -r requirements.txt`
*   **启动应用**: `streamlit run main.py`
*   **环境变量**:
    *   `PREDICTION_USE_PROCESS_POOL=1`: 启用预测进程池。
    *   `PREDICTION_MAX_WORKERS`: 限制并发数。
*   **测试**: `pytest tests`: 包含单元测试，压测等

## 缓存与持久化

*   **预测缓存**: `st.cache_data(ttl=600)`。
*   **数据缓存**: `st.cache_data(ttl=3600)` (案例数据)。
*   **资源缓存**: `st.cache_resource` (模型加载)。
*   **文本加成**: 内存级 LRU 缓存。
*   **用户表单**: 会话态自动保存 (无落盘)。

## 常见问题排查

*   **模型无法加载**: 检查 `.ubj` 文件及 Booster 属性；缺失属性时检查同名 JSON。
*   **特征不对齐**: 确保线上与训练的 `feature_names` 一致；分类特征需统一编码。
*   **文本加成无效**: 检查产物文件 (`.joblib`, `.npz`, `.json`) 及门控阈值配置。
*   **相似度为 0**: 检查缓存文件及键名格式 (`major1|major2` 字母序)。

---

> **维护人**: lijiapeng8@xdf.cn
> **版本**: v3.0
