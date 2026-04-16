# [company]留学数据科学平台

基于 Streamlit 的多页 Web 应用：入口为根目录 [main.py](main.py)，门户页标题与 `init_page` 中一致。经 E2 OAuth 登录后，按邮箱白名单与模块权限（见 [src/utils/auth/permission_checker.py](src/utils/auth/permission_checker.py)）展示可用入口；维护模式由 `config/auth_config.json` 中的 `MAINTENANCE_MODE` 控制（管理员除外）。

**主要功能入口（与 `main.py` 中按钮一致）**

- EasyApply 留学择校（`pages/hk.py`，模块 `hk`）：录取概率预测与相关编排逻辑。
- 人力数据中心（`pages/hr_hub.py`）：在具备 `hr_dashboard`、`hr_profile` 或 `hr_structure_dashboard` 任一权限时可见，分别进入成本/人力看板、绩效档案、结构分析。
- 管理员：`pages/admin.py`（权限管理）、`pages/algorithm_lab.py`（算法实验）。
- 外链：Power BI 案例库（需 `hk` 权限时与 EasyApply 一并展示）。

其他 `pages/` 下的脚本（如 `guide.py`、`redirect.py`）由 Streamlit 自动注册为独立页面，是否从门户进入取决于部署与书签。

## 架构分层

| 层级 | 说明 |
|------|------|
| `pages/*.py` | Streamlit 页面路由，宜保持薄，调用 `src/pages/` |
| `src/pages/` | 业务实现：预测在 `prediction/`，人力在 `hr_dashboard/`、`hr_profile/`、`hr_structure_dashboard/` 等 |
| `src/utils/` | 认证（`page_init`、`auth/`）、数据与模型加载、通用 UI |
| `src/machine_learning_models/` | 离线训练与特征配置 |
| `src/agent/` | 边界场景等辅助逻辑（如与预测流水线配合的 Agent） |

预测子系统在线流程可概括为：表单校验与归一（`input_form_components/`）→ 编排与推理（`flow/pipeline.py`、`prediction_execution/executor.py`）→ 结果合并与后处理（`result_modifier/`，含 TF-IDF 文本 logit uplift 等）。进程内预测封装见 `src/pages/prediction/api/json_api.py`（模块注释标明非生产级、与 Streamlit 同进程，独立部署需另行设计服务层）。

## 机器学习与数据

- **训练目标**：`src/machine_learning_models/data_config.py` 中 `TARGET_COLUMN = "admitted"`，二分类；训练入口见 `train.py` / `model_trainer.py`（XGBoost、采样与校准等逻辑以代码为准）。
- **线上推理**：页面侧通过 `page_data_loader` 等加载预训练模型与案例数据；批量推理并发行为见 `prediction_execution/executor.py`（如 `PREDICTION_USE_PROCESS_POOL`、`PREDICTION_SINGLE_THREAD_THRESHOLD`、`PREDICTION_MIN_CHUNK_SIZE`、`PREDICTION_OVERALL_TIMEOUT_SEC`）。
- **数据与缓存**：案例与院校/专业表多为 Feather（如 `cases.feather`、`school_base.feather`、`school_major_details.feather`）；专业相似度等缓存见 `cache/` 下文件。具体默认路径以 `app_data_loader` 与训练脚本为准。

## 配置

常用文件包括：`config/app_config.json`、`config/dev_config.json`、`config/auth_config.json`、`config/hr_permissions.json`、`config/gpa_conversion_rules.json`、`config/similarity_adjustment_rules.json`、`config/prediction_rules.json` 等。仓库中提供若干 `*.example.json` 供复制后本地填写。敏感项勿提交版本库。

## 本地运行

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run main.py
```

依赖见 [requirements.txt](requirements.txt)（如 Streamlit、pandas、scikit-learn、xgboost-cpu 等）。

## 仓库结构（摘要）

```
config/          应用与权限、业务规则 JSON
pages/           Streamlit 多页入口
src/pages/       各业务模块实现
src/utils/       认证、加载器、UI 组件
src/machine_learning_models/  训练与模型产物路径相关逻辑
src/agent/       Agent 辅助逻辑
scripts/         离线脚本（如相似度预计算等）
tests/           pytest
docs/            模块级说明（预测、表单、训练、结果修正等）
```

## 测试

```bash
pytest tests
```

（`requirements.txt` 中 pytest 为注释状态时，需自行安装 pytest 后再运行。）

## 延伸阅读

- [docs/ml_training_api.md](docs/ml_training_api.md)
- [docs/prediction_api.md](docs/prediction_api.md)
- [docs/input_form_components_api.md](docs/input_form_components_api.md)
- [docs/result_modifier_api.md](docs/result_modifier_api.md)
- [docs/major_similarity_precompute.md](docs/major_similarity_precompute.md)
- 预测子系统源码旁说明（与 docs 互补）：[src/pages/prediction/README.md](src/pages/prediction/README.md)，[flow/README.md](src/pages/prediction/flow/README.md)，[result_modifier/README.md](src/pages/prediction/result_modifier/README.md)

## 常见问题（运维）

- 模型无法加载：检查预训练文件路径与 `feature_names` 是否与线上一致。
- 文本加成不生效：核对 TF-IDF 相关产物路径及 `result_modifier` 与 `config` 中的门控与规则。
- 相似度异常：核对缓存键格式与列名（如 `major1|major2` 等）是否与加载逻辑一致。

## 免责声明与使用边界

本仓库所载软件系统、文档及配置示例由维护人**独立设计、开发与维护**；在依法须保留署名的开源第三方库之外，核心业务逻辑与工程实现均为自主完成。**本仓库的展示、分发或内部部署，不构成与任何自然人、法人或其他组织之间就本代码或文档另行订立委托开发、技术转让、独家许可等合同关系**；任何第三方未因阅读、克隆或使用本仓库而与维护人自动形成权利义务关系。

仓库内业务数据、统计样例、缓存及示例配置均已做**脱敏或替换**，不包含真实客户、员工可识别信息，亦不应将示例路径与键名等同于生产环境密钥管理要求。使用者若在自有或所在单位环境中加载真实数据，须自行满足个人信息保护、网络安全及内部合规要求。

本软件及文档按**「现状」**提供，不作任何明示或默示担保，包括但不限于对适销性、特定用途适用性、不间断或无错误运行的保证。因使用或无法使用本软件而产生的任何直接、间接、附带或后果性损害，在适用法律允许的最大范围内，维护人不承担责任。

若本系统部署于所在单位基础设施之上，**部署行为及运行安全责任由部署与运维方承担**；本 README 及仓库内容**不构成所在单位对产品功能、服务质量、数据安全或合规性的官方陈述、承诺或担保**，亦不代表所在单位对代码知识产权归属以外的立场。

---

维护人：lijiapeng8@xdf.cn · 文档版本随仓库演进
