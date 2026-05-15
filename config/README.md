
# Config 模块说明

## 1. 概述

`config/` 是所有运行时配置的单一真相源。所有 API key、OAuth 密钥、白名单、模块权限、预测规则、GPA 转换规则、文本信号词等均在此集中管理。配置文件均为 JSON 格式，通过 `src/utils/env_config_loader.py` 按环境（test/prod）加载。

## 2. 目录结构

```
config/
├── __init__.py                      # 空
├── app_config.json                  # 环境配置：E2 OAuth key、OpenAI API key/URL
├── auth_config.json                 # 认证配置：白名单（~730人）、模块权限、维护模式
├── dev_config.json                  # 开发配置：DEBUG_MODE + debug 用户身份
├── prediction_rules.json            # 预测规则：23所学校难度排序、国家-学校映射、专业列表
├── hot_paths.json                   # 热门组合快速路径（hot_major_substrings + hot_schools，2026-05 新增）
├── major_name_mapping.json          # 专业名别名 → 标准名归一化映射（2026-05 新增）
├── gpa_conversion_rules.json        # GPA 转换规则：北大/中科大/上交等校特定转换
├── similarity_adjustment_rules.json  # 相似度调整规则
├── text_high_signal_terms.json      # 文本高信号词（TF-IDF 文本提升）
├── hr_permissions.json              # HR 细粒度权限：管理者映射、部门范围、HR 管理员
├── redirect_whitelist.json          # E2 回调重定向白名单
└── cs_survey/                       # CS 问卷配置
    ├── 2026_jp_product.yaml         # 日本产品线问卷
    └── layout.json                  # 布局配置
```

## 3. 各配置文件说明

### 3.1 app_config.json

环境分离配置（`APP_ENV` 环境变量切换 profile）：
- **test**：测试环境 E2 key、OpenAI API key/URL、Streamlit 部署路径
- **prod**：生产环境对应配置
- 加载方式：`load_app_config()` → 返回当前环境配置 dict

### 3.2 auth_config.json

认证与权限配置（~39 KB）：
- **whitelist**：全局用户白名单（~730人）
- **module_permissions**：模块级权限（EasyApply / HR Dashboard / HR Profile 等）
- **admin_users**：管理员列表
- **maintenance_mode**：维护模式开关

### 3.3 prediction_rules.json

预测规则配置：
- 23所学校的难度排序（冲刺/适中/保底三级）
- 国家 → 学校列表映射
- 专业学位类型（授课型/研究型）列表
- 职业学位降级规则

### 3.4 hot_paths.json（2026-05 新增）

热门组合快速路径配置，用于跳过语义相似度计算，加速组合生成：
- **hot_major_substrings**：热门专业子串列表（如 SMART、ACCT、IT、Finance），命中即直接纳入候选
- **hot_schools**：热门院校列表（预留，当前仅 major_substrings 生效）
- 数据来源：`cache/usage_stats.json` 的使用统计 → admin 页面手动更新 → git commit 部署
- 加载：`_load_hot_paths()` 在模块导入时读取，文件不存在或损坏时 fallback 到硬编码默认值

### 3.5 major_name_mapping.json（2026-05 新增）

专业名别名归一化映射表，解决 LLM 提取 / 用户输入的专业名不统一问题：
- 格式：`{"别名": "标准专业名", "CS": "Computer Science", ...}`
- 使用方：`form_bridge._fuzzy_match_major()` 在 fuzzy 匹配前先查映射表
- 与 `school_major_details.feather` 中的英文专业名保持同步

### 3.6 gpa_conversion_rules.json

GPA 转换规则：
- 院校级规则：如北京大学特定转换表
- 国家级规则：如加拿大院校 GPA 转换
- 支持区间映射（ranges: min/max → target_gpa）
- 兜底：`fallback_multiplier` 线性换算或 `is_percentage` 百分制

### 3.5 hr_permissions.json

HR 权限配置：
- **manager_map**：管理者名称 → 邮箱列表
- **admin_hr_emails**：HR 管理员邮箱列表
- **admin_hr_permissions**：管理员可见部门范围

### 3.6 cs_survey/

CS 问卷模块配置：
- **YAML** 文件：调研定义（数据源、维度、评分规则、视图）
- **layout.json**：页面布局配置

## 4. 使用方式

```python
# 环境配置（带 @st.cache_data 缓存）
from src.utils.env_config_loader import load_app_config
config = load_app_config()  # → dict

# 静态配置（直接 json.load）
import json
with open("config/prediction_rules.json") as f:
    rules = json.load(f)
```

## 5. 依赖

- [Utils](src/utils/README.md) — `env_config_loader`（`load_app_config()` 按环境加载）
- `streamlit`：`@st.cache_data` 缓存配置
