# Utils 模块技术文档

## 1. 模块概述

`src/utils` 是所有页面的共享基础设施层。提供页面初始化（E2 登录守卫 + CSS 注入 + 水印）、OAuth 认证、强类型 Session 管理、日志系统、配置加载、数据安全（剪贴板防护 + 水印）、院校/难度服务、管理后台工具等。每个 Streamlit 页面启动时都会调用 `init_page()`。

## 2. 目录结构

```
utils/
├── __init__.py                  # 空（各子模块独立 import）
├── page_init.py                 # init_page()：页面初始化总入口
├── page_auth.py                 # handle_e2_login()：E2 OAuth 登录守卫
├── session_manager.py           # SessionManager + UserDataModel + PredictionResultModel
├── logger.py                    # setup_logger()：结构化日志
├── app_data_loader.py           # 应用级数据加载
├── model_loader.py              # ML 模型加载工具
├── school_constants.py          # 院校常量
├── school_level_service.py      # 院校等级/海外判定/语言加成
├── university_difficulty_service.py # 院校难度分级
├── env_config_loader.py         # 环境配置加载（test/prod）
├── auth/                        # 认证子模块
│   ├── e2_handler.py            # E2 OAuth 签名/回调处理
│   ├── permission_checker.py    # 全局白名单 / 模块权限 / 功能权限三级检查
│   ├── auth_json_loader.py      # auth_config.json 加载
│   ├── dev_config_loader.py     # 开发环境假用户注入
│   └── config_processor.py      # 配置预处理
├── admin/                       # 管理后台工具
│   ├── config_manager.py        # 配置文件读写
│   ├── ui_helpers.py            # 管理 UI 辅助
│   ├── delete_confirmation.py   # 删除确认弹窗
│   └── permissions.py           # 管理权限工具
├── data_safety/                 # 数据安全
│   ├── clipboard_guard.py       # 剪贴板防护
│   └── watermark.py             # 水印 CSS 生成
└── ui/                          # UI 工具
    ├── page_header.py           # 页面标题栏
    ├── button.py                # 通用按钮组件
    ├── ui_utils.py              # UI 辅助函数
    └── hk_shield.py             # HK Shield 组件
```

## 3. 初始化链条

```
init_page(page_title, current_page_path, ...)
    │
    ├── 1. st.set_page_config（标题、布局、favicon）
    │
    ├── 2. CSS 注入
    │       ├── assets/style.css（全局样式）
    │       └── additional_css_files（页面专属 CSS，如 HK 品牌样式）
    │
    ├── 3. E2 登录守卫（handle_e2_login）
    │       ├── 生产环境：检查 session TTL（24h），过期重定向 E2 登录
    │       ├── 开发环境：DEBUG_MODE 注入假用户
    │       ├── 权限检查：全局白名单 → 模块权限 → 功能权限
    │       └── admin_only：管理员限定
    │
    ├── 4. SessionManager() 初始化
    │       └── st.session_state["user_data_model"] = UserDataModel()
    │
    └── 5. 水印注入
            └── generate_watermark_css(user_nickname) → st.markdown()
```

## 4. 核心组件

### 4.1 page_init.py

`init_page()` 是所有页面的统一入口，参数：
- `page_title`、`current_page_path`、`layout`（默认 wide）
- `additional_css_files`：页面专属 CSS
- `watermark_config`：水印样式覆盖
- `skip_auth`：跳过登录（调试用）
- `skip_watermark`：跳出水印
- `module_name`：模块权限检查标识
- `admin_only`：仅管理员可访问
- `hide_sidebar`：隐藏 Streamlit 侧边栏

返回值：`{"user_nickname": str, "user_email": str}`

### 4.2 page_auth.py

`handle_e2_login(current_page_path, module_name, admin_only)`：
- 生产环境：检查 `e2_user_email` 是否在 session 中且未过期
- 未登录 → 构建 E2 授权 URL → 重定向
- 回调 → E2 handler 验证签名 → 种 session
- 开发环境：`config/dev_config.json` 中 `DEBUG_MODE=true` 时注入 debug 用户

### 4.3 session_manager.py

**SessionManager**：强类型 session state 访问层，持有 `UserDataModel` dataclass：
- `session_id`：会话 UUID
- `is_logged_in`、`user_info`：用户身份
- `input_data`：表单数据
- `prediction_results`：`PredictionResultModel`（similarity/cross/user_specified/unified + meta）
- `prediction_submit_lock`：提交锁
- `other_states`：扩展字段 dict

提供 `get()` / `set()` / `batch_set()` / `delete()` / `clear_session()` 方法。

**PredictionResultModel**：预测结果 dataclass，包含三类推荐 + 统一结果 + 元数据。

### 4.4 logger.py

`setup_logger(name, category)`：返回带 `SessionIDFilter` 的 logger，日志自动附带 `session_id`。

### 4.5 auth/ 子包

三级权限体系：
1. **全局白名单**（`auth_config.json`，~730人）：`is_user_in_whitelist()`
2. **模块权限**（EasyApply / HR Dashboard / HR Profile 等）：`check_module_permission()`
3. **功能权限**（HR 细粒度，如仅查看特定部门）：`check_user_access_permission()`

管理员列表：`cuiting3`、`lijiapeng8`

### 4.6 data_safety/ 子包

- **clipboard_guard.py**：注入 JS 拦截复制事件
- **watermark.py**：`generate_watermark_css()` 生成平铺用户名水印 CSS

### 4.7 env_config_loader.py

`load_app_config()`：根据 `APP_ENV` 环境变量（默认 `test`）从 `config/app_config.json` 加载对应 profile 的 OAuth key、API key、URL 等。

### 4.8 school_level_service.py / university_difficulty_service.py

- **school_level_service**：院校等级查询、海内外判定、海外院校语言成绩加成
- **university_difficulty_service**：23所学校难度分级（冲刺/适中/保底）

## 5. 依赖

- `streamlit`：UI 框架
- E2 SSO SDK：企业统一认证
- [Config](config/README.md) — `app_config.json`、`auth_config.json`、`dev_config.json`
