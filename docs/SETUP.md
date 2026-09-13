# 本地运行与数据准备

## 环境

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows 会自动跳过 uvloop
# .venv/bin/pip install -r requirements.txt       # Linux / macOS

cp .streamlit/config.toml.example .streamlit/config.toml
# 主题段别删。缺这个文件会退回 Streamlit 默认亮色主题，页面是按暗色底设计的。

.venv/bin/streamlit run main.py
```

http://localhost:8501 ，访问码 `SIGNALS2026`。没填 API key 也能打开，模型相关能力静默降级。

## 模型接口

全部走 DeepSeek 官方 API。

```bash
cp config/app_config.json.example config/app_config.json   # 填 key
```

或者用环境变量，优先级高于配置文件：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com/v1/chat/completions
export OPENAI_MODEL=deepseek-v4-flash
export GHOST_API_KEY=sk-...      # 浏览器端补全专用
export APP_ENV=test              # 选择 app_config.json 里的段落
```

`GHOST_API_KEY` 不要和主 key 共用。它会被下发到浏览器，主 key 没有这个暴露面。

## 数据

仓库不含案例库和模型权重，这是脱敏的结果。共 53.4 MB：

| 路径 | 大小 | 说明 |
|---|---:|---|
| `src/ml/data/cases.feather` | 21 MB | 历史案例 |
| `src/ml/data/school_base.feather` | 0.4 MB | 院校基础表 |
| `src/ml/data/school_major_details.feather` | 0.6 MB | 院校-专业对照 |
| `src/ml/pre-trained_models/xgboost_*.ubj.gz` | 17 MB | XGBoost 权重 |
| `src/ml/pre-trained_models/tfidf_*.{joblib,npz}` | 0.4 MB | 文本提升模型 |
| `cache/background_target_similarity.feather` | 14 MB | 相似度预计算 |

换算权重换了的话，相似度矩阵要重跑 `scripts/precompute_similarities.py`，SHAP 图重跑 `scripts/compute_shap.py`。

## 部署

见 [`../deploy/README.md`](../deploy/README.md)。
