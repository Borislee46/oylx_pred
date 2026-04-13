# 专业相似度离线预计算（E5）

| 项 | 说明 |
|----|------|
| 脚本 | `scripts/precompute_similarities.py` |
| 模型加载 | `scripts/model_utils.py`（优先本地目录，否则 Hugging Face `intfloat/multilingual-e5-large-instruct`） |
| 输出文件 | 仓库根目录 `cache/background_target_similarity.feather` |
| 线上读取 | `src/utils/app_data_loader.load_bg_target_similarity_cache`；检索辅助函数 `src/pages/prediction/core/utils.py::get_cached_major_similarity` |

## 1. 目的

对「背景专业 × 目标专业」批量预计算嵌入相似度，避免在线对每个组合实时编码，缩短预测路径延迟。

## 2. 输入数据

| 路径 | 用途 |
|------|------|
| `src/machine_learning_models/data/cases.feather` | 汇总 `background_major`、`target_major` 去重集合 |
| `src/machine_learning_models/data/school_major_details.feather` | 补充目标侧专业英文名集合；构建英→中映射 |

## 3. 编码与指令格式

- 背景与目标专业名字符串经 `strip().lower()` 去重；嵌入输入可使用映射后的中文显示名（`eng_to_chi_map`），**缓存列仍存原始小写键**（与线上一致）。
- 背景侧 query 格式（与脚本一致）：

  `Instruct: {task}\nQuery: {text}`

  其中 `task` 为固定英文任务句：`Given an academic major, retrieve the most relevant target major for university admission`。

- 目标侧：直接使用表示名 `text`，无 Instruct 前缀。
- `model_utils` 中编码开启 `normalize_embeddings=True`，相似度矩阵为归一化向量点积（等价余弦）。

## 4. 输出 Schema

Feather 三列：

- `bg_major`、`target_major`：小写字符串
- `similarity`：`float`，范围 \([-1, 1]\)（点积）

脚本将笛卡尔积写入 `BG_TARGET_CACHE_PATH`；目录不存在时自动创建。

## 5. 运行

```bash
python scripts/precompute_similarities.py
```

可调参数：`scripts/precompute_similarities.py` 内 `BATCH_SIZE`（默认 64）、`MODEL_NAME`。大批量时主要受 GPU/内存与 PyTorch 设备影响。

## 6. 运维与排障

| 现象 | 建议 |
|------|------|
| 跨学科相似度异常偏高 | 核对是否使用 Instruct 模板与同一模型版本；清空缓存后全量重算 |
| 键匹配失败 | 线上检索前对 major 做与离线一致的 `strip().lower()`；检查中英映射表是否覆盖 |
| 缓存体积过大 | 源数据去重后仍可能为 \|Bg\|×\|Target\| 行数；仅在业务层截断或阈值过滤，不改变缓存生成逻辑 |

---

维护：与 `scripts/precompute_similarities.py`、`scripts/model_utils.py` 同步更新。
