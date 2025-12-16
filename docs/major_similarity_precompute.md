# 专业相似度预计算 (E5 Embedding)

参考模型：
- **Multilingual E5 Instruct**: [https://huggingface.co/intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

本说明描述如何使用 E5 嵌入模型预计算“背景专业-目标专业”的相似度缓存，以加速预测/推荐阶段的相似度查询。

## 1. 模型与加载 (`scripts/model_utils.py`)

- **优先本地加载**：`src/services/multilingual-e5-large-instruct`
- 本地不存在时回退到在线模型 `intfloat/multilingual-e5-large-instruct`。
- 支持 GPU/CPU，CPU 可选动态量化（线性层）。
- 批量大小：`compute_embeddings_batch_local(..., batch_size=64)` 可按机器性能调整（默认值在脚本中配置）。
- 编码阶段开启 `normalize_embeddings=True`，直接返回单位向量，便于后续用点积计算相似度。

## 2. 预计算脚本 (`scripts/precompute_similarities.py`)

### 输入数据路径
- **案例库**：`src/machine_learning_models/data/cases.feather`
- **学校专业详情**：`src/machine_learning_models/data/school_major_details.feather`

### 目标缓存
- **背景专业-目标专业 (BG-Target)**：预测流水线使用（脚本当前仅生成这一份）

### 表示名选择
用于 embedding 输入，而非缓存 key：
- 若详情表同时包含“专业英文名称/专业中文名称”，会把英文专业映射为中文名作为 embedding 输入；无法映射时使用原字符串。
- embedding 会对“去重后的表示名集合”计算一次向量，并通过相似度矩阵查表复用（同一表示名共享 embedding）。

### 提示词
- 仅当 `MODEL_NAME` 包含 `e5`（不区分大小写）时，输入会自动加 `query: ` 前缀；否则直接使用原文本。

### 相似度
- 编码阶段已做 L2 归一化（`normalize_embeddings=True`），相似度使用向量点积（等价于余弦）。

### 缓存键与存储
- **键规则**：按字母序排序后拼接为 `major1|major2`（参考 `src/pages/prediction/prediction_utils.py`）。
- **注意**：key 使用的是**原始专业名字符串**（`cases.feather` 里的 `background_major/target_major`），不是 embedding 的中文表示名。
- **输出文件** (feather)：
  - `cache/background_target_similarity.feather`
- **文件结构**：两列 `key`、`similarity`；线上由 `utils.app_data_loader.load_bg_target_similarity_cache` 读取为字典。
- **兼容性**：
  - 脚本里仍保留了历史路径常量，但当前版本不会写入该文件。
  - 当前脚本不做旧缓存自动清理。

## 3. 常见问题排查

- **相似度异常偏高（跨学科被错判为高相似）**：
  - 确保使用 E5 并采用 `query:` 提示词风格。
  - 重新运行预计算脚本生成新缓存后再观察推荐结果。

- **相似度列表偏少/偏多**：
  - 可调 `batch_size` 提升吞吐；阈值在业务层（如 `result_modifier`）调节，不影响缓存本身。

## 4. 路径配置与运行

运行脚本将缓存写入仓库根目录下的 `cache/`（线上加载函数默认从该目录读取：`cache/background_target_similarity.feather`）。

### 运行

```bash
python scripts/precompute_similarities.py
```

完成后，缓存将被写入根目录 `cache/`，供线上 `utils.app_data_loader.load_bg_target_similarity_cache` 加载使用。

---

> **维护人**: lijiapeng8@xdf.cn
> **版本**: v2.8
