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

### 提示词与指令 (Instruction)
- 系统使用 **Instruct** 模式提升匹配精度。
- **背景专业 (Query Side)**：增加指令前缀 `Instruct: Given an academic major, retrieve the most relevant target major for university admission\nQuery: {专业名称}`。
- **目标专业 (Candidate Side)**：直接使用专业名称，不添加前缀。
- 若 `eng_to_chi_map` 存在映射，则使用中文专业名进行 Embedding 计算，但缓存 Key 仍保留为原始字符串（通常为英文）。

### 相似度计算
- 编码阶段已做 L2 归一化（`normalize_embeddings=True`），相似度使用向量点积（等价于余弦相似度）。

### 存储规范与检索
- **存储格式**：使用 Feather 格式，包含 `bg_major`、`target_major`、`similarity` 三列。
- **检索逻辑**：线上由 `src/pages/prediction/core/utils.py::get_cached_major_similarity` 执行。
- **一致性保证**：为了保证匹配成功率，所有的专业名在存储和检索前都会进行 **`.strip().lower()`** 处理（包括 Embedding 计算时的中文映射查找）。
- **检索顺序**：检索 Key 顺序为 `(background_major, target_major)`。
- **输出文件**：`cache/background_target_similarity.feather`。
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

> **维护人**: support@demo.local
> **版本**: v2.9
> **最后更新**: 2026-06-07 — 路径核实；调用方同步（utils + app_data_loader + knn_retrieval）；E5 Instruct 模型引用确认
