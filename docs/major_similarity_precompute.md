## 专业相似度预计算（E5 Embedding）

参考模型：
- Multilingual E5 Instruct: https://huggingface.co/intfloat/multilingual-e5-large-instruct

本说明描述如何使用 E5 嵌入模型预计算专业相似度缓存，以加速预测/推荐阶段的相似度查询（脚本暂不公开，有需要请联系lijiapeng8@xdf.cn）。

### 模型与加载（scripts/model_utils.py）
- 优先本地加载：`src/services/multilingual-e5-large-instruct`
- 本地不存在时回退到在线模型 `intfloat/multilingual-e5-large-instruct`。
- 支持 GPU/CPU，CPU 可选动态量化（线性层）。
- 批量大小：`compute_embeddings_batch_local(..., batch_size=64)` 默认 64，可按机器性能调整。
- 编码阶段开启 `normalize_embeddings=True`，直接返回单位向量，便于后续用点积计算相似度。

### 预计算脚本（scripts/precompute_similarities.py）
- 输入数据路径：
  - 案例库：`src/machine_learning_models/data/cases.feather`
  - 学校专业详情：`src/machine_learning_models/data/school_major_details.feather`
- 目标缓存：
  1) 背景专业-背景专业（BG-BG）：相似案例使用
  2) 背景专业-目标专业（BG-Target）：预测流水线使用
- 名称表示：
  - 若详情表含“专业中文名称”，优先使用中文；否则使用英文。
- 提示词：E5 instruct 使用前缀 `query: {major_name}`
- 相似度：
  - 编码阶段已做 L2 归一化（`normalize_embeddings=True`），相似度使用向量点积（等价于余弦）。
- 缓存键与存储：
  - 键规则：按字母序排序后拼接为 `major1|major2`（参考 `src/pages/prediction/prediction_utils.py#_create_major_similarity_key` 与 `scripts/precompute_similarities.py#_create_major_similarity_key`）。
  - 输出文件（feather）：
    - `cache/background_target_similarity.feather`
  - 文件结构：两列 `key`、`similarity`；线上由 `utils.app_data_loader.load_bg_target_similarity_cache` 读取为字典。
  - 兼容性：历史 `src/machine_learning_models/data/cache/major_similarity_cache.{json,pkl}` 已废弃，当前脚本不做自动清理。

#### 模型与提示词
- 预计算脚本固定采用 E5，并使用 `query:` 提示词风格。

#### 常见问题排查
- 相似度异常偏高（跨学科被错判为高相似）：
  - 确保使用 E5 并采用 `query:` 提示词风格。
  - 重新运行预计算脚本生成新缓存后再观察推荐结果。
- 相似度列表偏少/偏多：
  - 可调 `batch_size` 提升吞吐；阈值在业务层（如 `result_modifier`）调节，不影响缓存本身。

### 路径配置
- 运行脚本默认将缓存写入仓库根目录下的 `cache/`（线上加载函数默认从该目录读取）。
  - 资源读取仍可按需从配置覆盖（如 cases/详情表路径），但相似度缓存默认不再写入 `src/machine_learning_models/data/cache/`。

### 运行
```bash
python scripts/precompute_similarities.py
```

完成后，缓存将被写入根目录 `cache/`，供线上 `utils.app_data_loader.load_bg_target_similarity_cache` 加载使用。

---
维护人：lijiapeng8@xdf.cn
版本：v2.8


