## 专业相似度预计算（Gemma Embedding）(https://huggingface.co/google/embeddinggemma-300m)

本说明描述如何使用 Gemma 嵌入模型预计算专业相似度缓存，以加速预测/推荐阶段的相似度查询（脚本暂不公开，有需要请联系lijiapeng8@xdf.cn）。

### 模型与加载（scripts/model_utils.py）
- 优先本地加载：
  - embeddinggemma: `src/services/embeddinggemma-300m`
  - 备用 E5: `src/services/multilingual-e5-large-instruct`
- 本地不存在时回退到在线模型（如 `google/embeddinggemma-300m`）。
- 支持 GPU/CPU，CPU 可选动态量化（线性层）。
- 批量大小：`compute_embeddings_batch_local(..., batch_size=64)` 默认 64，可按机器性能调整。

### 预计算脚本（scripts/precompute_similarities.py）
- 输入数据路径：
  - 案例库：`src/machine_learning_models/data/cases.feather`
  - 学校专业详情：`src/machine_learning_models/data/school_major_details.feather`
- 目标缓存：
  1) 背景专业-背景专业（BG-BG）：相似案例使用
  2) 背景专业-目标专业（BG-Target）：预测流水线使用
- 名称表示：
  - 若详情表含“专业中文名称”，优先使用中文；否则使用英文。
  - 使用带指令的提示词编码：
    - `Instruct: Represent this academic major for similarity comparison.\nQuery: {major_name}`
- 相似度：
  - 句向量 L2 归一化后做点积（等价余弦）。
- 缓存键与存储：
  - 键：`prediction_utils.get_cached_major_similarity_key(major_A, major_B)`（按字母序排序后拼接为 `major1|major2`）。
  - 键函数位置：`src/pages/prediction/prediction_utils.py#get_cached_major_similarity_key`
  - 输出文件（feather）：
    - `cache/background_background_similarity.feather`
    - `cache/background_target_similarity.feather`
  - 文件结构：两列 `key`、`similarity`，线上由 `utils.app_data_loader.load_bg_*_similarity_cache` 读取。
  - 兼容性清理：会自动删除旧的 `src/machine_learning_models/data/cache/major_similarity_cache.{json,pkl}`。

### 路径配置
- 运行脚本默认将缓存写入仓库根目录下的 `cache/`（线上加载函数默认从该目录读取）。
  - 资源读取仍可按需从配置覆盖（如 cases/详情表路径），但相似度缓存默认不再写入 `src/machine_learning_models/data/cache/`。

### 运行
```bash
python scripts/precompute_similarities.py
```

完成后，两类缓存将被写入根目录 `cache/`，供线上 `utils.app_data_loader.load_bg_target_similarity_cache` 等加载使用。

---
维护人：lijiapeng8@xdf.cn
版本：v2.5


