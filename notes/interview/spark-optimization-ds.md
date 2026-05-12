# Spark SQL 优化：DS 面试够用版

> 你的当前水平：SQL 闭眼写，做过 map_from_entries 代替 join、不用 LIKE %% 等优化
> 你的缺口：能优化但不一定能用原理语言说出"为什么"
> 本文目标：补 DS 面试会用到的 Spark 概念，不涉及 DE 深度内容

---

## 先说结论：DS 面 Spark 只考三个东西

面试官不是 DE，不会问你 tungsten 执行引擎、AQE 自适应查询、动态分区合并。他们只想知道：
1. 你知道什么操作最贵（shuffle）
2. 你能识别并处理数据倾斜
3. 你在写 SQL 时有优化的意识

---

## 1. Shuffle — 最贵的操作

**一句话**：数据需要跨节点重新分区时，触发 **Shuffle**。网络传输 + 磁盘 IO + 序列化反序列化，比单节点内存计算慢 10-100 倍。

**哪些操作触发 Shuffle**：

| 操作 | 是否 Shuffle | 原因 |
|------|------------|------|
| `SELECT * FROM t WHERE x > 10` | 否 | 单个 partition 内就能算 |
| `JOIN` (两个大表) | **是** | 相同 key 的数据必须在同一节点才能 join |
| `GROUP BY` | **是** | 相同 group key 必须汇集 |
| `DISTINCT` | **是** | 去重需要把相同值汇集 |
| `ORDER BY` (全局) | **是** | 排序需要全局比较 → 最后聚到一个 partition |
| `ROW_NUMBER() OVER (PARTITION BY ...)` | **是** | 窗口函数按分区汇集 |

**你对号入座**：你做的 `map_from_entries` 代替 join → map 操作只在单 partition 内，**避免了 shuffle**。这就是你在无意识中做的优化。

**面试话术**：
> "我知道 shuffle 是 Spark 最重的操作。写 SQL 时会有意识地减少不必要的 shuffle，比如用 map 操作（map_from_entries）替代小表的 join，在 JOIN 和 GROUP BY 之前先过滤缩小数据量。"

---

## 2. Broadcast Join — 小表免 Shuffle 的法宝

**一句话**：如果一张表 < 100MB（默认 broadcast threshold），Spark 会把它完整复制到每个 Executor 的内存中。JOIN 时不需要 shuffle 大表——每个 partition 直接查本地内存中的小表副本。

**你做的 `map_from_entries` 其实就是手动版的 broadcast**——把小表变成内存字典，避免 join。Spark 的 `broadcast()` hint 做的事一模一样，只是自动化了。

```
-- Spark SQL 写法（比 map_from_entries 更通用）
SELECT /*+ BROADCAST(small_table) */ *
FROM large_table
JOIN small_table ON large_table.key = small_table.key
```

**什么时候用**：
- 小表 < 100MB（如维度表、配置表、院校字典）
- 你的场景：学校信息表在整个系统中就是典型的小表 → broadcast

**面试话术**：
> "维度表通常很小（院校信息几十行），用 broadcast join 避免了大表 shuffle。如果小表是大表 JOIN 后的结果，我会先把子查询结果物化再 broadcast。"

---

## 3. 数据倾斜 — 面试必问

**一句话**：某个 key 的数据量占总量的 30%+，导致 shuffle 后一个 partition 特别大 → 这个 partition 的 task 跑 10 分钟，其他 task 跑 10 秒 → 整体被拖死。

**你用到的场景（面试中可以说的）**：搜索查询词倾斜——"天气""游戏""抖音"这类头部词的数据量是长尾词的几百倍。BY query 聚合时，头部词永远最慢。

**怎么发现**：Spark UI → Stages → 看每个 Task 的耗时分布 → 如果 199 个 Task 1 分钟 + 1 个 Task 20 分钟 = 数据倾斜。

**面试最关心的：你怎么处理**：

| 方法 | 适用场景 | 原理 |
|------|---------|------|
| **加盐打散** | GROUP BY / JOIN 都有倾斜 key | 给倾斜 key 加随机后缀(0-N)，分散到 N 个 partition 各自算，最后再合并 |
| **预聚合 / 两阶段聚合** | GROUP BY 场景 | 先加盐做一次局部分组，去掉后缀再全局分组 |
| **Map-side Join** | 大表 JOIN 小表（维度倾斜） | 把维度表 broadcast，避免 shuffle，倾斜自然消失 |
| **单独处理** | 只有少数几个 key 倾斜 | 倾斜 key 拆出来单独 join（给小表加盐），非倾斜 key 正常 join |

**核心原则**：**不要让一个 task 处理 30% 的数据**。把它打散成 N 个小 task。

**面试话术**：
> "在搜索日志分析中遇到过查询词倾斜——头部词（'天气''游戏'）数据量占 20%。解决方案是给倾斜 key 加随机后缀（0-9），打散成 10 个 task 各自聚合，最后再加一层去后缀聚合。跟 PM 确认过，搜索指标对随机拆分是无偏的。"

---

## 4. 你已经在做的优化 + 对应的原理

| 你做的事 | 你用的方法 | 背后的 Spark 原理 |
|---------|-----------|-----------------|
| `map_from_entries` 代替 join | 内存映射 | 避免 Shuffle |
| 不用 `LIKE %%` | 全模糊避免 | LIKE %% 无法走分区裁剪 → 全表扫描 |
| 子查询先过滤再 join | 缩小中间数据 | Shuffle 数据量 = 瓶颈，缩小它 |
| 分区字段放进 WHERE | 分区裁剪 | 只读需要的 partition，跳过不相关的 |

**你唯一需要补的**：`EXPLAIN` 看执行计划。Spark SQL 也一样，`EXPLAIN EXTENDED` 能看到哪步是全表扫描、哪步是 broadcast join、哪步是 shuffle。面试时说"我写完复杂 SQL 会 EXPLAIN 确认执行计划"就够了。

---

## 5. DS 面试不会问但你应该知道的名词（防身）

| 名词 | 一句话 | 要不要学 |
|------|--------|---------|
| **AQE** | 运行时自动调整执行计划（Spark 3.0+） | DE 的事 |
| **Dynamic Partition Pruning** | JOIN 时自动用小表过滤大表分区 | 知道就行 |
| **Tungsten** | Spark 的内存管理引擎 | 不用管 |
| **Catalyst Optimizer** | Spark 的 SQL 优化器（解析→优化→物理计划） | 知道名字就好 |
| **Columnar Storage (Parquet/ORC)** | 列存 = 聚合/过滤只读需要的列 | 知道为什么快 |

---

## 6. 面试完整话术（30秒版）

> "我在 Spark SQL 优化上有实战经验。核心思路是减少 shuffle——小表用 broadcast join 或 map 操作代替，尽量在 JOIN/GROUP BY 之前缩小数据量，利用分区裁剪过滤。
> 处理过搜索查询词的数据倾斜问题——头部词数据量是长尾的几十倍，通过加盐打散 + 两阶段聚合解决。
> 写复杂 SQL 后会用 EXPLAIN 确认执行计划没有意外的全表扫描。"

---

## 7. 你在留学项目里就能找到的 Spark 场景

你的系统虽然现在用 Streamlit + feather 文件，但**如果数据量涨 100 倍**，面试官问"你怎么迁移到 Spark"，你可以直接说：

> "院校维表 → broadcast join。历史案例表如果按年份分区 → 可以按年份分区裁剪，最近三年的数据占大多数查询。背景专业相似度矩阵预计算 → Sparse representation 存成 Parquet，查询时只读相关行。"
