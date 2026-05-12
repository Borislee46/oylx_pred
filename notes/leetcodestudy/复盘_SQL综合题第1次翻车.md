# SQL 综合题第1次翻车复盘

**日期**：2026-05-09
**题目**：WAU 用户构成拆解（新用户/持续活跃/复活用户/流失用户）
**结果**：40分钟未完成，CTE 链写到第2层绕晕

---

## 犯的错

### 1. LAG 方向写反

```sql
-- 我写的（错）
lag(week_num, 1) over (...) - week_num = 1

-- 正确
week_num - lag(week_num, 1) over (...) = 1
```

**根因**：没在脑子里跑一遍数值。对 week=2 的行，LAG=1，1-2=-1，不可能等于1。写完没自测。

### 2. CASE WHEN 嵌套语法错误

```sql
-- 我写的（错）
case when ... then 'retained'  
     case when ... then 'resurrected'
     case when lag(week_num,1)
end as label1

-- 正确
CASE 
    WHEN condition1 THEN 'retained'
    WHEN condition2 THEN 'resurrected'
    ELSE 'something'
END
```

**根因**：三个错叠加——CASE 套 CASE（不需要）、第二个和第三个 CASE 没 END、最后一个 WHEN 条件不完整。

### 3. 试图一个 CTE 算出所有分类

把 NEW / RETAINED / RESURRECTED / CHURNED 全塞进一个 CASE WHEN。这是**架构层面**的错误：

- 前三类回答"本周活跃的人从哪来" → 从本周活跃表出发
- Churn 回答"上周活跃的人去哪了" → 从上周活跃表出发，LEFT JOIN 本周
- **两个问题的数据源不一样，不能放在同一个 FROM 里**

### 4. 没意识到 LAG 对"缺行"无效

user 2 活跃周是 [1, 3]，week 2 在表中**不存在**。对 week=3 的行，LAG 拿到的是 week=1。虽然碰巧 `3-1=2` 能判断复活，但如果跳三周（week [1,5]），`5-1=4`，用 `=2` 就漏了。

**正确的复活判定**：排除法 — 不是 NEW，也不是 RETAINED，就是复活。不需要算间隔。

### 5. Churn 完全漏掉

Churn 用户本周不活跃，不存在于 `user_activity` 中。从同一个表出发根本算不出来。正确做法需要生成"所有周列表"或从上周活跃 LEFT JOIN 本周活跃。

---

## 做得对的

- `DISTINCT` 去重意识 ✓
- `MIN() OVER (PARTITION BY user_id)` 找首周 ✓
- 想到用 LAG 判断相邻周 ✓（方向写反是执行问题，不是方向问题）
- CTE 分层拆解的意识已经有了 ✓（base → lfa → labels，虽然中间绕晕）

---

## 学到的

1. **SQL 组合题的难点不是"会不会某个函数"，而是"CTE 链怎么拆"**。这跟 pandas 写 pipeline 一样——每个操作都会，连起来就找不到中间变量名。

2. **写之前先画 CTE 树**：每个 CTE 的输入是什么、输出是什么、下个 CTE 怎么用。不要上来就写 WITH。

3. **写完每个 CTE 在脑子里跑一行数据**，验证数值对不对。

4. **当一个问题有"两个不同数据源"时（本周活跃 vs 上周活跃），一定需要分叉成两个 CTE 链，最后 JOIN**，不能强塞进一个。

5. **排除法是 SQL 的利器**。复活用户不需要精确定义"间隔多久"，只需要"不是 NEW + 不是 RETAINED"。简单且不漏。

---

## 下一步

- 重写这道题，不用看答案
- 再练 2-3 道综合题，直到 CTE 拆解变成肌肉记忆
