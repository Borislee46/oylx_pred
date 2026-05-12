-- ═══════════════════════════════════════════════════════════════════════════════
-- 手写测试题：周活跃用户(WAU)构成拆解
-- ═══════════════════════════════════════════════════════════════════════════════
-- 时间限制：40分钟
-- 规则：不查AI，不查笔记，纯手写
-- 数据：下面已建好表 + 插入示例数据，跑通就算过
-- ═══════════════════════════════════════════════════════════════════════════════


-- ============================================================================
-- 【需求描述】
-- ============================================================================
-- 你是字节跳动某产品的数据分析师。PM 问你：
-- "我们每周的活跃用户(WAU)到底是哪些人？是新来的、一直来的、还是走了又回来的？"
--
-- 请写一段 SQL，把每个自然周的 WAU 拆成以下 4 类用户，输出每周各类的人数。


-- ============================================================================
-- 【指标定义 — 请仔细读】
-- ============================================================================
--
-- 自然周定义：
--   用 WEEK(date, 1) 获取 ISO 周编号。
--   "2024年第1周" = WEEK('2024-01-01', 1)。
--   同一年内用 week_num 即可，不必跨年。
--
-- 四种用户类型（每人每周只能属于一种，按优先级判定）：
--
--   (1) 新用户 (New)：
--       定义：本周是 TA 在系统中的首次活跃周。
--       公式：本周 = MIN(活跃周) OVER (PARTITION BY user_id)
--       优先级：最高 — 先判定，判定完就排除。
--
--   (2) 持续活跃 (Retained)：
--       定义：上周也活跃 且 本周也活跃。
--       公式：存在一条登录记录，登录周 = 本周 - 1。
--       注意：这里"上周活跃"是指自然周，不是7天前。用 week_num - 1。
--       前提：不是新用户。
--
--   (3) 复活用户 (Resurrected)：
--       定义：不是新用户、也不是持续活跃，但本周活跃了。
--       也就是说：之前来过（至少某周活跃过），中间断了至少一周，本周又来了。
--       判定逻辑：不是 New 也不是 Retained，就落在这里。
--
--   (4) 流失用户 (Churned)：
--       定义：本周不活跃。但上周活跃过。
--       注意：这类用户不出现在本周的活跃名单里，但 PM 想知道"上周还在、这周就没了"的人数。
--       所以每个周要算的是：上周活跃 且 本周不活跃。
--       这个计数是独立于前三类的 — 前三类统计的是"本周活跃的人怎么来的"，
--       流失用户统计的是"上周活跃的人去哪了"。
--
-- 输出格式（示例）：
--   year | week_num | new_users | retained_users | resurrected_users | churned_users
--   2024 | 1        | 120       | 300            | 45                | 18
--   2024 | 2        | 80        | 280            | 52                | 22
--   ...  (churned_users 是"上周活跃本周不活跃"的人数)


-- ============================================================================
-- 【示例数据】
-- ============================================================================

DROP TABLE IF EXISTS user_activity;
CREATE TABLE user_activity (
    user_id INT,
    active_date DATE
);

-- 插入一些精心设计的样本数据，方便手工验证：
-- user 1: 连续3周活跃 (week1, week2, week3)
-- user 2: 第1周活跃，第2周断，第3周回来 (复活)
-- user 3: 第1周活跃，第2周断，第3周断 (流失)
-- user 4: 第2周首次出现，第3周继续 (持续)
-- user 5: 只在第1周活跃 (第2周流失)
-- user 6: 第1第3周活跃，第2周断 (复活)
-- user 7: 第2周新用户，第3周流失

INSERT INTO user_activity VALUES
-- Week 1 (2023-12-31 ~ 2024-01-06) → week_num=1
(1, '2024-01-01'),
(1, '2024-01-03'),
(2, '2024-01-02'),
(3, '2024-01-04'),
(5, '2024-01-01'),
(6, '2024-01-05'),
-- Week 2 (2024-01-07 ~ 2024-01-13) → week_num=2
(1, '2024-01-08'),
(1, '2024-01-10'),
(4, '2024-01-09'),
(7, '2024-01-11'),
-- Week 3 (2024-01-14 ~ 2024-01-20) → week_num=3
(1, '2024-01-15'),
(2, '2024-01-16'),
(4, '2024-01-17'),
(6, '2024-01-18');

-- ============================================================================
-- 【期望输出 — 你可以对照验证】
-- ============================================================================
--
-- 手算一遍：
--
-- Week 1 (week_num=1):
--   活跃用户: 1,2,3,5,6
--   New: 全部5人 (对所有人来说这都是首次活跃周)
--   Retained: 0 (没有上周)
--   Resurrected: 0 (没有上周，不存在"之前来过"的人)
--   Churned: 无法计算 (没有上周数据) → 输出 0 或 NULL
--
-- Week 2 (week_num=2):
--   活跃用户: 1,4,7
--   New: 4,7 (首次活跃周 = week2) → 2人
--   Retained: 1 (上周也活跃) → 1人
--   Resurrected: 0 (剩下来的没有，都是New或Retained)
--   Churned: 上周(1,2,3,5,6)中本周不活跃的: 2,3,5,6 → 4人
--
-- Week 3 (week_num=3):
--   活跃用户: 1,2,4,6
--   New: 0 (所有人都不是首次活跃)
--   Retained: 1,4 (上周week2也活跃) → 2人
--   Resurrected: 2,6 (不是New，不是Retained，但本周活跃)
--     - user 2: 第1周来过，第2周断，第3周来 → 复活
--     - user 6: 第1周来过，第2周断，第3周来 → 复活
--     → 2人
--   Churned: 上周(1,4,7)中本周不活跃的: 7 → 1人
--
-- 期望输出：
--   year | week_num | new | retained | resurrected | churned
--   2024 | 1        | 5   | 0        | 0           | 0
--   2024 | 2        | 2   | 1        | 0           | 4
--   2024 | 3        | 0   | 2        | 2           | 1


-- ============================================================================
with base as (select distinct user_id, week(date,1) as week_num, year(date) as year from user_activity),
lfa as (select *, min(week_num) over (partition by user_id) as new, case when lag(week_num, 1) over (partition by user_id, year) - week_num = 1 then 
'retained'  
 case when lag(week_num, 1) over (partition by user_id, year) - week_num = 2 then 'resurrected'
 case when lag(week_num,1)
end as label1
from base),
labels as (select lfa.*, coaleses(case when faw.new is not null then 'new' end,label1) as label2 faw.year as fy
left join (select * from lfa where faw is not null) faw 
on faw.user_id = lfa.user_id
), select year, week_num, count(new) as new,  from labels group by year, week_num

-- ============================================================================

-- 提示：
-- 1. 先用 DISTINCT 去重（同一周同一用户多次活跃只算一次）
-- 2. 用 YEARWEEK(date, 1) 或 WEEK(date, 1) 获取周编号
-- 3. 用 MIN() OVER (PARTITION BY user_id) 找每个用户的首次活跃周
-- 4. 用 LAG() 或自 JOIN 判断上周是否活跃
-- 5. 流失用户需要把"上周活跃"和"本周活跃"做 LEFT JOIN
-- 6. 多拆几个 CTE，不要试图一个查询写完













-- ═══════════════════════════════════════════════════════════════════════════════
-- 参考答案
-- ═══════════════════════════════════════════════════════════════════════════════

-- 第1步：每周每用户去重
WITH weekly_active AS (
    SELECT DISTINCT
        user_id,
        YEAR(active_date) AS year,
        WEEK(active_date, 1) AS week_num
    FROM user_activity
),

-- 第2步：每个用户的首次活跃周（NEW 判定用）
first_week AS (
    SELECT
        user_id,
        MIN(week_num) AS first_week_num
    FROM weekly_active
    GROUP BY user_id
),

-- 第3步：给每条活跃记录打标签
--   NEW:   本周 = 首次活跃周
--   RETAINED: 上周也在 weekly_active 中
--   RESURRECTED: 剩下的（不是NEW、不是RETAINED，但本周活跃）
user_label AS (
    SELECT
        wa.user_id,
        wa.year,
        wa.week_num,
        CASE
            WHEN wa.week_num = fw.first_week_num THEN 'new'
            -- 用 LAG 判断和上一次活跃周是否相邻
            -- 注意：LAG 拿的是"表中存在的上一行"，不是严格的上周
            -- 但这里我们用自关联的方式更稳妥
            ELSE NULL  -- 先标 NEW，剩下的后面再分
        END AS label
    FROM weekly_active wa
    JOIN first_week fw ON wa.user_id = fw.user_id
),

-- 第3步改进版：用自 JOIN 判断"上周是否也活跃"
--   这样比 LAG 更准确（LAG 对缺行情况不友好）
classified AS (
    SELECT
        wa.user_id,
        wa.year,
        wa.week_num,
        CASE
            -- NEW: 本周 = 首次活跃周
            WHEN wa.week_num = fw.first_week_num THEN 'new'
            -- RETAINED: 上周也活跃
            WHEN prev.user_id IS NOT NULL THEN 'retained'
            -- RESURRECTED: 活跃但不是 NEW 也不是 RETAINED
            ELSE 'resurrected'
        END AS user_type
    FROM weekly_active wa
    JOIN first_week fw ON wa.user_id = fw.user_id
    LEFT JOIN weekly_active prev
        ON wa.user_id = prev.user_id
        AND prev.week_num = wa.week_num - 1   -- 上周
),

-- 第4步：计算前三类（本周活跃用户）的汇总
active_summary AS (
    SELECT
        year,
        week_num,
        COUNT(CASE WHEN user_type = 'new'         THEN 1 END) AS new_users,
        COUNT(CASE WHEN user_type = 'retained'    THEN 1 END) AS retained_users,
        COUNT(CASE WHEN user_type = 'resurrected' THEN 1 END) AS resurrected_users
    FROM classified
    GROUP BY year, week_num
),

-- 第5步：计算流失用户（上周活跃但本周不活跃）
--   方法：取所有"上一周活跃"的用户，LEFT JOIN 本周活跃用户，对不上的就是流失
churned AS (
    SELECT
        wa.year,
        wa.week_num,
        COUNT(DISTINCT wa.user_id) AS churned_users
    FROM weekly_active wa
    LEFT JOIN weekly_active this_week
        ON wa.user_id = this_week.user_id
        AND this_week.week_num = wa.week_num + 1   -- 看下周还在不在
    WHERE this_week.user_id IS NULL                 -- 下周不在 = 流失
    GROUP BY wa.year, wa.week_num
)

-- 第6步：合并输出
SELECT
    a.year,
    a.week_num,
    COALESCE(a.new_users, 0)          AS new_users,
    COALESCE(a.retained_users, 0)     AS retained_users,
    COALESCE(a.resurrected_users, 0)  AS resurrected_users,
    COALESCE(c.churned_users, 0)      AS churned_users
FROM active_summary a
LEFT JOIN churned c
    ON a.year = c.year AND a.week_num = c.week_num
ORDER BY a.year, a.week_num;


-- ═══════════════════════════════════════════════════════════════════════════════
-- 关键点总结（你踩的坑）
-- ═══════════════════════════════════════════════════════════════════════════════
--
-- 坑1: LAG 只能看"表中存在的上一行"，不能判断"上周有没有"
--      → 用 LEFT JOIN ON prev.week_num = wa.week_num - 1 更准确
--
-- 坑2: Churn 必须从"上周活跃"的视角出发，LEFT JOIN 本周
--      → Churn 用户在本周活跃表中不存在，所以不能从本周活跃表推导
--
-- 坑3: CASE WHEN 不需要嵌套，多个 WHEN 平级即可
--      → CASE WHEN A THEN X WHEN B THEN Y ELSE Z END
--
-- 坑4: 首周 = MIN(week_num) 针对的是去重后的 weekly_active，
--      不能单独开窗口对 user_activity 直接 MIN(active_date)
--      → 因为是按周分组，不是按天








