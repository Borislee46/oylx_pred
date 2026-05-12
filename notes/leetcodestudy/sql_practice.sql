-- ═══════════════════════════════════════════════════════════════════════════════
-- 字节DS面试 · SQL专项（20题）
-- ═══════════════════════════════════════════════════════════════════════════════
-- 覆盖: 窗口函数 / 留存率 / 连续登录 / 漏斗分析 / 复购率 / 累计 / TopN
-- 语法以 Hive/MySQL 为准, 面试时确认方言
-- ═══════════════════════════════════════════════════════════════════════════════

-- ============================================================================
-- Part 0: 创建示例表 + 数据
-- ============================================================================

-- 用户表
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    register_date DATE,
    city VARCHAR(20)
);
INSERT INTO users VALUES
(1, '2024-01-01', '北京'),
(2, '2024-01-05', '上海'),
(3, '2024-01-10', '北京'),
(4, '2024-01-15', '深圳'),
(5, '2024-02-01', '杭州');

-- 订单表
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT,
    order_date DATE,
    amount DECIMAL(10, 2),
    category VARCHAR(20)
);
INSERT INTO orders VALUES
(1,  1, '2024-01-01', 100.0, '电子'),
(2,  1, '2024-01-02', 200.0, '服装'),
(3,  2, '2024-01-05', 150.0, '电子'),
(4,  1, '2024-01-15', 300.0, '图书'),
(5,  3, '2024-01-10', 50.0,  '图书'),
(6,  1, '2024-02-01', 250.0, '电子'),
(7,  2, '2024-02-05', 100.0, '服装'),
(8,  3, '2024-02-10', 80.0,  '电子'),
(9,  4, '2024-02-15', 200.0, '图书'),
(10, 1, '2024-02-20', 300.0, '电子'),
(11, 2, '2024-03-01', 120.0, '图书'),
(12, 3, '2024-03-05', 90.0,  '服装'),
(13, 1, '2024-03-10', 400.0, '电子'),
(14, 4, '2024-03-15', 60.0,  '图书'),
(15, 5, '2024-03-20', 180.0, '电子');

-- 登录表
DROP TABLE IF EXISTS login_log;
CREATE TABLE login_log (
    user_id INT,
    login_date DATE
);
INSERT INTO login_log VALUES
(1, '2024-01-01'), (1, '2024-01-02'), (1, '2024-01-03'),
(1, '2024-01-05'), (1, '2024-01-06'),
(2, '2024-01-01'), (2, '2024-01-02'),
(2, '2024-01-04'),
(3, '2024-01-01'), (3, '2024-01-02'), (3, '2024-01-03'),
(3, '2024-01-04'), (3, '2024-01-05');

-- 页面访问日志表
DROP TABLE IF EXISTS page_log;
CREATE TABLE page_log (
    user_id INT,
    page_name VARCHAR(20),
    visit_time DATETIME
);
INSERT INTO page_log VALUES
(1, '首页',  '2024-01-01 10:00:00'),
(1, '搜索页','2024-01-01 10:01:00'),
(1, '商品页','2024-01-01 10:02:00'),
(1, '下单页','2024-01-01 10:05:00'),
(1, '支付页','2024-01-01 10:06:00'),
(2, '首页',  '2024-01-01 11:00:00'),
(2, '搜索页','2024-01-01 11:01:00'),
(2, '商品页','2024-01-01 11:02:00'),
(2, '首页',  '2024-01-01 11:05:00'), -- 又回首页
(3, '首页',  '2024-01-01 12:00:00'),
(3, '搜索页','2024-01-01 12:01:00');


-- ============================================================================
-- Part 1: 窗口函数基础（6题）
-- ============================================================================

-- ── SQL-1. 每个用户的订单按金额排名（ROW_NUMBER / RANK / DENSE_RANK）──────
-- 面试要点: 讲清楚三者区别
SELECT
    user_id,
    order_id,
    amount,
    ROW_NUMBER()   OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn,
    RANK()         OVER (PARTITION BY user_id ORDER BY amount DESC) AS rk,
    DENSE_RANK()   OVER (PARTITION BY user_id ORDER BY amount DESC) AS dr
FROM orders;
/*
ROW_NUMBER: 1,2,3,4... (同值也不同号)
RANK:       1,1,3,4... (同值同号, 跳号)
DENSE_RANK: 1,1,2,3... (同值同号, 不跳号)
*/


-- ── SQL-2. 每个用户的首次订单和末次订单 ────────────────────────────────────
-- FIRST_VALUE / LAST_VALUE 或 子查询 + ROW_NUMBER
-- 方法1: 窗口函数
SELECT DISTINCT
    user_id,
    FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_date
                              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_amount,
    LAST_VALUE(amount)  OVER (PARTITION BY user_id ORDER BY order_date
                              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_amount
FROM orders;

-- 方法2: ROW_NUMBER（更通用，Hive也支持）
WITH ranked AS (
    SELECT
        user_id,
        amount,
        order_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date ASC)  AS rn_asc,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date DESC) AS rn_desc
    FROM orders
)
SELECT
    user_id,
    MAX(CASE WHEN rn_asc = 1  THEN amount END)  AS first_amount,
    MAX(CASE WHEN rn_desc = 1 THEN amount END)  AS last_amount
FROM ranked
GROUP BY user_id;


-- ── SQL-3. 计算每月金额环比增长 ──────────────────────────────────────────────
-- LAG 获取前一行值
WITH monthly AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') AS mon,
        SUM(amount) AS total
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT
    mon,
    total,
    LAG(total, 1) OVER (ORDER BY mon) AS prev_total,
    ROUND((total - LAG(total, 1) OVER (ORDER BY mon))
          / LAG(total, 1) OVER (ORDER BY mon) * 100, 2) AS growth_pct
FROM monthly;


-- ── SQL-4. 计算累计销售额 ────────────────────────────────────────────────────
-- SUM() OVER (ORDER BY ...) 滚动求和
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS cumulative_sum
FROM orders
ORDER BY order_date;

-- 按月累计
WITH monthly AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') AS mon,
        SUM(amount) AS total
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT
    mon,
    total,
    SUM(total) OVER (ORDER BY mon) AS cumulative_sales
FROM monthly;


-- ── SQL-5. 移动平均（近3天滚动）──────────────────────────────────────────────
-- ROWS BETWEEN N PRECEDING AND CURRENT ROW
SELECT
    order_date,
    SUM(amount) AS daily_total,
    AVG(SUM(amount)) OVER (ORDER BY order_date
                           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3day
FROM orders
GROUP BY order_date
ORDER BY order_date;


-- ── SQL-6. 中位数计算 ────────────────────────────────────────────────────────
-- PERCENTILE 或 ROW_NUMBER 定位中间行
-- 方法: ROW_NUMBER 双排序
WITH ranked AS (
    SELECT
        amount,
        ROW_NUMBER() OVER (ORDER BY amount) AS rn,
        COUNT(*) OVER () AS total
    FROM orders
)
SELECT
    AVG(amount) AS median
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0));


-- ============================================================================
-- Part 2: 留存分析（3题）—— 字节DS必考
-- ============================================================================

-- ── SQL-7. 次日留存率 ────────────────────────────────────────────────────────
-- 定义: 某天注册的用户, 第二天还登录的比例
WITH first_login AS (
    SELECT user_id, MIN(login_date) AS first_date
    FROM login_log
    GROUP BY user_id
),
retention AS (
    SELECT
        f.first_date,
        COUNT(DISTINCT f.user_id) AS new_users,
        COUNT(DISTINCT l.user_id) AS day1_retained
    FROM first_login f
    LEFT JOIN login_log l
        ON f.user_id = l.user_id
        AND l.login_date = DATE_ADD(f.first_date, INTERVAL 1 DAY)
    GROUP BY f.first_date
)
SELECT
    first_date,
    new_users,
    day1_retained,
    ROUND(day1_retained * 100.0 / new_users, 2) AS day1_retention_pct
FROM retention
ORDER BY first_date;


-- ── SQL-8. 多日留存（次日/3日/7日）───────────────────────────────────────────
WITH first_login AS (
    SELECT user_id, MIN(login_date) AS first_date
    FROM login_log
    GROUP BY user_id
)
SELECT
    f.first_date,
    COUNT(DISTINCT f.user_id) AS new_users,
    COUNT(DISTINCT d1.user_id) AS d1,
    COUNT(DISTINCT d3.user_id) AS d3,
    COUNT(DISTINCT d7.user_id) AS d7,
    ROUND(COUNT(DISTINCT d1.user_id) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d1_pct,
    ROUND(COUNT(DISTINCT d3.user_id) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d3_pct,
    ROUND(COUNT(DISTINCT d7.user_id) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d7_pct
FROM first_login f
LEFT JOIN login_log d1 ON f.user_id = d1.user_id
    AND d1.login_date = DATE_ADD(f.first_date, INTERVAL 1 DAY)
LEFT JOIN login_log d3 ON f.user_id = d3.user_id
    AND d3.login_date = DATE_ADD(f.first_date, INTERVAL 3 DAY)
LEFT JOIN login_log d7 ON f.user_id = d7.user_id
    AND d7.login_date = DATE_ADD(f.first_date, INTERVAL 7 DAY)
GROUP BY f.first_date
ORDER BY f.first_date;


-- ── SQL-9. Cohort 留存矩阵 ────────────────────────────────────────────────────
-- 按注册月份分群, 看每群在各月的留存
WITH first_month AS (
    SELECT user_id, DATE_FORMAT(MIN(login_date), '%Y-%m') AS cohort_month
    FROM login_log
    GROUP BY user_id
),
monthly_login AS (
    SELECT DISTINCT user_id, DATE_FORMAT(login_date, '%Y-%m') AS active_month
    FROM login_log
)
SELECT
    f.cohort_month,
    f.user_id,
    m.active_month
FROM first_month f
LEFT JOIN monthly_login m ON f.user_id = m.user_id AND m.active_month >= f.cohort_month;
-- 然后在 Excel / Python 做透视: 行=cohort, 列=月份偏移量


-- ============================================================================
-- Part 3: 连续登录 / 连续行为（3题）—— 字节DS高频
-- ============================================================================

-- ── SQL-10. 连续登录天数 ─────────────────────────────────────────────────────
-- 核心技巧: 日期 - ROW_NUMBER = 连续组的标识
-- login_date - RN 对于连续日期是相同的
WITH t1 AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
    FROM (SELECT DISTINCT user_id, login_date FROM login_log) t
),
t2 AS (
    SELECT
        user_id,
        DATE_SUB(login_date, INTERVAL rn DAY) AS grp_key   -- 连续组标识
    FROM t1
)
SELECT
    user_id,
    grp_key,
    COUNT(*) AS consecutive_days
FROM t2
GROUP BY user_id, grp_key
ORDER BY user_id, consecutive_days DESC;

-- 面试口述:
-- 1. 先去重(同一天多次登录算一次)
-- 2. ROW_NUMBER 按用户分组、按日期排序
-- 3. 日期 - 行号: 连续的日期会得到相同的差值, 差值相同 = 同一连续段
-- 4. GROUP BY 用户+差值, COUNT(*) = 连续天数

-- ── SQL-11. 最长连续登录天数 ────────────────────────────────────────────────
WITH t1 AS (
    SELECT DISTINCT user_id, login_date FROM login_log
),
t2 AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
    FROM t1
),
t3 AS (
    SELECT
        user_id,
        COUNT(*) AS streak
    FROM t2
    GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
)
SELECT
    user_id,
    MAX(streak) AS max_consecutive_days
FROM t3
GROUP BY user_id
ORDER BY max_consecutive_days DESC;


-- ── SQL-12. 连续3天以上登录的用户 ────────────────────────────────────────────
WITH t1 AS (
    SELECT DISTINCT user_id, login_date FROM login_log
),
t2 AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
    FROM t1
),
t3 AS (
    SELECT
        user_id,
        COUNT(*) AS streak
    FROM t2
    GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
    HAVING COUNT(*) >= 3
)
SELECT DISTINCT user_id FROM t3;


-- ============================================================================
-- Part 4: 漏斗分析（2题）
-- ============================================================================

-- ── SQL-13. 页面漏斗转化率 ───────────────────────────────────────────────────
-- 首页 → 搜索页 → 商品页 → 下单页 → 支付页
-- 每个用户只取第一次访问该页
WITH steps AS (
    SELECT
        user_id,
        page_name,
        MIN(visit_time) AS first_visit
    FROM page_log
    WHERE page_name IN ('首页', '搜索页', '商品页', '下单页', '支付页')
    GROUP BY user_id, page_name
),
funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN page_name = '首页'  THEN 1 ELSE 0 END) AS s1,
        MAX(CASE WHEN page_name = '搜索页' THEN 1 ELSE 0 END) AS s2,
        MAX(CASE WHEN page_name = '商品页' THEN 1 ELSE 0 END) AS s3,
        MAX(CASE WHEN page_name = '下单页' THEN 1 ELSE 0 END) AS s4,
        MAX(CASE WHEN page_name = '支付页' THEN 1 ELSE 0 END) AS s5
    FROM steps
    GROUP BY user_id
    -- 严格漏斗: 确保每一步时间在上一步之后
    -- (简化版: 只检查是否访问过)
)
SELECT
    SUM(s1) AS 首页,
    SUM(s2) AS 搜索页,
    SUM(s3) AS 商品页,
    SUM(s4) AS 下单页,
    SUM(s5) AS 支付页,
    ROUND(SUM(s2) * 100.0 / SUM(s1), 2) AS rate_1to2,
    ROUND(SUM(s3) * 100.0 / SUM(s2), 2) AS rate_2to3,
    ROUND(SUM(s4) * 100.0 / SUM(s3), 2) AS rate_3to4,
    ROUND(SUM(s5) * 100.0 / SUM(s4), 2) AS rate_4to5
FROM funnel;


-- ── SQL-14. 严格漏斗（步骤必须按时间先后）────────────────────────────────────
-- 使用 LEAD 或 LAG 确保步骤顺序
WITH user_page_seq AS (
    SELECT
        user_id,
        page_name,
        visit_time,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY visit_time) AS step
    FROM page_log
),
-- 找每个用户第一个到达的页面
first_pages AS (
    SELECT
        user_id,
        page_name,
        step,
        LEAD(page_name, 1) OVER (PARTITION BY user_id ORDER BY step) AS next_page,
        LEAD(page_name, 2) OVER (PARTITION BY user_id ORDER BY step) AS next_next_page,
        LEAD(page_name, 3) OVER (PARTITION BY user_id ORDER BY step) AS next3_page,
        LEAD(page_name, 4) OVER (PARTITION BY user_id ORDER BY step) AS next4_page
    FROM user_page_seq
)
SELECT
    COUNT(DISTINCT CASE WHEN page_name = '首页' THEN user_id END) AS u1,
    COUNT(DISTINCT CASE WHEN page_name = '首页' AND next_page = '搜索页' THEN user_id END) AS u2,
    COUNT(DISTINCT CASE WHEN page_name = '首页' AND next_page = '搜索页'
                             AND next_next_page = '商品页' THEN user_id END) AS u3
FROM first_pages;


-- ============================================================================
-- Part 5: 复购 / 用户分层（3题）
-- ============================================================================

-- ── SQL-15. 复购率 ───────────────────────────────────────────────────────────
-- 定义: 有过 ≥2 次购买的用户占比
WITH user_order_cnt AS (
    SELECT
        user_id,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY user_id
)
SELECT
    COUNT(CASE WHEN order_count >= 2 THEN 1 END) AS repeat_users,
    COUNT(*) AS total_users,
    ROUND(COUNT(CASE WHEN order_count >= 2 THEN 1 END) * 100.0 / COUNT(*), 2) AS repurchase_rate
FROM user_order_cnt;


-- ── SQL-16. 用户价值分层（RFM 简化版）────────────────────────────────────────
-- Recency: 距今天数, Frequency: 订单数, Monetary: 总金额
SELECT
    user_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency,
    COUNT(*) AS frequency,
    SUM(amount) AS monetary,
    CASE
        WHEN DATEDIFF(CURRENT_DATE, MAX(order_date)) <= 30
             AND COUNT(*) >= 3 AND SUM(amount) >= 500 THEN '高价值'
        WHEN DATEDIFF(CURRENT_DATE, MAX(order_date)) <= 30
             AND COUNT(*) >= 2 THEN '活跃用户'
        WHEN DATEDIFF(CURRENT_DATE, MAX(order_date)) > 60
             AND COUNT(*) < 2 THEN '流失用户'
        ELSE '普通用户'
    END AS segment
FROM orders
GROUP BY user_id;


-- ── SQL-17. 每个用户最近N次购买 → 用于判断是否流失 ───────────────────────
WITH ranked AS (
    SELECT
        user_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date DESC) AS rn
    FROM orders
)
SELECT
    user_id,
    MAX(CASE WHEN rn = 1 THEN order_date END) AS last_order,
    MAX(CASE WHEN rn = 2 THEN order_date END) AS prev_order,
    MAX(CASE WHEN rn = 3 THEN order_date END) AS prev2_order
FROM ranked
WHERE rn <= 3
GROUP BY user_id;


-- ============================================================================
-- Part 6: 分组 Top N + 综合（3题）
-- ============================================================================

-- ── SQL-18. 每个城市订单金额最高的用户 ───────────────────────────────────────
WITH ranked AS (
    SELECT
        u.city,
        u.user_id,
        SUM(o.amount) AS total_amount,
        ROW_NUMBER() OVER (PARTITION BY u.city ORDER BY SUM(o.amount) DESC) AS rn
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.city, u.user_id
)
SELECT city, user_id, total_amount
FROM ranked
WHERE rn = 1;


-- ── SQL-19. GMV = 用户数 × 客单价 × 购买频次 (拆解分析法) ───────────────
-- 面试会问: "GMV下降了怎么办? 怎么拆?"
-- SQL层面: 先把三要素算出来
SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS mon,
    COUNT(DISTINCT user_id) AS buyer_cnt,        -- 购买用户数
    COUNT(*) / COUNT(DISTINCT user_id) AS avg_freq,  -- 人均频次
    SUM(amount) / COUNT(*) AS avg_order_value,    -- 笔单价
    SUM(amount) AS gmv
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY mon;


-- ── SQL-20. 新老用户贡献 ──────────────────────────────────────────────────────
WITH first_order AS (
    SELECT user_id, MIN(order_date) AS first_date FROM orders GROUP BY user_id
)
SELECT
    DATE_FORMAT(o.order_date, '%Y-%m') AS mon,
    CASE WHEN o.order_date = f.first_date THEN '新用户' ELSE '老用户' END AS user_type,
    COUNT(DISTINCT o.user_id) AS users,
    SUM(o.amount) AS gmv
FROM orders o
JOIN first_order f ON o.user_id = f.user_id
GROUP BY
    DATE_FORMAT(o.order_date, '%Y-%m'),
    CASE WHEN o.order_date = f.first_date THEN '新用户' ELSE '老用户' END
ORDER BY mon, user_type;


-- ═══════════════════════════════════════════════════════════════════════════════
-- Part 7: 字节难度综合题（5题）🆕
-- ═══════════════════════════════════════════════════════════════════════════════
-- 特点: 每道题 = 2-4个模板嵌套，CTE连环套，字数多
-- 字节面试真实难度: 给你一道题，30分钟写SQL+解释
-- 本文件需要已有的 users / orders / login_log / page_log 表
-- ═══════════════════════════════════════════════════════════════════════════════


-- ── COMPLEX-1. 高质量用户的留存率 ───────────────────────────────────────
/*
需求:
  定义"高质量用户"= 注册后7天内下单≥2次
  分别计算高质量用户和普通用户的:
  - 注册人数
  - 次日留存率、3日留存率、7日留存率
  - 按注册月份分组

涉及的模板:
  - 用户分层 (注册7天内下单次数)
  - 多日留存 (1/3/7日 LEFT JOIN)
  - 按月份分组
  - 条件聚合
*/

WITH user_quality AS (
    -- 判断每个用户是否高质量: 注册7天内下单≥2次
    SELECT
        u.user_id,
        u.register_date,
        DATE_FORMAT(u.register_date, '%Y-%m') AS cohort,
        CASE WHEN COUNT(o.order_id) >= 2 THEN '高质量' ELSE '普通' END AS quality
    FROM users u
    LEFT JOIN orders o
        ON u.user_id = o.user_id
        AND o.order_date BETWEEN u.register_date
                             AND DATE_ADD(u.register_date, INTERVAL 7 DAY)
    GROUP BY u.user_id, u.register_date
),
retention AS (
    SELECT
        q.cohort,
        q.quality,
        COUNT(DISTINCT q.user_id) AS total_users,
        COUNT(DISTINCT d1.user_id) AS d1_retained,
        COUNT(DISTINCT d3.user_id) AS d3_retained,
        COUNT(DISTINCT d7.user_id) AS d7_retained
    FROM user_quality q
    LEFT JOIN login_log d1
        ON q.user_id = d1.user_id
        AND d1.login_date = DATE_ADD(q.register_date, INTERVAL 1 DAY)
    LEFT JOIN login_log d3
        ON q.user_id = d3.user_id
        AND d3.login_date = DATE_ADD(q.register_date, INTERVAL 3 DAY)
    LEFT JOIN login_log d7
        ON q.user_id = d7.user_id
        AND d7.login_date = DATE_ADD(q.register_date, INTERVAL 7 DAY)
    GROUP BY q.cohort, q.quality
)
SELECT
    cohort,
    quality,
    total_users,
    ROUND(d1_retained * 100.0 / total_users, 1) AS d1_rate,
    ROUND(d3_retained * 100.0 / total_users, 1) AS d3_rate,
    ROUND(d7_retained * 100.0 / total_users, 1) AS d7_rate
FROM retention
ORDER BY cohort, quality;


-- ── COMPLEX-2. 连续活跃用户的GMV贡献分析 ─────────────────────────────
/*
需求:
  找连续登录≥3天的所有用户, 列出:
  - user_id, 连续段起止日期, 连续天数
  - 该连续段内的购买金额(GMV)
  - 该GMV占该用户历史总GMV的比例
  按连续天数降序排列, 取Top 20

涉及的模板:
  - 连续登录 (ROW_NUMBER + date_diff分组)
  - 订单关联 + 区间求和
  - 窗口函数计算总GMV
  - TopN排序
*/

WITH dedup_login AS (
    SELECT DISTINCT user_id, login_date FROM login_log
),
login_rn AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
    FROM dedup_login
),
streaks AS (
    SELECT
        user_id,
        DATE_SUB(login_date, INTERVAL rn DAY) AS grp_key,
        MIN(login_date) AS streak_start,
        MAX(login_date) AS streak_end,
        COUNT(*) AS streak_days
    FROM login_rn
    GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
    HAVING COUNT(*) >= 3
),
streak_gmv AS (
    SELECT
        s.user_id,
        s.streak_start,
        s.streak_end,
        s.streak_days,
        COALESCE(SUM(o.amount), 0) AS streak_gmv
    FROM streaks s
    LEFT JOIN orders o
        ON s.user_id = o.user_id
        AND o.order_date BETWEEN s.streak_start AND s.streak_end
    GROUP BY s.user_id, s.streak_start, s.streak_end, s.streak_days
),
user_total AS (
    SELECT user_id, SUM(amount) AS total_gmv FROM orders GROUP BY user_id
)
SELECT
    g.user_id,
    g.streak_start,
    g.streak_end,
    g.streak_days,
    g.streak_gmv,
    u.total_gmv,
    ROUND(g.streak_gmv * 100.0 / NULLIF(u.total_gmv, 0), 1) AS gmv_pct
FROM streak_gmv g
JOIN user_total u ON g.user_id = u.user_id
ORDER BY g.streak_days DESC, g.streak_gmv DESC
LIMIT 20;


-- ── COMPLEX-3. 漏斗各环节的每日转化率 + 月度环比 ─────────────────────
/*
需求:
  漏斗: 首页 → 搜索 → 商品 → 下单 → 支付
  按天计算每个环节的访问人数(UV)和转化率
  再按月汇总, 计算各环节转化率的月度环比

涉及的模板:
  - 漏斗 (CASE WHEN + GROUP BY)
  - 按天聚合 → 按月聚合 (两层GROUP BY)
  - 环比 (LAG)
*/

WITH daily_funnel AS (
    SELECT
        DATE(visit_time) AS dt,
        COUNT(DISTINCT CASE WHEN page_name = '首页'  THEN user_id END) AS s1,
        COUNT(DISTINCT CASE WHEN page_name = '搜索页' THEN user_id END) AS s2,
        COUNT(DISTINCT CASE WHEN page_name = '商品页' THEN user_id END) AS s3,
        COUNT(DISTINCT CASE WHEN page_name = '下单页' THEN user_id END) AS s4,
        COUNT(DISTINCT CASE WHEN page_name = '支付页' THEN user_id END) AS s5
    FROM page_log
    GROUP BY DATE(visit_time)
),
monthly_funnel AS (
    SELECT
        DATE_FORMAT(dt, '%Y-%m') AS mon,
        SUM(s1) AS m1, SUM(s2) AS m2, SUM(s3) AS m3,
        SUM(s4) AS m4, SUM(s5) AS m5
    FROM daily_funnel
    GROUP BY DATE_FORMAT(dt, '%Y-%m')
),
rates AS (
    SELECT
        mon,
        ROUND(m2 * 100.0 / NULLIF(m1, 0), 2) AS rate_1to2,
        ROUND(m3 * 100.0 / NULLIF(m2, 0), 2) AS rate_2to3,
        ROUND(m4 * 100.0 / NULLIF(m3, 0), 2) AS rate_3to4,
        ROUND(m5 * 100.0 / NULLIF(m4, 0), 2) AS rate_4to5
    FROM monthly_funnel
)
SELECT
    mon,
    rate_1to2,
    LAG(rate_1to2) OVER (ORDER BY mon) AS prev_rate_1to2,
    ROUND(rate_1to2 - LAG(rate_1to2) OVER (ORDER BY mon), 2) AS change_1to2,
    rate_2to3,
    LAG(rate_2to3) OVER (ORDER BY mon) AS prev_rate_2to3,
    ROUND(rate_2to3 - LAG(rate_2to3) OVER (ORDER BY mon), 2) AS change_2to3,
    rate_3to4,
    LAG(rate_3to4) OVER (ORDER BY mon) AS prev_rate_3to4,
    ROUND(rate_3to4 - LAG(rate_3to4) OVER (ORDER BY mon), 2) AS change_3to4,
    rate_4to5,
    LAG(rate_4to5) OVER (ORDER BY mon) AS prev_rate_4to5,
    ROUND(rate_4to5 - LAG(rate_4to5) OVER (ORDER BY mon), 2) AS change_4to5
FROM rates
ORDER BY mon;


-- ── COMPLEX-4. 用户LTV与留存周期关系 ──────────────────────────────────
/*
需求:
  计算每个用户的:
  - 存活天数 (末次登录 - 首次登录)
  - 总GMV
  - 总订单数
  然后按"存活天数"分段: 0天 / 1-7天 / 8-30天 / 31-90天 / 90天+
  每段输出: 用户数、人均GMV、人均频次、客单价
  再按城市拆分

涉及的模板:
  - 用户生命周期 (MIN/MAX/DATEDIFF)
  - 分段统计 (CASE WHEN 分段)
  - 多维度拆解 (城市)
  - 多个聚合指标嵌套
*/

WITH user_ltv AS (
    SELECT
        u.user_id,
        u.city,
        MIN(l.login_date) AS first_login,
        MAX(l.login_date) AS last_login,
        DATEDIFF(MAX(l.login_date), MIN(l.login_date)) AS lifespan,
        COALESCE(SUM(o.amount), 0) AS total_gmv,
        COUNT(DISTINCT o.order_id) AS order_cnt
    FROM users u
    LEFT JOIN login_log l ON u.user_id = l.user_id
    LEFT JOIN orders o    ON u.user_id = o.user_id
    GROUP BY u.user_id, u.city
),
segmented AS (
    SELECT
        city,
        CASE
            WHEN lifespan = 0 THEN '0天(仅注册日)'
            WHEN lifespan <= 7  THEN '1-7天'
            WHEN lifespan <= 30 THEN '8-30天'
            WHEN lifespan <= 90 THEN '31-90天'
            ELSE '90天+'
        END AS lifespan_segment,
        user_id,
        total_gmv,
        order_cnt
    FROM user_ltv
)
SELECT
    city,
    lifespan_segment,
    COUNT(DISTINCT user_id) AS user_cnt,
    ROUND(AVG(total_gmv), 2) AS avg_ltv,
    ROUND(AVG(order_cnt), 2) AS avg_freq,
    ROUND(SUM(total_gmv) / NULLIF(SUM(order_cnt), 0), 2) AS avg_aov
FROM segmented
GROUP BY city, lifespan_segment
ORDER BY city,
    FIELD(lifespan_segment, '0天(仅注册日)', '1-7天', '8-30天', '31-90天', '90天+');


-- ── COMPLEX-5. 复购周期 + 用户分层 + 中位数 ─────────────────────────
/*
需求:
  对每个有≥2次购买的用户:
  - 计算相邻购买间隔天数 (用LAG)
  - 计算该用户复购间隔的均值和中位数
  按用户价值分层(高/中/低)后:
  - 每层的人数、平均复购间隔、中位复购间隔
  - 每层"7天内复购"的用户占比

涉及的模板:
  - LAG 计算间隔
  - 中位数 (ROW_NUMBER双排序)
  - 用户分层 (RFM简化)
  - 条件聚合 + 多层CTE
*/

WITH order_gap AS (
    SELECT
        user_id,
        order_date,
        amount,
        LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS prev_date,
        DATEDIFF(order_date,
                 LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date)
                ) AS gap_days
    FROM orders
),
user_gap_stats AS (
    SELECT
        user_id,
        AVG(gap_days) AS avg_gap,
        COUNT(*) AS order_cnt,
        SUM(amount) AS total_gmv,
        MAX(order_date) AS last_date
    FROM order_gap
    WHERE gap_days IS NOT NULL                   -- 过滤掉首次购买(无间隔)
    GROUP BY user_id
    HAVING COUNT(*) >= 1                         -- 至少2次购买 = 至少1个间隔
),
user_segment AS (
    SELECT
        user_id,
        avg_gap,
        order_cnt,
        total_gmv,
        CASE
            WHEN total_gmv >= 500 AND order_cnt >= 4 THEN '高价值'
            WHEN total_gmv >= 200 AND order_cnt >= 2 THEN '中价值'
            ELSE '低价值'
        END AS segment
    FROM user_gap_stats
),
-- 中位数: 对每层的所有 avg_gap 排序取中间
gap_ranked AS (
    SELECT
        segment,
        avg_gap,
        ROW_NUMBER() OVER (PARTITION BY segment ORDER BY avg_gap) AS rn,
        COUNT(*) OVER (PARTITION BY segment) AS total
    FROM user_segment
),
segment_median AS (
    SELECT
        segment,
        AVG(avg_gap) AS median_gap               -- 取中间1-2个值的平均
    FROM gap_ranked
    WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0))
    GROUP BY segment
),
segment_stats AS (
    SELECT
        segment,
        COUNT(*) AS user_cnt,
        ROUND(AVG(avg_gap), 1) AS mean_gap,
        ROUND(SUM(CASE WHEN avg_gap <= 7 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_repurchase_7d
    FROM user_segment
    GROUP BY segment
)
SELECT
    s.segment,
    s.user_cnt,
    s.mean_gap,
    m.median_gap,
    s.pct_repurchase_7d
FROM segment_stats s
JOIN segment_median m ON s.segment = m.segment
ORDER BY FIELD(s.segment, '高价值', '中价值', '低价值');


-- ═══════════════════════════════════════════════════════════════════════════════
-- 综合题自检清单
-- ═══════════════════════════════════════════════════════════════════════════════
-- □ COMPLEX-1 高质量留存: 用户分层+留存+按月 ← 3模板嵌套
-- □ COMPLEX-2 连续活跃GMV: 连续登录+窗口+分组TopN ← 4模板嵌套
-- □ COMPLEX-3 漏斗环比: 漏斗+聚合+环比 ← 3模板嵌套
-- □ COMPLEX-4 LTV分段: 生命周期+分段+城市拆解 ← 3模板嵌套
-- □ COMPLEX-5 复购中位数: LAG+中位数+分层 ← 4模板嵌套
--
-- 如果这5道能30分钟内脱手写出 → 字节DS SQL轮铁过


-- ═══════════════════════════════════════════════════════════════════════════════
-- 面试 Checklist
-- ═══════════════════════════════════════════════════════════════════════════════
-- □ 窗口函数: ROW_NUMBER / RANK / DENSE_RANK / LAG / LEAD / SUM OVER
-- □ 连续登录: login_date - ROW_NUMBER = grp_key (背下来)
-- □ 留存率: LEFT JOIN + DATE_ADD / DATE_SUB
-- □ 漏斗: GROUP BY + CASE WHEN 或 LEAD
-- □ 复购: GROUP BY + HAVING COUNT(*) >= 2
-- □ TOP N: ROW_NUMBER + WHERE rn = 1/2/3
-- □ 日期函数: DATE_FORMAT / DATE_ADD / DATE_SUB / DATEDIFF
-- □ 注意: COUNT(*) vs COUNT(DISTINCT) 的区别
-- □ 注意: WHERE vs HAVING 的区别
-- □ 注意: LEFT JOIN 用于留存 (要保留所有新用户, 即使是0留存)
