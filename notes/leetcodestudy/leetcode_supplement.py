"""
字节DS面试 · 扩充题库（53题）
════════════════════════════════════════════════════════
在 50 题基础上补全：DP / 回溯 / 图（18题）/ 概率统计 / 字节高频
每道题：能脱手AC + 能讲WHY + 能变体
"""

from collections import Counter, defaultdict, deque
from heapq import heappush, heappop
from typing import List, Optional
import random
import math

# ═══════════════════════════════════════════════════════════════════════════════
# 一、动态规划 补全（10题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── DP-1. 打家劫舍 ─────────────────────────────────────────────────────────
# [1,2,3,1] → 4 (1+3)
def rob(nums: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    dp[i] = 到第i间房子的最大收益
    状态转移: dp[i] = max(不偷当前=dp[i-1], 偷当前=dp[i-2]+nums[i])
    WHY 两个变量: 只依赖前两个状态, 滚动优化到O(1)空间
    """
    prev2 = prev1 = 0                           # prev2=dp[i-2], prev1=dp[i-1]
    for num in nums:
        cur = max(prev1, prev2 + num)           # 偷或不偷
        prev2, prev1 = prev1, cur
    return prev1


# ── DP-2. 打家劫舍 II（环形）────────────────────────────────────────────
# [2,3,2] → 3 (首尾不能同时偷)
def rob_ii(nums: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    WHY 两次DP: 环形问题 → 拆成两个线性问题
    情况1: 偷 [0, n-2] (不偷最后一间)
    情况2: 偷 [1, n-1] (不偷第一间)
    取 max
    """
    if len(nums) == 1:
        return nums[0]

    def rob_range(lo, hi):
        prev2 = prev1 = 0
        for i in range(lo, hi + 1):
            cur = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, cur
        return prev1

    return max(rob_range(0, len(nums) - 2), rob_range(1, len(nums) - 1))


# ── DP-3. 打家劫舍 III（树形）────────────────────────────────────────────
# 二叉树, 相邻节点不能同时偷
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def rob_iii(root: Optional[TreeNode]) -> int:
    """
    时间 O(n)  空间 O(h)
    后序遍历, 每个节点返回 (偷当前, 不偷当前)
    WHY 返回元组: 偷当前 → 左右都不能偷; 不偷当前 → 左右可以偷(但不一定偷)
    这样每个节点独立决策
    """
    def dfs(node):
        if not node:
            return (0, 0)                       # (偷, 不偷)
        left = dfs(node.left)
        right = dfs(node.right)
        # 偷当前: 左右都不能偷
        rob_cur = node.val + left[1] + right[1]
        # 不偷当前: 左右各自取最大 (偷或不偷都可以)
        not_rob = max(left) + max(right)
        return (rob_cur, not_rob)

    return max(dfs(root))


# ── DP-4. 分割等和子集（01背包）────────────────────────────────────────────
# [1,5,11,5] → True (1+5+5=11)
def can_partition(nums: List[int]) -> bool:
    """
    时间 O(n·sum)  空间 O(sum)
    转化为: 能否选一些数使和 = total/2
    dp[j] = 用前i个数能否凑出和j
    01背包内层倒序: 保证每个数只用一次
    WHY 倒序: 正序会变成完全背包(可重复用), 倒序保证每数只用一次
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2

    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):    # 必须倒序!
            dp[j] = dp[j] or dp[j - num]
    return dp[target]


# ── DP-5. 最后一块石头的重量 II（01背包变体）───────────────────────────────
# [2,7,4,1,8,1] → 1
def last_stone_weight_ii(stones: List[int]) -> int:
    """
    时间 O(n·sum)  空间 O(sum)
    转化: 分成最接近的两堆, 差 = 最后重量
    找最接近 total/2 的子集和
    """
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = dp[j] or dp[j - stone]
    # 找最大的可凑出的值
    for j in range(target, -1, -1):
        if dp[j]:
            return total - 2 * j
    return 0


# ── DP-6. 零钱兑换II（完全背包组合数）─────────────────────────────────────
# amount=5, coins=[1,2,5] → 4
def change(amount: int, coins: List[int]) -> int:
    """
    时间 O(n·amount)  空间 O(amount)
    完全背包求组合数: dp[j] = 凑成j的组合数
    WHY 外层coin内层j: 这样保证硬币顺序固定 → 组合数(不是排列数)
    如果内外层反过来, 会变成排列数 (1+2 和 2+1 算两种)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    return dp[amount]


# ── DP-7. 买卖股票的最佳时机 ───────────────────────────────────────────────
# [7,1,5,3,6,4] → 5 (1买6卖)
def max_profit_i(prices: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    只能买卖一次: 遍历时记录历史最低价, 当天卖出的利润 = price - min_price
    WHY one pass: 每天只需要知道"之前的最低价", 不需要未来信息
    """
    min_price = float("inf")
    ans = 0
    for price in prices:
        min_price = min(min_price, price)
        ans = max(ans, price - min_price)
    return ans


# ── DP-8. 买卖股票的最佳时机 II（多次交易）───────────────────────────────
# [7,1,5,3,6,4] → 7 (1→5+3→6)
def max_profit_ii(prices: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    可以任意多次买卖 → 贪心: 只要明天涨, 今天买明天卖
    WHY 贪心正确: 不限交易次数 → 所有上涨区间都吃到
    等价于把所有的 prices[i+1] - prices[i] > 0 累加起来
    """
    return sum(max(0, prices[i + 1] - prices[i]) for i in range(len(prices) - 1))


# ── DP-9. 买卖股票的最佳时机 III（最多2次交易）───────────────────────────
# [3,3,5,0,0,3,1,4] → 6
def max_profit_iii(prices: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    状态机DP: 5个状态 → 滚动到4个变量
    buy1: 完成1次买入后的最大余额
    sell1: 完成1次卖出后的最大余额
    buy2: 完成2次买入后的最大余额
    sell2: 完成2次卖出后的最大余额

    WHY 状态机: 股票问题的通用框架, 最多k次都可以扩展
    初始化: buy = -inf (还没买入时, "完成买入"是不可能的)
    """
    buy1 = buy2 = float("-inf")
    sell1 = sell2 = 0
    for price in prices:
        buy1 = max(buy1, -price)               # 第一次买入: 本金0 - price
        sell1 = max(sell1, buy1 + price)        # 第一次卖出
        buy2 = max(buy2, sell1 - price)         # 第二次买入 (用第一次利润)
        sell2 = max(sell2, buy2 + price)        # 第二次卖出
    return sell2


# ── DP-10. 最长公共子序列 ─────────────────────────────────────────────────
# text1="abcde", text2="ace" → 3 ("ace")
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    时间 O(m·n)  空间 O(min(m,n))
    dp[i][j] = text1[:i] 与 text2[:j] 的 LCS 长度
    若 text1[i-1]==text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    否则: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    空间优化到一维 + 一个 pre 变量
    """
    m, n = len(text1), len(text2)
    if m < n:                                   # 保证 text2 是短的, 节省空间
        return longest_common_subsequence(text2, text1)

    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        pre = 0                                 # dp[i-1][j-1]
        for j in range(1, n + 1):
            tmp = dp[j]                         # 保存旧的 dp[j] = dp[i-1][j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = pre + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            pre = tmp
    return dp[n]

def lcs_2d(text1: str, text2: str) -> int:
    """二维DP版本, 更直观, 面试先写这个再优化"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ═══════════════════════════════════════════════════════════════════════════════
# 二、回溯（8题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── BT-1. 全排列 ───────────────────────────────────────────────────────────
# [1,2,3] → [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
def permute(nums: List[int]) -> List[List[int]]:
    """
    时间 O(n·n!)  空间 O(n)
    回溯模板: path记录当前路径, used标记用过的元素
    WHY 回溯: 排列问题 = 决策树, 每个位置选一个未用的数
    """
    n = len(nums)
    used = [False] * n
    ans = []
    path = []

    def backtrack():
        if len(path) == n:
            ans.append(path[:])                 # 必须拷贝!
            return
        for i in range(n):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False

    backtrack()
    return ans


# ── BT-2. 全排列 II（有重复元素）─────────────────────────────────────────
# [1,1,2] → [[1,1,2],[1,2,1],[2,1,1]]
def permute_unique(nums: List[int]) -> List[List[int]]:
    """
    时间 O(n·n!)  空间 O(n)
    去重策略: 排序 + 同级剪枝
    如果 nums[i]==nums[i-1] 且 used[i-1]==False,
    说明在这个决策层级上, 相同值已经选过了 → 跳过
    WHY used[i-1]==False 而不是 True: False说明是同层选择, True说明是上一层选的
    """
    nums.sort()                                 # 排序使重复值相邻
    n = len(nums)
    used = [False] * n
    ans = []
    path = []

    def backtrack():
        if len(path) == n:
            ans.append(path[:])
            return
        for i in range(n):
            if used[i]:
                continue
            # 同层去重: 前面相同的没被选 → 说明同层已走过
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return ans


# ── BT-3. 子集 ────────────────────────────────────────────────────────────
# [1,2,3] → [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
def subsets(nums: List[int]) -> List[List[int]]:
    """
    时间 O(n·2^n)  空间 O(n)
    方法1: 回溯, 每个元素"选/不选"
    方法2: 迭代, 每次在已有子集上追加新元素
    """
    ans = []
    path = []

    def backtrack(start):
        ans.append(path[:])                     # 每个节点都是答案
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return ans

def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """迭代法: 初始[[]], 遍历nums, 每次把已有子集+'当前数'加进去"""
    ans = [[]]
    for num in nums:
        ans += [sub + [num] for sub in ans]
    return ans


# ── BT-4. 组合总和 ────────────────────────────────────────────────────────
# candidates=[2,3,6,7], target=7 → [[2,2,3],[7]]
def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    时间 O(n^(target/min))  空间 O(target/min)
    每个数可以重复选 → start 不递增 (或每次从同一位置开始)
    WHY start: 避免重复组合(如[2,3]和[3,2]), start保证按序选取
    """
    ans = []
    path = []

    def backtrack(start, remain):
        if remain == 0:
            ans.append(path[:])
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])  # i 不+1: 可以重复选
            path.pop()

    backtrack(0, target)
    return ans


# ── BT-5. 括号生成 ───────────────────────────────────────────────────────
# n=3 → ["((()))","(()())","(())()","()(())","()()()"]
def generate_parenthesis(n: int) -> List[str]:
    """
    时间 O(4^n/√n) Catalan数  空间 O(n)
    约束: 左括号数 <= n, 右括号数 <= 左括号数
    WHY 约束剪枝: 保证生成的括号序列始终合法
    不用暴力生成所有序列再过滤
    """
    ans = []

    def backtrack(left, right, cur):
        if left == n and right == n:
            ans.append(cur)
            return
        if left < n:                            # 还能加左括号
            backtrack(left + 1, right, cur + "(")
        if right < left:                        # 右括号不能超过左括号
            backtrack(left, right + 1, cur + ")")

    backtrack(0, 0, "")
    return ans


# ── BT-6. 电话号码的字母组合 ──────────────────────────────────────────────
# "23" → ["ad","ae","af","bd","be","bf","cd","ce","cf"]
def letter_combinations(digits: str) -> List[str]:
    """
    时间 O(4^n)  空间 O(n)
    每个数字映射3-4个字母, 回溯组合
    """
    if not digits:
        return []
    mapping = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
    ans = []

    def backtrack(idx, cur):
        if idx == len(digits):
            ans.append(cur)
            return
        for ch in mapping[int(digits[idx]) - 2]:
            backtrack(idx + 1, cur + ch)

    backtrack(0, "")
    return ans


# ── BT-7. 单词搜索 ───────────────────────────────────────────────────────
# board=[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word="ABCCED" → True
def exist(board: List[List[str]], word: str) -> bool:
    """
    时间 O(m·n·4^L) L=len(word)  空间 O(L) 递归栈
    DFS 四个方向, 用 board 本身标记已访问(临时改值)
    WHY 原地标记: 避免 used 矩阵的额外空间, 回溯时恢复即可
    """
    m, n = len(board), len(board[0])

    def dfs(i, j, idx):
        if idx == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[idx]:
            return False

        tmp = board[i][j]
        board[i][j] = "#"                       # 标记已访问
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if dfs(i + di, j + dj, idx + 1):
                return True
        board[i][j] = tmp                       # 回溯恢复
        return False

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False


# ── BT-8. N皇后 ───────────────────────────────────────────────────────────
# n=4 → [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
def solve_n_queens(n: int) -> List[List[str]]:
    """
    时间 O(n!)  空间 O(n)
    逐行放置, 用三个set记录列/主对角线/副对角线的占用
    WHY set而非逐格检查: O(1)判断冲突, 不需要遍历棋盘
    对角线编码: 主对角=row-col (恒定), 副对角=row+col (恒定)
    """
    cols = set()
    diag1 = set()                               # 主对角线 row-col
    diag2 = set()                               # 副对角线 row+col
    ans = []
    board = []

    def backtrack(row):
        if row == n:
            ans.append(board[:])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board.append("." * col + "Q" + "." * (n - col - 1))
            backtrack(row + 1)
            board.pop()
            cols.discard(col)
            diag1.discard(row - col)
            diag2.discard(row + col)

    backtrack(0)
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# 三、图论（18题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── G-1. 岛屿数量 ─────────────────────────────────────────────────────────
def num_islands(grid: List[List[str]]) -> int:
    """
    时间 O(m·n)  空间 O(m·n) (递归栈)
    DFS 沉岛: 遇到'1' → ans++ → DFS把整座岛变成'0'
    WHY 沉岛法: 不需要visited矩阵, 直接在grid上修改
    """
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j] == "1":
            grid[i][j] = "0"                    # 沉
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj)

    ans = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                ans += 1
                dfs(i, j)
    return ans

def num_islands_bfs(grid: List[List[str]]) -> int:
    """BFS版本, 避免递归栈溢出"""
    m, n = len(grid), len(grid[0])
    ans = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                ans += 1
                q = deque([(i, j)])
                grid[i][j] = "0"
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == "1":
                            grid[nx][ny] = "0"
                            q.append((nx, ny))
    return ans


# ── G-2. 岛屿最大面积 ─────────────────────────────────────────────────────
def max_area_of_island(grid: List[List[int]]) -> int:
    """
    时间 O(m·n)  空间 O(m·n)
    DFS返回面积: 1 + 四个方向的面积
    """
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j]:
            grid[i][j] = 0
            return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)
        return 0

    ans = 0
    for i in range(m):
        for j in range(n):
            ans = max(ans, dfs(i, j))
    return ans


# ── G-3. 课程表（拓扑排序 / 检测环）─────────────────────────────────────
# numCourses=2, prereqs=[[1,0]] → True (0→1)
def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    时间 O(V+E)  空间 O(V+E)
    Kahn 算法 (BFS拓扑排序):
    入度为0的节点入队 → 逐个处理, 把后继节点入度-1
    最后看处理的节点数是否等于总节点数
    WHY 入度法: 入度为0 = 没有前置依赖 = 现在可以学
    """
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for cur, pre in prerequisites:
        graph[pre].append(cur)                  # pre → cur
        indegree[cur] += 1

    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    taken = 0
    while q:
        course = q.popleft()
        taken += 1
        for nxt in graph[course]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return taken == num_courses


# ── G-4. 课程表 II（输出拓扑序）────────────────────────────────────────
def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    输出一个合法的学习顺序, 如果存在环返回[]
    拓扑排序同上, 多一个order列表记录顺序
    """
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for cur, pre in prerequisites:
        graph[pre].append(cur)
        indegree[cur] += 1

    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    order = []
    while q:
        course = q.popleft()
        order.append(course)
        for nxt in graph[course]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    return order if len(order) == num_courses else []


# ── G-5. 克隆图 ─────────────────────────────────────────────────────────
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """
    时间 O(V+E)  空间 O(V)
    DFS + 哈希表: visited存"原节点→克隆节点"的映射
    WHY 哈希表: 既记录已克隆(避免重复), 又提供O(1)查找
    """
    if not node:
        return None
    visited = {}

    def dfs(n):
        if n in visited:
            return visited[n]
        clone = Node(n.val)
        visited[n] = clone
        for nb in n.neighbors:
            clone.neighbors.append(dfs(nb))
        return clone

    return dfs(node)


# ── G-6. 省份数量（并查集）──────────────────────────────────────────────
# isConnected=[[1,1,0],[1,1,0],[0,0,1]] → 2
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:       # 按秩合并
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        self.count -= 1

def find_circle_num(is_connected: List[List[int]]) -> int:
    """
    时间 O(n²·α(n))  空间 O(n)
    并查集: 连通的合并, 最后看有几个集合
    WHY 并查集: 比DFS/BFS更适合"动态合并"和"查连通分量数"
    """
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j]:
                uf.union(i, j)
    return uf.count


# ── G-7. 腐烂的橘子 ─────────────────────────────────────────────────────
# [[2,1,1],[1,1,0],[0,1,1]] → 4
def oranges_rotting(grid: List[List[int]]) -> int:
    """
    时间 O(m·n)  空间 O(m·n)
    多源BFS: 所有初始腐烂的橘子同时入队
    每一层 = 一分钟, 最后检查是否还有新鲜橘子
    WHY BFS: 求"最短时间" = 求"最短路", BFS天然适合
    WHY 多源: 同时腐烂 → 等价于虚拟超级源点
    """
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1

    if fresh == 0:
        return 0                                    # 没有新鲜橘子

    minutes = -1
    while q:
        minutes += 1
        for _ in range(len(q)):                     # 按层处理
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    fresh -= 1
                    q.append((nx, ny))

    return minutes if fresh == 0 else -1


# ── G-8. 单词接龙 ───────────────────────────────────────────────────────
# beginWord="hit", endWord="cog", wordList=["hot","dot","dog","lot","log","cog"] → 5
def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """
    时间 O(n·k²)  空间 O(n)  k=单词长度
    BFS 最短路径: 每个单词是一个节点, 差一个字母 = 有一条边
    优化: 用通配模式 key 建图, 避免O(n²)两两比较
    h*t → hot, hit (所有匹配这个模式的词都相连)
    WHY 通配符: 建图从 O(n²·k) 降到 O(n·k²), n可能很大
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    # 建图: pattern → 匹配的单词列表
    pattern_map = defaultdict(list)
    for word in word_set:
        for i in range(len(word)):
            key = word[:i] + "*" + word[i + 1:]
            pattern_map[key].append(word)

    q = deque([begin_word])
    visited = {begin_word}
    step = 1
    while q:
        step += 1
        for _ in range(len(q)):
            word = q.popleft()
            for i in range(len(word)):
                key = word[:i] + "*" + word[i + 1:]
                for nxt in pattern_map[key]:
                    if nxt == end_word:
                        return step
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
    return 0


# ── G-9. 网络延迟时间（Dijkstra）────────────────────────────────────────
# times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2 → 2
def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """
    时间 O(E log V)  空间 O(V+E)
    标准 Dijkstra: 优先队列 + 距离数组
    WHY Dijkstra: 带权最短路径, 不是BFS(无权)能解决的
    从起点 k 出发, 求到所有节点的最短距离, 取最大值
    如果存在不可达节点 → 返回 -1

    变体: 如果是稀疏图, 用邻接表; 稠密图用邻接矩阵 + 朴素Dijkstra O(V²)
    """
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))

    dist = [float("inf")] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]                             # (距离, 节点)

    while heap:
        d, u = heappop(heap)
        if d > dist[u]:                         # 旧数据跳过
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))

    ans = max(dist[1:])                         # 注意 0 号索引不用
    return ans if ans != float("inf") else -1


# ── G-10. 判断二分图（染色法）─────────────────────────────────────────
# graph=[[1,3],[0,2],[1,3],[0,2]] → True
def is_bipartite(graph: List[List[int]]) -> bool:
    """
    时间 O(V+E)  空间 O(V)
    染色法 BFS: 相邻节点染不同颜色
    如果能完整染色不冲突 → 二分图
    WHY BFS: 逐层染色, 发现相邻同色 → 不是二分图
    也可以用 DFS 递归染色

    WHY 考二分图: 很多分组问题本质是二分图 (演员-电影, 用户-物品)
    """
    n = len(graph)
    color = [0] * n                             # 0=未染色, 1=红色, -1=蓝色

    def bfs(start):
        q = deque([start])
        color[start] = 1
        while q:
            u = q.popleft()
            for v in graph[u]:
                if color[v] == 0:
                    color[v] = -color[u]        # 染相反颜色
                    q.append(v)
                elif color[v] == color[u]:       # 相邻同色 → 冲突
                    return False
        return True

    for i in range(n):                          # 图可能不连通
        if color[i] == 0 and not bfs(i):
            return False
    return True


# ── G-11. 可能的二分 ─────────────────────────────────────────────────────
# dislikes=[[1,2],[1,3],[2,4]], n=4 → True (分两组, 互不喜欢的不在一组)
def possible_bipartition(n: int, dislikes: List[List[int]]) -> bool:
    """
    二分图实战题: 人=节点, 不喜欢=边, 能否分成两组(二分图)
    染色法同上, 建图时注意是无向图
    字节考这题时喜欢追问: "如果分成 k 组呢?" → 变成 k-分图(NP-hard), 非多项式
    """
    graph = [[] for _ in range(n + 1)]
    for a, b in dislikes:
        graph[a].append(b)
        graph[b].append(a)                      # 无向图

    color = [0] * (n + 1)

    def dfs(u, c):
        color[u] = c
        for v in graph[u]:
            if color[v] == c:
                return False
            if color[v] == 0 and not dfs(v, -c):
                return False
        return True

    for i in range(1, n + 1):
        if color[i] == 0 and not dfs(i, 1):
            return False
    return True


# ── G-12. 冗余连接（并查集实战）────────────────────────────────────────
# edges=[[1,2],[1,3],[2,3]] → [2,3] (最后一条形成环的边)
def find_redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    时间 O(n·α(n))  空间 O(n)
    并查集: 遍历每条边, 如果两端已经在同一集合 → 这条边是冗余的
    WHY 并查集: 无向图检测环 → 如果两个端点已经连通, 再加边就形成环
    题目要求返回"最后一条"冗余边, 所以遍历到底, 最后一个冲突的边就是答案
    """
    n = len(edges)
    uf = UnionFind(n + 1)
    ans = []
    for u, v in edges:
        if uf.find(u) == uf.find(v):
            ans = [u, v]                        # 这轮会成环
        else:
            uf.union(u, v)
    return ans


# ── G-13. 所有可能的路径 ───────────────────────────────────────────────
# graph=[[1,2],[3],[3],[]] → [[0,1,3],[0,2,3]]
def all_paths_source_target(graph: List[List[int]]) -> List[List[int]]:
    """
    时间 O(n·2^n) 最坏  空间 O(n)
    DFS回溯: 从 0 出发, 每条路径走到底, 到 n-1 时记录下来
    WHY 回溯: 找出所有路径 → 必须穷举, 回溯是最直接的方式
    面试追问: "如果图有环怎么办?" → 加 visited set
    """
    n = len(graph)
    ans = []
    path = [0]

    def dfs(u):
        if u == n - 1:
            ans.append(path[:])
            return
        for v in graph[u]:
            path.append(v)
            dfs(v)
            path.pop()

    dfs(0)
    return ans


# ── G-14. 钥匙和房间 ───────────────────────────────────────────────────
# rooms=[[1],[2],[3],[]] → True (从0出发能访问所有房间)
def can_visit_all_rooms(rooms: List[List[int]]) -> bool:
    """
    时间 O(V+E)  空间 O(V)
    DFS可达性: 从房间0出发, 用钥匙开门, 看能否访问所有房间
    WHY 简单DFS: 每个房间里的钥匙 = 邻接表, 从0开始遍历, 看visited数量
    """
    n = len(rooms)
    visited = set()

    def dfs(room):
        visited.add(room)
        for key in rooms[room]:
            if key not in visited:
                dfs(key)

    dfs(0)
    return len(visited) == n


# ── G-15. 找到小镇的法官 ──────────────────────────────────────────────
# trust=[[1,3],[2,3]], n=3 → 3 (1,2都信任3, 且3不信任任何人)
def find_judge(n: int, trust: List[List[int]]) -> int:
    """
    时间 O(n+m)  空间 O(n)
    出度/入度: 法官: 出度=0, 入度=n-1
    trust[i]=[a,b] 表示 a 信任 b
    outdegree[a]++, indegree[b]++
    WHY 出入度: 不需要建图, 只需要统计每个人信任谁和被谁信任
    """
    outd = [0] * (n + 1)
    ind = [0] * (n + 1)
    for a, b in trust:
        outd[a] += 1
        ind[b] += 1
    for i in range(1, n + 1):
        if outd[i] == 0 and ind[i] == n - 1:
            return i
    return -1


# ── G-16. 01矩阵（多源BFS）─────────────────────────────────────────────
# mat=[[0,0,0],[0,1,0],[1,1,1]] → [[0,0,0],[0,1,0],[1,2,1]]
def update_matrix(mat: List[List[int]]) -> List[List[int]]:
    """
    时间 O(m·n)  空间 O(m·n)
    多源BFS: 所有0同时入队, 距离逐层+1
    和腐烂的橘子同样的多源BFS模板
    WHY 多源: 一次遍历得到所有格子的最近0距离
    单源BFS每次查一个格子 → O(mn·mn), 多源BFS → O(mn)
    """
    m, n = len(mat), len(mat[0])
    dist = [[-1] * n for _ in range(m)]
    q = deque()

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                dist[i][j] = 0
                q.append((i, j))

    while q:
        x, y = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


# ── G-17. 最小高度树（拓扑剥洋葱）─────────────────────────────────────
# n=4, edges=[[1,0],[1,2],[1,3]] → [1]
def find_min_height_trees(n: int, edges: List[List[int]]) -> List[int]:
    """
    时间 O(n)  空间 O(n)
    拓扑排序剥洋葱: 从所有叶子节点(度=1)开始, 逐层剥去
    最后剩下的1-2个节点就是最小高度树的根
    WHY 剥洋葱: 最小高度树的根一定是"图的中心", 等价于不断删叶子直到剩1-2个
    这个思路很巧妙, 面试官爱问 WHY
    """
    if n == 1:
        return [0]

    graph = [set() for _ in range(n)]
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    leaves = [i for i in range(n) if len(graph[i]) <= 1]

    while n > 2:
        n -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()        # 叶子只有一个邻居
            graph[neighbor].discard(leaf)       # 从邻居的邻接表删掉叶子
            if len(graph[neighbor]) == 1:       # 邻居变成新叶子
                new_leaves.append(neighbor)
        leaves = new_leaves
    return leaves


# ── G-18. 太平洋大西洋水流 ────────────────────────────────────────────
# heights=[[1,2,2,3,5],[3,2,3,4,4],...] → 能同时流到太平洋和大西洋的格子
def pacific_atlantic(heights: List[List[int]]) -> List[List[int]]:
    """
    时间 O(m·n)  空间 O(m·n)
    逆向DFS/BFS: 从两个大洋的边界开始, 逆流而上(找"高度≥当前"的格子)
    WHY 逆向: 正向每个格子都要搜一次 → O((mn)²)
    逆向从边界搜, 太平洋和大西洋各搜一次 → O(mn)
    两洋都能到达的格子 → 取交集
    """
    m, n = len(heights), len(heights[0])
    pacific = [[False] * n for _ in range(m)]
    atlantic = [[False] * n for _ in range(m)]

    def dfs(i, j, ocean, prev_height):
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        if ocean[i][j] or heights[i][j] < prev_height:  # 水不能往低处流
            return
        ocean[i][j] = True
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(i + di, j + dj, ocean, heights[i][j])

    # 太平洋: 左边界 + 上边界
    for i in range(m):
        dfs(i, 0, pacific, -1)
    for j in range(n):
        dfs(0, j, pacific, -1)
    # 大西洋: 右边界 + 下边界
    for i in range(m):
        dfs(i, n - 1, atlantic, -1)
    for j in range(n):
        dfs(m - 1, j, atlantic, -1)

    ans = []
    for i in range(m):
        for j in range(n):
            if pacific[i][j] and atlantic[i][j]:
                ans.append([i, j])
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# 四、概率与统计编程（5题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── PROB-1. 用 rand7 实现 rand10 ───────────────────────────────────────────
def rand7() -> int:
    """给定 API: 返回 1~7 均匀随机"""
    return random.randint(1, 7)

def rand10() -> int:
    """
    拒绝采样: (rand7()-1)*7 + rand7() → 1~49 均匀
    舍弃 41~49, 只取 1~40, 映射到 1~10
    WHY 拒绝采样: 构造更大的均匀分布 → 截取10的倍数 → 取模
    期望调用次数: 49/40 * 2 ≈ 2.45 次
    """
    while True:
        num = (rand7() - 1) * 7 + rand7()       # 1..49 均匀
        if num <= 40:                           # 拒绝 41..49
            return (num - 1) % 10 + 1

def rand10_v2() -> int:
    """优化: 利用被拒绝的 41~49 减少 rand7 调用"""
    while True:
        a = rand7()
        b = rand7()
        num = (a - 1) * 7 + b
        if num <= 40:
            return (num - 1) % 10 + 1
        # 41-49: 9个数, 用基数 rand9
        a = num - 40                            # 1..9
        b = rand7()
        num = (a - 1) * 7 + b                   # 1..63
        if num <= 60:
            return (num - 1) % 10 + 1
        # 61-63: 3个数, 用基数 rand3
        a = num - 60                            # 1..3
        b = rand7()
        num = (a - 1) * 7 + b                   # 1..21
        if num <= 20:
            return (num - 1) % 10 + 1


# ── PROB-2. 蓄水池抽样 ─────────────────────────────────────────────────────
# 从未知大小的数据流中, 等概率选取 k 个元素
def reservoir_sample(stream, k: int) -> List[int]:
    """
    时间 O(n)  空间 O(k)
    算法:
    - 前 k 个直接入池
    - 第 i 个 (i>=k): 以 k/i 的概率替换池中随机一个
    WHY 这成立: 数学归纳法可证每个元素最终在池中的概率 = k/n
    """
    import random
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:                           # 概率 = k / (i+1)
                reservoir[j] = item
    return reservoir


# ── PROB-3. 带权重的随机选择 ────────────────────────────────────────────────
# w=[1,3] → pickIndex() 返回 0(1/4概率) 或 1(3/4概率)
class WeightedRandom:
    """
    前缀和 + 二分查找
    WHY 前缀和: 把权重转换为区间, 生成随机数落在哪个区间就选哪个
    [1,3] → 前缀和 [1,4] → 区间: [0,1)=0, [1,4)=1
    """
    def __init__(self, w: List[int]):
        self.prefix = []
        total = 0
        for weight in w:
            total += weight
            self.prefix.append(total)
        self.total = total

    def pick_index(self) -> int:
        import random
        import bisect
        r = random.random() * self.total
        return bisect.bisect_left(self.prefix, r)


# ── PROB-4. 洗牌算法 (Fisher-Yates) ───────────────────────────────────────
def shuffle(nums: List[int]) -> List[int]:
    """
    时间 O(n)  空间 O(1)
    从后往前: 对于位置 i, 从 [0, i] 中等概率选一个位置交换
    WHY 从后往前: 每个元素在当前位置的概率 = 1/(i+1)
    每个排列概率 = 1/n!, 完全均匀
    """
    import random
    for i in range(len(nums) - 1, 0, -1):
        j = random.randint(0, i)                # [0, i] 随机
        nums[i], nums[j] = nums[j], nums[i]
    return nums


# ── PROB-5. 均匀硬币产生不均匀概率 ──────────────────────────────────────────
# 给定 fair_coin() 返回 0/1 各 50%, 实现以概率 p 返回 True
def biased_coin(p: float) -> bool:
    """
    用二进制展开: 用 fair coin 生成二进制小数, 与 p 比较
    期望调用次数: 2 (p的二进制每一位用2次fair coin)
    WHY 二进制展开: 均匀随机数可以表示为二进制小数 0.b1b2b3...
    每一位用 fair coin 生成, 比较这个小数和 p
    """
    while True:
        # 生成一个 [0, 1) 均匀随机数
        x = 0.0
        bit = 0.5
        while True:
            x += bit * random.randint(0, 1)
            bit /= 2
            # 提前退出 (Pratt 优化)
            if x + bit < p:                     # 即使后面全是1也小于p
                return True
            if x > p:                           # 即使后面全是0也大于p
                return False


# ═══════════════════════════════════════════════════════════════════════════════
# 五、字节高频额外题（12题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── EXT-1. 合并区间 ────────────────────────────────────────────────────────
# [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    时间 O(n log n)  空间 O(n)
    排序后贪心: 按左边界排序, 维护一个 merged 列表
    如果当前区间与 merged[-1] 重叠 (cur[0] <= last[1]) → 合并
    否则 → 新区间
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur in intervals[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            last[1] = max(last[1], cur[1])      # 合并: 右边界取大
        else:
            merged.append(cur)
    return merged


# ── EXT-2. 螺旋矩阵 ────────────────────────────────────────────────────────
# [[1,2,3],[4,5,6],[7,8,9]] → [1,2,3,6,9,8,7,4,5]
def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    时间 O(m·n)  空间 O(1)
    四个边界: top/bottom/left/right, 每走完一边收缩边界
    WHY 边界收缩: 模拟顺时针绕圈, 每次走一条边后该边界内移一格
    """
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    top, bottom, left, right = 0, m - 1, 0, n - 1
    ans = []

    while top <= bottom and left <= right:
        # 向右
        for j in range(left, right + 1):
            ans.append(matrix[top][j])
        top += 1
        # 向下
        for i in range(top, bottom + 1):
            ans.append(matrix[i][right])
        right -= 1
        # 向左 (检查是否还有行)
        if top <= bottom:
            for j in range(right, left - 1, -1):
                ans.append(matrix[bottom][j])
            bottom -= 1
        # 向上 (检查是否还有列)
        if left <= right:
            for i in range(bottom, top - 1, -1):
                ans.append(matrix[i][left])
            left += 1
    return ans


# ── EXT-3. 字符串解码 ──────────────────────────────────────────────────────
# "3[a]2[bc]" → "aaabcbc", "3[a2[c]]" → "accaccacc"
def decode_string(s: str) -> str:
    """
    时间 O(n)  空间 O(n)
    栈: 遇到 ']' 时弹出直到 '[', 复制 num 次, 再入栈
    WHY 栈: 嵌套结构 → 后进先出, 类似括号匹配
    """
    stack = []
    cur_num = 0
    cur_str = ""
    for ch in s:
        if ch.isdigit():
            cur_num = cur_num * 10 + int(ch)    # 处理多位数
        elif ch == "[":
            stack.append((cur_str, cur_num))    # 保存外层状态
            cur_str = ""
            cur_num = 0
        elif ch == "]":
            prev_str, num = stack.pop()
            cur_str = prev_str + cur_str * num  # 拼接: 外层 + 当前重复
        else:
            cur_str += ch
    return cur_str


# ── EXT-4. 每日温度 ────────────────────────────────────────────────────────
# [73,74,75,71,69,72,76,73] → [1,1,4,2,1,1,0,0]
def daily_temperatures(temps: List[int]) -> List[int]:
    """
    时间 O(n)  空间 O(n)
    单调递减栈: 栈存索引, 遇到更高温时弹出并计算答案
    WHY 单调栈: 对每个元素, 找"右侧第一个比它大的" → 单调栈标准模板
    """
    n = len(temps)
    ans = [0] * n
    stack = []                                  # 单调递减栈 (存索引)
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            ans[j] = i - j
        stack.append(i)
    return ans


# ── EXT-5. 下一个更大元素 II（环形数组）───────────────────────────────────
# [1,2,1] → [2,-1,2]
def next_greater_elements(nums: List[int]) -> List[int]:
    """
    时间 O(n)  空间 O(n)
    环形数组 → 遍历两遍 (i < 2n), 索引取模
    WHY 两遍: 环形 = 数组后面接一个自己
    """
    n = len(nums)
    ans = [-1] * n
    stack = []
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            j = stack.pop()
            ans[j] = nums[idx]
        if i < n:                               # 第一遍才入栈
            stack.append(idx)
    return ans


# ── EXT-6. 最小栈 ──────────────────────────────────────────────────────────
class MinStack:
    """
    时间 O(1) 所有操作  空间 O(n)
    双栈: 一个正常栈, 一个存"当前最小值"的栈
    WHY 两个栈: getMin 也要 O(1), 必须额外记录每个状态下的最小值
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []                     # 栈顶 = 当前 min

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.min_stack[-1]


# ── EXT-7. 用栈实现队列 ────────────────────────────────────────────────────
class MyQueue:
    """
    均摊 O(1)  空间 O(n)
    双栈: in_stack 入队, out_stack 出队
    out_stack 空了就把 in_stack 全倒进去
    WHY 双栈: 栈倒一次 → 顺序反转, 两次反转 = 先进先出
    均摊分析: 每个元素只被 push/pop 各两次 → O(1)
    """
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def _transfer(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def pop(self) -> int:
        self._transfer()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._transfer()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack


# ── EXT-8. 二叉树的直径 ────────────────────────────────────────────────────
def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """
    时间 O(n)  空间 O(h)
    直径 = 任意两节点最长路径 (可能不经过根)
    后序遍历: 对每个节点, 左深+右深 可能是直径
    WHY 后序: 需要先知道左右子树深度
    WHY 全局变量: 直径不一定经过根, 需要在遍历过程中记录全局最大值
    """
    ans = 0

    def depth(node):
        nonlocal ans
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        ans = max(ans, left + right)            # 经过当前节点的路径
        return 1 + max(left, right)             # 当前节点为根的最大深度

    depth(root)
    return ans


# ── EXT-9. 寻找峰值 ────────────────────────────────────────────────────────
# nums=[1,2,3,1] → 2 (3是峰值)
def find_peak_element(nums: List[int]) -> int:
    """
    时间 O(log n)  空间 O(1)
    二分: 比较 mid 和 mid+1, 往大的方向走
    - nums[mid] < nums[mid+1] → 峰值在右边
    - nums[mid] > nums[mid+1] → 峰值在左边 (包括mid)
    WHY 二分正确: 任意两个相邻元素不等 + 数组边界视为 -inf
    往高的方向走一定能走到峰值, 因为两边边界都是 -inf
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1                        # 右边更高
        else:
            hi = mid                            # 左边更高 (mid可能是峰值)
    return lo


# ── EXT-10. 寻找旋转排序数组中的最小值 ─────────────────────────────────────
# [3,4,5,1,2] → 1
def find_min_rotated(nums: List[int]) -> int:
    """
    时间 O(log n)  空间 O(1)
    二分: 比较 mid 和 right
    - nums[mid] < nums[right] → 右边有序, 最小值在左边 (含mid)
    - nums[mid] > nums[right] → 最小值在右边 (不含mid)
    WHY 比 right 而不是比 left: 因为右边更有信息量
    如果 nums[mid] > nums[right], 转折点必在右边
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < nums[hi]:
            hi = mid                            # mid可能是最小值
        else:
            lo = mid + 1                        # mid不可能是最小值
    return nums[lo]


# ── EXT-11. 完全平方数 ─────────────────────────────────────────────────────
# n=12 → 3 (4+4+4)
def num_squares(n: int) -> int:
    """
    时间 O(n√n)  空间 O(n)
    等同零钱兑换: 硬币 = 完全平方数 [1,4,9,...]
    dp[i] = 最少平方数个数凑成 i
    dp[i] = min(dp[i - j*j] + 1) for j*j <= i
    """
    dp = [float("inf")] * (n + 1)
    dp[0] = 0
    squares = [i * i for i in range(1, int(math.sqrt(n)) + 1)]
    for i in range(1, n + 1):
        for sq in squares:
            if sq > i:
                break
            dp[i] = min(dp[i], dp[i - sq] + 1)
    return dp[n]

def num_squares_bfs(n: int) -> int:
    """
    BFS版本: 求最短路径，节点=剩余值，边=减去一个平方数
    适合 n 较大时可能更快
    """
    squares = [i * i for i in range(1, int(math.sqrt(n)) + 1)]
    q = deque([n])
    visited = {n}
    level = 0
    while q:
        level += 1
        for _ in range(len(q)):
            remain = q.popleft()
            for sq in squares:
                nxt = remain - sq
                if nxt == 0:
                    return level
                if nxt > 0 and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
    return n


# ── EXT-12. 回文子串个数 ────────────────────────────────────────────────────
# "abc" → 3 ("a","b","c"), "aaa" → 6
def count_substrings(s: str) -> int:
    """
    时间 O(n²)  空间 O(1)
    中心扩展: 每个位置作为中心 (奇回文) 或中心-left (偶回文)
    扩展到不满足回文为止, 每次成功扩展 count++
    WHY 中心扩展优于DP: DP是O(n²)时间和空间, 中心扩展O(n²)时间O(1)空间
    """
    n = len(s)
    ans = 0
    for i in range(n):
        # 奇回文
        l = r = i
        while l >= 0 and r < n and s[l] == s[r]:
            ans += 1
            l -= 1
            r += 1
        # 偶回文
        l, r = i, i + 1
        while l >= 0 and r < n and s[l] == s[r]:
            ans += 1
            l -= 1
            r += 1
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# 自测入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("补充题库 冒烟测试")
    print("=" * 60)

    # DP
    assert rob([1, 2, 3, 1]) == 4
    assert rob_ii([2, 3, 2]) == 3
    assert can_partition([1, 5, 11, 5]) is True
    assert change(5, [1, 2, 5]) == 4
    assert max_profit_i([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit_ii([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit_iii([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert longest_common_subsequence("abcde", "ace") == 3
    assert last_stone_weight_ii([2, 7, 4, 1, 8, 1]) == 1

    # 回溯
    assert len(permute([1, 2, 3])) == 6
    assert len(permute_unique([1, 1, 2])) == 3
    assert len(subsets([1, 2, 3])) == 8
    assert len(combination_sum([2, 3, 6, 7], 7)) == 2
    assert len(generate_parenthesis(3)) == 5
    assert len(letter_combinations("23")) == 9
    assert exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED") is True
    assert len(solve_n_queens(4)) == 2

    # 图
    grid1 = [["1", "1", "0"], ["1", "1", "0"], ["0", "0", "1"]]
    assert num_islands([row[:] for row in grid1]) == 2
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False
    assert find_order(2, [[1, 0]]) == [0, 1]
    assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    # 图补充
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]]) is True
    assert is_bipartite([[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]) is False
    assert possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]) is True
    assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert all_paths_source_target([[1, 2], [3], [3], []]) == [[0, 1, 3], [0, 2, 3]]
    assert can_visit_all_rooms([[1], [2], [3], []]) is True
    assert find_judge(3, [[1, 3], [2, 3]]) == 3
    assert find_judge(3, [[1, 3], [2, 3], [3, 1]]) == -1

    # 概率
    for _ in range(20):
        r = rand10()
        assert 1 <= r <= 10

    # 高频
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert decode_string("3[a]2[bc]") == "aaabcbc"
    assert decode_string("3[a2[c]]") == "accaccacc"
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert find_peak_element([1, 2, 3, 1]) in [2]
    assert find_min_rotated([3, 4, 5, 1, 2]) == 1
    assert num_squares(12) == 3
    assert count_substrings("aaa") == 6

    print("✅ 全部补充题库冒烟测试通过")
