"""
50 道算法题 · 全量注释版
================================
原则: 先暴力 → 再优化 → 讲 WHY 不是 WHAT
"""

from collections import Counter, defaultdict, deque, OrderedDict
from heapq import heappush, heappop, heappushpop, heapify
from typing import List, Optional
import bisect

# ═══════════════════════════════════════════════════════════════════════════════
# 一、哈希表 + 字符串（15题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. 两数之和 ──────────────────────────────────────────────────────────
# 给定 nums=[2,7,11,15], target=9 → [0,1]
# 核心洞察: a+b=target → b=target-a → 遍历时用 dict 记住"我见过谁"
def two_sum(nums: List[int], target: int) -> List[int]:
    """
    时间 O(n)  空间 O(n)
    WHY dict: 空间换时间, 把"找互补值"从 O(n) 压到 O(1)
    数组有序时可用双指针 O(n) 空间 O(1)
    """
    seen = {}                         # val → index, 存已经遍历过的值
    for i, val in enumerate(nums):
        need = target - val           # 我还缺多少
        if need in seen:              # 缺的这个之前出现过吗？
            return [seen[need], i]    # 出现过 → 直接返回
        seen[val] = i                 # 没出现 → 把当前值存起来供后面用
    return []


# ── 2. 字母异位词分组 ────────────────────────────────────────────────────
# 输入 ["eat","tea","tan","ate","nat","bat"]
# → [["bat"],["nat","tan"],["ate","eat","tea"]]
def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    时间 O(n·k·log k)  排序作 key; 或 O(n·k) 用计数作 key
    WHY 排序作 key: 字母异位词 → 排序后完全一致, 天然聚集
    """
    groups = defaultdict(list)               # 避免 key 不存在的判断
    for s in strs:
        key = "".join(sorted(s))             # "eat"→"aet", 所有异位词收敛到同一key
        groups[key].append(s)
    return list(groups.values())

def group_anagrams_v2(strs: List[str]) -> List[List[str]]:
    """O(n·k) 版本: 计数元组作 key, 适合字符串都很长的情况"""
    groups = defaultdict(list)
    for s in strs:
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - ord("a")] += 1
        groups[tuple(cnt)].append(s)         # tuple 可哈希, list 不可
    return list(groups.values())


# ── 3. 最长不重复子串 ────────────────────────────────────────────────────
# "abcabcbb" → 3 ("abc")
def length_of_longest_substring(s: str) -> int:
    """
    时间 O(n)  空间 O(min(n, 26))
    核心: 滑动窗口, 右指针扩张, 左指针在重复时收缩
    用 dict 记录字符最后出现的位置, 左指针一次性跳到"上次出现+1"
    """
    last_pos = {}                           # char → 最后一次出现的 index
    left = 0                                # 滑动窗口左边界(包含)
    ans = 0
    for right, ch in enumerate(s):
        if ch in last_pos and last_pos[ch] >= left:
            # 这个字符在当前窗口内出现过 → 左边界跳到"出现位置+1"
            # >= left 是关键: 只处理"在当前窗口内"的重复
            left = last_pos[ch] + 1
        last_pos[ch] = right
        ans = max(ans, right - left + 1)
    return ans


# ── 4. 最小覆盖子串 ──────────────────────────────────────────────────────
# s="ADOBECODEBANC", t="ABC" → "BANC"
# 这是 Hard, 但是滑动窗口最经典模板题
def min_window(s: str, t: str) -> str:
    """
    时间 O(n)  空间 O(|Σ|)
    核心: 右指针扩张收字符, 左指针收缩找最优解
    两个哈希表: need(需要的) vs window(当前窗口内的)
    """
    if not s or not t:
        return ""

    need = Counter(t)                       # 每个字符需要几个
    window = defaultdict(int)               # 当前窗口里各有多少
    required = len(need)                    # 需要多少种字符"达标"
    formed = 0                              # 已达标几种

    left = 0
    ans_start, ans_len = 0, float("inf")

    for right, ch in enumerate(s):
        window[ch] += 1
        if ch in need and window[ch] == need[ch]:
            formed += 1                     # 这个字符刚好够了 → +1种达标

        # 收缩: 当所有字符都达标, 尝试从左边缩
        while left <= right and formed == required:
            if right - left + 1 < ans_len:
                ans_len = right - left + 1
                ans_start = left

            left_ch = s[left]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1                 # 不够了 → 失去一种达标
            left += 1

    return "" if ans_len == float("inf") else s[ans_start:ans_start + ans_len]


# ── 5. 有效字母异位词 ────────────────────────────────────────────────────
def is_anagram(s: str, t: str) -> bool:
    """O(n) / O(1) — 两个 Counter 直接比, 或排序"""
    return Counter(s) == Counter(t)


# ── 6. 前K个高频元素 (哈希解法) ──────────────────────────────────────────
# 注: 本题也出现在堆分类, 此处展示桶排序解法 O(n)
def top_k_frequent_bucket(nums: List[int], k: int) -> List[int]:
    """
    时间 O(n)  空间 O(n)
    桶排序: 频率作为下标, 出现频率为 f 的数放进 bucket[f]
    WHY 优于堆: 堆是 O(n log k), 当 k 接近 n 时退化为 O(n log n)
    """
    freq = Counter(nums)
    n = len(nums)
    bucket = [[] for _ in range(n + 1)]     # bucket[i] = 出现i次的元素们

    for num, cnt in freq.items():
        bucket[cnt].append(num)

    result = []
    for cnt in range(n, 0, -1):             # 从高频往低频取
        for num in bucket[cnt]:
            result.append(num)
            if len(result) == k:
                return result
    return result


# ── 7. 最长连续序列 ──────────────────────────────────────────────────────
# nums=[100,4,200,1,3,2] → 4 ([1,2,3,4])
def longest_consecutive(nums: List[int]) -> int:
    """
    时间 O(n)  空间 O(n)
    WHY set: 查找 O(1), 核心逻辑是"只有当它是序列起点时才往后数"
    如果是序列中间元素, 跳过不数, 保证每个元素只会被数一次
    """
    num_set = set(nums)
    ans = 0
    for x in num_set:
        # 只有当 x 是序列起点时 (x-1 不在集合中) 才展开计数
        if x - 1 not in num_set:
            cur = x
            streak = 1
            while cur + 1 in num_set:
                cur += 1
                streak += 1
            ans = max(ans, streak)
    return ans


# ── 8. 和为K的子数组 ────────────────────────────────────────────────────
# nums=[1,1,1], k=2 → 2 (两个[1,1])
def subarray_sum(nums: List[int], k: int) -> int:
    """
    时间 O(n)  空间 O(n)
    核心: 前缀和 prefix_sum[i] - prefix_sum[j] = k
    → prefix_sum[j] = prefix_sum[i] - k
    遍历时哈希表存"此前出现过的前缀和次数"
    """
    prefix_count = defaultdict(int)
    prefix_count[0] = 1                     # 空前缀和=0出现1次, 处理[0..i]直接=k的情况
    cur_sum = 0
    ans = 0
    for num in nums:
        cur_sum += num
        need = cur_sum - k
        ans += prefix_count.get(need, 0)    # 之前有几个前缀和=need, 就有几个子数组
        prefix_count[cur_sum] += 1
    return ans


# ── 9. 找到字符串中所有字母异位词 ────────────────────────────────────────
# s="cbaebabacd", p="abc" → [0,6]
def find_anagrams(s: str, p: str) -> List[int]:
    """
    时间 O(n)  空间 O(1) (26字母)
    定长滑动窗口: 窗口长度始终 = len(p)
    比较窗口内字符计数 vs p的字符计数, 用数组代替Counter比对
    """
    if len(s) < len(p):
        return []
    p_cnt = [0] * 26
    window_cnt = [0] * 26
    for ch in p:
        p_cnt[ord(ch) - ord("a")] += 1

    ans = []
    for i, ch in enumerate(s):
        window_cnt[ord(ch) - ord("a")] += 1
        # 窗口长度超过 len(p), 移除最左边字符
        if i >= len(p):
            left_ch = s[i - len(p)]
            window_cnt[ord(left_ch) - ord("a")] -= 1
        if window_cnt == p_cnt:
            ans.append(i - len(p) + 1)
    return ans


# ── 10. 最长回文子串 ─────────────────────────────────────────────────────
# s="babad" → "bab" 或 "aba"
def longest_palindrome(s: str) -> str:
    """
    时间 O(n²)  空间 O(1)
    中心扩展: 回文分两类: 单中心 "aba" vs 双中心 "abba"
    每个位置尝试两种中心, 向两边扩展, 更新最大长度
    进阶: Manacher 算法 O(n)
    """
    if not s:
        return ""
    n = len(s)
    start, max_len = 0, 1                   # 至少一个字符

    def expand_around_center(left: int, right: int) -> tuple[int, int]:
        """返回以 left,right 为中心的最长回文 (start, length)"""
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        # 循环退出时 left,right 已经不满足 → 回退一格
        length = right - left - 1
        return left + 1, length

    for i in range(n):
        # 奇中心: 单个字符
        l, ln = expand_around_center(i, i)
        if ln > max_len:
            start, max_len = l, ln
        # 偶中心: 相邻两个
        l, ln = expand_around_center(i, i + 1)
        if ln > max_len:
            start, max_len = l, ln
    return s[start:start + max_len]


# ── 11. 单词拆分 ─────────────────────────────────────────────────────────
# s="leetcode", wordDict=["leet","code"] → True
def word_break(s: str, word_dict: List[str]) -> bool:
    """
    时间 O(n·m) n=len(s), m=字典最大长度  空间 O(n)
    dp[i] = s[:i] 能否被拼接
    dp[i] = any(dp[j] and s[j:i] in word_set for j < i)
    """
    word_set = set(word_dict)
    max_len = max((len(w) for w in word_dict), default=0)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True                            # 空字符串可拼接

    for i in range(1, n + 1):
        # 从 i-1 往回查, 但只查到 max_len (剪枝)
        for j in range(i - 1, max(i - max_len - 1, -1), -1):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]


# ── 12. LRU缓存 ───────────────────────────────────────────────────────────
# get/put 都 O(1), 容量满时淘汰最久未使用
class LRUCache:
    """
    OrderedDict 实现: Python 3.7+ dict 已经有序, 但 OrderedDict.move_to_end 更清晰
    核心: 每次访问时把 key 移到末尾(最新), 淘汰时从头部(最旧)删除
    """
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)         # 刚用过 → 最新
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)     # 更新也算"用过"
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)  # FIFO: 删最老的


# ── 13. 同构字符串 ────────────────────────────────────────────────────────
# s="egg", t="add" → True (e→a, g→d)
def is_isomorphic(s: str, t: str) -> bool:
    """双映射: s2t 和 t2s 都必须一一对应, 防止 s='ab' t='aa' (两个s字符映射到同一个t)"""
    s2t, t2s = {}, {}
    for ch_s, ch_t in zip(s, t):
        if (ch_s in s2t and s2t[ch_s] != ch_t) or (ch_t in t2s and t2s[ch_t] != ch_s):
            return False
        s2t[ch_s] = ch_t
        t2s[ch_t] = ch_s
    return True


# ── 14. 回文对 ────────────────────────────────────────────────────────────
# words=["abcd","dcba","lls","s","sssll"] → [[0,1],[1,0],[3,2],[2,4]]
def palindrome_pairs(words: List[str]) -> List[List[int]]:
    """
    时间 O(n·k²) k=最大单词长度  空间 O(n)
    逐单词逐切割点, 检查逆串是否在哈希表中
    三种情况: 空串+"", 前缀为回文+剩余逆在表, 后缀为回文+剩余逆在表
    """
    word_index = {w: i for i, w in enumerate(words)}
    ans = []

    for i, w in enumerate(words):
        # 情况1: 逆序本身在表中 (包含空串情况)
        rev = w[::-1]
        if rev in word_index and word_index[rev] != i:
            ans.append([i, word_index[rev]])

        for cut in range(1, len(w)):
            prefix, suffix = w[:cut], w[cut:]
            # 前缀是回文 → 后缀的逆序在表中 → 拼在前面: [rev(suffix), w]
            if prefix == prefix[::-1]:
                rev_suffix = suffix[::-1]
                if rev_suffix in word_index and word_index[rev_suffix] != i:
                    ans.append([word_index[rev_suffix], i])
            # 后缀是回文 → 前缀的逆序在表中 → 拼在后面: [w, rev(prefix)]
            if suffix == suffix[::-1]:
                rev_prefix = prefix[::-1]
                if rev_prefix in word_index and word_index[rev_prefix] != i:
                    ans.append([i, word_index[rev_prefix]])
    return ans


# ── 15. 重复的DNA序列 ────────────────────────────────────────────────────
# s="AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT" → ["AAAAACCCCC","CCCCCAAAAA"]
def find_repeated_dna_sequences(s: str) -> List[str]:
    """
    时间 O(n)  空间 O(n)
    滚动哈希: 每次窗口向右 1 位, 用 Rabin-Karp 思想 O(1) 更新哈希
    但 Python 切片 s[i:i+10] 已经是 O(1) (字符串不可变, 切片创建新对象但很快)
    直接用 set 存已见过的10位子串
    """
    if len(s) < 10:
        return []
    seen = set()
    repeated = set()
    for i in range(len(s) - 9):
        sub = s[i:i + 10]
        if sub in seen:
            repeated.add(sub)
        seen.add(sub)
    return list(repeated)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、数组 + 双指针（12题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 16. 盛最多水的容器 ──────────────────────────────────────────────────
# height=[1,8,6,2,5,4,8,3,7] → 49
def max_area(height: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    左右双指针往中间收, 每次移动较矮的板:
    面积 = min(左高, 右高) × 宽度
    移动高板一定不会增大面积(因为宽度在减小, 且高度受限于矮板)
    WHY 贪心正确: 矮板决定了上限, 放弃矮板才可能有更大的面积
    """
    left, right = 0, len(height) - 1
    ans = 0
    while left < right:
        h = min(height[left], height[right])
        ans = max(ans, h * (right - left))
        if height[left] < height[right]:
            left += 1                       # 左边是短板 → 让左边的可能性
        else:
            right -= 1
    return ans


# ── 17. 三数之和 ─────────────────────────────────────────────────────────
# nums=[-1,0,1,2,-1,-4] → [[-1,-1,2],[-1,0,1]]
def three_sum(nums: List[int]) -> List[List[int]]:
    """
    时间 O(n²)  空间 O(1) (排序不计)
    排序后固定一个数, 对剩余的数用双指针找两数之和 = -fixed
    关键是去重: 跳过相邻相同的值
    """
    nums.sort()
    n = len(nums)
    ans = []

    for i in range(n - 2):
        if nums[i] > 0:
            break                           # 最小的数 >0 → 三数和不可能=0
        if i > 0 and nums[i] == nums[i - 1]:
            continue                        # 跳过重复的固定值

        left, right = i + 1, n - 1
        target = -nums[i]

        while left < right:
            cur = nums[left] + nums[right]
            if cur < target:
                left += 1
            elif cur > target:
                right -= 1
            else:
                ans.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                # 跳过重复
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
    return ans


# ── 18. 接雨水 ────────────────────────────────────────────────────────────
# height=[0,1,0,2,1,0,1,3,2,1,2,1] → 6
def trap(height: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)  (双指针版本)
    核心洞察: 每个位置存水量 = min(left_max, right_max) - height[i]
    双指针: left_max 和 right_max 谁小就处理谁
    WHY 双指针优于DP: 空间从O(n)降到O(1)
    """
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        # 关键判断: 谁小处理谁
        if left_max < right_max:
            water += left_max - height[left]
            left += 1
        else:
            water += right_max - height[right]
            right -= 1
    return water

def trap_dp(height: List[int]) -> int:
    """DP 版本: O(n) 时间 O(n) 空间, 更直观"""
    n = len(height)
    if n < 3:
        return 0
    left_max = [0] * n
    right_max = [0] * n
    # 从左往右扫 left_max
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])
    # 从右往左扫 right_max
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])
    # 每个格子算水
    return sum(min(l, r) - height[i] for i, (l, r) in enumerate(zip(left_max, right_max)))


# ── 19. 移动零 ────────────────────────────────────────────────────────────
# [0,1,0,3,12] → [1,3,12,0,0]
def move_zeroes(nums: List[int]) -> None:
    """
    时间 O(n)  空间 O(1)
    快慢指针: slow 指向"下个非零值应该放的位置"
    fast 遍历, 遇到非零就 swap
    """
    slow = 0                                # 下一个非零元素的位置
    for fast, val in enumerate(nums):
        if val != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# ── 20. 删除有序数组中的重复项 ──────────────────────────────────────────
# [1,1,2] → 2 ([1,2,_])
def remove_duplicates(nums: List[int]) -> int:
    """快慢指针: slow 指向"下个写入位置", fast 找新值"""
    if not nums:
        return 0
    slow = 1                                # [0] 已经是第一个元素
    for fast in range(1, len(nums)):
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
    return slow


# ── 21. 下一个排列 ────────────────────────────────────────────────────────
# [1,2,3]→[1,3,2], [3,2,1]→[1,2,3]
def next_permutation(nums: List[int]) -> None:
    """
    时间 O(n)  空间 O(1)
    四步法:
    1. 从右往左找第一个"打破升序"的位置 i (nums[i] < nums[i+1])
    2. 从右往左找第一个 > nums[i] 的数, 交换
    3. 反转 i+1 到末尾 (从降序变升序)
    4. 如果 i 不存在 → 已经是最大排列 → 全体反转
    """
    n = len(nums)
    # 1. 找第一个下降点
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        # 2. 找刚好比 nums[i] 大的数
        j = n - 1
        while j > i and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    # 3. 反转 i+1..末尾
    left, right = i + 1, n - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1


# ── 22. 颜色分类 ─────────────────────────────────────────────────────────
# [2,0,2,1,1,0] → [0,0,1,1,2,2]
def sort_colors(nums: List[int]) -> None:
    """
    时间 O(n)  空间 O(1)
    荷兰国旗三指针:
    p0: 下一个0应该放的位置
    p2: 下一个2应该放的位置
    cur: 当前扫描指针
    """
    p0, cur, p2 = 0, 0, len(nums) - 1
    while cur <= p2:
        if nums[cur] == 0:
            nums[p0], nums[cur] = nums[cur], nums[p0]
            p0 += 1
            cur += 1
        elif nums[cur] == 2:
            nums[p2], nums[cur] = nums[cur], nums[p2]
            p2 -= 1                         # cur 不自增: 换过来的值还需要判断
        else:
            cur += 1


# ── 23. 合并两个有序数组 ─────────────────────────────────────────────────
# nums1=[1,2,3,0,0,0], m=3, nums2=[2,5,6], n=3 → [1,2,2,3,5,6]
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    逆向双指针: 从末尾开始, 谁大谁放最后
    WHY 逆向: 避免覆盖 nums1 中尚未处理的元素
    """
    p1, p2 = m - 1, n - 1
    tail = m + n - 1
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[tail] = nums1[p1]
            p1 -= 1
        else:
            nums1[tail] = nums2[p2]
            p2 -= 1
        tail -= 1


# ── 24. 轮转数组 ─────────────────────────────────────────────────────────
# [1,2,3,4,5,6,7], k=3 → [5,6,7,1,2,3,4]
def rotate(nums: List[int], k: int) -> None:
    """三次反转: 全体反 → 前k反 → 后n-k反"""
    def reverse(arr, l, r):
        while l < r:
            arr[l], arr[r] = arr[r], arr[l]
            l += 1
            r -= 1

    k %= len(nums)
    if k == 0: return
    reverse(nums, 0, len(nums) - 1)         # [7,6,5,4,3,2,1]
    reverse(nums, 0, k - 1)                 # [5,6,7,4,3,2,1]
    reverse(nums, k, len(nums) - 1)         # [5,6,7,1,2,3,4]


# ── 25. 在排序数组中查找元素的第一个和最后一个位置 ──────────────────────
# [5,7,7,8,8,10], target=8 → [3,4]
def search_range(nums: List[int], target: int) -> List[int]:
    """
    时间 O(log n)  空间 O(1)
    二分找左边界 + 二分找右边界, 两个二分微妙不同:
    左边界: even if nums[mid]==target, right=mid-1, 继续向左缩
    右边界: even if nums[mid]==target, left=mid+1, 继续向右缩
    """
    def find_left() -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1                # nums[mid]==target 时也往左缩
        return lo if lo < len(nums) and nums[lo] == target else -1

    def find_right() -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] <= target:
                lo = mid + 1                # nums[mid]==target 时也往右缩
            else:
                hi = mid - 1
        return hi if hi >= 0 and nums[hi] == target else -1

    return [find_left(), find_right()]


# ── 26. 搜索旋转排序数组 ─────────────────────────────────────────────────
# [4,5,6,7,0,1,2], target=0 → 4
def search_rotated(nums: List[int], target: int) -> int:
    """
    时间 O(log n)  空间 O(1)
    核心: 二分后必有一侧是有序的, 判断 target 是否在有序侧
    如果在有序侧 → 二分这一侧, 否则二分另一侧
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        # 左半部分有序
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1                # target 在有序的左半
            else:
                lo = mid + 1                # target 在无序的右半
        # 右半部分有序
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1                # target 在有序的右半
            else:
                hi = mid - 1                # target 在无序的左半
    return -1


# ── 27. 长度最小的子数组 ─────────────────────────────────────────────────
# target=7, nums=[2,3,1,2,4,3] → 2 (子数组[4,3])
def min_subarray_len(target: int, nums: List[int]) -> int:
    """
    时间 O(n)  空间 O(1)
    滑动窗口: 右指针扩张累加, 和>=target时收缩左指针
    """
    left = 0
    cur_sum = 0
    ans = float("inf")
    for right, val in enumerate(nums):
        cur_sum += val
        while cur_sum >= target:
            ans = min(ans, right - left + 1)
            cur_sum -= nums[left]
            left += 1
    return 0 if ans == float("inf") else ans


# ═══════════════════════════════════════════════════════════════════════════════
# 三、堆 + Top K（10题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 28. 数组中的第K大元素 ────────────────────────────────────────────────
# [3,2,1,5,6,4], k=2 → 5
def find_kth_largest_heap(nums: List[int], k: int) -> int:
    """O(n log k) / O(k): 维护k大小的小顶堆, 堆顶=第k大"""
    heap = []
    for num in nums:
        heappush(heap, num)
        if len(heap) > k:
            heappop(heap)                   # 踢掉最小的, 最终堆里剩k个最大的
    return heap[0]                          # 小顶堆的堆顶=第k大

def find_kth_largest_quick_select(nums: List[int], k: int) -> int:
    """
    O(n) 平均, O(n²) 最坏 / O(1) 空间
    快速选择: 类似快排的 partition, 但只递归目标所在的那一侧
    """
    import random

    def partition(lo: int, hi: int) -> int:
        pivot_idx = random.randint(lo, hi)  # 随机选pivot防退化
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        pivot = nums[hi]
        i = lo                              # i 指向下个 > pivot 的位置
        for j in range(lo, hi):
            if nums[j] > pivot:             # 大于pivot放左边 (降序)
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    target_idx = k - 1                      # 第k大 → 降序排列后下标 k-1
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        p = partition(lo, hi)
        if p == target_idx:
            return nums[p]
        elif p < target_idx:
            lo = p + 1
        else:
            hi = p - 1
    return -1


# ── 29. 前K个高频元素 (堆版本) ────────────────────────────────────────────
def top_k_frequent_heap(nums: List[int], k: int) -> List[int]:
    """O(n log k): 最小堆维护前k"""
    freq = Counter(nums)
    heap = []
    for num, cnt in freq.items():
        heappush(heap, (cnt, num))
        if len(heap) > k:
            heappop(heap)
    return [num for _, num in heap]


# ── 30. 数据流中位数 ─────────────────────────────────────────────────────
class MedianFinder:
    """
    双堆法: 大顶堆存较小一半, 小顶堆存较大一半
    Python 没有大顶堆 → 存负数模拟
    始终保持 len(小) >= len(大) (最多差1)
    median = 小顶堆堆顶 (奇数) 或 两堆顶平均 (偶数)
    """
    def __init__(self):
        self.small = []                     # 大顶堆(存负数), 较小的一半
        self.large = []                     # 小顶堆, 较大的一半

    def add_num(self, num: int) -> None:
        # 先放进大顶堆(小的一半), 再把最大的踢到小顶堆
        heappush(self.small, -num)
        # 平衡: 把大顶堆的最大值 → 小顶堆
        heappush(self.large, -heappop(self.small))
        # 保证 large 长度 >= small
        if len(self.large) > len(self.small) + 1:
            heappush(self.small, -heappop(self.large))

    def find_median(self) -> float:
        if len(self.large) > len(self.small):
            return float(self.large[0])
        return (self.large[0] - self.small[0]) / 2


# ── 31. 合并K个升序链表 ─────────────────────────────────────────────────
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    O(n log k) / O(k): 优先队列存每个链表的头节点
    WHY 堆: 每次取最小的头 O(log k), 共 n 个节点, 比每次 O(k) 找最小快
    """
    heap = []
    for i, node in enumerate(lists):
        if node:
            # 元组 (val, i, node): i 是 tiebreaker, 防止 ListNode 未实现 __lt__
            heappush(heap, (node.val, i, node))

    dummy = ListNode()
    tail = dummy
    while heap:
        _, i, node = heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heappush(heap, (node.next.val, i, node.next))
    return dummy.next


# ── 32. 有序矩阵中第K小的元素 ────────────────────────────────────────────
# matrix=[[1,5,9],[10,11,13],[12,13,15]], k=8 → 13
def kth_smallest_matrix_heap(matrix: List[List[int]], k: int) -> int:
    """O(k log r): 归并思想, 逐行推进"""
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]  # 每行第一个
    heapify(heap)

    for _ in range(k - 1):
        _, row, col = heappop(heap)
        if col + 1 < n:
            heappush(heap, (matrix[row][col + 1], row, col + 1))
    return heap[0][0]

def kth_smallest_matrix_binary(matrix: List[List[int]], k: int) -> int:
    """
    O(n log(max-min)) / O(1)
    二分答案: 猜一个值 mid, 数有多少个元素 <= mid
    如果 count >= k → 答案 <= mid; 否则 > mid
    """
    n = len(matrix)
    lo, hi = matrix[0][0], matrix[-1][-1]

    def count_le(mid: int) -> int:
        """从矩阵左下角开始, O(n) 计数 <= mid 的元素个数"""
        c = 0
        row, col = n - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] <= mid:
                c += row + 1                # 这一列从 row 往上的都 <= mid
                col += 1
            else:
                row -= 1
        return c

    while lo < hi:
        mid = (lo + hi) // 2
        if count_le(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── 33. 最接近原点的K个点 ────────────────────────────────────────────────
def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """大顶堆存前K小, 距离用负数模拟"""
    heap = []
    for x, y in points:
        dist = x * x + y * y                # 不需要 sqrt, 比较平方就够了
        heappush(heap, (-dist, [x, y]))     # 负距离 = 最大值优先(先踢)
        if len(heap) > k:
            heappop(heap)
    return [p for _, p in heap]


# ── 34. 任务调度器 ────────────────────────────────────────────────────────
# tasks=["A","A","A","B","B","B"], n=2 → 8
def least_interval(tasks: List[str], n: int) -> int:
    """
    O(n) / O(1)
    贪心: 高频任务先排, 冷却期用其他任务填充
    公式: max((max_freq-1) * (n+1) + max_freq_count, len(tasks))
    拆解: 用(max_freq-1)个完整冷却周期 × (n+1)槽位 + 最后一排的最频任务数
    """
    freq = list(Counter(tasks).values())
    max_freq = max(freq)
    max_freq_count = freq.count(max_freq)
    return max((max_freq - 1) * (n + 1) + max_freq_count, len(tasks))


# ── 35. 重构字符串 ───────────────────────────────────────────────────────
# "aab" → "aba", "aaab" → ""
def reorganize_string(s: str) -> str:
    """
    堆解法 O(n log k): 每次取频率最高的两个字符交替放置
    WHY 两个: 防止最高频字符相邻
    """
    freq = Counter(s)
    # 最大频率 > (n+1)/2 则不可能
    if max(freq.values()) > (len(s) + 1) // 2:
        return ""

    heap = [(-cnt, ch) for ch, cnt in freq.items()]
    heapify(heap)
    result = []

    while len(heap) > 1:
        cnt1, ch1 = heappop(heap)
        cnt2, ch2 = heappop(heap)
        result.extend([ch1, ch2])
        if cnt1 + 1 < 0:
            heappush(heap, (cnt1 + 1, ch1)) # 用掉1次, 负值+1 = -变少
        if cnt2 + 1 < 0:
            heappush(heap, (cnt2 + 1, ch2))

    if heap:                                # 还剩一个字符, 一定只剩1次
        result.append(heap[0][1])
    return "".join(result)


# ── 36. 滑动窗口最大值 ───────────────────────────────────────────────────
# [1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]
def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """
    O(n) / O(k)
    单调递减队列: 队首永远是窗口最大值的 index
    新元素入队时, 把队尾所有比它小的踢掉 (它们再也不可能成为最大值)
    队首出窗口时踢掉
    """
    dq = deque()
    ans = []
    for i, val in enumerate(nums):
        # 1. 踢队首: 已滑出窗口
        if dq and dq[0] <= i - k:
            dq.popleft()
        # 2. 踢队尾: 比当前值小的都出队 (它们再没机会当最大值了)
        while dq and nums[dq[-1]] < val:
            dq.pop()
        # 3. 当前索引入队
        dq.append(i)
        # 4. 窗口满后记录最大值
        if i >= k - 1:
            ans.append(nums[dq[0]])
    return ans


# ── 37. 丑数II ────────────────────────────────────────────────────────────
# n=10 → 12 (1,2,3,4,5,6,8,9,10,12)
def nth_ugly_number(n: int) -> int:
    """
    O(n) / O(n)
    多指针DP: 每个新丑数来自已有丑数 ×2 或 ×3 或 ×5
    三个指针追踪"乘以哪个因子能刚好超过当前最大值"
    """
    ugly = [1]
    p2 = p3 = p5 = 0                        # 当前可用 ×2/3/5 的最小丑数索引
    for _ in range(1, n):
        cand2, cand3, cand5 = ugly[p2] * 2, ugly[p3] * 3, ugly[p5] * 5
        nxt = min(cand2, cand3, cand5)
        ugly.append(nxt)
        if nxt == cand2: p2 += 1            # 用了哪个指针就推进哪个
        if nxt == cand3: p3 += 1            # 不用 elif: 处理重复(如6=2×3=3×2)
        if nxt == cand5: p5 += 1
    return ugly[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# 四、树 + 递归（8题）
# ═══════════════════════════════════════════════════════════════════════════════

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# ── 38. 二叉树层序遍历 ───────────────────────────────────────────────────
def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS 模板: 队列, 每次处理一层
    WHY 内层 for: 固定 size=len(queue), 保证按层分组
    """
    if not root:
        return []
    q = deque([root])
    ans = []
    while q:
        level = []
        for _ in range(len(q)):             # 固定当前层大小
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        ans.append(level)
    return ans


# ── 39. 二叉树最大深度 ───────────────────────────────────────────────────
def max_depth(root: Optional[TreeNode]) -> int:
    """DFS: 深度 = 1 + max(左深, 右深)"""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


# ── 40. 路径总和 ─────────────────────────────────────────────────────────
def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """DFS: 到达叶子且 sum==target 时返回 True"""
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target_sum
    remaining = target_sum - root.val
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)


# ── 41. 二叉树的最近公共祖先 ─────────────────────────────────────────────
def lowest_common_ancestor(root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    """
    后序遍历:
    - 当前节点是 p 或 q → 返回当前
    - 左右子树各找到一个 → 当前=LCA
    - 只有一侧找到 → 返回那侧 (p,q 在同一个子树)
    """
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root                          # p,q 分别在我的两边 → 我是 LCA
    return left or right                     # 都在这边子树, 上传即可


# ── 42. 二叉树的锯齿形层序遍历 ───────────────────────────────────────────
def zigzag_level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """BFS + 标志位: 偶数层正序, 奇数层逆序"""
    if not root:
        return []
    q = deque([root])
    ans = []
    left_to_right = True
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        ans.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return ans


# ── 43. 从前序与中序遍历构造二叉树 ──────────────────────────────────────
# preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]
def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    时间 O(n)  空间 O(n)
    前序的第一个是 root, 在中序里找到 root 的位置 → 左边=左子树, 右边=右子树
    用哈希表 O(1) 查索引, 递归构造
    """
    inorder_map = {val: idx for idx, val in enumerate(inorder)}
    pre_idx = 0

    def build(lo: int, hi: int) -> Optional[TreeNode]:
        nonlocal pre_idx
        if lo > hi:
            return None
        root_val = preorder[pre_idx]
        pre_idx += 1
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        root.left = build(lo, root_idx - 1) # 中序左边 = 左子树
        root.right = build(root_idx + 1, hi)
        return root

    return build(0, len(inorder) - 1)


# ── 44. 二叉树的序列化与反序列化 ─────────────────────────────────────────
class Codec:
    """
    BFS 序列化: None → "N", 分隔符 ","
    BFS 反序列化: 队列逐节点重建
    """
    def serialize(self, root: Optional[TreeNode]) -> str:
        if not root:
            return "N"
        q = deque([root])
        parts = []
        while q:
            node = q.popleft()
            if node:
                parts.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
            else:
                parts.append("N")
        return ",".join(parts)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        if data == "N":
            return None
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        q = deque([root])
        i = 1
        while q and i < len(vals):
            node = q.popleft()
            if vals[i] != "N":
                node.left = TreeNode(int(vals[i]))
                q.append(node.left)
            i += 1
            if i < len(vals) and vals[i] != "N":
                node.right = TreeNode(int(vals[i]))
                q.append(node.right)
            i += 1
        return root


# ── 45. 验证二叉搜索树 ───────────────────────────────────────────────────
def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    传递区间法: 左子树上界=root.val, 右子树下界=root.val
    WHY 区间优于中序: 中序需要 O(n) 空间, 区间只需栈递归
    """
    def validate(node: Optional[TreeNode], lo: float, hi: float) -> bool:
        if not node:
            return True
        if not (lo < node.val < hi):        # BST 必须是严格大小, 不能等号
            return False
        return validate(node.left, lo, node.val) and validate(node.right, node.val, hi)

    return validate(root, float("-inf"), float("inf"))


# ═══════════════════════════════════════════════════════════════════════════════
# 五、动态规划基础（5题）
# ═══════════════════════════════════════════════════════════════════════════════

# ── 46. 爬楼梯 ────────────────────────────────────────────────────────────
# n=3 → 3 (1+1+1, 1+2, 2+1)
def climb_stairs(n: int) -> int:
    """
    O(n) / O(1)
    dp[i] = dp[i-1] + dp[i-2] (最后一步跨1级 或 跨2级)
    = 斐波那契: f(n) = f(n-1) + f(n-2)
    空间优化: 只存前两个值
    """
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


# ── 47. 最大子数组和 ─────────────────────────────────────────────────────
# [-2,1,-3,4,-1,2,1,-5,4] → 6 ([4,-1,2,1])
def max_subarray(nums: List[int]) -> int:
    """
    O(n) / O(1)
    Kadane 算法:
    dp[i] = 以 i 结尾的最大子数组和
    dp[i] = max(nums[i], dp[i-1] + nums[i]) → 要么续, 要么重新开始
    滚动优化: cur_max = max(num, cur_max + num)
    """
    cur_max = ans = nums[0]
    for num in nums[1:]:
        cur_max = max(num, cur_max + num)   # 续? 还是重新开始?
        ans = max(ans, cur_max)
    return ans


# ── 48. 零钱兑换 ─────────────────────────────────────────────────────────
# coins=[1,2,5], amount=11 → 3 (5+5+1)
def coin_change(coins: List[int], amount: int) -> int:
    """
    O(n·m) / O(n)  n=amount, m=len(coins)
    完全背包: dp[i] = 凑成金额 i 的最少硬币数
    dp[i] = min(dp[i], dp[i - coin] + 1)
    """
    dp = [amount + 1] * (amount + 1)        # amount+1 代表无穷大
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1


# ── 49. 最长递增子序列 ───────────────────────────────────────────────────
# [10,9,2,5,3,7,101,18] → 4 ([2,3,7,101])
def length_of_lis_dp(nums: List[int]) -> int:
    """O(n²): dp[i] = 以 i 结尾的最长递增子序列长度"""
    n = len(nums)
    dp = [1] * n                             # 每个元素自身构成长度1
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def length_of_lis_greedy(nums: List[int]) -> int:
    """
    O(n log n) 贪心+二分:
    维护 tails 数组, tails[i] = 长度为 i+1 的递增子序列的最小末尾值
    遍历 nums, 在 tails 中二分查找插入位置:
    - 找到了 → 替换 (压低末尾值, 为后续创造更多可能)
    - 没找到 → 扩展 tails
    """
    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num                 # 压低相同长度子序列的末尾值
    return len(tails)


# ── 50. 编辑距离 ─────────────────────────────────────────────────────────
# word1="horse", word2="ros" → 3
def min_distance(word1: str, word2: str) -> int:
    """
    O(mn) / O(mn)
    dp[i][j] = word1[:i] → word2[:j] 的最少操作数
    三种操作:
    - 插入: dp[i][j-1] + 1  (word2[j-1] 插入到 word1)
    - 删除: dp[i-1][j] + 1  (word1[i-1] 删除)
    - 替换: dp[i-1][j-1] + (0 if 相等 else 1)
    空间优化: 滚动数组 → O(min(m,n))
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i                        # word1[:i] → "" 全删
    for j in range(n + 1):
        dp[0][j] = j                        # "" → word2[:j] 全插

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j],           # 删除 word1[i-1]
                    dp[i][j - 1],           # 插入 word2[j-1]
                    dp[i - 1][j - 1]        # 替换
                ) + 1
    return dp[m][n]


# ═══════════════════════════════════════════════════════════════════════════════
# 自测入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("50 题冒烟测试")
    print("=" * 60)

    # 快速冒烟测试
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert length_of_longest_substring("abcabcbb") == 3
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert is_anagram("anagram", "nagaram") is True
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert subarray_sum([1, 1, 1], 2) == 2
    assert find_anagrams("cbaebabacd", "abc") == [0, 6]
    assert len(longest_palindrome("babad")) == 3
    assert word_break("leetcode", ["leet", "code"]) is True
    assert is_isomorphic("egg", "add") is True
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert find_kth_largest_heap([3, 2, 1, 5, 6, 4], 2) == 5
    assert climb_stairs(3) == 3
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert coin_change([1, 2, 5], 11) == 3
    assert length_of_lis_dp([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert min_distance("horse", "ros") == 3

    print("✅ 全部冒烟测试通过")
