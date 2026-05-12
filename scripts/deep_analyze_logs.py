"""Deep-dive cross-analysis of user behavior logs — beyond surface stats."""
import re, ast
from collections import Counter, defaultdict
from datetime import datetime

LOG_PATH = r"C:\Users\lijia\Desktop\0927\oylx_hk_pred_test-master\logs (2).txt"
DEV_EMAILS = {"lijiapeng8@xdf.cn"}


def parse(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = re.match(r"\t?[\d\-:. ]+\t([\d\-: ]+) - (.+?) <(.+?)> - (\w+) - 用户输入: (.+)$", line)
            if m:
                ts, name, email, sid, raw = m.groups()
                try: inp = ast.literal_eval(raw)
                except: inp = {}
                records.append({"ts": ts.strip(), "name": name.strip(), "email": email.strip(), "session_id": sid, "input": inp})
    return [r for r in records if r["email"] not in DEV_EMAILS]


def deep_analyze(RR):
    R = len(RR)
    print("=" * 64)
    print("  深度交叉分析")
    print("=" * 64)

    # ═══════════════════════════════════════════════
    # 1. "等 N 项" 模式：用户是认真选了专业，还是一键全选？
    # ═══════════════════════════════════════════════
    print("\n── 1. 专业筛选行为：精准 vs 海投 ──")
    bulk = 0   # 包含 "等 N 项" → 未做专业筛选
    precise = 0  # 没有 "等 N 项"
    bulk_sessions = set()
    precise_sessions = set()
    for r in RR:
        ms = r["input"].get("target_majors", "")
        if re.search(r'等\s*\d+\s*项', ms):
            bulk += 1
            bulk_sessions.add(r["session_id"])
        elif ms.strip():
            precise += 1
            precise_sessions.add(r["session_id"])

    print(f"  海投模式 (含'等N项'): {bulk} 条 ({bulk/R*100:.0f}%), {len(bulk_sessions)} 个 session")
    print(f"  精准模式 (手动选专业): {precise} 条 ({precise/R*100:.0f}%), {len(precise_sessions)} 个 session")

    # What's the typical N?
    ns = []
    for r in RR:
        ms = r["input"].get("target_majors", "")
        for m in re.finditer(r'等\s*(\d+)\s*项', ms):
            ns.append(int(m.group(1)))
    if ns:
        print(f"  海投平均覆盖专业数: {sum(ns)/len(ns):.0f}, min={min(ns)}, max={max(ns)}")

    # Among bulk users, do they specify college?
    bulk_with_cat = sum(1 for r in RR
                        if re.search(r'等\s*\d+\s*项', r["input"].get("target_majors", ""))
                        and r["input"].get("major_categories", ""))
    print(f"  海投用户中填写了学院的比例: {bulk_with_cat}/{bulk} ({bulk_with_cat/bulk*100:.0f}%)")
    print(f"  → 海投用户几乎全不选学院，可能是不知道有这个功能，或者觉得不需要")

    # ═══════════════════════════════════════════════
    # 2. 用户调参方向分析：往上调还是往下调？
    # ═══════════════════════════════════════════════
    print("\n── 2. 调参方向：用户在试探什么？──")

    session_groups = defaultdict(list)
    for r in RR:
        session_groups[r["session_id"]].append(r)

    adjustments = {"gpa_up": 0, "gpa_down": 0, "lang_up": 0, "lang_down": 0,
                   "add_school": 0, "remove_school": 0, "add_major": 0, "remove_major": 0}
    for sid, recs in session_groups.items():
        for i in range(1, len(recs)):
            prev, curr = recs[i-1]["input"], recs[i]["input"]
            try:
                pg, cg = float(prev.get("gpa_score", 0)), float(curr.get("gpa_score", 0))
                if cg > pg: adjustments["gpa_up"] += 1
                elif cg < pg: adjustments["gpa_down"] += 1
            except: pass
            try:
                pl, cl = float(prev.get("language_score", 0)), float(curr.get("language_score", 0))
                if cl > pl: adjustments["lang_up"] += 1
                elif cl < pl: adjustments["lang_down"] += 1
            except: pass
            # school count
            pu = len([x for x in re.split(r'[,，]', prev.get("target_universities", "")) if x.strip() and '等' not in x])
            cu = len([x for x in re.split(r'[,，]', curr.get("target_universities", "")) if x.strip() and '等' not in x])
            if cu > pu: adjustments["add_school"] += 1
            elif cu < pu: adjustments["remove_school"] += 1

    print(f"  GPA: 上调 {adjustments['gpa_up']}次 vs 下调 {adjustments['gpa_down']}次")
    print(f"  语言: 上调 {adjustments['lang_up']}次 vs 下调 {adjustments['lang_down']}次")
    print(f"  院校: 增加 {adjustments['add_school']}次 vs 减少 {adjustments['remove_school']}次")
    total_adj = sum(adjustments.values())
    if total_adj:
        optimistic = adjustments['gpa_up'] + adjustments['lang_up'] + adjustments['add_school']
        print(f"  → 乐观调参(上调分数/增加院校) vs 悲观调参 = {optimistic} vs {total_adj-optimistic}")

    # ═══════════════════════════════════════════════
    # 3. Session 内时间间隔分析
    # ═══════════════════════════════════════════════
    print("\n── 3. Session 内操作节奏 ──")
    intervals = []
    for sid, recs in session_groups.items():
        if len(recs) < 2: continue
        for i in range(1, len(recs)):
            try:
                t1 = datetime.strptime(recs[i-1]["ts"], "%Y-%m-%d %H:%M:%S")
                t2 = datetime.strptime(recs[i]["ts"], "%Y-%m-%d %H:%M:%S")
                intervals.append(abs((t2 - t1).total_seconds()))
            except: pass

    if intervals:
        intervals.sort()
        print(f"  提交间隔 (秒): median={intervals[len(intervals)//2]:.0f}s, "
              f"p25={intervals[len(intervals)//4]:.0f}s, p75={intervals[len(intervals)*3//4]:.0f}s")

        # Bucket
        quick = sum(1 for x in intervals if x <= 30)
        medium = sum(1 for x in intervals if 30 < x <= 120)
        slow = sum(1 for x in intervals if 120 < x <= 600)
        vslow = sum(1 for x in intervals if x > 600)
        print(f"  ≤30s (快速调参): {quick}, 30s-2min: {medium}, 2-10min: {slow}, >10min: {vslow}")

        if quick / len(intervals) > 0.3:
            print(f"  → 大量快速调参 ({quick/len(intervals)*100:.0f}%)，暗示用户在用系统做实时 what-if 对比")
            print(f"  → 如果当前结果展示不支持并行对比，用户只能反复提交来看差异")

    # ═══════════════════════════════════════════════
    # 4. 用户"成熟度"分层：熟练用户 vs 新手行为差异
    # ═══════════════════════════════════════════════
    print("\n── 4. 用户成熟度分层 ──")

    def user_sophistication(recs):
        """Higher score = more sophisticated usage."""
        score = 0
        for r in recs:
            inp = r["input"]
            if inp.get("major_categories", ""): score += 3  # 指定学院=高级操作
            ms = inp.get("target_majors", "")
            if ms and not re.search(r'等\s*\d+\s*项', ms): score += 2  # 精准选专业
            if inp.get("exam_score", ""): score += 1
            if inp.get("research_details", "") or inp.get("internship_details", ""): score += 1
        return score / len(recs)

    per_user = defaultdict(list)
    for r in RR:
        per_user[r["email"]].append(r)

    user_scores = {}
    for email, recs in per_user.items():
        user_scores[email] = (len(recs), user_sophistication(recs), recs[0]["name"])

    # Split into quartiles
    scored = sorted(user_scores.values(), key=lambda x: -x[1])
    top_n = max(1, len(scored) // 4)
    top_users = scored[:top_n]
    bot_users = scored[-top_n:]

    print(f"  高成熟度用户 (TOP 25%):")
    for cnt, score, name in top_users[:5]:
        print(f"    {name:<8} 提交{cnt}次, 成熟度={score:.1f}")

    print(f"  低成熟度用户 (BOT 25%):")
    for cnt, score, name in bot_users[:5]:
        print(f"    {name:<8} 提交{cnt}次, 成熟度={score:.1f}")

    # Correlation: usage count vs sophistication
    high_usage_emails = [e for e, (c, s, n) in user_scores.items() if c >= 10]
    low_usage_emails = [e for e, (c, s, n) in user_scores.items() if c <= 2]
    high_avg_score = sum(user_scores[e][1] for e in high_usage_emails) / max(len(high_usage_emails), 1)
    low_avg_score = sum(user_scores[e][1] for e in low_usage_emails) / max(len(low_usage_emails), 1)
    print(f"\n  重度用户(≥10次)平均成熟度: {high_avg_score:.1f}")
    print(f"  轻度用户(≤2次)平均成熟度:  {low_avg_score:.1f}")

    # ═══════════════════════════════════════════════
    # 5. 首次提交 vs 末次提交：用户最终往哪个方向收敛
    # ═══════════════════════════════════════════════
    print("\n── 5. Session 内收敛方向：首次→末次 ──")
    first_last_gpa_deltas = []
    first_last_lang_deltas = []
    first_last_school_deltas = []

    for sid, recs in session_groups.items():
        if len(recs) < 2: continue
        first, last = recs[-1]["input"], recs[0]["input"]  # chronological: recs[-1] is earliest
        try:
            first_last_gpa_deltas.append(float(last.get("gpa_score", 0)) - float(first.get("gpa_score", 0)))
        except: pass
        try:
            first_last_lang_deltas.append(float(last.get("language_score", 0)) - float(first.get("language_score", 0)))
        except: pass

    if first_last_gpa_deltas:
        gpa_inc = sum(1 for d in first_last_gpa_deltas if d > 0)
        gpa_dec = sum(1 for d in first_last_gpa_deltas if d < 0)
        gpa_same = sum(1 for d in first_last_gpa_deltas if d == 0)
        print(f"  GPA 变化: 最终更高 {gpa_inc}, 更低 {gpa_dec}, 不变 {gpa_same}")
        print(f"  → 用户倾向于{'上调' if gpa_inc > gpa_dec else '下调' if gpa_dec > gpa_inc else '保持'} GPA")

    if first_last_lang_deltas:
        lang_inc = sum(1 for d in first_last_lang_deltas if d > 0)
        lang_dec = sum(1 for d in first_last_lang_deltas if d < 0)
        print(f"  语言变化: 最终更高 {lang_inc}, 更低 {lang_dec}")

    # ═══════════════════════════════════════════════
    # 6. 空字段的关联性：缺了一个字段的人，还缺什么？
    # ═══════════════════════════════════════════════
    print("\n── 6. 字段空值关联矩阵 ──")
    fields = {
        "exam_score": "GRE",
        "target_universities": "目标院校",
        "major_categories": "目标学院",
    }
    for f1, l1 in fields.items():
        empty_f1 = [r for r in RR if not r["input"].get(f1, "")]
        if not empty_f1: continue
        for f2, l2 in fields.items():
            if f1 >= f2: continue
            both_empty = sum(1 for r in empty_f1 if not r["input"].get(f2, ""))
            pct = both_empty / len(empty_f1) * 100
            print(f"  {l1}为空 → {l2}也为空: {both_empty}/{len(empty_f1)} ({pct:.0f}%)")

    # ═══════════════════════════════════════════════
    # 7. 背景院校聚类：用户来自什么层次的学校？
    # ═══════════════════════════════════════════════
    print("\n── 7. 本科院校聚类 (按出现频次) ──")

    # Manual tier tagging for top schools
    c9 = {"清华大学", "北京大学", "复旦大学", "上海交通大学", "南京大学", "浙江大学", "中国科学技术大学", "哈尔滨工业大学", "西安交通大学"}
    other_985 = {"武汉大学", "四川大学", "山东大学", "北京工业大学", "电子科技大学"}
    bg_counter = Counter(r["input"].get("background_university", "") for r in RR)

    c9_count = sum(c for u, c in bg_counter.items() if u in c9)
    other_985_count = sum(c for u, c in bg_counter.items() if u in other_985)
    other_count = sum(bg_counter.values()) - c9_count - other_985_count

    print(f"  C9 院校: {c9_count} 次")
    print(f"  其他985/211: {other_985_count} 次")
    print(f"  其他院校: {other_count} 次")
    print(f"  院校种类数: {len(bg_counter)}")

    # Do students from lower-tier schools target more ambitiously?
    # Ambition = number of target schools
    for label, unis in [("C9", c9), ("其他985", other_985)]:
        matching = [r for r in RR if r["input"].get("background_university", "") in unis]
        if matching:
            avg_targets = sum(
                len([x for x in re.split(r'[,，]', r["input"].get("target_universities", "")) if x.strip()])
                for r in matching
            ) / len(matching)
            print(f"  {label}背景平均目标院校数: {avg_targets:.1f}")

    other_matching = [r for r in RR if r["input"].get("background_university", "") not in c9
                      and r["input"].get("background_university", "") not in other_985]
    if other_matching:
        avg = sum(len([x for x in re.split(r'[,，]', r["input"].get("target_universities", "")) if x.strip()])
                  for r in other_matching) / len(other_matching)
        print(f"  其他院校背景平均目标院校数: {avg:.1f}")

    # ═══════════════════════════════════════════════
    # 8. Session 复杂度 vs 完成度
    # ═══════════════════════════════════════════════
    print("\n── 8. Session 复杂度 vs 表单完整度 ──")
    for sid, recs in sorted(session_groups.items(), key=lambda x: -len(x[1]))[:8]:
        n = len(recs)
        last = recs[0]["input"]  # most recent in reverse-chronological
        completeness = sum([
            bool(last.get("target_universities", "")),
            bool(last.get("target_majors", "")),
            bool(last.get("major_categories", "")),
            bool(last.get("exam_score", "")),
        ])
        user = recs[0]["name"]
        has_cat = "✓学院" if last.get("major_categories") else "✗学院"
        has_target = "✓院校" if last.get("target_universities") else "✗院校"
        print(f"  {user:<8} {n}次提交 | {has_target} {has_cat} | 完整度={completeness}/4")

    # ═══════════════════════════════════════════════
    # 9. 最后一条：月度趋势 — 系统在被更多人用还是同一批人在用？
    # ═══════════════════════════════════════════════
    print("\n── 9. 月度新用户 vs 老用户 ──")
    monthly_new = defaultdict(set)
    monthly_all = defaultdict(set)
    first_seen = {}
    for r in RR:
        try:
            month = datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m")
        except:
            month = "unknown"
        email = r["email"]
        if email not in first_seen:
            first_seen[email] = month
        monthly_all[month].add(email)
        if first_seen[email] == month:
            monthly_new[month].add(email)

    for m in sorted(monthly_all.keys()):
        all_u = len(monthly_all[m])
        new_u = len(monthly_new.get(m, set()))
        returning = all_u - new_u
        print(f"  {m}: 总{all_u}人 (新{new_u} + 回訪{returning})")


if __name__ == "__main__":
    RR = parse(LOG_PATH)
    deep_analyze(RR)
