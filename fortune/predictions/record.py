#!/usr/bin/env python3
"""
预测记录工具 — 增/查/验 命理预测条目。

用法:
  python record.py add                          # 交互式录入
  python record.py add --prediction "..." ...   # 命令行录入
  python record.py list                         # 列出全部
  python record.py list --status pending        # 按状态筛选
  python record.py list --due                   # 窗口已到期的
  python record.py verify <id>                  # 交互式应验
  python record.py show <id>                    # 查看详情
"""

import argparse
import json
import os
import sys
from datetime import datetime, date

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PREDICTIONS_DIR, "predictions.json")
BACKUPS_DIR = os.path.join(PREDICTIONS_DIR, "backups")

CATEGORIES = ["career", "relationship", "health", "finance", "housing", "other"]
SOURCES = ["事前预测", "事后匹配"]
STATUSES = ["pending", "ongoing", "verified", "missed"]
MATCH_LEVELS = ["吻合", "部分吻合", "不吻合", "暂无证据"]


def _load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_db(db):
    _backup_db()
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _backup_db():
    if not os.path.exists(DB_PATH):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUPS_DIR, f"predictions_{ts}.json")
    os.makedirs(BACKUPS_DIR, exist_ok=True)
    with open(DB_PATH, "r", encoding="utf-8") as src:
        with open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    _cleanup_backups()


def _cleanup_backups():
    files = sorted(
        [f for f in os.listdir(BACKUPS_DIR) if f.startswith("predictions_") and f.endswith(".json")]
    )
    while len(files) > 10:
        os.remove(os.path.join(BACKUPS_DIR, files.pop(0)))


def _next_id(predictions):
    max_n = 0
    for p in predictions:
        pid = p["id"]
        if pid.startswith("p") and "_" in pid:
            try:
                n = int(pid.split("_")[1])
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    return f"p{date.today().strftime('%Y')}_{max_n + 1:03d}"


def cmd_add(args):
    db = _load_db()
    preds = db["predictions"]

    if args.prediction:
        prediction_text = args.prediction
        category = args.category or "other"
        confidence = args.confidence or "中"
        source_type = args.source_type or "事前预测"
        chart_signal = args.chart_signal or ""
        json_ref = args.json_ref or ""
        doc_ref = args.doc_ref or ""
        window_start = args.window_start or str(date.today())
        window_end = args.window_end or str(date.today().replace(year=date.today().year + 1))
    else:
        print("\n  新增预测\n")
        prediction_text = input("  预测内容: ").strip()
        if not prediction_text:
            print("  已取消。")
            return

        print(f"\n  类别: {' / '.join(CATEGORIES)}")
        category = input(f"  [默认: other]: ").strip() or "other"

        print(f"\n  置信度: 高 / 中 / 低")
        confidence = input(f"  [默认: 中]: ").strip() or "中"

        print(f"\n  来源类型: {' / '.join(SOURCES)}")
        source_type = input(f"  [默认: 事前预测]: ").strip() or "事前预测"

        chart_signal = input("  命盘信号描述: ").strip()
        json_ref = input("  JSON字段引用 (如 palaces[1].stars.main): ").strip()
        doc_ref = input("  文档引用 (如 紫微斗数命盘综合分析_完整版.md §十二): ").strip()

        window_start = input(f"  应期开始 [默认: {date.today()}]: ").strip() or str(date.today())
        window_end = input(f"  应期结束 [默认: {date.today().replace(year=date.today().year + 1)}]: ").strip()
        window_end = window_end or str(date.today().replace(year=date.today().year + 1))

    new_pred = {
        "id": _next_id(preds),
        "created": str(date.today()),
        "prediction": prediction_text,
        "category": category,
        "confidence": confidence,
        "combined_confidence": confidence,
        "bazi_signals": [],
        "ziwei_signals": [{
            "source": chart_signal,
            "signal": chart_signal,
            "direction": "平",
            "reason": doc_ref,
        }] if chart_signal else [],
        "cross_check": "仅紫微" if chart_signal else "待交叉验证",
        "tags": args.tags if args.tags else [],
        "window": {"start": window_start, "end": window_end},
        "status": "pending",
        "verification": None,
    }

    preds.append(new_pred)
    _save_db(db)
    print(f"\n  [OK] 已创建预测 {new_pred['id']}: {prediction_text[:60]}...")


def _filter_and_print(preds, args):
    today = str(date.today())
    filtered = []

    for p in preds:
        if args.status and p["status"] != args.status:
            continue
        if args.category and p["category"] != args.category:
            continue
        if args.due and p["window"]["end"] > today:
            continue
        if args.due and p["status"] in ("verified", "missed"):
            continue
        filtered.append(p)

    if not filtered:
        print("  (无匹配记录)")
        return

    status_icon = {"pending": "[..]", "ongoing": "[>>]", "verified": "[OK]", "missed": "[NO]"}
    cat_abbr = {"career": "事业", "relationship": "感情", "health": "健康", "finance": "财运", "housing": "田宅", "other": "其他"}

    print(f"\n  {'ID':<14} {'状态':<4} {'类别':<4} {'应期':<22} {'预测摘要'}")
    print(f"  {'─' * 14} {'─' * 4} {'─' * 4} {'─' * 22} {'─' * 50}")
    for p in filtered:
        icon = status_icon.get(p["status"], "  ")
        cat = cat_abbr.get(p["category"], p["category"])
        win = f"{p['window']['start']}~{p['window']['end']}"
        if len(win) > 22:
            win = win[:19] + "..."
        summary = p["prediction"][:50]
        print(f"  {p['id']:<14} {icon:<4} {cat:<4} {win:<22} {summary}")

    total = len(filtered)
    verified = sum(1 for p in filtered if p["status"] == "verified")
    missed = sum(1 for p in filtered if p["status"] == "missed")
    print(f"\n  共 {total} 条 | 已验证 {verified} | 未应验 {missed}")


def cmd_list(args):
    db = _load_db()
    _filter_and_print(db["predictions"], args)


def cmd_verify(args):
    db = _load_db()
    target = None
    for p in db["predictions"]:
        if p["id"] == args.id:
            target = p
            break

    if target is None:
        print(f"  [NO] 未找到预测 {args.id}")
        return

    print(f"\n  验证: [{target['id']}] {target['prediction'][:80]}")
    print(f"     来源: {target['source']['type']} | 应期: {target['window']['start']} ~ {target['window']['end']}")
    print(f"     当前状态: {target['status']}\n")

    if args.outcome:
        outcome = args.outcome
        match_level = args.match_level or "暂无证据"
        adverse = args.adverse or ""
        intervention_applied = args.intervention_applied
        temporal_precision = args.temporal_precision
    else:
        outcome = input("  实际发生了什么？（留空取消）: ").strip()
        if not outcome:
            print("  已取消。")
            return

        print(f"\n  吻合度: {' / '.join(MATCH_LEVELS)}")
        match_level = input(f"  [默认: 部分吻合]: ").strip() or "部分吻合"

        adverse = input("  不利证据（命盘预测了但实际没发生的，直接回车跳过）: ").strip()

        print(f"\n  时间精度: 窗口内 / 相邻月 / 隔季 / 跨年")
        temporal_precision = input(f"  [默认: 窗口内]: ").strip() or "窗口内"

        ia = input(f"  观测者效应？（预测本身改变了行为）[y/N]: ").strip().lower()
        intervention_applied = ia == "y"

    target["verification"] = {
        "date": str(date.today()),
        "outcome": outcome,
        "match_level": match_level,
        "adverse_evidence": adverse,
        "is_post_hoc": target["source"]["type"] == "事后匹配",
        "intervention_applied": intervention_applied,
        "temporal_precision": temporal_precision,
    }

    if match_level == "吻合" or match_level == "部分吻合":
        target["status"] = "verified"
    elif match_level == "不吻合":
        target["status"] = "missed"
    # "暂无证据" → leave status unchanged

    _save_db(db)
    if match_level == "暂无证据":
        status_text = "暂无证据（状态未变更）"
    elif target["status"] == "verified":
        status_text = "应验"
    else:
        status_text = "未应验"
    print(f"\n  [OK] 已更新 {target['id']} -> {status_text} ({match_level})")
    if adverse:
        print(f"     [!] 不利证据已记录: {adverse[:60]}")


def cmd_show(args):
    db = _load_db()
    target = None
    for p in db["predictions"]:
        if p["id"] == args.id:
            target = p
            break

    if target is None:
        print(f"  [NO] 未找到预测 {args.id}")
        return

    print(f"\n  {'=' * 50}")
    print(f"  预测 ID:     {target['id']}")
    print(f"  创建日期:     {target['created']}")
    print(f"  状态:         {target['status']}")
    print(f"  类别:         {target['category']}")
    print(f"  来源类型:     {target['source']['type']}")
    print(f"  命盘信号:     {target['source']['chart_signal']}")
    print(f"  JSON引用:     {target['source']['json_ref']}")
    print(f"  文档引用:     {target['source']['document_ref']}")
    print(f"  应期:         {target['window']['start']} ~ {target['window']['end']}")
    print(f"  标签:         {', '.join(target['tags']) if target['tags'] else '(无)'}")
    print(f"\n  ── 预测内容 ──")
    print(f"  {target['prediction']}")

    v = target.get("verification")
    if v:
        print(f"\n  ── 验证结果 ──")
        print(f"  验证日期:     {v['date']}")
        print(f"  吻合度:       {v['match_level']}")
        print(f"  实际结果:     {v['outcome']}")
        if v.get("adverse_evidence"):
            print(f"  [!] 不利证据:   {v['adverse_evidence']}")
        print(f"  事后匹配:     {'是' if v.get('is_post_hoc') else '否'}")
    print(f"  {'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="命理预测追踪 — 记录与验证工具")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add", help="新增预测")
    p_add.add_argument("--prediction", help="预测内容")
    p_add.add_argument("--category", choices=CATEGORIES, help="类别")
    p_add.add_argument("--source-type", choices=SOURCES, help="来源类型")
    p_add.add_argument("--chart-signal", help="命盘信号描述")
    p_add.add_argument("--json-ref", help="JSON字段引用")
    p_add.add_argument("--doc-ref", help="文档引用")
    p_add.add_argument("--confidence", choices=["高", "中", "低"], default="中", help="预测置信度")
    p_add.add_argument("--tags", nargs="*", help="标签列表")
    p_add.add_argument("--window-start", help="应期开始 (YYYY-MM-DD)")
    p_add.add_argument("--window-end", help="应期结束 (YYYY-MM-DD)")

    p_list = sub.add_parser("list", help="列出预测")
    p_list.add_argument("--status", choices=STATUSES, help="按状态筛选")
    p_list.add_argument("--category", choices=CATEGORIES, help="按类别筛选")
    p_list.add_argument("--due", action="store_true", help="只看应期已到但仍pending/ongoing的")

    p_verify = sub.add_parser("verify", help="验证预测")
    p_verify.add_argument("id", help="预测ID")
    p_verify.add_argument("--outcome", help="实际结果")
    p_verify.add_argument("--match-level", choices=MATCH_LEVELS, help="吻合度")
    p_verify.add_argument("--adverse", help="不利证据")
    p_verify.add_argument("--intervention-applied", action="store_true", help="预测本身是否改变了行为（观测者效应）")
    p_verify.add_argument("--temporal-precision", choices=["窗口内", "相邻月", "隔季", "跨年"], default="窗口内", help="实际发生时间相对预测窗口的精度")

    p_show = sub.add_parser("show", help="查看预测详情")
    p_show.add_argument("id", help="预测ID")

    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"  [NO] 数据库未找到: {DB_PATH}")
        sys.exit(1)

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "show":
        cmd_show(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
