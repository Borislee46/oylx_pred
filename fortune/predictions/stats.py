#!/usr/bin/env python3
"""
预测统计报告 — 分析预测追踪数据，输出应验率、宫位/四化/类别细分。

用法:
  python stats.py              # 终端输出完整报告
  python stats.py --json       # JSON 格式输出
  python stats.py --summary    # 仅摘要
"""

import argparse
import json
import os
import sys
from datetime import date
from collections import defaultdict

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PREDICTIONS_DIR, "predictions.json")

CAT_LABELS = {
    "career": "事业", "relationship": "感情", "health": "健康",
    "finance": "财运", "housing": "田宅", "other": "其他",
}

PALACE_NAMES = ["命宫", "兄弟宫", "夫妻宫", "子女宫", "财帛宫", "疾厄宫",
                "迁移宫", "交友宫", "官禄宫", "田宅宫", "福德宫", "父母宫"]

SI_HUA_NAMES = ["禄", "权", "科", "忌"]


def _load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _rate(numerator, denominator):
    if denominator == 0:
        return "—"
    return f"{numerator / denominator:.0%}"


def _compute_stats(predictions):
    verified = [p for p in predictions if p["status"] == "verified"]
    missed = [p for p in predictions if p["status"] == "missed"]
    resolved = verified + missed

    by_category = defaultdict(lambda: {"total": 0, "verified": 0, "missed": 0})
    for p in predictions:
        by_category[p["category"]]["total"] += 1
        if p["status"] == "verified":
            by_category[p["category"]]["verified"] += 1
        elif p["status"] == "missed":
            by_category[p["category"]]["missed"] += 1

    by_palace = defaultdict(lambda: {"total": 0, "verified": 0, "missed": 0})
    for p in predictions:
        for palace in PALACE_NAMES:
            if palace in " ".join(p.get("tags", [])) or palace in p["source"].get("chart_signal", ""):
                by_palace[palace]["total"] += 1
                if p["status"] == "verified":
                    by_palace[palace]["verified"] += 1
                elif p["status"] == "missed":
                    by_palace[palace]["missed"] += 1

    by_sihua = defaultdict(lambda: {"total": 0, "verified": 0, "missed": 0})
    for p in predictions:
        for sh in SI_HUA_NAMES:
            signal = p["source"].get("chart_signal", "")
            tags = " ".join(p.get("tags", []))
            if sh in tags or sh in signal:
                by_sihua[sh]["total"] += 1
                if p["status"] == "verified":
                    by_sihua[sh]["verified"] += 1
                elif p["status"] == "missed":
                    by_sihua[sh]["missed"] += 1

    by_source = defaultdict(lambda: {"total": 0, "verified": 0, "missed": 0})
    for p in predictions:
        st = p["source"]["type"]
        by_source[st]["total"] += 1
        if p["status"] == "verified":
            by_source[st]["verified"] += 1
        elif p["status"] == "missed":
            by_source[st]["missed"] += 1

    adverse = [p for p in resolved if p.get("verification", {}).get("adverse_evidence", "").strip()]

    today = str(date.today())
    due = [p for p in predictions
           if p["window"]["end"] <= today and p["status"] in ("pending", "ongoing")]

    return {
        "total": len(predictions),
        "verified": len(verified),
        "missed": len(missed),
        "pending": len([p for p in predictions if p["status"] == "pending"]),
        "ongoing": len([p for p in predictions if p["status"] == "ongoing"]),
        "resolved": len(resolved),
        "by_category": dict(by_category),
        "by_palace": dict(by_palace),
        "by_sihua": dict(by_sihua),
        "by_source": dict(by_source),
        "adverse_count": len(adverse),
        "adverse_items": [{"id": p["id"], "evidence": p["verification"]["adverse_evidence"][:80]} for p in adverse],
        "due_count": len(due),
        "due_items": [{"id": p["id"], "prediction": p["prediction"][:60], "window_end": p["window"]["end"]} for p in due],
        "coverage": _compute_coverage(predictions),
        "calibration": _compute_calibration(predictions),
    }


def _print_summary(s):
    print(f"\n  {'=' * 50}")
    print(f"  命理预测追踪 · 统计报告")
    print(f"  生成日期: {date.today()}  |  数据: predictions.json")
    print(f"  {'=' * 50}")

    print(f"\n  ── 全局 ──")
    print(f"  总预测数:     {s['total']}")
    print(f"  已验证(吻合):  {s['verified']}  ({_rate(s['verified'], s['total'])})")
    print(f"  已验证(不吻合):{s['missed']}  ({_rate(s['missed'], s['total'])})")
    print(f"  待验证:        {s['pending']}")
    print(f"  进行中:        {s['ongoing']}")
    if s["resolved"] > 0:
        acc = (s["verified"] - s["missed"]) / s["resolved"]
        print(f"  净准确率:      {acc:+.0%}  (吻合-不吻合)/已解决")
    if s["verified"] + s["missed"] > 0:
        print(f"  应验率:        {_rate(s['verified'], s['verified'] + s['missed'])}  (吻合/已解决)")


def _print_table(title, data, label_map=None):
    if not data:
        return
    print(f"\n  ── {title} ──")
    header = f"  {'维度':<10} {'总数':<5} {'验证':<5} {'未验':<5} {'应验率'}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    items = sorted(data.items(), key=lambda x: x[1]["total"], reverse=True)
    for key, v in items:
        if v["total"] == 0:
            continue
        label = label_map.get(key, key) if label_map else key
        total_r = v["verified"] + v["missed"]
        print(f"  {label:<10} {v['total']:<5} {v['verified']:<5} {v['missed']:<5} {_rate(v['verified'], total_r)}")


def _print_due(s):
    if not s["due_items"]:
        print(f"\n  ── 待跟进 ──")
        print(f"  [OK] 无到期未验证的预测")
        return
    print(f"\n  ── [!] 到期未验证 ({s['due_count']}条) ──")
    for item in s["due_items"]:
        print(f"  [{item['id']}] 应期止 {item['window_end']} | {item['prediction']}")


def _print_adverse(s):
    if not s["adverse_items"]:
        return
    print(f"\n  ── [!] 不利证据记录 ({s['adverse_count']}条) ──")
    for item in s["adverse_items"]:
        print(f"  [{item['id']}] {item['evidence']}")


def _compute_coverage(predictions):
    """Identify palaces and si-hua with zero predictions."""
    covered_palaces = set()
    covered_sihua = set()
    category_count = defaultdict(int)

    for p in predictions:
        signal = p["source"].get("chart_signal", "")
        tags = " ".join(p.get("tags", []))
        text = signal + " " + tags

        for palace in PALACE_NAMES:
            if palace in text:
                covered_palaces.add(palace)
        for sh in SI_HUA_NAMES:
            if sh in text:
                covered_sihua.add(sh)
        category_count[p["category"]] += 1

    blind_palaces = [p for p in PALACE_NAMES if p not in covered_palaces]
    blind_sihua = [s for s in SI_HUA_NAMES if s not in covered_sihua]

    return {
        "covered_palaces": sorted(covered_palaces),
        "blind_spots_palaces": blind_palaces,
        "covered_sihua": sorted(covered_sihua),
        "blind_spots_sihua": blind_sihua,
        "category_count": dict(category_count),
    }


def _compute_calibration(predictions):
    """Group predictions by confidence level for calibration tracking."""
    by_confidence = defaultdict(lambda: {"total": 0, "verified": 0, "missed": 0, "pending": 0})
    for p in predictions:
        conf = p.get("confidence", "中")
        by_confidence[conf]["total"] += 1
        if p["status"] == "verified":
            by_confidence[conf]["verified"] += 1
        elif p["status"] == "missed":
            by_confidence[conf]["missed"] += 1
        elif p["status"] in ("pending", "ongoing"):
            by_confidence[conf]["pending"] += 1

    resolved = sum(1 for p in predictions if p["status"] in ("verified", "missed"))

    intervention_count = 0
    by_temporal = defaultdict(int)
    for p in predictions:
        v = p.get("verification")
        if v:
            if v.get("intervention_applied"):
                intervention_count += 1
            tp = v.get("temporal_precision", "未记录")
            by_temporal[tp] += 1

    return {
        "by_confidence": dict(by_confidence),
        "intervention_count": intervention_count,
        "by_temporal": dict(by_temporal),
        "note": "N too small for statistical significance — calibration curve requires >= 30 verified predictions. This is infrastructure, not yet a scorecard." if resolved < 10 else "",
    }


def _print_coverage(s):
    cov = s["coverage"]
    print(f"\n  ── 覆盖盲区分析 ──")
    print(f"  已覆盖宫位 ({len(cov['covered_palaces'])}/12): {', '.join(cov['covered_palaces'])}")
    if cov["blind_spots_palaces"]:
        print(f"  [!] 盲区宫位 ({len(cov['blind_spots_palaces'])}): {', '.join(cov['blind_spots_palaces'])}")
    else:
        print(f"  [OK] 全部十二宫均有预测覆盖")
    print(f"  已覆盖四化 ({len(cov['covered_sihua'])}/4): {', '.join(cov['covered_sihua'])}")
    if cov["blind_spots_sihua"]:
        print(f"  [!] 盲区四化 ({len(cov['blind_spots_sihua'])}): {', '.join(cov['blind_spots_sihua'])}")
    print(f"  按类别分布: {', '.join(f'{k}({v})' for k, v in sorted(cov['category_count'].items()))}")


def _print_calibration(s):
    cal = s["calibration"]
    if not cal["by_confidence"]:
        return
    print(f"\n  ── 校准追踪（按置信度）──")
    header = f"  {'置信度':<6} {'总数':<5} {'已验证':<5} {'未应验':<5} {'待验证':<5} {'应验率'}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for level in ["高", "中", "低"]:
        v = cal["by_confidence"].get(level)
        if not v:
            continue
        resolved = v["verified"] + v["missed"]
        print(f"  {level:<6} {v['total']:<5} {v['verified']:<5} {v['missed']:<5} {v['pending']:<5} {_rate(v['verified'], resolved)}")
    if cal["intervention_count"]:
        print(f"\n  观测者效应: {cal['intervention_count']} 条验证标记了 intervention_applied")
    if cal["by_temporal"]:
        print(f"  时间精度分布: {', '.join(f'{k}({v})' for k, v in sorted(cal['by_temporal'].items()))}")
    if cal["note"]:
        print(f"  [!] {cal['note']}")


def main():
    parser = argparse.ArgumentParser(description="预测统计报告")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--summary", action="store_true", help="仅输出摘要")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"  ❌ 数据库未找到: {DB_PATH}")
        sys.exit(1)

    db = _load_db()
    s = _compute_stats(db["predictions"])

    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return

    _print_summary(s)

    if args.summary:
        _print_due(s)
        return

    _print_table("按类别", s["by_category"], CAT_LABELS)
    _print_table("按宫位", s["by_palace"])
    _print_table("按四化", s["by_sihua"])
    _print_table("事前预测 vs 事后匹配", s["by_source"])

    _print_coverage(s)
    _print_calibration(s)

    print(f"\n  ── 方法学备注 ──")
    print(f"  事前预测: {s['by_source'].get('事前预测', {}).get('total', 0)}条")
    print(f"  事后匹配: {s['by_source'].get('事后匹配', {}).get('total', 0)}条")
    print(f"  事后匹配的'吻合'不计入模型验证力（见记录规则.md §2.3）")

    _print_adverse(s)
    _print_due(s)

    print(f"\n  {'=' * 50}\n")


if __name__ == "__main__":
    main()
