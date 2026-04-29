"""WritePrint — CLI diagnostic output and main entry point."""

import sys

from src.pages.write_print.diagnosis import _DIFFICULTY, _sentence_rhythm_chart
from src.pages.write_print.engine import analyze
from src.pages.write_print.features import _ensure_nltk_data, read_input
from src.pages.write_print.model import fit_model, get_model, print_fit_report


def diagnose(text: str, scaler, model):
    result = analyze(text, scaler, model)
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    verdict = result["verdict"]
    ts = result["text_stats"]

    print("=" * 72)
    print(f"  Verdict: {verdict['label']}")
    print(f"  AI Score: {result['score']:.0f}%  ({result['confidence']})")
    print("=" * 72)
    print(f"  {verdict['description']}")
    print()
    print(
        f"  {ts['words']:,} words  |  {ts['sentences']} sentences  |  {ts['paragraphs']} paragraphs"
    )
    print()

    print("  What drives the score:")
    print(f"  {'Factor':<30} {'Weight':>8}  {'Impact'}")
    print(f"  {'-' * 30} {'-' * 8}  {'-' * 6}")
    for _name, fb in result["features"].items():
        bar = "█" * max(0, min(10, int(abs(fb["contribution"]) / 2.5)))
        direction = {
            "AI": "↑ pushes score up",
            "human": "↓ pushes score down",
            "neutral": "— little effect",
        }[fb["direction"]]
        print(f"  {fb['label']:<30} {fb['value']:>7.3f}  {bar} {direction}")
    print()

    print("  Sentence rhythm (▌= one sentence; 'uniform' = too regular → AI-like):")
    chart = _sentence_rhythm_chart(result["sentence_lengths"])
    print(chart.replace("#", "▌"))
    print()

    local_items = result["local_fixes"]
    print("=" * 72)
    print("  LOCAL FIXES  — specific words & sentences to change")
    print("=" * 72)

    if local_items:
        local_impact = sum(f["impact"] for f in local_items[:8])
        print(f"\n  Fixing these could lower your score by ~{local_impact:.0f}%")
        print(f"  (from {result['score']:.0f}% → ~{result['estimated_new_score']:.0f}%)\n")

    for i, item in enumerate(local_items):
        if i >= 15:
            print(f"  ... {len(local_items) - 15} more fixes")
            break
        diff = _DIFFICULTY.get(item["category"], {})
        print(f"  [{item['category'].upper()}]  Difficulty: {diff.get('label', '?')}")
        print(f"         {item['label']}")
        print(f"         {item['detail']}")
        print(f"         Fix: {item['action']}")
        print()

    if not local_items:
        print("\n  No local fixes needed — your word choices look natural.\n")

    print("=" * 72)
    print("  GLOBAL FIXES  — overall structure & writing patterns")
    print("=" * 72)

    if result["global_fixes"]:
        for i, fix in enumerate(result["global_fixes"], 1):
            print(f"\n  [{i}] {fix}")
    else:
        print("\n  No structural issues detected — your rhythm and variety look good.")

    if len(result["quick_wins"]) >= 2:
        print("\n  ⚡ QUICK WINS  (5 minutes, no rewriting skill needed):")
        for w in result["quick_wins"]:
            print(f"  - {w['action']}")

    print()
    print("=" * 72)
    print("  BOTTOM LINE")
    print("=" * 72)
    print(f"  {verdict['description']}")

    local_count = len(local_items)
    global_count = len(result["global_fixes"])

    if result["score"] < 40:
        print(
            f"\n  This reads well. You have {local_count} minor word-level tweaks "
            f"and {global_count} structural notes — none are urgent."
        )
    elif result["score"] < 60:
        print(
            f"\n  Worth a review. Focus on the {local_count} LOCAL fixes first "
            f"(word swaps, template phrases). Then check the {global_count} "
            f"structural suggestions if you want extra polish."
        )
    else:
        print(
            f"\n  Needs work. Start with the {global_count} GLOBAL structural "
            f"changes, then tackle the {local_count} LOCAL fixes. The quick "
            f"wins above will give you the fastest improvement."
        )

    print()


def main():
    _ensure_nltk_data()

    if len(sys.argv) >= 3 and sys.argv[1] == "--diagnose":
        input_path = sys.argv[2]
        text = read_input(input_path)
        if len(text) < 50:
            print(f"Error: text too short ({len(text)} chars)")
            sys.exit(1)
        model, scaler = get_model()
        diagnose(text, scaler, model)

    elif len(sys.argv) >= 2 and sys.argv[1] == "--diagnose":
        text = sys.stdin.read().strip()
        if len(text) < 50:
            print(f"Error: text too short ({len(text)} chars)")
            sys.exit(1)
        model, scaler = get_model()
        diagnose(text, scaler, model)

    else:
        print("Fitting model on sample PDFs (6 features)...")
        model, scaler, X, y, names_list = fit_model()
        print_fit_report(model, scaler, X, y, names_list)
        print('\nUsage: py score_fitter.py --diagnose <file.pdf|file.txt|"raw text">')


if __name__ == "__main__":
    main()
