"""
Pilot: Compare TF-IDF vs E5 embedding for text-major similarity.

Phase 1: Direct comparison on real cases
Phase 2: Negative control test - can E5 discriminate correct vs random majors?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CASES_PATH = PROJECT_ROOT / "src" / "machine_learning_models" / "data" / "cases.feather"
VECTORIZER_PATH = (
    PROJECT_ROOT
    / "src"
    / "machine_learning_models"
    / "pre-trained_models"
    / "tfidf_vectorizer.joblib"
)
CENTROIDS_PATH = (
    PROJECT_ROOT
    / "src"
    / "machine_learning_models"
    / "pre-trained_models"
    / "tfidf_centroids.npz"
)
E5_MODEL_PATH = PROJECT_ROOT / "src" / "services" / "multilingual-e5-large-instruct"
N_SAMPLES = 5


def load_tfidf():
    vec = joblib.load(VECTORIZER_PATH)
    data = np.load(CENTROIDS_PATH)
    centroids = {}
    for k in data.files:
        arr = np.asarray(data[k], dtype=np.float32)
        norm = np.linalg.norm(arr)
        centroids[k] = arr / norm if norm > 0 else arr
    return vec, centroids


def load_e5():
    from sentence_transformers import SentenceTransformer
    print(f"Loading E5 from {E5_MODEL_PATH} ...")
    return SentenceTransformer(str(E5_MODEL_PATH))


def compute_tfidf_sim(vec, centroids, text: str, field: str) -> float:
    if not text or not isinstance(text, str) or text.strip() == "":
        return 0.0
    X = vec.transform([text])
    row = X.getrow(0)
    if row.nnz == 0:
        return 0.0
    centroid = centroids.get(field)
    if centroid is None:
        return 0.0
    dot = row.dot(centroid)
    val = float(np.asarray(dot).flat[0])
    return float(np.clip(val, 0.0, 1.0))


def compute_e5_sim(model, text: str, target_major: str, variant: str = "instruct") -> float:
    """Compute E5 similarity.

    variant='instruct': proper E5 instruct query/passage format
    variant='plain': just encode and compare directly
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return 0.0

    if variant == "instruct":
        query = (
            "Instruct: Given an applicant's experience description, "
            "assess how relevant it is to the academic major.\n"
            f"Query: {text.strip()}"
        )
        passage = f"Passage: {target_major.strip()}"
    else:
        query = text.strip()
        passage = target_major.strip()

    emb_q = model.encode(query, normalize_embeddings=True)
    emb_p = model.encode(passage, normalize_embeddings=True)
    return float(np.dot(emb_q, emb_p))


def run_negative_test(model, vec, centroids, df, n_cases=5, n_negs=5):
    """Test if E5 can rank the correct major higher than random ones."""
    rng = np.random.RandomState(99)
    all_majors = df["target_major"].dropna().unique()

    # Pick cases with good internship text
    mask = df["internship_detail"].notna() & (df["internship_detail"].str.len() > 50)
    candidates = df[mask].copy()

    results = []
    for i, (_, row) in enumerate(candidates.sample(n=n_cases, random_state=rng).iterrows()):
        major = row["target_major"]
        text = row["internship_detail"]

        negs = [m for m in rng.choice(all_majors, size=n_negs + 3, replace=False) if m != major][:n_negs]

        for variant in ["instruct", "plain"]:
            pos = compute_e5_sim(model, text, major, variant=variant)
            neg_vals = [compute_e5_sim(model, text, nm, variant=variant) for nm in negs]
            neg_mean = float(np.mean(neg_vals))
            margin = pos - neg_mean
            rank = sum(1 for s in neg_vals if s >= pos) + 1
            results.append({
                "case": i + 1,
                "variant": variant,
                "major": major[:60],
                "pos": round(pos, 4),
                "neg_mean": round(neg_mean, 4),
                "neg_std": round(float(np.std(neg_vals)), 4),
                "margin": round(margin, 4),
                "rank": rank,
            })

    return results


def main():
    print("=" * 70)
    print("PILOT: TF-IDF vs E5 Text-Major Similarity")
    print("=" * 70)

    df = pd.read_feather(CASES_PATH)
    print(f"\nLoaded {len(df):,} cases")

    vec, centroids = load_tfidf()
    print(f"TF-IDF vocab: {len(vec.vocabulary_):,}, centroids: {list(centroids.keys())}")

    e5 = load_e5()

    # ===== Phase 1: direct comparison =====
    print(f"\n{'=' * 70}")
    print("PHASE 1: Direct TF-IDF vs E5 on real cases")
    print(f"{'=' * 70}")

    text_fields = {
        "research_detail": "research_details",
        "internship_detail": "internship_details",
        "award_detail": "award_details",
        "paper_detail": "paper_details",
    }

    mask = df["internship_detail"].notna() & (df["internship_detail"].str.len() > 50)
    candidates = df[mask]
    rng = np.random.RandomState(42)
    samples = candidates.sample(n=min(N_SAMPLES * 3, len(candidates)), random_state=rng)
    seen_majors = set()
    picked = []
    for _, row in samples.iterrows():
        major = row["target_major"]
        if major not in seen_majors or len(picked) < N_SAMPLES:
            if len(picked) < N_SAMPLES:
                picked.append(row)
                seen_majors.add(major)

    phase1_results = []
    for i, row in enumerate(picked):
        major = row["target_major"]
        print(f"\n--- Case {i+1}: {major[:80]} ---")

        for field_name, centroid_key in text_fields.items():
            text = row.get(field_name)
            if not isinstance(text, str) or len(text.strip()) < 10:
                continue

            tfidf_sim = compute_tfidf_sim(vec, centroids, text, centroid_key)
            e5_sim = compute_e5_sim(e5, text, major, variant="instruct")

            label = field_name.replace("_detail", "")
            print(f"  [{label}] TF-IDF={tfidf_sim:.4f}  E5={e5_sim:.4f}  Δ={e5_sim - tfidf_sim:+.4f}")
            print(f"          {text[:100]}...")
            phase1_results.append({
                "case": i + 1, "field": label, "tfidf": round(tfidf_sim, 4),
                "e5": round(e5_sim, 4), "delta": round(e5_sim - tfidf_sim, 4),
            })

    # Phase 1 summary
    tfidf_vals = [r["tfidf"] for r in phase1_results]
    e5_vals = [r["e5"] for r in phase1_results]
    deltas = [r["delta"] for r in phase1_results]
    print(f"\n  Phase 1 Summary ({len(phase1_results)} pairs):")
    print(f"  TF-IDF: mean={np.mean(tfidf_vals):.4f} std={np.std(tfidf_vals):.4f}")
    print(f"  E5:     mean={np.mean(e5_vals):.4f} std={np.std(e5_vals):.4f}")
    print(f"  Delta:  mean={np.mean(deltas):+.4f}")
    print(f"  Spearman r = {np.corrcoef(tfidf_vals, e5_vals)[0,1]:.4f}")

    # ===== Phase 2: negative control test =====
    print(f"\n{'=' * 70}")
    print("PHASE 2: Can E5 discriminate correct vs random majors?")
    print(f"{'=' * 70}")

    neg_results = run_negative_test(e5, vec, centroids, df, n_cases=5, n_negs=5)

    for variant in ["instruct", "plain"]:
        subset = [r for r in neg_results if r["variant"] == variant]
        margins = [r["margin"] for r in subset]
        top1_count = sum(1 for r in subset if r["rank"] == 1)
        print(f"\n  [{variant}]")
        for r in subset:
            flag = "✓ TOP-1" if r["rank"] == 1 else f"✗ rank={r['rank']}"
            print(f"    Case {r['case']}: pos={r['pos']:.4f} neg_mean={r['neg_mean']:.4f} "
                  f"margin={r['margin']:+.4f} {flag}  | {r['major']}")
        print(f"    → Top-1 accuracy: {top1_count}/{len(subset)}")
        print(f"    → Mean margin: {np.mean(margins):+.4f}")

    # ===== Phase 3: TF-IDF discrimination check =====
    print(f"\n{'=' * 70}")
    print("PHASE 3: TF-IDF Centroid Discrimination (for reference)")
    print(f"{'=' * 70}")

    # TF-IDF centroid sim is independent of target major
    # But we can check: for a given text, how much does similarity vary across fields?
    case = picked[0]
    text = case.get("internship_detail")
    print(f"\n  Text: {text[:100]}...")
    print(f"  Target major: {case['target_major'][:60]}")
    print(f"\n  TF-IDF similarity varies by FIELD (not by major):")
    for field_name, centroid_key in text_fields.items():
        sim = compute_tfidf_sim(vec, centroids, text, centroid_key)
        label = field_name.replace("_detail", "")
        print(f"    [{label}] → centroid: {sim:.4f}")

    print(f"\n  KEY INSIGHT: TF-IDF compares text against 'ideal admitted student' centroid")
    print(f"  per field. E5 compares text against major description. These measure")
    print(f"  fundamentally different things - one is quality, the other is relevance.")

    # Save
    out_path = PROJECT_ROOT / "reports" / "e5_vs_tfidf_pilot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase1_summary": {
                "n_pairs": len(phase1_results),
                "tfidf_mean": round(float(np.mean(tfidf_vals)), 4),
                "e5_mean": round(float(np.mean(e5_vals)), 4),
                "spearman_r": round(float(np.corrcoef(tfidf_vals, e5_vals)[0, 1]), 4),
            },
            "phase1_details": phase1_results,
            "phase2_results": neg_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
