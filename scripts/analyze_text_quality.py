"""
Text "含金量" Analysis: What measurable text properties predict admission?

Research question: After controlling for hard credentials (GPA, school, language),
what text characteristics are associated with higher admission probability?

Approach:
1. Establish tabular baseline (logistic regression on hard credentials)
2. Compute multi-dimensional text features:
   - TF-IDF centroid similarity (current production)
   - E5 centroid similarity (same logic, dense embedding space)
   - Surface metrics (length, entropy, specificity markers)
   - E5 semantic features (relevance to major, embedding diversity)
3. Measure incremental predictive power of each feature set
4. Identify which dimensions of "text quality" actually matter
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────
CASES_PATH = PROJECT_ROOT / "src/machine_learning_models/data/cases.feather"
VEC_PATH = PROJECT_ROOT / "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib"
CEN_PATH = PROJECT_ROOT / "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz"
E5_PATH = PROJECT_ROOT / "src/services/multilingual-e5-large-instruct"

SEED = 42
N_ANALYSIS = 5000  # sample size for analysis (5090 goes brrr)
N_CENTROID_POOL = 1000  # admitted students to build E5 centroid

TEXT_KEYS = ["research_detail", "internship_detail", "award_detail", "paper_detail"]
CENTROID_KEYS = ["research_details", "internship_details", "award_details", "paper_details"]


# ═══════════════════════════════════════════════════════
# 1. Data Loading & Preparation
# ═══════════════════════════════════════════════════════

def load_and_prepare() -> pd.DataFrame:
    df = pd.read_feather(CASES_PATH)
    print(f"Raw cases: {len(df):,}")

    # Keep cases with internship text (most common)
    df = df[df["internship_detail"].notna() & (df["internship_detail"].str.len() > 30)].copy()
    print(f"With internship text: {len(df):,}")

    # Create school tier (simplified: C9 / 985 / 211 / other)
    # For now, use has_text as indicator of "has any text"
    df["has_text"] = df["internship_detail"].notna().astype(int)

    # Fill missing values
    df["gpa"] = df["gpa"].fillna(df["gpa"].median())
    df["ielts"] = df["ielts"].fillna(df["ielts"].median())
    df["toefl"] = df["toefl"].fillna(df["toefl"].median())

    # Normalize language scores to a single scale (approximate IELTS)
    # TOEFL 100 ≈ IELTS 7.0, rough linear mapping
    has_toefl = df["toefl"].notna() & (df["toefl"] > 0)
    has_ielts = df["ielts"].notna() & (df["ielts"] > 0)
    df["lang_score"] = np.where(
        has_ielts, df["ielts"],
        np.where(has_toefl, df["toefl"] / 100 * 7.0, 6.0)
    )
    df["lang_score"] = df["lang_score"].clip(4.0, 9.0)

    # Sample for analysis
    if len(df) > N_ANALYSIS:
        df = df.sample(n=N_ANALYSIS, random_state=SEED).copy()
        print(f"Sampled {N_ANALYSIS:,} for analysis")

    print(f"Admission rate: {df['admitted'].mean():.3f}")
    return df


# ═══════════════════════════════════════════════════════
# 2. Feature Computation
# ═══════════════════════════════════════════════════════

def compute_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """Hard credentials only."""
    feats = pd.DataFrame(index=df.index)
    feats["gpa"] = df["gpa"]
    feats["lang_score"] = df["lang_score"]
    # One-hot encode faculties if available
    if "faculty" in df.columns:
        top_faculties = df["faculty"].value_counts().head(10).index
        for fac in top_faculties:
            feats[f"faculty_{fac}"] = (df["faculty"] == fac).astype(int)
    feats = feats.fillna(0)
    return feats.values.astype(np.float64)


def compute_surface_features(df: pd.DataFrame) -> np.ndarray:
    """Surface text metrics: length, entropy, specificity markers."""
    feats = pd.DataFrame(index=df.index)

    for key in TEXT_KEYS:
        label = key.replace("_detail", "")
        texts = df[key].fillna("")

        # Length features
        feats[f"{label}_len"] = texts.apply(lambda t: len(str(t)))
        feats[f"{label}_len_log"] = np.log1p(feats[f"{label}_len"])

        # Number of distinct entries (separated by ; or ；)
        feats[f"{label}_n_entries"] = texts.apply(
            lambda t: len([x for x in str(t).replace("；", ";").split(";") if x.strip()])
        )

        # Specificity: count numbers (dates, percentages, metrics)
        import re
        feats[f"{label}_n_numbers"] = texts.apply(
            lambda t: len(re.findall(r'\d+', str(t)))
        )

        # Count proper nouns (Chinese/English names, organizations)
        # Simplified: count uppercase sequences and 《》 references
        feats[f"{label}_n_refs"] = texts.apply(
            lambda t: len(re.findall(r'《[^》]+》', str(t)))
        )

    feats = feats.fillna(0)
    return feats.values.astype(np.float64)


def compute_tfidf_features(df: pd.DataFrame, vec, centroids) -> np.ndarray:
    """Current production TF-IDF centroid similarities."""
    from scipy.sparse import vstack

    feats = pd.DataFrame(index=df.index)

    for text_key, centroid_key in zip(TEXT_KEYS, CENTROID_KEYS):
        label = text_key.replace("_detail", "")
        texts = df[text_key].fillna("")

        # Transform all texts at once for efficiency
        X = vec.transform(texts.tolist())
        centroid = centroids[centroid_key]

        # Compute dot product for each row
        dots = X.dot(centroid)
        if hasattr(dots, 'toarray'):
            dots = dots.toarray().flatten()
        else:
            dots = np.asarray(dots).flatten()

        feats[f"{label}_tfidf_sim"] = np.clip(dots, 0.0, 1.0)

    # Log-entropy richness (replication of production logic)
    for text_key in TEXT_KEYS:
        label = text_key.replace("_detail", "")
        texts = df[text_key].fillna("")
        feats[f"{label}_richness"] = texts.apply(_fast_entropy)

    feats = feats.fillna(0)
    return feats.values.astype(np.float64)


def _fast_entropy(text: str) -> float:
    """Replicate production entropy computation."""
    if not text or len(text) < 10:
        return 0.0
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return 0.0
    if len(b) < 10:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(b)
    entropy = -np.sum(probs * np.log2(probs))
    byte_rich = float(np.clip(entropy / 5.0, 0.0, 1.0))

    n = len(text)
    if n >= 12:
        span = max(12.0, float(n ** 0.55))
        char_f = float(np.clip(len(set(text)) / span, 0.0, 1.0))
        byte_rich *= 0.35 + 0.65 * char_f

    return float(np.clip(byte_rich, 0.0, 1.0))


def compute_e5_features(df: pd.DataFrame, e5_model, df_admitted: pd.DataFrame) -> np.ndarray:
    """E5 embedding-based features.

    1. E5 centroid similarity (same logic as TF-IDF, but in E5 space)
    2. E5 relevance to target major
    """
    from sentence_transformers import SentenceTransformer

    model: SentenceTransformer = e5_model
    feats = pd.DataFrame(index=df.index)

    # Build E5 centroids from admitted students
    print("  Building E5 centroids from admitted students...")
    e5_centroids = {}
    for text_key in TEXT_KEYS:
        label = text_key.replace("_detail", "")
        admitted_texts = df_admitted[text_key].dropna()
        admitted_texts = admitted_texts[admitted_texts.str.len() > 30]
        if len(admitted_texts) > N_CENTROID_POOL:
            admitted_texts = admitted_texts.sample(n=N_CENTROID_POOL, random_state=SEED)

        print(f"    Encoding {len(admitted_texts)} {label} texts for centroid...")
        embeddings = model.encode(
            admitted_texts.tolist(),
            normalize_embeddings=True,
            show_progress_bar=False, batch_size=256,
        )
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        e5_centroids[text_key] = centroid

    # Compute similarities
    for text_key in TEXT_KEYS:
        label = text_key.replace("_detail", "")
        texts = df[text_key].fillna("").tolist()
        # Replace empty with placeholder
        texts_enc = [t if (isinstance(t, str) and len(t) >= 10) else " " for t in texts]
        centroid = e5_centroids[text_key]

        print(f"    Computing E5 centroid similarity for {label} ({len(texts)} texts)...")
        embeddings = model.encode(texts_enc, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        sims = np.dot(embeddings, centroid)
        feats[f"{label}_e5_centroid_sim"] = np.clip(sims, 0.0, 1.0)

    # E5 relevance to major
    print("  Computing E5 major relevance...")
    majors = df["target_major"].fillna("").tolist()
    for text_key in ["internship_detail"]:  # just one field for now (most informative)
        label = text_key.replace("_detail", "")
        texts = df[text_key].fillna("").tolist()
        texts_enc = [t if (isinstance(t, str) and len(t) >= 10) else " " for t in texts]

        # Encode text as query, major as passage
        queries = [f"Instruct: Given an applicant's experience, assess relevance to the major.\nQuery: {t}" for t in texts_enc]
        passages = [f"Passage: {m}" for m in majors]

        print(f"    Encoding {len(queries)} query-passage pairs...")
        emb_q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        emb_p = model.encode(passages, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        # Cosine similarity per pair
        sims = np.sum(emb_q * emb_p, axis=1)
        feats[f"{label}_e5_major_rel"] = np.clip(sims, 0.0, 1.0)

    feats = feats.fillna(0)
    return feats.values.astype(np.float64)


# ═══════════════════════════════════════════════════════
# 3. Analysis
# ═══════════════════════════════════════════════════════

def evaluate_feature_set(
    name: str,
    X_base: np.ndarray,
    X_extra: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate if X_extra adds predictive power beyond X_base."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Baseline (tabular only)
    model_base = LogisticRegression(max_iter=2000, random_state=SEED)
    auc_base = cross_val_score(model_base, X_base, y, cv=cv, scoring="roc_auc").mean()

    # Tabular + extra features
    X_combined = np.hstack([X_base, X_extra])
    model_combined = LogisticRegression(max_iter=2000, random_state=SEED)
    auc_combined = cross_val_score(model_combined, X_combined, y, cv=cv, scoring="roc_auc").mean()

    # Fit on full data for coefficient analysis
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    model_combined.fit(X_combined_scaled, y)

    # Get coefficients for extra features
    n_base = X_base.shape[1]
    extra_coefs = model_combined.coef_[0][n_base:]

    return {
        "name": name,
        "n_features": X_extra.shape[1],
        "auc_tabular_only": round(auc_base, 4),
        "auc_with_text": round(auc_combined, 4),
        "delta_auc": round(auc_combined - auc_base, 4),
        "extra_coefs_mean": round(float(np.mean(np.abs(extra_coefs))), 6),
        "extra_coefs_max": round(float(np.max(np.abs(extra_coefs))), 6),
    }


def analyze_text_dimensions(df: pd.DataFrame) -> dict:
    """Break down text quality into dimensions and test each."""
    # Tabular baseline
    X_tab = compute_tabular_features(df)
    y = df["admitted"].values

    print(f"\nTabular features: {X_tab.shape[1]}")
    print(f"Admission rate: {y.mean():.3f}")

    results = {}

    # Dimension 1: Surface features (length, entropy, specificity)
    print("\n[Dimension 1] Surface text metrics...")
    X_surface = compute_surface_features(df)
    results["surface"] = evaluate_feature_set("Surface (length/entropy/specificity)", X_tab, X_surface, y)
    print(f"  AUC improvement: {results['surface']['delta_auc']:+.4f}")

    # Dimension 2: TF-IDF centroid (current production)
    print("\n[Dimension 2] TF-IDF centroid similarity...")
    vec = joblib.load(VEC_PATH)
    centroids = dict(np.load(CEN_PATH))
    for k in centroids:
        arr = np.asarray(centroids[k], dtype=np.float32)
        n = np.linalg.norm(arr)
        if n > 0:
            centroids[k] = arr / n
    X_tfidf = compute_tfidf_features(df, vec, centroids)
    results["tfidf"] = evaluate_feature_set("TF-IDF Centroid", X_tab, X_tfidf, y)
    print(f"  AUC improvement: {results['tfidf']['delta_auc']:+.4f}")

    return results, X_tab, y, vec, centroids


# ═══════════════════════════════════════════════════════
# 4. Case Studies: What does "good text" look like?
# ═══════════════════════════════════════════════════════

def case_study(df: pd.DataFrame, vec, centroids):
    """Identify cases where text likely helped or hurt."""
    # Fit simple model to get residuals
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({
        "gpa": df["gpa"],
        "lang_score": df["lang_score"],
    }).fillna(0).values
    y = df["admitted"].values

    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(X, y)
    prob_tab = model.predict_proba(X)[:, 1]

    # Residual: actual - predicted (from tabular only)
    df_eval = df.copy()
    df_eval["prob_tabular"] = prob_tab
    df_eval["residual"] = y - prob_tab

    # Compute text quality score (TF-IDF sim average across fields)
    sims = {}
    for text_key, centroid_key in zip(TEXT_KEYS, CENTROID_KEYS):
        texts = df[text_key].fillna("")
        X_sparse = vec.transform(texts.tolist())
        centroid = centroids[centroid_key]
        dots = X_sparse.dot(centroid)
        if hasattr(dots, 'toarray'):
            dots = dots.toarray().flatten()
        else:
            dots = np.asarray(dots).flatten()
        sims[text_key] = np.clip(dots, 0.0, 1.0)

    df_eval["text_score"] = sum(sims.values()) / len(sims)

    # Find interesting cases
    # HIGH residual + admitted: text likely helped
    # LOW residual + rejected + high text_score: text should have helped but didn't
    mask_helped = (df_eval["residual"] > 0.15) & (y == 1)
    mask_hurt = (df_eval["residual"] < -0.2) & (y == 0) & (df_eval["text_score"] > df_eval["text_score"].median())

    cases_helped = df_eval[mask_helped].nlargest(5, "text_score")
    cases_confounding = df_eval[mask_hurt].nlargest(5, "text_score")

    print(f"\n{'=' * 70}")
    print("CASE STUDY: High-residual admitted (text likely helped)")
    print(f"{'=' * 70}")
    for i, (_, row) in enumerate(cases_helped.iterrows()):
        print(f"\n--- Helped Case {i+1} ---")
        print(f"  GPA={row['gpa']:.2f}  Lang={row.get('ielts', row.get('lang_score','?'))}  Admitted=YES")
        print(f"  Tabular prob={row['prob_tabular']:.3f}  Residual={row['residual']:+.3f}")
        print(f"  Text score: {row['text_score']:.3f}")
        print(f"  Major: {row['target_major'][:60]}")
        for text_key in TEXT_KEYS:
            text = row.get(text_key)
            if isinstance(text, str) and len(text) > 20:
                print(f"  [{text_key.replace('_detail','')}]: {text[:150]}...")

    print(f"\n{'=' * 70}")
    print("CASE STUDY: High text score but rejected (text not enough)")
    print(f"{'=' * 70}")
    for i, (_, row) in enumerate(cases_confounding.iterrows()):
        print(f"\n--- Confounding Case {i+1} ---")
        print(f"  GPA={row['gpa']:.2f}  Lang={row.get('ielts', row.get('lang_score','?'))}  Admitted=NO")
        print(f"  Tabular prob={row['prob_tabular']:.3f}  Residual={row['residual']:+.3f}")
        print(f"  Text score: {row['text_score']:.3f}")
        print(f"  Major: {row['target_major'][:60]}")
        for text_key in TEXT_KEYS:
            text = row.get(text_key)
            if isinstance(text, str) and len(text) > 20:
                print(f"  [{text_key.replace('_detail','')}]: {text[:150]}...")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("TEXT QUALITY (含金量) ANALYSIS")
    print("=" * 70)

    # Check if we should include E5 analysis
    include_e5 = "--e5" in sys.argv
    if include_e5:
        print("Mode: Full analysis including E5 embeddings")
    else:
        print("Mode: Surface + TF-IDF (add --e5 for E5 embeddings)")

    df = load_and_prepare()

    # Run surface + TF-IDF analysis
    results, X_tab, y, vec, centroids = analyze_text_dimensions(df)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Feature Set':<40} {'N Feat':<8} {'AUC Base':<10} {'AUC +Text':<10} {'Δ AUC':<10}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{r['name']:<40} {r['n_features']:<8} {r['auc_tabular_only']:<10} {r['auc_with_text']:<10} {r['delta_auc']:<+10}")

    # Run case studies
    case_study(df, vec, centroids)

    # If E5 requested, run it
    if include_e5:
        print(f"\n{'=' * 70}")
        print("E5 EMBEDDING ANALYSIS")
        print(f"{'=' * 70}")
        from sentence_transformers import SentenceTransformer
        e5 = SentenceTransformer(str(E5_PATH))
        df_admitted = df[df["admitted"] == 1]
        X_e5 = compute_e5_features(df, e5, df_admitted)
        r_e5 = evaluate_feature_set("E5 Embedding Features", X_tab, X_e5, y)
        results["e5"] = r_e5
        print(f"  AUC improvement: {r_e5['delta_auc']:+.4f}")

    # Save results
    out_path = PROJECT_ROOT / "reports" / "text_quality_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
