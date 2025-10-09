import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _text_prep(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s


def _build_corpus(df: pd.DataFrame) -> list[str]:
    texts = []
    column_map = {
        "research_details": ["research_details", "research_detail"],
        "award_details": ["award_details", "award_detail"],
        "internship_details": ["internship_details", "internship_detail"],
        "paper_details": ["paper_details", "paper_detail"],
    }
    for canonical, candidates in column_map.items():
        for col in candidates:
            if col in df.columns:
                col_texts = df[col].fillna("").astype(str).map(_text_prep).tolist()
                texts.extend(col_texts)
    return [t for t in texts if t]


def _fit_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
        norm="l2",
    )
    vec.fit(corpus)
    return vec


def _compute_centroids(df: pd.DataFrame, vec: TfidfVectorizer) -> dict[str, np.ndarray]:
    centroids = {}
    column_map = {
        "research_details": ["research_details", "research_detail"],
        "award_details": ["award_details", "award_detail"],
        "internship_details": ["internship_details", "internship_detail"],
        "paper_details": ["paper_details", "paper_detail"],
    }
    for canonical, candidates in column_map.items():
        texts_collected: list[str] = []
        for col in candidates:
            if col in df.columns:
                texts_collected.extend(df[col].fillna("").astype(str).map(_text_prep).tolist())
        texts = [t for t in texts_collected if t]
        if not texts:
            continue
        X = vec.transform(texts)
        mean_vec = X.mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        centroids[canonical] = mean_vec
    return centroids


def main():
    data_path = Path("src/machine_learning_models/data/cases.feather")
    df = pd.read_feather(data_path)

    corpus = _build_corpus(df)
    if not corpus:
        print("没有足够的文本语料用于训练 TF-IDF 向量器")
        return

    vec = _fit_vectorizer(corpus)
    centroids = _compute_centroids(df, vec)

    out_dir = Path("src/machine_learning_models/pre-trained_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = out_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vec, vec_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"已保存 TF-IDF 向量器: {vec_path}")

    centroids_path = out_dir / "tfidf_centroids.npz"
    np.savez_compressed(centroids_path, **{k: v.astype(np.float32) for k, v in centroids.items()})
    print(f"已保存 TF-IDF 质心: {centroids_path}")


if __name__ == "__main__":
    main()
