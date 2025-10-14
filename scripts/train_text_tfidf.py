import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

try:
    from scipy.optimize import nnls

    _HAS_NNLS = True
except Exception:
    _HAS_NNLS = False


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


def _compute_similarities(df: pd.DataFrame, vec: TfidfVectorizer) -> pd.DataFrame:
    column_map = {
        "research_details": ["research_details", "research_detail"],
        "award_details": ["award_details", "award_detail"],
        "internship_details": ["internship_details", "internship_detail"],
        "paper_details": ["paper_details", "paper_detail"],
    }
    canonical_keys = ["research_details", "award_details", "internship_details", "paper_details"]
    sims: dict[str, list[float]] = {"sr": [], "sa": [], "si": [], "sp": []}
    centroids_path = Path("src/machine_learning_models/pre-trained_models/tfidf_centroids.npz")
    if not centroids_path.exists():
        raise FileNotFoundError("请先生成 tfidf_centroids.npz")
    data = np.load(centroids_path, mmap_mode="r")
    centroids = {k: data[k] for k in data.files}
    for k in list(centroids.keys()):
        v = np.asarray(centroids[k], dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        centroids[k] = v

    def _prep(s: str) -> str:
        if not isinstance(s, str):
            return ""
        return s.strip()

    for _, row in df.iterrows():
        merged_texts = []
        for name in canonical_keys:
            parts = []
            for col in column_map[name]:
                if col in df.columns:
                    parts.append(_prep(row.get(col, "")))
            merged_texts.append(" ".join([p for p in parts if p]))

        X = vec.transform(merged_texts)
        vals = []
        for idx, name in enumerate(canonical_keys):
            r = X.getrow(idx)
            if r is None or getattr(r, "data", None) is None or r.data.size == 0:
                vals.append(0.0)
                continue
            dot_val = r.dot(centroids.get(name, np.zeros_like(next(iter(centroids.values())))))
            try:
                dot_scalar = float(np.asarray(dot_val).ravel()[0])
            except Exception:
                dot_scalar = 0.0
            vals.append(float(np.clip(dot_scalar, 0.0, 1.0)))
        sims["sr"].append(vals[0])
        sims["sa"].append(vals[1])
        sims["si"].append(vals[2])
        sims["sp"].append(vals[3])
    return pd.DataFrame(sims)


def _fit_uplift_weights(
    df: pd.DataFrame,
    sims_df: pd.DataFrame,
    p_base: Optional[np.ndarray],
) -> dict[str, float]:
    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 1e-4, 1 - 1e-4)

    if "admitted" not in df.columns:
        raise ValueError("cases.feather 缺少目标列 admitted")
    y = df["admitted"].astype(np.float32).to_numpy()
    p_true = _clip01(y)
    logit_true = np.log(p_true / (1 - p_true))

    sims_df = sims_df.fillna(0.0)
    sr, sa, si, sp = (
        sims_df["sr"].to_numpy(dtype=np.float32),
        sims_df["sa"].to_numpy(dtype=np.float32),
        sims_df["si"].to_numpy(dtype=np.float32),
        sims_df["sp"].to_numpy(dtype=np.float32),
    )

    def _count_series(name: str) -> pd.Series:
        if name in df.columns:
            s = df[name]
        else:
            s = pd.Series(0, index=df.index)
        return s.fillna(0).astype(np.float32)

    rc = np.log1p(_count_series("research_count").to_numpy(dtype=np.float32))
    ac = np.log1p(_count_series("award_count").to_numpy(dtype=np.float32))
    ic = np.log1p(_count_series("internship_count").to_numpy(dtype=np.float32))
    pc = np.log1p(_count_series("paper_count").to_numpy(dtype=np.float32))

    X = np.stack(
        [
            sr,
            sa,
            si,
            sp,
            sr * rc,
            sa * ac,
            si * ic,
            sp * pc,
        ],
        axis=1,
    ).astype(np.float64)

    if p_base is None:
        logit_base = np.zeros_like(logit_true, dtype=np.float64)
    else:
        p_base = _clip01(p_base.astype(np.float64))
        logit_base = np.log(p_base / (1 - p_base))

    y_vec = (logit_true - logit_base).astype(np.float64)
    y_vec = np.maximum(y_vec, 0.0)
    signal = (sr + sa + si + sp) > 0.05
    strength = (rc + ac + ic + pc) > 0.0
    mask = np.logical_or(signal, strength)
    if mask.any() and mask.sum() >= 50:
        X = X[mask]
        y_vec = y_vec[mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

    if _HAS_NNLS:
        coef, _ = nnls(X, y_vec)
        if float(np.sum(coef)) <= 1e-12:
            ridge = Ridge(alpha=6.0, fit_intercept=False, positive=True)
            ridge.fit(X, y_vec)
            coef = ridge.coef_
    else:
        ridge = Ridge(alpha=6.0, fit_intercept=False, positive=True)
        ridge.fit(X, y_vec)
        coef = ridge.coef_

    out = {
        "b": 0.0,
        "w_r": float(coef[0]),
        "w_a": float(coef[1]),
        "w_i": float(coef[2]),
        "w_p": float(coef[3]),
        "u_r": float(coef[4]),
        "u_a": float(coef[5]),
        "u_i": float(coef[6]),
        "u_p": float(coef[7]),
    }
    for k, v in list(out.items()):
        if v < 0:
            out[k] = 0.0
    return out


def _load_latest_xgb_model_paths() -> tuple[Path | None, Path | None, Path | None]:
    base_dir = Path("src/machine_learning_models/pre-trained_models")
    if not base_dir.exists():
        return None, None, None
    models = sorted(base_dir.glob("xgboost_*.model"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        return None, None, None
    model_path = models[0]
    stem = model_path.stem
    features_path = base_dir / f"{stem}_features.json"
    calib_path = base_dir / f"{stem}_calibration.json"
    return (
        model_path,
        (features_path if features_path.exists() else None),
        (calib_path if calib_path.exists() else None),
    )


def _compute_p_base_with_xgb(df: pd.DataFrame) -> np.ndarray | None:
    try:
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import xgboost as xgb

        from src.machine_learning_models.feature_engineer import FeatureEngineer

        model_path, features_path, calib_path = _load_latest_xgb_model_paths()
        if model_path is None or features_path is None:
            return None

        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        if not isinstance(feature_names, list) or not feature_names:
            return None

        fe = FeatureEngineer()
        X = fe.fit_transform(df.copy())
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dmat = xgb.DMatrix(X, enable_categorical=True)
        p_raw = booster.predict(dmat)
        p_raw = np.asarray(p_raw, dtype=np.float64)

        if calib_path is not None and calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    calib = json.load(f)
                if calib and calib.get("method") == "sigmoid":
                    a = float(calib.get("params", {}).get("a", 0.0))
                    b = float(calib.get("params", {}).get("b", 0.0))
                    p_raw = 1.0 / (1.0 + np.exp(a * p_raw + b))
            except Exception:
                pass

        return np.clip(p_raw, 1e-6, 1 - 1e-6)
    except Exception:
        return None


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

    sims_df = _compute_similarities(df, vec)
    p_base = _compute_p_base_with_xgb(df)
    weights = _fit_uplift_weights(df, sims_df, p_base=p_base)
    weights_path = out_dir / "text_uplift_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"已保存文本增益权重: {weights_path}")


if __name__ == "__main__":
    main()
