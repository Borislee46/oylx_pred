"""TF-IDF文本加成模型训练脚本

本脚本用于训练文本TF-IDF向量器、计算质心并拟合logit uplift权重。
生成的模型用于线上LogitUpliftProvider进行文本加成。
"""

import json
import logging
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
except ImportError:
    _HAS_NNLS = False

# 配置常量
# 列名映射配置（统一管理）
COLUMN_MAP: dict[str, list[str]] = {
    "research_details": ["research_details", "research_detail"],
    "award_details": ["award_details", "award_detail"],
    "internship_details": ["internship_details", "internship_detail"],
    "paper_details": ["paper_details", "paper_detail"],
}

CANONICAL_KEYS: list[str] = list(COLUMN_MAP.keys())

# TF-IDF向量器参数
TFIDF_ANALYZER: str = "char_wb"
TFIDF_NGRAM_RANGE: tuple[int, int] = (2, 4)
TFIDF_MIN_DF: int = 1
TFIDF_MAX_FEATURES: int = 20000
TFIDF_SUBLINEAR_TF: bool = True
TFIDF_NORM: str = "l2"

# 相似度计算阈值
SIMILARITY_SIGNAL_THRESHOLD: float = 0.05
MIN_SAMPLES_FOR_FILTERING: int = 50

# Ridge回归参数
RIDGE_ALPHA: float = 6.0

# 概率裁剪参数
PROB_CLIP_MIN: float = 1e-4
PROB_CLIP_MAX: float = 1.0 - 1e-4
PROB_XGB_CLIP_MIN: float = 1e-6
PROB_XGB_CLIP_MAX: float = 1.0 - 1e-6

# 数值稳定性参数
EPSILON: float = 1e-12

# 路径配置
DATA_PATH: Path = Path("src/machine_learning_models/data/cases.feather")
OUTPUT_DIR: Path = Path("src/machine_learning_models/pre-trained_models")
VECTORIZER_FILENAME: str = "tfidf_vectorizer.joblib"
CENTROIDS_FILENAME: str = "tfidf_centroids.npz"
WEIGHTS_FILENAME: str = "text_uplift_weights.json"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _text_prep(s: str) -> str:
    """
    文本预处理：去除首尾空白

    Args:
        s: 原始文本

    Returns:
        处理后的文本
    """
    if not isinstance(s, str):
        return ""
    return s.strip()


def _build_corpus(df: pd.DataFrame) -> list[str]:
    """
    从DataFrame构建文本语料库

    Args:
        df: 数据框

    Returns:
        文本列表
    """
    texts = []
    for canonical, candidates in COLUMN_MAP.items():
        for col in candidates:
            if col in df.columns:
                col_texts = df[col].fillna("").astype(str).map(_text_prep).tolist()
                texts.extend(col_texts)
    return [t for t in texts if t]


def _fit_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """
    训练TF-IDF向量器

    Args:
        corpus: 文本语料库

    Returns:
        训练好的TF-IDF向量器
    """
    vec = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        norm=TFIDF_NORM,
    )
    vec.fit(corpus)
    logger.info(f"TF-IDF向量器训练完成，特征数: {len(vec.vocabulary_)}")
    return vec


def _compute_centroids(df: pd.DataFrame, vec: TfidfVectorizer) -> dict[str, np.ndarray]:
    """
    计算各类文本的质心向量

    Args:
        df: 数据框
        vec: TF-IDF向量器

    Returns:
        质心字典，键为规范列名，值为归一化的质心向量
    """
    centroids = {}
    for canonical, candidates in COLUMN_MAP.items():
        texts_collected: list[str] = []
        for col in candidates:
            if col in df.columns:
                texts_collected.extend(df[col].fillna("").astype(str).map(_text_prep).tolist())
        texts = [t for t in texts_collected if t]
        if not texts:
            logger.warning(f"列 {canonical} 没有有效文本，跳过质心计算")
            continue

        try:
            X = vec.transform(texts)
            mean_vec = X.mean(axis=0)
            mean_vec = np.asarray(mean_vec).ravel()
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            else:
                logger.warning(f"列 {canonical} 的质心向量范数为0")
                continue
            centroids[canonical] = mean_vec
        except Exception as e:
            logger.error(f"计算列 {canonical} 的质心时出错: {str(e)}", exc_info=True)
            continue

    logger.info(f"成功计算 {len(centroids)} 个质心")
    return centroids


def _compute_similarities(
    df: pd.DataFrame, vec: TfidfVectorizer, centroids: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    计算每个样本与各类质心的相似度

    Args:
        df: 数据框
        vec: TF-IDF向量器
        centroids: 质心字典

    Returns:
        相似度数据框，列名为 sr, sa, si, sp
    """
    if not centroids:
        raise ValueError("质心字典为空，请先计算质心")

    sims: dict[str, list[float]] = {"sr": [], "sa": [], "si": [], "sp": []}

    # 获取默认质心向量形状（用于处理缺失质心的情况）
    default_centroid_shape = None
    if centroids:
        default_centroid_shape = next(iter(centroids.values())).shape

    for _, row in df.iterrows():
        merged_texts = []
        for name in CANONICAL_KEYS:
            parts = []
            for col in COLUMN_MAP[name]:
                if col in df.columns:
                    parts.append(_text_prep(row.get(col, "")))
            merged_texts.append(" ".join([p for p in parts if p]))

        try:
            X = vec.transform(merged_texts)
            vals = []
            for idx, name in enumerate(CANONICAL_KEYS):
                r = X.getrow(idx)
                if r is None or getattr(r, "data", None) is None or r.data.size == 0:
                    vals.append(0.0)
                    continue

                centroid = centroids.get(name)
                if centroid is None:
                    if default_centroid_shape is not None:
                        centroid = np.zeros(default_centroid_shape, dtype=np.float32)
                    else:
                        vals.append(0.0)
                        continue

                try:
                    dot_val = r.dot(centroid)
                    dot_scalar = float(np.asarray(dot_val).ravel()[0])
                except (IndexError, ValueError, AttributeError) as e:
                    logger.debug(f"计算相似度时出错: {str(e)}")
                    dot_scalar = 0.0
                vals.append(float(np.clip(dot_scalar, 0.0, 1.0)))

            sims["sr"].append(vals[0])
            sims["sa"].append(vals[1])
            sims["si"].append(vals[2])
            sims["sp"].append(vals[3])
        except Exception as e:
            logger.warning(f"处理样本时出错: {str(e)}")
            sims["sr"].append(0.0)
            sims["sa"].append(0.0)
            sims["si"].append(0.0)
            sims["sp"].append(0.0)

    return pd.DataFrame(sims)


def _fit_uplift_weights(
    df: pd.DataFrame,
    sims_df: pd.DataFrame,
    p_base: Optional[np.ndarray],
) -> dict[str, float]:
    """
    拟合文本增益权重

    Args:
        df: 数据框
        sims_df: 相似度数据框
        p_base: 基础概率数组（可选）

    Returns:
        权重字典，包含 b, w_r, w_a, w_i, w_p, u_r, u_a, u_i, u_p

    Raises:
        ValueError: 如果数据框缺少必需的列
    """

    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, PROB_CLIP_MIN, PROB_CLIP_MAX)

    if "admitted" not in df.columns:
        raise ValueError("数据框缺少目标列 'admitted'")

    if len(df) != len(sims_df):
        raise ValueError(f"数据框长度不匹配: df={len(df)}, sims_df={len(sims_df)}")

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
        """获取计数序列，缺失时返回0"""
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
        logger.info("未提供基础概率，使用零作为logit基准")
    else:
        if len(p_base) != len(logit_true):
            raise ValueError(f"基础概率长度不匹配: p_base={len(p_base)}, y={len(logit_true)}")
        p_base = _clip01(p_base.astype(np.float64))
        logit_base = np.log(p_base / (1 - p_base))

    y_vec = (logit_true - logit_base).astype(np.float64)
    y_vec = np.maximum(y_vec, 0.0)

    # 过滤有效样本
    signal = (sr + sa + si + sp) > SIMILARITY_SIGNAL_THRESHOLD
    strength = (rc + ac + ic + pc) > 0.0
    mask = np.logical_or(signal, strength)

    if mask.any() and mask.sum() >= MIN_SAMPLES_FOR_FILTERING:
        X = X[mask]
        y_vec = y_vec[mask]
        logger.info(f"过滤后样本数: {mask.sum()}/{len(mask)}")
    else:
        logger.warning(f"有效样本数不足({mask.sum() if mask.any() else 0})，使用全部样本")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # 拟合权重
    if _HAS_NNLS:
        try:
            coef, _ = nnls(X, y_vec)
            if float(np.sum(coef)) <= EPSILON:
                logger.warning("NNLS结果接近零，回退到Ridge回归")
                ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, positive=True)
                ridge.fit(X, y_vec)
                coef = ridge.coef_
            else:
                logger.info("使用NNLS拟合权重")
        except Exception as e:
            logger.warning(f"NNLS拟合失败，回退到Ridge回归: {str(e)}")
            ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, positive=True)
            ridge.fit(X, y_vec)
            coef = ridge.coef_
    else:
        logger.info("NNLS不可用，使用Ridge回归")
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, positive=True)
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

    # 确保权重非负
    negative_count = sum(1 for v in out.values() if v < 0)
    if negative_count > 0:
        logger.warning(f"发现 {negative_count} 个负权重，将截断为0")

    for k, v in list(out.items()):
        if v < 0:
            out[k] = 0.0

    logger.info(f"权重拟合完成: {out}")
    return out


def _load_latest_xgb_model_paths() -> tuple[Path | None, Path | None, Path | None]:
    """
    加载最新的XGBoost模型路径

    Returns:
        (模型路径, 特征路径, 校准路径) 元组
    """
    if not OUTPUT_DIR.exists():
        logger.warning(f"输出目录不存在: {OUTPUT_DIR}")
        return None, None, None

    try:
        models = sorted(
            OUTPUT_DIR.glob("xgboost_*.model"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not models:
            logger.warning("未找到XGBoost模型文件")
            return None, None, None

        model_path = models[0]
        stem = model_path.stem
        features_path = OUTPUT_DIR / f"{stem}_features.json"
        calib_path = OUTPUT_DIR / f"{stem}_calibration.json"

        logger.info(f"找到XGBoost模型: {model_path}")
        return (
            model_path,
            (features_path if features_path.exists() else None),
            (calib_path if calib_path.exists() else None),
        )
    except Exception as e:
        logger.error(f"加载XGBoost模型路径时出错: {str(e)}", exc_info=True)
        return None, None, None


def _compute_p_base_with_xgb(df: pd.DataFrame) -> np.ndarray | None:
    """
    使用XGBoost模型计算基础概率

    Args:
        df: 数据框

    Returns:
        基础概率数组，如果计算失败则返回None
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import xgboost as xgb

        from src.machine_learning_models.feature_engineer import FeatureEngineer

        model_path, features_path, calib_path = _load_latest_xgb_model_paths()
        if model_path is None or features_path is None:
            logger.warning("未找到XGBoost模型或特征文件，跳过基础概率计算")
            return None

        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        if not isinstance(feature_names, list) or not feature_names:
            logger.warning("特征列表无效")
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
                    logger.info("应用了sigmoid校准")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"加载校准参数失败: {str(e)}")

        return np.clip(p_raw, PROB_XGB_CLIP_MIN, PROB_XGB_CLIP_MAX)
    except ImportError:
        logger.warning("xgboost或FeatureEngineer不可用，跳过基础概率计算")
        return None
    except Exception as e:
        logger.error(f"计算基础概率时出错: {str(e)}", exc_info=True)
        return None


def main() -> None:
    """主函数：执行完整的训练流程"""
    try:
        # 检查数据文件
        if not DATA_PATH.exists():
            logger.error(f"数据文件不存在: {DATA_PATH}")
            sys.exit(1)

        logger.info(f"加载数据: {DATA_PATH}")
        df = pd.read_feather(DATA_PATH)
        logger.info(f"数据加载完成，样本数: {len(df)}")

        # 构建语料库
        corpus = _build_corpus(df)
        if not corpus:
            logger.error("没有足够的文本语料用于训练 TF-IDF 向量器")
            sys.exit(1)
        logger.info(f"语料库大小: {len(corpus)}")

        # 训练向量器
        vec = _fit_vectorizer(corpus)

        # 计算质心
        centroids = _compute_centroids(df, vec)
        if not centroids:
            logger.error("质心计算失败")
            sys.exit(1)

        # 创建输出目录
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 保存向量器
        vec_path = OUTPUT_DIR / VECTORIZER_FILENAME
        joblib.dump(vec, vec_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"已保存 TF-IDF 向量器: {vec_path}")

        # 保存质心
        centroids_path = OUTPUT_DIR / CENTROIDS_FILENAME
        np.savez_compressed(
            centroids_path, **{k: v.astype(np.float32) for k, v in centroids.items()}
        )
        logger.info(f"已保存 TF-IDF 质心: {centroids_path}")

        # 计算相似度
        sims_df = _compute_similarities(df, vec, centroids)
        logger.info(f"相似度计算完成，形状: {sims_df.shape}")

        # 计算基础概率
        p_base = _compute_p_base_with_xgb(df)
        if p_base is not None:
            logger.info(f"基础概率计算完成，范围: [{p_base.min():.4f}, {p_base.max():.4f}]")

        # 拟合权重
        weights = _fit_uplift_weights(df, sims_df, p_base=p_base)

        # 保存权重
        weights_path = OUTPUT_DIR / WEIGHTS_FILENAME
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存文本增益权重: {weights_path}")

        logger.info("训练完成！")

    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
