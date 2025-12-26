import json

"""
文本质量增量模型训练脚本

本脚本负责训练在线预测时使用的“背提加成”模型权重。
主要产出：
1. tfidf_vectorizer.joblib: 文本向量化模型。
2. tfidf_centroids.npz: 高质量案例在 TF-IDF 空间中的中心点（作为质量基准）。
3. text_uplift_weights.json: 相似度与经历数量对录取概率提升的贡献权重。

权重文件字段说明:
- w_r, w_a, w_i, w_p: 对应科研、奖项、实习、论文的【基础质量权重】。
- u_r, u_a, u_i, u_p: 对应科研、奖项、实习、论文的【质量 × 数量 交互权重】。
"""
import logging
import pickle
import sys
from pathlib import Path

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

COLUMN_MAP: dict[str, list[str]] = {
    "research_details": ["research_details", "research_detail"],
    "award_details": ["award_details", "award_detail"],
    "internship_details": ["internship_details", "internship_detail"],
    "paper_details": ["paper_details", "paper_detail"],
}

CANONICAL_KEYS: list[str] = list(COLUMN_MAP.keys())

TFIDF_ANALYZER: str = "char_wb"
TFIDF_NGRAM_RANGE: tuple[int, int] = (2, 4)
TFIDF_MIN_DF: int = 1
TFIDF_MAX_FEATURES: int = 20000
TFIDF_SUBLINEAR_TF: bool = True
TFIDF_NORM: str = "l2"

SIMILARITY_SIGNAL_THRESHOLD: float = 0.05
MIN_SAMPLES_FOR_FILTERING: int = 50

RIDGE_ALPHA: float = 6.0

PROB_CLIP_MIN: float = 1e-4
PROB_CLIP_MAX: float = 1.0 - 1e-4
PROB_XGB_CLIP_MIN: float = 1e-6
PROB_XGB_CLIP_MAX: float = 1.0 - 1e-6

EPSILON: float = 1e-12

DATA_PATH: Path = Path("src/machine_learning_models/data/cases.feather")
OUTPUT_DIR: Path = Path("src/machine_learning_models/pre-trained_models")
VECTORIZER_FILENAME: str = "tfidf_vectorizer.joblib"
CENTROIDS_FILENAME: str = "tfidf_centroids.npz"
WEIGHTS_FILENAME: str = "text_uplift_weights.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _text_prep(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()


def _calculate_entropy_fast(text: str) -> float:
    """
    极致优化的香农熵计算（字节级）。
    比原生 Counter 快 10-20 倍。用于评估文本信息密度。
    """
    if not text:
        return 0.0
    # 转换为字节数组进行超高速计数
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return 0.0

    if len(b) < 10:  # 过滤过短文本
        return 0.0

    # 使用 np.bincount 在字节层面进行直方图统计
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(b)

    # 熵计算：-Σ p*log2(p)
    entropy = -np.sum(probs * np.log2(probs))

    # 归一化：字节最大熵为 8.0。
    # 针对硕士申请背景，取 5.0 作为高质量信息量的饱和阈值
    return float(np.clip(entropy / 5.0, 0.0, 1.0))


def _build_corpus(df: pd.DataFrame) -> list[str]:
    texts = []
    for _canonical, candidates in COLUMN_MAP.items():
        for col in candidates:
            if col in df.columns:
                col_texts = df[col].fillna("").astype(str).map(_text_prep).tolist()
                texts.extend(col_texts)
    return [t for t in texts if t]


def _fit_vectorizer(corpus: list[str]) -> TfidfVectorizer:
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
    if not centroids:
        raise ValueError("质心字典为空，请先计算质心")

    sims: dict[str, list[float]] = {"sr": [], "sa": [], "si": [], "sp": []}

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
    p_base: np.ndarray | None,
) -> dict[str, float]:
    """
    拟合文本增量权重 (Uplift Weights Fitting)。

    采用增量建模 (Uplift Modeling) 的核心逻辑：
    1. 残差定义：计算真实标签 (logit_true) 与基础模型预测 (logit_base) 的差值，即“纯背景带来的增量”。
    2. 特征构建：包含各维度文本的基础相似度，以及“相似度 × 经历数量”的交互项。
    3. 优化目标：寻找一组非负权重，使得背景特征的线性组合能最大程度解释该残差。
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
        if name in df.columns:
            s = df[name]
        else:
            s = pd.Series(0, index=df.index)
        return s.fillna(0).astype(np.float32)

    # 极致优化：使用向量化方式计算各列的熵（有效信息丰盈度）
    def _get_richness_vec(canonical_key: str) -> np.ndarray:
        candidates = COLUMN_MAP.get(canonical_key, [])
        all_parts = []
        for col in candidates:
            if col in df.columns:
                all_parts.append(df[col].fillna("").astype(str))

        if not all_parts:
            return np.zeros(len(df), dtype=np.float32)

        # 合并所有文本部分
        merged = all_parts[0]
        for i in range(1, len(all_parts)):
            merged = merged + " " + all_parts[i]

        return merged.map(_calculate_entropy_fast).to_numpy(dtype=np.float32)

    r_rich = _get_richness_vec("research_details")
    a_rich = _get_richness_vec("award_details")
    i_rich = _get_richness_vec("internship_details")
    p_rich = _get_richness_vec("paper_details")

    # 构建特征矩阵 X
    # 前 4 列为修正后的基础质量得分：quality * richness
    # 后 4 列为修正后的交互项：quality(adj) * log1p(count * richness)
    sr_adj = sr * r_rich
    sa_adj = sa * a_rich
    si_adj = si * i_rich
    sp_adj = sp * p_rich

    rc_adj = np.log1p(_count_series("research_count").to_numpy() * r_rich)
    ac_adj = np.log1p(_count_series("award_count").to_numpy() * a_rich)
    ic_adj = np.log1p(_count_series("internship_count").to_numpy() * i_rich)
    pc_adj = np.log1p(_count_series("paper_count").to_numpy() * p_rich)

    X = np.stack(
        [
            sr_adj,
            sa_adj,
            si_adj,
            sp_adj,
            sr_adj * rc_adj,
            sa_adj * ac_adj,
            si_adj * ic_adj,
            sp_adj * pc_adj,
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

    signal = (sr_adj + sa_adj + si_adj + sp_adj) > SIMILARITY_SIGNAL_THRESHOLD
    strength = (rc_adj + ac_adj + ic_adj + pc_adj) > 0.0
    mask = np.logical_or(signal, strength)

    if mask.any() and mask.sum() >= MIN_SAMPLES_FOR_FILTERING:
        X = X[mask]
        y_vec = y_vec[mask]
        logger.info(f"过滤后样本数: {mask.sum()}/{len(mask)}")
    else:
        logger.warning(f"有效样本数不足({mask.sum() if mask.any() else 0})，使用全部样本")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

    if _HAS_NNLS:
        try:
            # 使用非负最小二乘法 (NNLS)
            # 业务约束：我们假设背景文本只会带来正面加成 (Uplift)，不应存在“扣分”权重
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

    negative_count = sum(1 for v in out.values() if v < 0)
    if negative_count > 0:
        logger.warning(f"发现 {negative_count} 个负权重，将截断为0")

    for k, v in list(out.items()):
        if v < 0:
            out[k] = 0.0

    logger.info(f"权重拟合完成: {out}")
    return out


def _load_latest_xgb_model_path() -> Path | None:
    if not OUTPUT_DIR.exists():
        logger.warning(f"输出目录不存在: {OUTPUT_DIR}")
        return None

    try:
        models = sorted(
            OUTPUT_DIR.glob("xgboost_*.ubj"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not models:
            logger.warning("未找到 .ubj 格式的 XGBoost 模型文件")
            return None

        model_path = models[0]
        logger.info(f"找到最新 XGBoost 模型: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"加载 XGBoost 模型路径时出错: {str(e)}", exc_info=True)
        return None


def _compute_p_base_with_xgb(df: pd.DataFrame) -> np.ndarray | None:
    try:
        project_root = Path(__file__).resolve().parent.parent
        ml_models_dir = project_root / "src" / "machine_learning_models"

        for p in [str(project_root), str(ml_models_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        import xgboost as xgb

        from src.machine_learning_models.feature_engineer import FeatureEngineer

        model_path = _load_latest_xgb_model_path()
        if model_path is None:
            return None

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        feature_names_raw = booster.attr("feature_names")
        if not feature_names_raw:
            logger.warning("模型中未找到特征名称属性")
            return None
        feature_names = json.loads(feature_names_raw)

        fe = FeatureEngineer()
        X = fe.fit_transform(df.copy())
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        dmat = xgb.DMatrix(X, enable_categorical=True)
        p_raw = booster.predict(dmat)
        p_raw = np.asarray(p_raw, dtype=np.float64)

        calib_params_raw = booster.attr("calibration_params")
        if calib_params_raw:
            try:
                calib = json.loads(calib_params_raw)
                params = calib.get("params", {}) if isinstance(calib, dict) else {}
                if "a" in params and "b" in params:
                    a = float(params["a"])
                    b = float(params["b"])
                    p_raw = 1.0 / (1.0 + np.exp(a * p_raw + b))
                    logger.info("应用了模型内置的 Sigmoid 校准")
                elif "a" in calib and "b" in calib:
                    a = float(calib["a"])
                    b = float(calib["b"])
                    p_raw = 1.0 / (1.0 + np.exp(a * p_raw + b))
                    logger.info("应用了模型内置的 Sigmoid 校准")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"解析内置校准参数失败: {str(e)}")

        return np.clip(p_raw, PROB_XGB_CLIP_MIN, PROB_XGB_CLIP_MAX)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"xgboost 或 FeatureEngineer 不可用: {str(e)}，跳过基础概率计算")
        return None
    except Exception as e:
        logger.error(f"计算基础概率时出错: {str(e)}", exc_info=True)
        return None


def main() -> None:
    try:
        if not DATA_PATH.exists():
            logger.error(f"数据文件不存在: {DATA_PATH}")
            sys.exit(1)

        logger.info(f"加载数据: {DATA_PATH}")
        df = pd.read_feather(DATA_PATH)
        logger.info(f"数据加载完成，样本数: {len(df)}")

        corpus = _build_corpus(df)
        if not corpus:
            logger.error("没有足够的文本语料用于训练 TF-IDF 向量器")
            sys.exit(1)
        logger.info(f"语料库大小: {len(corpus)}")

        vec = _fit_vectorizer(corpus)

        centroids = _compute_centroids(df, vec)
        if not centroids:
            logger.error("质心计算失败")
            sys.exit(1)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        vec_path = OUTPUT_DIR / VECTORIZER_FILENAME
        joblib.dump(vec, vec_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"已保存 TF-IDF 向量器: {vec_path}")

        centroids_path = OUTPUT_DIR / CENTROIDS_FILENAME
        np.savez_compressed(
            centroids_path, **{k: v.astype(np.float32) for k, v in centroids.items()}
        )
        logger.info(f"已保存 TF-IDF 质心: {centroids_path}")

        sims_df = _compute_similarities(df, vec, centroids)
        logger.info(f"相似度计算完成，形状: {sims_df.shape}")

        p_base = _compute_p_base_with_xgb(df)
        if p_base is not None:
            logger.info(f"基础概率计算完成，范围: [{p_base.min():.4f}, {p_base.max():.4f}]")

        weights = _fit_uplift_weights(df, sims_df, p_base=p_base)

        weights_path = OUTPUT_DIR / WEIGHTS_FILENAME
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存文本增益权重: {weights_path}")

        logger.info("训练完成")

    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
