"""分类与聚类 (Classification & Clustering) — 核心实现

复刻 SPSS Classify 菜单:
- K-Means 聚类 (WCSS, 中心)
- 层次聚类 (Ward/Complete/Average/Single linkage)
- 轮廓系数 (Silhouette Score)
- 简单决策树 (Gini 不纯度, 最大深度)

══════════════════════════════════════════════════════════════════════
统计概念速查
══════════════════════════════════════════════════════════════════════

【K-Means — 聚类的工作马】

算法: 1. 初始化 k 个中心 → 2. 分配每个点到最近中心 → 3. 更新中心
      → 4. 重复 2-3 直到收敛。

WCSS (Within-Cluster Sum of Squares) / Inertia:
  每个点到其所属中心的距离平方和。WCSS 越小, 聚类越紧凑。
  随着 k 增大, WCSS 必然减小 (每个簇更小更紧), 但不能直接最小化 WCSS。

Elbow 法: 画 WCSS ~ k 的曲线, 找"弯曲点" (类似人胳膊肘)。
  但实际中很难有清晰的 elbow。

K-Means++ 初始化: 选择相互距离较远的初始中心。
  第一个随机选 → 后续点被选的概率与其到最近已选中心的距离平方成正比。
  这比纯随机初始化收敛更快, 且更容易找到更好的局部最优。

K-Means 的局限: 假设簇是球形的 (各方向方差相同) 且大小相似。
  如果数据有长条形簇或大小差异大的簇, 效果差。

【轮廓系数 (Silhouette Score) — 聚类好不好】

s(i) = (b(i) - a(i)) / max(a(i), b(i))
a(i): 点 i 到同簇其他点的平均距离 (凝聚度)
b(i): 点 i 到最近的其他簇所有点的平均距离 (分离度)

s(i) ≈ 1: 点很好地属于当前簇
s(i) ≈ 0: 点接近两个簇的边界
s(i) < 0: 点可能分配给了错误的簇

平均轮廓系数: 所有点的 s(i) 的均值。>0.5 可接受, >0.7 良好。

【层次聚类 — 4 种 linkage 方法的含义】

凝聚法 (Agglomerative): 从每个点是一个簇开始, 逐步合并最接近的两个簇。
区别在于"簇间距离"的定义:

Ward (推荐): 合并使得 WCSS 增量最小的两个簇。
  倾向于产生大小均衡的簇。最常用于社会科学。

Complete linkage: 两簇中最远的点对的距离。
  倾向于产生紧凑的簇 (球形的)。
Single linkage: 两簇中最近的点对的距离。
  可以处理长条形/非球形的簇, 但对噪音和离群值极其敏感。
Average linkage: 两簇中所有点对的距离的平均值。
  介于 complete 和 single 之间。

【决策树 (CART) — Gini 不纯度】

Gini = 1 - Σ(p_j²), p_j = 类别 j 在节点中的比例。
  如果节点纯 (只有一个类) → Gini = 0 (最好)
  如果节点完全混杂 (各类别均匀) → Gini 接近 1 (最差)

CART 在每个节点选择使 Gini 降低最多的特征和切分点。
Gini gain = Gini(parent) - [n_left/n × Gini_left + n_right/n × Gini_right]

特征重要性: 该特征在所有节点上的 Gini gain 总和 (按节点样本量加权)。

这是一个简单的教学型实现 (深度受限, 最小分裂样本固定)。
生产环境请使用 sklearn 的 DecisionTreeClassifier (有 pruning 和更多超参)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════
# K-Means
# ═══════════════════════════════════════════


@dataclass
class KMeansResult:
    """K-Means 聚类结果"""

    k: int
    labels: np.ndarray
    centers: np.ndarray
    inertia: float  # WCSS
    silhouette_scores: np.ndarray
    avg_silhouette: float
    n_iter: int


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    n_init: int = 10,
    random_seed: int | None = 42,
    standardize: bool = True,
) -> KMeansResult:
    """K-Means 聚类 (Lloyd 算法 + 多次初始化)。

    Args:
        X: (n, p) 数据矩阵。
        k: 聚类数。
        max_iter: 最大迭代次数。
        n_init: 随机初始化次数 (选最优)。
        random_seed: 随机种子。
        standardize: 是否 Z-score 标准化。

    Returns:
        KMeansResult。
    """
    arr = np.asarray(X, dtype=np.float64)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n, p = arr.shape

    if standardize:
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0, ddof=1) + 1e-10)

    rng = np.random.default_rng(random_seed)

    best_inertia = float("inf")
    best_labels = np.zeros(n, dtype=int)
    best_centers = np.zeros((k, p))

    for _ in range(n_init):
        # 随机初始化中心 (k-means++)
        centers = _kmeans_plus_plus(arr, k, rng)

        for iteration in range(max_iter):
            # 分配
            dists = np.zeros((n, k))
            for j in range(k):
                dists[:, j] = np.sum((arr - centers[j]) ** 2, axis=1)
            labels = np.argmin(dists, axis=1)

            # 更新中心
            new_centers = np.zeros((k, p))
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centers[j] = arr[mask].mean(axis=0)
                else:
                    new_centers[j] = arr[rng.choice(n)]

            if np.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers

        inertia = np.sum((arr - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    # 轮廓系数
    sil_scores = silhouette_samples(arr, best_labels) if k > 1 else np.zeros(n)
    avg_sil = float(np.mean(sil_scores))

    return KMeansResult(
        k=k,
        labels=best_labels,
        centers=best_centers,
        inertia=float(best_inertia),
        silhouette_scores=sil_scores,
        avg_silhouette=avg_sil,
        n_iter=max_iter,
    )


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means++ 初始化。"""
    n = len(X)
    centers = np.zeros((k, X.shape[1]))
    centers[0] = X[rng.choice(n)]

    for j in range(1, k):
        dists = np.min([np.sum((X - centers[c]) ** 2, axis=1) for c in range(j)], axis=0)
        probs = dists / dists.sum()
        centers[j] = X[rng.choice(n, p=probs)]

    return centers


# ═══════════════════════════════════════════
# 轮廓系数
# ═══════════════════════════════════════════


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """每个样本的轮廓系数 s(i)。

    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i): 样本 i 到同类其他样本的平均距离
    b(i): 样本 i 到最近异类的平均距离
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.zeros(n)

    scores = np.zeros(n)
    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = [l for l in unique_labels if l != labels[i]]

        # a(i)
        if same_cluster.sum() <= 1:
            scores[i] = 0.0
            continue
        a_i = np.mean(np.sqrt(np.sum((X[i] - X[same_cluster]) ** 2, axis=1)))

        # b(i) = min_j mean dist to cluster j
        b_i = float("inf")
        for oc in other_clusters:
            oc_mask = labels == oc
            dist_oc = np.mean(np.sqrt(np.sum((X[i] - X[oc_mask]) ** 2, axis=1)))
            b_i = min(b_i, dist_oc)

        scores[i] = (b_i - a_i) / max(a_i, b_i)

    return scores


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """平均轮廓系数。"""
    return float(np.mean(silhouette_samples(X, labels)))


# ═══════════════════════════════════════════
# 层次聚类
# ═══════════════════════════════════════════


@dataclass
class HierarchicalResult:
    """层次聚类结果"""

    linkage: np.ndarray  # (n-1, 4): [cluster_a, cluster_b, distance, n_members]
    n: int
    method: str


def hierarchical_clustering(
    X: np.ndarray,
    method: str = "ward",
    standardize: bool = True,
) -> HierarchicalResult:
    """层次聚类 (凝聚法)。

    Args:
        X: (n, p) 数据矩阵 (n < 5000 推荐)。
        method: ``"ward"`` (默认, 最小化方差增量) /
                ``"complete"`` (最大距离) / ``"average"`` / ``"single"`` (最近)。
        standardize: 是否 Z-score 标准化。

    Returns:
        HierarchicalResult。
    """
    arr = np.asarray(X, dtype=np.float64)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n = arr.shape[0]

    if standardize:
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0, ddof=1) + 1e-10)

    # 距离矩阵 (欧氏)
    from scipy.spatial.distance import pdist, squareform

    dist_mat = squareform(pdist(arr, metric="euclidean"))

    # 初始化: 每个点一个类
    n_clusters = n
    cluster_sizes = np.ones(n, dtype=int)
    # 聚类内平方和 (对 Ward)
    if method == "ward":
        cluster_dist = np.zeros(n)

    linkage = np.zeros((n - 1, 4))

    for step in range(n - 1):
        # 找最近的两个类
        min_dist = float("inf")
        min_i, min_j = -1, -1

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if method == "ward":
                    # Ward: Δ = (n_i * n_j) / (n_i + n_j) * ||c_i - c_j||²
                    d_ij = (cluster_sizes[i] * cluster_sizes[j]) / (cluster_sizes[i] + cluster_sizes[j]) * dist_mat[i, j]
                else:
                    d_ij = dist_mat[i, j]

                if d_ij < min_dist:
                    min_dist = d_ij
                    min_i, min_j = i, j

        # 合并类 i 和 j
        new_label = n + step  # 新类标签
        n_merged = cluster_sizes[min_i] + cluster_sizes[min_j]
        linkage[step] = [float(min_i), float(min_j), math.sqrt(min_dist) if method == "ward" else min_dist, float(n_merged)]

        # 更新距离矩阵
        for k in range(n_clusters):
            if k != min_i and k != min_j:
                ni, nj, nk = cluster_sizes[min_i], cluster_sizes[min_j], cluster_sizes[k]
                if method == "ward":
                    d_new = ((ni + nk) * dist_mat[min_i, k] + (nj + nk) * dist_mat[min_j, k] - nk * dist_mat[min_i, min_j]) / (ni + nj + nk)
                elif method == "single":
                    d_new = min(dist_mat[min_i, k], dist_mat[min_j, k])
                elif method == "complete":
                    d_new = max(dist_mat[min_i, k], dist_mat[min_j, k])
                else:  # average
                    d_new = (ni * dist_mat[min_i, k] + nj * dist_mat[min_j, k]) / (ni + nj)
                dist_mat[min_i, k] = d_new
                dist_mat[k, min_i] = d_new

        cluster_sizes[min_i] = n_merged
        # 移除 min_j 列/行
        dist_mat = np.delete(dist_mat, min_j, axis=0)
        dist_mat = np.delete(dist_mat, min_j, axis=1)
        cluster_sizes = np.delete(cluster_sizes, min_j)
        n_clusters -= 1

    return HierarchicalResult(linkage=linkage, n=n, method=method)


# ═══════════════════════════════════════════
# 简单决策树 (CART)
# ═══════════════════════════════════════════


@dataclass
class TreeNode:
    """决策树节点"""

    feature: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    value: int | None = None  # 叶节点类别
    gini: float = 0.0
    n_samples: int = 0


@dataclass
class DecisionTreeResult:
    """决策树结果"""

    tree: TreeNode
    depth: int
    n_nodes: int
    feature_importances: list[float]


def decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int = 5,
    min_samples_split: int = 5,
    random_seed: int | None = 42,
) -> DecisionTreeResult:
    """决策树分类器 (CART, Gini 不纯度)。

    Args:
        X: (n, p) 特征矩阵。
        y: 类别标签。
        max_depth: 最大深度。
        min_samples_split: 最小分裂样本数。

    Returns:
        DecisionTreeResult。
    """
    arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)

    mask = ~np.isnan(arr).any(axis=1)
    arr, labels = arr[mask], labels[mask]

    classes = sorted(set(labels))
    class_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([class_map[label] for label in labels])

    rng = np.random.default_rng(random_seed)

    feature_importances = np.zeros(arr.shape[1])
    n_nodes = [0]

    def _build_tree(X_node, y_node, depth):
        n_samples = len(y_node)
        n_nodes[0] += 1

        # 停止条件
        unique_classes = np.unique(y_node)
        if len(unique_classes) == 1 or depth >= max_depth or n_samples < min_samples_split:
            value = int(np.bincount(y_node, minlength=len(classes)).argmax() if len(y_node) > 0 else 0)
            gini = _gini(y_node, len(classes))
            return TreeNode(value=value, gini=gini, n_samples=n_samples)

        # 找最佳分裂
        best_gini = float("inf")
        best_feat = -1
        best_thresh = None
        best_left_idx = None
        best_right_idx = None

        for f in range(arr.shape[1]):
            thresholds = np.unique(X_node[:, f])
            if len(thresholds) < 2:
                continue
            # 抽样候选切分点
            if len(thresholds) > 20:
                thresholds = np.percentile(thresholds, np.linspace(5, 95, 15))

            for thr in thresholds:
                left = X_node[:, f] <= thr
                right = ~left
                y_left, y_right = y_node[left], y_node[right]

                if len(y_left) < min_samples_split or len(y_right) < min_samples_split:
                    continue

                gini_left = _gini(y_left, len(classes))
                gini_right = _gini(y_right, len(classes))
                gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / n_samples

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feat = f
                    best_thresh = thr
                    best_left_idx = left
                    best_right_idx = right

        if best_feat == -1:
            value = int(np.bincount(y_node, minlength=len(classes)).argmax() if len(y_node) > 0 else 0)
            return TreeNode(value=value, gini=_gini(y_node, len(classes)), n_samples=n_samples)

        parent_gini = _gini(y_node, len(classes))
        importance_gain = n_samples * (parent_gini - best_gini)
        feature_importances[best_feat] += importance_gain

        left_tree = _build_tree(X_node[best_left_idx], y_node[best_left_idx], depth + 1)
        right_tree = _build_tree(X_node[best_right_idx], y_node[best_right_idx], depth + 1)

        return TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_tree,
            right=right_tree,
            gini=parent_gini,
            n_samples=n_samples,
        )

    tree = _build_tree(arr, y_int, 0)

    # 归一化特征重要性
    total = feature_importances.sum()
    if total > 0:
        feature_importances /= total

    return DecisionTreeResult(
        tree=tree,
        depth=max_depth,
        n_nodes=n_nodes[0],
        feature_importances=[float(v) for v in feature_importances],
    )


def _gini(y: np.ndarray, n_classes: int) -> float:
    """Gini 不纯度。"""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)


def predict_tree(node: TreeNode, X: np.ndarray) -> np.ndarray:
    """决策树预测。"""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    preds = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        n = node
        while n.feature is not None:
            if X[i, n.feature] <= n.threshold:
                n = n.left
            else:
                n = n.right
        preds[i] = n.value
    return preds
