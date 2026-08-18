"""模型与数据的权威事实常量 —— 三视角共享的单一真相源。

避免在 hero / 技术线 / AI prompt 等多处硬编码 ECE/AUC/样本量，
导致重训后台面数字自相矛盾。

来源优先级：
1. reports/ 评估 JSON（动态，重训后自动更新）
2. 模块内 fallback 常量（与 reports/README.md「关键数字速查」对齐，标注证据等级）
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

_PREDICTION_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_PREDICTION_DIR, "..", ".."))
_REPORTS_DIR = os.path.abspath(os.path.join(_SRC_DIR, "..", "reports"))

# ── Fallback 常量（来源：reports/README.md 关键数字速查）──
BETA_BINOMIAL_PRIOR_STRENGTH = (
    5.0  # Beta-Binomial 层次收缩先验强度（2026-06 替代离散级联的 n=5 硬阈值）
)
# Note: FALLBACK_N_THRESHOLD (5) 保留在 config.py 中作为旧级联的遗存配置，不再被 fallback.py 使用
N_SAMPLES = 81830  # 训练样本总数（2026-08 合并 2026 硕士案例后：61,716 + 20,114，清洗后实测）
N_COMBOS = (
    68942  # 唯一院校-专业组合数（4-key: bg_uni, bg_major, tgt_uni, tgt_major；strip 口径实测）
)
SPARSITY_PCT = 99.29  # 组合 ≤4 样本占比（实测）
# 4-key combo 稀疏度分箱（bg_uni, bg_major, tgt_uni, tgt_major；来源 V7 / cases.feather）
SPARSITY_BIN_LABELS = ["1", "2", "3 − 4", "5 − 10", "11 − 30", "30 +"]
SPARSITY_BIN_PCT = [88.22, 8.42, 2.65, 0.66, 0.05, 0.0]
SPARSITY_BIN_ONE_PCT = SPARSITY_BIN_PCT[0]
BASE_MODEL_ECE = (
    0.0263  # Base model ECE（sigmoid-era 全量测量，2026-06-12 切换为 isotonic 后未重测全量）
)
FULL_CHAIN_ECE = 0.1016  # 全链路 ECE（2026-08-10 新基线 in-sample 全量 / [int]）
# ── 2026-06-12 校准方法切换：sigmoid → isotonic ──
# 原因：(a) 校准集 12.3K ≫ 1000，isotonic 安全；(b) 旧注释错误声称"~数百样本"
# 500样本实验: sigmoid 全链路 ECE 0.0857 → isotonic 0.0792 (-7.6%), Brier 0.1467→0.1507
CALIBRATION_METHOD = "isotonic"  # 当前校准方法
CALIBRATION_SET_SIZE = (
    12501  # 校准集实际大小（time-split 训练集 62,505 × 20%；训练器 StratifiedShuffleSplit）
)
ISOTONIC_500_SAMPLE_ECE = 0.0792  # isotonic 全链路 ECE (500样本 stratified / [int])
SIGMOID_500_SAMPLE_ECE = 0.0857  # sigmoid 全链路 ECE (同 500 样本 baseline / [int])
ISOTONIC_ECE_DELTA_PCT = -7.6  # isotonic vs sigmoid ECE 改善幅度 (%)
DEFAULT_AUC = 0.7225  # XGBoost 新模型 AUC（2026-08-10 time-split held-out 评估，JSON 优先自动读取）
ECE_ACCEPTABLE = 0.10  # 行业可接受 ECE 上限（全链路 0.1016 超出此值）
ECE_WARN = ECE_ACCEPTABLE  # warn 线（= 可接受上限；held-out 0.1006 恰好越线）
ECE_FAIL = 0.15  # fail 线（内部口径：全链路 ECE ≥ 0.15 视为不通过；R-075 登记，此前仅存于注释）
DATA_DENSITY = N_SAMPLES / N_COMBOS  # ≈1.187 样本/组合
PENALTY_CEILING_PCT = 37.2  # 触及 70% 惩罚上限的 case 占比（V5 / [int]）
MISSING_GPA_ADMIT_DELTA = (
    +14.0
)  # 无 GPA vs 有 GPA 录取率差（pp, V7 / [int], 2026-08-11: 44.9%−31.0%）
EXTERNAL_ECE_AS = 0.4417  # ApplySquare 外部 ECE（V12b / [ext-AS], n=488 全 cache-hit, 2026-08-11）
FULL_CHAIN_ECE_EXCEEDS = True  # 全链路 ECE 超过可接受上限（0.1016 > 0.10）

# ── ApplySquare 外部验证（来源：V12b + V21 / [ext-AS]）──
APPLYSQUARE_FULL_N = 507  # 全量数据集行数
APPLYSQUARE_STRATIFICATION_N = 417  # V21 分层偏差分析子集（含 raw_prob 估计的 case, 2026-08-11）
APPLYSQUARE_ADMIT_RATE = 0.85  # 录取率（vs 训练集 34%，label shift 标志）
APPLYSQUARE_OVERALL_BIAS_PP = -44.2  # V12b n=488 全 cache-hit 整体偏差（pp），2026-08-11

# ── V21 分层校准偏差（来源：ApplySquare 子集 n=417 / [ext-AS], 2026-08-11）──
STRATIFICATION_C9_BIAS_PP = -20.0  # C9 院校概率偏差
STRATIFICATION_985_BIAS_PP = -29.4  # 985/211 院校概率偏差
STRATIFICATION_DOUBLE_NON_BIAS_PP = -11.6  # 双非院校概率偏差

# ── 跨专业惩罚参数（来源：专业相似度分布 + 顾问校准 / [int]）──
CROSS_MAJOR_SIGMOID_CENTER = 0.87  # sigmoid 函数中心（相似度=0.87 时 multiplier=0.75）
CROSS_MAJOR_TRIGGER_THRESHOLD = 0.89  # 惩罚触发阈值（相似度 < 此值触发惩罚，来源：elbow point）

# ── AI 盲评 benchmark（来源：reports/experiments/blind_eval_scatter.json / [int], 2026-08-11 export, n=37 有效）──
BLINDEVAL_N = 37  # 盲评样本量（absolute estimation, 40 requested / 37 valid）
BLINDEVAL_PEARSON_R = 0.3757  # Pearson 相关系数（2026-08-11 重跑）
BLINDEVAL_SPEARMAN_R = 0.4265  # Spearman 秩相关系数（同 export 重算）
BLINDEVAL_CI_95 = (0.059, 0.624)  # Fisher z CI on export r (n=37)
# ── Pairwise blind eval（来源：scripts/pairwise_blind_eval.py / [int], 2026-06）──
# 2026-08-12：以 reports/experiments/pairwise_blind_eval.json（2026-06-14）为准对齐；
# actual accuracy 无独立来源，取 accuracy_vs_model 同值。重跑后以新结果覆盖（ISSUES R-026）
PAIRWISE_EVAL_N_PAIRS = 150  # 成对比较总数
PAIRWISE_EVAL_N_STUDENTS = 96  # 不重复学生数
PAIRWISE_EVAL_ACCURACY = 0.5967  # LLM 成对比较与模型排序一致率（barely > 0.5）
PAIRWISE_EVAL_SPEARMAN_R = 0.2966  # BT 排名 vs 模型排名的 Spearman r (p=0.002)
PAIRWISE_EVAL_HIGH_CONF_PCT = 95  # LLM 自评 high confidence 比例（142/150）
PAIRWISE_EVAL_ACTUAL_ACC_PCT = 60  # LLM 成对比较实际准确率（%）= accuracy_vs_model 0.5967
# 结论: Pairwise comparison 并未比 absolute estimation 提升排序质量（Spearman 0.30 < 0.56）。
# LLM 仍然严重高估自己的判断准确性（95% high confidence vs 60% accuracy）。

# ── 惩罚触发率漂移（来源：V21 内部 vs 外部对比 / [int]+[ext-AS]）──
PENALTY_TRIGGER_TRAIN_PCT = 37  # 训练集惩罚触发率（%）
PENALTY_TRIGGER_AS_PCT = 72  # ApplySquare 惩罚触发率（%），受相似度匹配差异驱动

# ── V11 消融实验（来源：v11_ablation_attribution / [int]）──
V11_ABLATION_N = 200  # 分层抽样子集大小
V11_ABLATION_BASELINE_ECE = 0.0875  # 子集 baseline（8.75%，不等同于全量 0.1016，见 §5.5 说明）

# ── Layer 1/2 消融实验（来源：test_calibration_report.py 消融实验 / [int], 2026-06）──
L12_ABLATION_N = 150  # 分层抽样样本量
L12_ABLATION_BASELINE_ECE = 0.1064  # 全五层 baseline ECE
L12_ABLATION_NO_L12_ECE = 0.1102  # 移除 L1+L2 后 ECE（+0.0038 ≈ noise）
L12_ABLATION_BASELINE_BRIER = 0.1501  # baseline Brier
L12_ABLATION_NO_L12_BRIER = 0.1417  # 移除 L1+L2 后 Brier（−0.0084 ✓）
L12_ABLATION_BASELINE_BIAS = -0.0976  # baseline 系统性偏差
L12_ABLATION_NO_L12_BIAS = -0.0747  # 移除 L1+L2 后 Bias（+2.3pp ✓）
L12_ABLATION_ECE_DELTA = L12_ABLATION_NO_L12_ECE - L12_ABLATION_BASELINE_ECE  # +0.0038
L12_ABLATION_BRIER_DELTA = L12_ABLATION_NO_L12_BRIER - L12_ABLATION_BASELINE_BRIER  # −0.0084
L12_ABLATION_BIAS_DELTA_PP = (L12_ABLATION_NO_L12_BIAS - L12_ABLATION_BASELINE_BIAS) * 100  # +2.3pp
# In-sample 结论: 初步支持 double-counting（ΔECE≈noise）——但 held-out 已推翻，见 HELDOUT_L12_*。

# ── NaN vs Imputation 实验（来源：scripts/nan_vs_imputation_experiment.py / [int], 2026-06）──
NAN_VS_IMP_N_TEST = 12344  # 测试集大小
NAN_VS_IMP_BRIER_MEDIAN = 0.2732  # median imputation Brier
NAN_VS_IMP_BRIER_NAN = 0.2706  # NaN-native Brier
NAN_VS_IMP_ECE_MEDIAN = 0.1929  # median imputation ECE
NAN_VS_IMP_ECE_NAN = 0.1935  # NaN-native ECE
NAN_VS_IMP_AUC_MEDIAN = 0.643  # median imputation AUC
NAN_VS_IMP_AUC_NAN = 0.648  # NaN-native AUC
NAN_VS_IMP_BIAS_MEDIAN = -0.1912  # median imputation systematic bias
NAN_VS_IMP_BIAS_NAN = -0.1902  # NaN-native systematic bias
# 结论: NaN-native 在 Brier/AUC/AP/Bias 上小幅领先，ECE 持平（噪声级）。
# XGBoost 原生 NaN 处理至少不输 median imputation，且保留了缺失作为潜在信号。

# ── E4 NaN-native + Adjustment Chain Skip 联合 A/B（来源：reports/experiments/run_nan_skip_joint_ab.py / [int], 2026-06）──
NAN_SKIP_BEST_ARM = "D_nan_penalty_skip"  # 缺失 GPA 子群 ECE 最优（0.055 vs baseline 0.102）
NAN_SKIP_GLOBAL_BEST_ARM = "A_median_penalty_on"  # 全局 ECE 最优（0.0270）
NAN_SKIP_DOUBLE_BOOSTING = False  # NaN+skip 未导致缺失 GPA 子群过度高估
MISSING_GPA_SUBGROUP_ECE_DELTA = -0.0474  # D vs A 缺失 GPA 子群 ECE 差（负=改善）
NAN_SKIP_N_TEST = 12344
NAN_SKIP_MISSING_GPA_PCT = 7.8
NAN_SKIP_MISSING_LANG_PCT = 16.5

# ── Split Conformal 实验（来源：scripts/venn_abers_prototype.py / [int], 2026-06；
#     2026-08-12 起覆盖率/宽度对齐 conditional_conformal_coverage.json 2025 主切分，见下）──
CONFORMAL_N_CALIB = 9874  # 校准集大小
CONFORMAL_N_TEST = 12344  # 测试集大小
CONFORMAL_ALPHA = 0.10  # 目标错误率（90% 覆盖率）
CONFORMAL_COVERAGE = 0.808  # 2025 主切分 marginal coverage（conditional_conformal_coverage.json primary_split, 2026-08-11）
# ⚠ CONFORMAL_AVG_WIDTH / PINBALL / POINT_* 与 split_conformal_prototype.json 不符
# （整体 coverage 0.9998 / point ECE 0.0117 / AUC 0.6128），无使用处，待重跑统一或删除（ISSUES R-054）
CONFORMAL_AVG_WIDTH = 0.344  # 平均区间宽度 (p1 - p0) —— 旧口径，待重跑
CONFORMAL_PINBALL = 0.126  # Pinball loss —— 旧口径，待重跑
CONFORMAL_POINT_ECE = 0.151  # 旧口径，待重跑
CONFORMAL_POINT_AUC = 0.773  # 旧口径，待重跑
# 2025 主切分 tercile 平均宽度（conditional_conformal_coverage.json split_2 coverage_by_tercile）
CONFORMAL_WIDTH_LOW = 0.2941  # 低概率区 (safety) 平均宽度
CONFORMAL_WIDTH_MID = 0.9844  # 中概率区宽度
CONFORMAL_WIDTH_HIGH = 0.9789  # 高概率区 (reach) 平均宽度
# 结论: 2025 主切分下区间整体很宽（low 0.294 / mid 0.984 / high 0.979，概率尺度），
# 且该切分各子组全面欠覆盖（NOT_READY），与旧 prototype 的「窄区间」口径不一致，勿混用。

# ── Label Shift 分解实验（来源：scripts/label_shift_multical.py / [ext-AS]+[int], 2026-08-11）──
LABEL_SHIFT_AS_N = 1624  # ApplySquare 有标签样本数
LABEL_SHIFT_TRAIN_RATE = 0.277  # 训练集录取率
LABEL_SHIFT_AS_RATE = 0.854  # ApplySquare 录取率
LABEL_SHIFT_PRIOR_RATIO = 3.08  # p_test/p_train ≈ 3.08
LABEL_SHIFT_RAW_ECE = 0.340  # 原始 ECE (before prior correction)
LABEL_SHIFT_RAW_BIAS_PP = -34.0  # 原始偏差 (pp)
LABEL_SHIFT_CORRECTED_ECE = 0.162  # simple prior correction 后 ECE（↓52%）
LABEL_SHIFT_CORRECTED_BIAS_PP = -12.1  # simple prior correction 后偏差（残余仍为负，无符号翻转）
# 结论: Label shift (p_train=0.277 vs p_test=0.854) 是外部偏差的主导成分。
# 2026-08-11 重跑：simple prior correction 将 ECE 降低 52%（0.340→0.162），
# 偏差从 −34pp 收窄至 −12pp（残余仍为负，不再出现旧数据的 +14pp 符号翻转）；
# EM prior correction（20 iterations）在新数据上发散（ECE 0.769），不可用。

# ── Multicalibration 实验（来源：scripts/label_shift_multical.py / [int], 2026-08-11）──
MULTICAL_GLOBAL_ECE = 0.1012  # 全局 isotonic calibration ECE
MULTICAL_MULTICAL_ECE = 0.1016  # 按背景院校 tier 分组建模后 ECE (Δ=−0.0005, 无显著改善)
MULTICAL_C9_ECE_DELTA = -0.0075  # ΔECE = global − multical；正=改善；C9 反遭恶化 (0.1166→0.1241)
MULTICAL_985_ECE_DELTA = -0.0112  # 985 子组同样恶化
MULTICAL_211_ECE_DELTA = -0.0313  # 211 子组恶化最明显 (0.0980→0.1294)
MULTICAL_DOUBLE_NON_ECE_DELTA = +0.0005  # 双非子组唯一微幅改善 (0.1054→0.1050)
MULTICAL_N_CAL_C9 = 529  # C9 校准集样本数 (极少!)
MULTICAL_N_CAL_985 = 1821
MULTICAL_N_CAL_211 = 197  # 211 校准集样本数 (极少!)
MULTICAL_N_CAL_DOUBLE_NON = 10545  # 双非校准集占绝对多数
# 结论（2026-08-11 重跑）: Per-tier calibration 在顶 tier 子组反而轻微恶化
# (C9/985/211 的 ECE 均上升)，双非略好——整体 ECE 几乎不变 (0.1012→0.1016)。
# 子组间偏差差距基本持平 (4.8→5.1pp)。per-tier calibration 只修映射、不修底层偏差的结论不变。
# Multicalibration 是必要但不充分的条件——真正的解法需要让模型特征包含 tier 信息。

# ── V20 组合优化回测（来源：v20_sandbox_backtest / [int], 2026-08-11 重跑）──
V20_BACKTEST_N = 200  # 沙盒回测学生数
V20_NASH_AT_LEAST_1_PCT = 65.5  # Nash 至少录 1 所（%）
V20_TOPK_AT_LEAST_1_PCT = 65.5  # Top-K 至少录 1 所（%）
V20_NASH_DELTA_AT_LEAST_1_PP = 0.0  # Nash vs Top-K 差异（pp）
V20_NASH_PRESTIGE = 0.4986  # Nash 院校档次（归一化）
V20_TOPK_PRESTIGE = 0.4945  # Top-K 院校档次（归一化）
# V20 backtest CI: Wald CI for difference of proportions (n=200 each)
# SE_diff = sqrt(p1*(1-p1)/n + p2*(1-p2)/n) ≈ 0.0475 (p1=p2=0.655)
# 95% CI ≈ (-9.3, +9.3) — crosses zero, difference not statistically significant
V20_DELTA_CI_LOWER = -9.3  # 差异 95% CI 下界（pp）
V20_DELTA_CI_UPPER = 9.3  # 差异 95% CI 上界（pp）

# ── V20-E1 Oracle 上界分析（来源：v20_sandbox_backtest/oracle_upper_bound.json / [int], 2026-08-11）──
ORACLE_PRESTIGE = 0.5100  # Oracle 院校档次（事后最优天花板）
ORACLE_TOPK_GAP = 0.0155  # Oracle − TopK 档次差（L1 杠杆理论天花板）
ORACLE_NASH_GAP = 0.0041  # Nash − TopK 档次差
ORACLE_NASH_REALIZATION = 0.265  # Nash 兑现率 (Nash−TopK)/(Oracle−TopK)
ORACLE_PCT_CEILING_UNDER_10PP = 0.98  # 天花板 ≤10pp 的学生占比
ORACLE_N = 200  # 学生数
E1_TOPK_PRESTIGE = 0.4945  # E1 实验自身的 TopK 档次（raw booster），非 E2 的 0.4945
E1_NASH_PRESTIGE = 0.4986  # E1 实验自身的 Nash 档次（raw booster），非 E2 的 0.4991
# 注意：E1 Oracle 实验使用 raw booster 概率（非校准后），因此 TopK/Nash 与 E2 不同。
# §8.5 中展示的三个数字必须来自同一实验（E1），不可混用 E2 的校准后值。

# ── V20-E2 修正配对重测（来源：v20_sandbox_backtest/paired_retest.json / [int], 2026-08-11）──
E2_CALIBRATION_DELTA_MEAN = -0.0418  # 校准概率偏移均值（isotonic − raw booster）
E2_CALIBRATION_DELTA_STD = 0.0708  # 校准概率偏移标准差
E2_RANK_FLIP_RATE = 0.40  # 校内排序翻转率（校准后排名变化的学生占比）
E2_NASH_PRESTIGE = 0.4991  # 校准后 Nash 档次
E2_TOPK_PRESTIGE = 0.4945  # 校准后 TopK 档次
E2_DELTA_PRESTIGE = 0.0045  # Δ (Nash − TopK) 档次差
E2_DELTA_PRESTIGE_CI = (0.0, 0.0136)  # 95% bootstrap CI（下界触及 0）
E2_DELTA_P1 = 0.005  # Δ P(≥1录) pp
E2_DELTA_P1_CI = (0.0, 0.015)  # 95% bootstrap CI
E2_MCNEMAR_CHI2 = 1.00  # McNemar χ²
E2_MCNEMAR_P = 0.317  # McNemar p-value（不显著）
E2_DISCORDANT_RATE = 0.06  # 策略选出不同组合的学生占比
E2_DISCORDANT_N = 12  # discordant 学生数
E2_N = 200  # 学生数
# V20 原始回测的三个硬伤：
#  ① 用 get_booster().predict() 绕过了 isotonic 校准
#  ② 对池外学校用模型概率≥0.5 填补（循环论证）
#  ③ 非配对聚合（报的是组间差不是配对差）
# E2 修复后：Nash > TopK 在 95% CI 下成立，但仅 24.5% 学生策略不同

# ── V20-E3 校准鲁棒性（来源：v20_sandbox_backtest/calibration_survival.json / [int], 2026-08-11）──
E3_CLEAN_DELTA_PRESTIGE = 0.0045  # 无噪声基线 Δ prestige
E3_UNIFORM_SIGMA_020_DELTA = 0.0033  # uniform σ=0.20 时 Δ
E3_STRATIFIED_SIGMA_020_DELTA = 0.0057  # stratified σ=0.20 时 Δ
E3_RANKPRES_SIGMA_020_DELTA = 0.0016  # rank-preserving σ=0.20 时 Δ
E3_BREAKEVEN_NONE = False  # breakeven σ 存在：uniform 0.02 / stratified 0.07 / rank-preserving 0.04
E3_NASH_WIN_RATE_CLEAN = 0.005  # 无噪声时 Nash 严格 > TopK 的学生占比
E3_N_REPEATS = 10  # 每 σ 重复次数
E3_NOISE_RANGE = (0.0, 0.20)  # σ 扫描范围（步长 0.01）
# 结论（2026-08-11 重跑）：Nash 优势幅度收窄（干净 Δ 仅 +0.45pp），
# 且在很小的噪声下即失去显著性（breakeven σ 0.02–0.07）——不再支持「结构性优势」表述。
# 杠杆绝对水平被校准质量锁死（E2 的 40% 排序翻转 + CI 下界触及 0）。

# ── V20-E4 Copula 决策翻转（来源：v20_sandbox_backtest/copula_decision_flip.json / [int], 2026-08-11）──
E4_N = 200
E4_DISCORDANT_RATE = 0.035  # Copula vs 独立假设下的策略差异率
E4_DISCORDANT_RATE_CI = (0.010, 0.065)  # 95% bootstrap CI
E4_PR_DIFF_MEAN = +0.0058  # P(全拒) copula−independent 均值（copula 更保守）
E4_PR_DIFF_P50 = +0.0064  # P(全拒) 差值中位数（<1pp）
E4_VERDICT = "SAFE TO REMOVE"  # 3.5% discordant + <1pp 修正 → 判死刑
# 结论（2026-08-11）：Copula 将 P(全拒) 抬高 ~0.6pp（更保守），
# 但 96.5% 学生策略不变。<1pp 的 P(全拒) 修正不足以改变任何业务决策。
# 400+ 行 MC 代码（Sobol + Cholesky + t-copula + 分块 + LRU cache）
# 的维护成本远超其决策价值。建议删除。

# ── Held-out evaluation（时间冻结留存集，year ≥ 2024 / [time-heldout]）──
# 来源：data/held_out_test.feather + evaluation_results/*_time_split.json
# 当 USE_HELD_OUT_IF_AVAILABLE=True 且 held-out 文件存在时，训练和评估自动使用此切分。
# 旧常量 FULL_CHAIN_ECE / BASE_MODEL_ECE 为 in-sample baseline，保留用于对比。
HELD_OUT_SPLIT_YEAR = 2024  # 留存集切分年份
HELD_OUT_N_TRAIN = 62505  # 训练集大小（2026-08 基线：全量 81,830 − 冻结测试 19,325）
HELD_OUT_N_TEST = 19325  # 测试集大小（≥2024）
# 以下为动态注入的 held-out 指标——运行时从最新 time-split eval JSON 读取
# 若 eval_metrics() 的 split_method != "time"，则回退到 None（标注不可用）
_HELD_OUT_METRICS: dict | None = None


def _ensure_held_out_metrics() -> dict | None:
    """延迟加载 held-out 评估指标（仅在首次访问时读取 eval JSON）。"""
    global _HELD_OUT_METRICS
    if _HELD_OUT_METRICS is not None:
        return _HELD_OUT_METRICS if _HELD_OUT_METRICS else None

    meta = latest_eval_meta()
    if meta.get("split_method") == "time":
        _HELD_OUT_METRICS = eval_metrics()
    else:
        _HELD_OUT_METRICS = {}  # sentinel: tried but not available
    return _HELD_OUT_METRICS if _HELD_OUT_METRICS else None


def held_out_ece() -> float | None:
    """Full-chain ECE on held-out set (from test_calibration_report.py, 2026-06-12).

    n=495 stratified sample, 10-bin equal-width ECE.
    Held-out (year≥2024) full-chain ECE = 0.1006 — 恰好越界 0.10 可接受线
    （warn 级、未达 fail 0.15；见 HELDOUT_ECE_PASSES=False）。
    """
    return HELDOUT_FULL_CHAIN_ECE


def held_out_brier() -> float | None:
    """Base model Brier score on held-out set (from training eval JSON)."""
    m = _ensure_held_out_metrics()
    return m.get("brier_score") if m else None


def held_out_auc() -> float | None:
    """Base model ROC-AUC on held-out set (from training eval JSON)."""
    m = _ensure_held_out_metrics()
    return m.get("roc_auc") if m else None


# ── V20-E4b Fixed t-copula 重测（来源：v20_sandbox_backtest/run_copula_fixed.py / [int], 2026-08-11）──
# t-copula 实现修复 — 添加共享 χ² 维度，构造正确的多元 t 分布
# Sobol(k+1) → Z = norm.ppf(:k) @ L^T, W = chi2.ppf(k), T = Z / sqrt(W/ν)
E4B_N = 200
E4B_GAUSSIAN_DISCORDANT = 0.035  # Gaussian copula (t_df=0) discordant rate
E4B_GAUSSIAN_PR_DIFF_MEAN = +0.0031  # ΔP(rej) mean
E4B_FIXED_T_DISCORDANT = 0.035  # Fixed t-copula (ν=4) discordant rate
E4B_FIXED_T_PR_DIFF_MEAN = +0.0106  # ΔP(rej) mean (方向: + → copula更保守)
E4B_FIXED_T_PR_DIFF_P50 = +0.0072  # ΔP(rej) median
E4B_OLD_BUGGY_PR_DIFF_MEAN = -0.0163  # Old buggy ΔP(rej) (方向错: − → copula更乐观)
E4B_TAIL_DEPENDENCE_DELTA = +0.0075  # Fixed t − Gaussian ΔP(rej) (尾依赖增量)
E4B_VERDICT = "SAFE TO REMOVE (3.5%)"  # 结论稳健，bug 不影响裁决

# ── V20-E5 DRO 非对称鲁棒优化（来源：v20_sandbox_backtest/run_dro_asymmetric.py / [int], 2026-08-11）──
DRO_N = 200
DRO_ASYM_RANK_FLIP_RATE = 0.885  # 非对称 haircut 翻转校内排序的比例
DRO_ASYM_DISCORDANT = 0.07  # DRO-Wilson vs Nash portfolio discordant
DRO_GAMMA1_DISCORDANT = 0.07  # DRO-Γ=1 vs Nash
DRO_GAMMA2_DISCORDANT = 0.065  # DRO-Γ=2 vs Nash
DRO_HAIRCUT_MEAN = 0.0523  # 非对称 haircut 均值
DRO_HAIRCUT_P50 = 0.0289  # 中位数
DRO_HAIRCUT_P90 = 0.3272  # p90 (10% 学校被削 >33pp)
DRO_HAIRCUT_OVER_10PP = 0.352  # haircut >10pp 的比例
DRO_NASH_DISCORDANT = 0.07  # DRO vs Nash discordant (与 DRO_ASYM_DISCORDANT 相同)
# 核心发现：88.5% 排序翻转 but only 7.0% portfolio 变化
# Tier 结构充当天然鲁棒性缓冲——同 tier 内学校可互换
DRO_VERDICT = "Tier系统已提供DRO想要的鲁棒性，增量杠杆仅7.0%"

# ── V20-E6 子模性证明与验证（来源：v20_sandbox_backtest/submodularity_proof.py / [int], 2026-08-11）──
SUBMODULARITY_N_VERIFIED = 96  # 全枚举验证的学生数
SUBMODULARITY_F1_OPTIMAL = 1.0  # P(≥1) greedy=精确最优 (100%)
SUBMODULARITY_F2_OPTIMAL = 1.0  # E[best prestige] greedy=精确最优 (100%)
SUBMODULARITY_F2_MAX_GAP = 0.0  # 最大gap
SUBMODULARITY_GREEDY_RATIO = 1.0  # Empirical: greedy严格最优(>1−1/e≈0.632)
SUBMODULARITY_VERDICT = "Greedy精确最优(非近似)，Nash贪心内循环有理论保证"

# ── V21 分层偏差置信区间（来源：n=417, 偏差为预测-实际差值 / [ext-AS], 2026-08-11）──
# 偏差的 SE 取决于预测-实际差值的方差，非简单比例
# CI 未计算——当前 n=377 下偏差方向可信，精确幅度需更大样本
STRATIFICATION_CI_NOTE = "n=417，偏差方向可信但幅度置信区间未计算——需 individual-level 差值方差"

# ── 调整链 ECE 放大倍数（来源：FULL_CHAIN_ECE / BASE_MODEL_ECE）──
ECE_AMPLIFICATION = (
    FULL_CHAIN_ECE / BASE_MODEL_ECE
)  # ≈3.86（0.1016/0.0263），调整链将 ECE 放大的倍数

# ── 消融实验 fallback（来源：v11_ablation_attribution，[int]）──
ABLATION_BASELINE_ECE = 0.1016  # 与 FULL_CHAIN_ECE 对齐（2026-08-10 新基线）
ABLATION_BASELINE_BRIER = 0.1622
ABLATION_BASELINE_GAP = -0.1016

# ── 关键实验发现 fallback（来源：各报告 JSON，[int]）──
# V13 阈值无关性
V13_ECE_SENSITIVITY_RANGE = 0.0064
V13_BEST_ECE = 0.124
V13_BEST_THRESHOLD = 0.88
# ── V9 相似度质量审计（来源：reports/similarity_system/v9 / [int]）──
V9_SIMILARITY_SPAN = 0.167  # 全距 [0.805, 0.972] — E5 对任何真实专业对不输出 <0.80
V9_SIMILARITY_STD = 0.028  # 标准差 — 阈值 0.89 切在分布最密集区
V9_THRESHOLD_NOISE_ZONE = True  # 相邻 bin (0.88-0.89 vs 0.89-0.90) 录取率差 0.8pp, z≈1.0, p>0.3
# ── V12 反事实相似度实验（来源：reports/similarity_system/v12 / [int]）──
V12_LINEAR_STRETCH_ECE_DELTA = +0.025  # 线性拉伸恶化 ECE（更多 case 触发惩罚）
V12_ORACLE_ECE_DELTA = 0.0  # Oracle 拉伸也无改善 → 相似度改善上界 ≈ 0
# ── KNN 检索器参数（来源：src/adjustment/knn_retrieval.py）──
KNN_DEFAULT_WEIGHTS = (
    0.30,
    0.20,
    0.15,
    0.35,
)  # 四维加权：院校·GPA·语言·专业接近（bg-to-bg 名称相似度，2026-08-12 第二轮）
# ── Underdog 检索阈值 — 学生侧（宽松）vs 池子侧（严格）──
KNN_UNDERDOG_STUDENT_SCHOOL = 0.65  # 学生侧：≤ 此值即尝试召回学校逆袭（500+及以下）
KNN_UNDERDOG_STUDENT_GPA = 2.8  # 学生侧：GPA < 此值即尝试召回 GPA 逆袭
KNN_UNDERDOG_WEAK_SCHOOL_SCORE = 0.55  # 参考上界；2026-08-12 起实际条件为“案例院校分 < 学生院校分”（500+ 学生等效 ≤0.55，普通本科学生更严）
KNN_UNDERDOG_LOW_GPA_CUTOFF = 2.5  # 池子侧：GPA < 此值视为真正低绩点逆袭
KNN_UNDERDOG_MAX_TIER = 3  # 仅 T1–T3 目标展示逆袭案例（T4 保底校不展示，避免 badge 稀释）
KNN_UNDERDOG_GPA_CUSHION = 0.3  # 学校逆袭 GPA 协变量控制：案例 GPA ≤ 学生 GPA + 此值
# V16 OOD 检测（2026-08-11 重跑）
V16_INTERNAL_SPEARMAN_R = 0.2075
V16_EXTERNAL_SPEARMAN_R = -0.3201
V16_EXTERNAL_DISTANCE_RATIO = 63.26
V16_FAR_VS_CLOSE_RATIO = 2.21
# V7 缺失即信号（2026-08-11 重跑）
V7_NO_GPA_ADMIT_RATE = 0.449  # 无GPA学生录取率
V7_HAS_GPA_ADMIT_RATE = 0.31  # 有GPA学生录取率
V7_GPA_ADMIT_DELTA_PP = MISSING_GPA_ADMIT_DELTA  # +14.0pp（与 MISSING_GPA_ADMIT_DELTA 对齐）
V7_GPA_ADMIT_DELTA = MISSING_GPA_ADMIT_DELTA / 100  # 0.134，tech_view 兼容（×100 → pp）
V7_C9_GPA_ADMIT_DELTA_PP = 18.3  # C9 子群：无 GPA vs 有 GPA 录取率差（pp / V7, 2026-08-11）
V7_DN_GPA_ADMIT_DELTA_PP = 12.4  # 双非子群：无 GPA vs 有 GPA 录取率差（pp / V7, 2026-08-11）
V7_NO_MAJOR_ADMIT_RATE = 0.599  # 未填本科专业学生录取率（V7）
V7_LANG_ADMIT_DELTA = -0.04  # 无语言 vs 有语言（pp, V7 2026-08-11: −0.0396）
V7_WOULD_FAIL_PCT = 7.4  # 预测失败率（%, V7 2026-08-11: 6,043/81,830）

# ── Held-out 全链路验证（来源：test_calibration_report.py / [ext-time], 2026-08-10 新基线）──
# 时间冻结留存集：year ≥ 2024, n_test=19,325, n_train=62,505（81,830 − 19,325）
# 正例率 train=31.1%, test=34.1% — 测试集冻结未动，2026 全进训练
HELDOUT_N_TEST = 19325
HELDOUT_N_TRAIN = 62505
HELDOUT_SPLIT_YEAR = 2024
SPLIT_METHOD = "time"  # "time" = 时间冻结留存集; "random" = 随机分层切分
HELDOUT_ECE_N = 495  # 全链路 ECE 分层评估子样本（全量测试集 n=19,325）
HELDOUT_FULL_CHAIN_ECE = (
    0.1006  # 全链路 ECE (held-out, n=495 stratified / [ext-time], 2026-08-10 新基线)
)
HELDOUT_INSAMPLE_COMPARABLE_ECE = 0.1016  # 同方法 in-sample ECE (n≈460 / [int], 2026-08-10)
HELDOUT_ECE_DELTA_VS_INSAMPLE = (
    -0.0010
)  # Held-out ECE − in-sample ECE（基本持平，in-sample 反略高）
HELDOUT_AUC = 0.6621  # 全链 held-out AUC（n=495 分层子样本；来源：
#   held_out_base_vs_chain_auc_20260812.json，2026-08-12 配对重跑验证：全链 0.6621 / 同样本 base 0.6699）
HELDOUT_BASE_AUC_PAIRED = 0.6699  # 同 n=495 子样本上 base（惩罚层全关）AUC——配对口径
HELDOUT_AUC_DIFF_PAIRED_CI = (-0.023, 0.0097)  # chain − base，paired bootstrap 95% CI（含 0）
HELDOUT_BRIER = 0.2128  # Held-out Brier (2026-08-10 新基线)
HELDOUT_ECE_PASSES = False  # 0.1006 > 0.10 → 恰好超出可接受阈值（旧模型 0.0939 通过）
# ── Held-out Reliability Diagram 分箱数据（来源：debiased_ece_robustness.json
#    held_out.reliability_bins_10, [ext-time], 2026-08-12 重跑；n 合计 421 = n_valid；
#    bins 反算 ECE=0.0676 与 10-bin ECE 一致）──
HELDOUT_RELIABILITY_BINS = [
    {"mean_prob": 0.0615, "mean_actual": 0.1589, "n": 107},
    {"mean_prob": 0.1473, "mean_actual": 0.2222, "n": 99},
    {"mean_prob": 0.2523, "mean_actual": 0.2368, "n": 76},
    {"mean_prob": 0.3406, "mean_actual": 0.4328, "n": 67},
    {"mean_prob": 0.4471, "mean_actual": 0.4324, "n": 37},
    {"mean_prob": 0.5362, "mean_actual": 0.4286, "n": 7},
    {"mean_prob": 0.6420, "mean_actual": 0.6429, "n": 14},
    {"mean_prob": 0.7357, "mean_actual": 0.8571, "n": 7},
    {"mean_prob": 0.8492, "mean_actual": 0.8333, "n": 6},
    {"mean_prob": 1.0000, "mean_actual": 0.0000, "n": 1},
]
# ── Held-out 分层偏差（来源：test_calibration_report.py per-segment / [ext-time]）──
# 对照的 "in-sample" V21 分层偏差见 STRATIFICATION_* 常量（ApplySquare 子集 n=417,
# 2026-08-11: C9 −20.0 / 985 −29.4 / 211双非 −11.6pp）；held-out 缩小幅度依 tier 不同（25%–83%）
HELDOUT_STRAT_C9_BIAS_PP = -13.2  # C9 held-out 偏差（2026-08-10 新基线）
HELDOUT_STRAT_985_BIAS_PP = -4.9  # 985 held-out 偏差（2026-08-10 新基线）
HELDOUT_STRAT_DOUBLE_NON_BIAS_PP = -8.7  # 211/双非 held-out 偏差（2026-08-10 新基线）
HELDOUT_STRAT_C9_N = 23  # C9 held-out 子样本量 (小!)
HELDOUT_STRAT_985_N = 90
HELDOUT_STRAT_DOUBLE_NON_N = 382
# ── Held-out L1/L2 消融（来源：test_calibration_report.py / [ext-time], 2026-08-10 新基线）──
# ★ 结论历史：in-sample Δ=+0.0038≈noise → "double-counting 可删"
#             旧模型 held-out（2026-06-12）Δ=+0.0173 → "L1/L2 有效, 不应删除"
#             新模型 held-out（2026-08-10）Δ=−0.0236（两次复现）→ "L1/L2 为负收益, 建议评估移除"
HELDOUT_L12_BASELINE_ECE = 0.1242  # Full chain ECE (held-out stratified n=148, 2026-08-10)
HELDOUT_L12_ABLATED_ECE = 0.1006  # No-L1/L2 ECE (held-out stratified n=148, 2026-08-10)
# 注意：该值恰好等于 HELDOUT_FULL_CHAIN_ECE（n=495）的 0.1006，纯属巧合，两者样本不同（n=148 vs n=495）
HELDOUT_L12_BASELINE_BIAS = -0.1115  # Baseline systematic bias
HELDOUT_L12_ABLATED_BIAS = -0.0935  # No-L1/L2 bias (改善 1.8pp)
HELDOUT_L12_ECE_DELTA = HELDOUT_L12_ABLATED_ECE - HELDOUT_L12_BASELINE_ECE  # -0.0236
HELDOUT_L12_BIAS_DELTA_PP = (HELDOUT_L12_ABLATED_BIAS - HELDOUT_L12_BASELINE_BIAS) * 100  # +1.8pp
HELDOUT_L12_N = 148  # 分层抽样样本量
# bootstrap 95% CI（来源：held_out_v11_waterfall_n150_check.json，n=148，seed=42）——
# 两区间高度重叠（共同区间 [0.0831, 0.1839]），Δ=−0.0236 为点估计、未达 95% 显著。
HELDOUT_L12_BASELINE_ECE_CI = (0.0831, 0.2058)
HELDOUT_L12_ABLATED_ECE_CI = (0.0678, 0.1839)
HELDOUT_L12_ECE_DELTA_NOTE = (
    "「两次复现」实为同 seed=42 的确定性重跑（held_out_v11_waterfall_n150_check.json），"
    "非独立抽样证据；bootstrap 95% CI 高度重叠（baseline [0.083, 0.206] vs ablated "
    "[0.068, 0.184]），差异未达 95% 显著，需更大样本或配对检验确认（R-066/R-067）。"
)
HELDOUT_L12_VERDICT = (
    "L1/L2 在新模型（2026-08-10）held-out 上点估计为负收益（ΔECE=-0.0236，"
    "同 seed 确定性复现）— 与 2026-06 旧模型结论（+0.0173 有益）相反；"
    "bootstrap CI 重叠、未达 95% 显著，建议在更大样本/配对检验后再定是否移除"
)
# ── Selection boundary heatmap（来源：reports/experiments/run_selection_boundary.py / [int]+[ext-Compass], 2026-06）──
# E5: GPA×tier density ratio (internal / Compass) quantifies agency selection bias.
# 2026-08-11 已执行（reports/experiments/selection_boundary_heatmap.json）；下方为落盘值。
SELECTION_BOUNDARY_PCT_UNDER_COVERED = 12.0  # E5: 25 cells 中 ratio<0.5 占比
SELECTION_BOUNDARY_N_LOW_COVERAGE_CELLS = 3
SELECTION_BOUNDARY_HELDOUT_LOW_COVERAGE_ECE = 0.0914  # 低覆盖区 held-out ECE (n=764)

# ── E1 Debiased ECE robustness（reports/experiments/debiased_ece_robustness.json / [ext-time], 2026-08-11）──
HELDOUT_DEBIASED_ECE = 0.0673  # Kumar 2019 debiased ECE (n=421 valid / 495 requested)
HELDOUT_ADAPTIVE_ECE = 0.0642  # Nixon equal-mass adaptive-bin ECE
HELDOUT_ECE_10BIN_ROBUSTNESS = 0.0676  # standard 10-bin ECE (same sample)
HELDOUT_ECE_ROBUSTNESS_N_VALID = 421  # n_valid in debiased_ece_robustness.json (495 requested)
ECE_BIN_SENSITIVITY_RANGE = 0.0303  # max-min ECE across 10/15/20 bins (held-out)

# ── E2 Rolling time-split（reports/experiments/rolling_time_split.json / [ext-time], 2026-08-11）──
ROLLING_SPLIT_MEAN_ECE = 0.1358
ROLLING_SPLIT_STD_ECE = 0.0871
ROLLING_SPLIT_MIN_ECE = 0.0498  # train<=2023, test=2024
ROLLING_SPLIT_MAX_ECE = 0.2552  # train<=2024, test=2025
ROLLING_SPLIT_ECE_2023 = 0.1025  # test year 2023
ROLLING_SPLIT_ECE_2024 = 0.0498  # test year 2024
ROLLING_SPLIT_ECE_2025 = 0.2552  # test year 2025
ROLLING_SPLIT_VERDICT = (
    "UNSTABLE: mean full-chain ECE=13.6% >= 10% (std=8.7%); "
    "2025 cohort label shift (+20.7pp admit rate) 段 ECE 0.255"
)

# ── E3 Beta-Binomial m sensitivity（reports/experiments/beta_binomial_sensitivity.json / [int], 2026-06-13）──
# 2026-08-12：新增 beta_binomial_ece_sweep.json（held-out n=500 分层子样本, fallback 层次收缩）；
# EB MLE 与最优 m 以新 sweep 为准（原 10.57 / 20 无来源，已废弃）
BETA_BINOMIAL_M5_CONTINUITY_DELTA_PP = 0.0  # n=4 vs n=6 synthetic continuity
BETA_BINOMIAL_EB_MLE_M = (
    2.5  # empirical-Bayes type-II MLE（beta_binomial_ece_sweep.json, 2026-08-12）
)
BETA_BINOMIAL_BEST_ECE_M = 2  # best held-out fallback ECE at m=2 (0.0245)
BETA_BINOMIAL_BEST_ECE = 0.0245  # held-out fallback ECE at m=2（n=500, 10-bin）
BETA_BINOMIAL_M5_ECE = 0.0283  # held-out fallback ECE at m=5（当前值，同 sweep）
BETA_BINOMIAL_SWEEP_NOTE = (
    "m=2（0.0245）与 m=4（0.0246）/m=1（0.0262）/m=6（0.0268）差异为噪声级"
    "（无 CI、单 seed=42 单次 sweep）；「ECE 最优 m=2」为点估计，EB type-II MLE=2.5 "
    "作为参考更稳健，不建议据此单独决策（R-074）。"
)

# ── E7 Conditional conformal temporal（reports/experiments/conditional_conformal_coverage.json / [ext-time], 2026-08-11）──
CONFORMAL_TEMPORAL_COVERAGE_2023 = 0.906  # train<=2022, test=2023
CONFORMAL_TEMPORAL_COVERAGE_2024 = 0.866  # train<=2023, test=2024
CONFORMAL_TEMPORAL_COVERAGE_2025 = 0.808  # train<=2024, test=2025 (primary)
CONFORMAL_TEMPORAL_READINESS = "NOT_READY"  # 主切分 2025 段全面欠覆盖（80.8%）

# ── Held-out Base Model vs Full Chain（来源：test_calibration_report.py / [ext-time], 2026-08-10 新基线）──
# ★ 核心反转：in-sample 调整链放大 ECE 3.9×；held-out（n=495, 2026-08-12 重跑）上
#    调整链整体同样放大 ECE（全链路 0.1006 → 惩罚层全关 0.0786，约 1.3×）——链在
#    held-out 上为净负收益，与 L1/L2 反转（R-001）同向。
#    旧值（0.1452→0.1339, 0.9×，2026-08-10 记录）在当前代码下无法复现（n=148 实测
#    0.1242→0.0970），已在 R-065 废弃，统一改用 held_out_v11_waterfall_20260812.json。
HELDOUT_BASE_ECE = (
    0.0786  # Base model ECE（5 层惩罚全关，n=495 / held_out_v11_waterfall_20260812.json）
)
HELDOUT_FULL_CHAIN_ECE_ABLATION_SUBSET = (
    0.1006  # 全链路 ECE（同 n=495 子集，= HELDOUT_FULL_CHAIN_ECE）
)
HELDOUT_ECE_AMPLIFICATION = round(0.1006 / 0.0786, 1)  # ≈1.3×（held-out 调整链放大 ECE，净负收益）
ECE_AMPLIFICATION_COMPARABILITY_NOTE = (
    "in-sample 3.9×（0.1016/0.0263）的 Base ECE=0.0263 为 sigmoid-era [L] 未重测全量；"
    "held-out 1.3×（0.1006/0.0786）为新基线实测——两者 Base 口径不同，倍数不可直接比较，"
    "也不应读作「held-out 上放大变小」（R-071）。"
)
HELDOUT_BASE_PRED_MEAN = (
    0.2652  # Base model pred_mean（n=495, held_out_v11_waterfall_20260812.json）
)
HELDOUT_FULL_CHAIN_PRED_MEAN = 0.2407  # Full chain pred_mean（n=495，压低 2.5pp）

# ── V11 held-out 积累正向消融（waterfall，来源：held_out_v11_waterfall_20260812.json / [ext-time], 2026-08-12）──
# 冻结 held-out（year >= 2024）分层抽样 n=495；生产全链路 predict()，10-bin ECE。
# 步骤：All On → −GPA → −Lang → −CrossMajor → −Faculty → All Off（accumulated forward）。
V11_HELDOUT_WATERFALL_N = 495
V11_HELDOUT_WATERFALL_ECE = [0.1006, 0.0977, 0.0902, 0.0902, 0.0789, 0.0786]
# 各步骤 bootstrap 95% CI（与 ECE 同序；来源同上 JSON）——相邻步骤区间高度重叠，
# 「逐层移除改善/持平」为点估计方向，差异未达 95% 显著（R-066）。
V11_HELDOUT_WATERFALL_ECE_CI = [
    (0.0767, 0.1422),
    (0.0694, 0.1358),
    (0.0596, 0.1293),
    (0.0596, 0.1293),
    (0.0519, 0.1177),
    (0.0517, 0.1175),
]
V11_HELDOUT_WATERFALL_NOTE = (
    "n=495、seed=42 单次抽样；ECE 差异为点估计。n=148 与 n=495 两组同 seed 分层抽样，"
    "样本是否嵌套未验证（R-068）；三个 0.1006 的「纯属巧合」判定同样待多 seed 验证。"
)

# ── Portfolio held-out replication E1–E3（来源：reports/experiments/portfolio_heldout_replication.json / [ext-time], 2026-08-11 重跑）──
# 全链路校准概率（json_api.predict），year >= 2024，n=199 pseudo-students（目标 200，1 人 predict 失败）
PORTFOLIO_HELDOUT_N = 199
PORTFOLIO_HELDOUT_N_APPS = 1171
PORTFOLIO_HELDOUT_PROB_SOURCE = "full_chain_calibrated"
PORTFOLIO_HELDOUT_NASH_PRESTIGE = 0.4582
PORTFOLIO_HELDOUT_TOPK_PRESTIGE = 0.4532
PORTFOLIO_HELDOUT_ADVISOR_PRESTIGE = 0.4381
PORTFOLIO_HELDOUT_DELTA_PRESTIGE = 0.0050  # Nash − TopK
PORTFOLIO_HELDOUT_DELTA_PRESTIGE_CI = (0.0, 0.0146)  # 95% bootstrap CI（下界触及 0）
PORTFOLIO_HELDOUT_NASH_MINUS_ADVISOR = 0.0201  # Nash − Advisor (Experiment A)
PORTFOLIO_HELDOUT_NASH_MINUS_ADVISOR_CI = (0.0050, 0.0388)  # 95% bootstrap CI — excludes 0
PORTFOLIO_HELDOUT_NASH_P1 = 0.558  # P(≥1录) Nash
PORTFOLIO_HELDOUT_TOPK_P1 = 0.553  # P(≥1录) TopK
PORTFOLIO_HELDOUT_ADVISOR_P1 = 0.538  # P(≥1录) AdvisorActual
PORTFOLIO_HELDOUT_DELTA_P1 = 0.005  # Nash − TopK（held-out 上差异近 0）
PORTFOLIO_HELDOUT_ORACLE_PRESTIGE = 0.4628
PORTFOLIO_HELDOUT_ORACLE_MINUS_TOPK = 0.0096  # Oracle 天花板仅 1.0pp（vs in-sample 4.8pp）
PORTFOLIO_HELDOUT_NASH_REALIZATION = 0.524
PORTFOLIO_HELDOUT_E3_CLEAN_GAIN = 0.005  # E3 无噪声 Nash−TopK prestige
PORTFOLIO_HELDOUT_DISCORDANT_RATE = 0.070  # E2 Nash vs TopK 组合差异率
PORTFOLIO_HELDOUT_MCNEMAR_P = 0.317  # E2 paired McNemar p-value
# vs in-sample deltas:
PORTFOLIO_HELDOUT_VS_INSAMPLE_NASH_PRESTIGE_DELTA = +0.0855
PORTFOLIO_HELDOUT_VS_INSAMPLE_ADVISOR_PRESTIGE_DELTA = +0.1031
PORTFOLIO_HELDOUT_VS_INSAMPLE_NASH_P1_DELTA = -0.1072
PORTFOLIO_HELDOUT_VS_INSAMPLE_DELTA_PRESTIGE_DELTA = -0.0454
PORTFOLIO_HELDOUT_SKIPPED_E4_E6 = True  # E4–E6 skipped per plan（n 足够 E1–E3）

# ── Experiment B: prediction-accuracy ablation（来源：reports/experiments/probability_ablation.json, 2026-06-24）──
# In-sample n=200, production feature encoding + production prestige. 3 probability sources.
EXP_B_UNIFORM_DELTA = 0.0409  # Nash−TopK Δprestige, uniform probability (objective function alone)
EXP_B_RAW_XGB_DELTA = 0.0223  # Nash−TopK Δprestige, raw XGBoost (production-like)
EXP_B_EMPIRICAL_DELTA = 0.0223  # Nash−TopK Δprestige, empirical school rate
EXP_B_N = 200
EXP_B_INTERPRETATION = (
    "Objective function contributes +4.1pp; prediction quality reduces gap to +2.2pp "
    "as TopK improves more from signal than Nash does."
)

# ── Experiment D: monetized economics（来源：reports/experiments/monetized_economics.json, 2026-06-24）──
# Corrected profit model: price − refund_ratio×price×P(all_reject) + E[LTV] − add_school_cost
EXP_D_NASH_PROFIT_YUAN = 15304  # ¥/student expected profit
EXP_D_TOPK_PROFIT_YUAN = 15427
EXP_D_DELTA_PROFIT_YUAN = -123  # Nash−TopK (Nash slightly less profitable)
EXP_D_NASH_P_REJECT = 0.245
EXP_D_TOPK_P_REJECT = 0.237
EXP_D_PRICE_CNY = 20000
EXP_D_REFUND_RATIO = 0.70


@lru_cache(maxsize=1)
def eval_metrics() -> dict:
    """加载 XGBoost 评估指标（AUC/Brier 等）。

    搜索顺序：
    1. 最新的 evaluation_results/*.json（split_method="time"，held-out 评估优先）
    2. 最新的 evaluation_results/*.json（任 split_method）
    3. 空 dict（回退到模块常量）
    """
    import glob as _glob

    result_dir = os.path.join(_SRC_DIR, "ml", "evaluation_results")
    json_files = sorted(
        _glob.glob(os.path.join(result_dir, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )

    def _load_json(path: str) -> dict | None:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # Prefer held-out (time-based) evaluation
    for jf in json_files:
        data = _load_json(jf)
        if data and data.get("split_method") == "time":
            return data.get("metrics", {})

    # Fall back to any evaluation
    for jf in json_files:
        data = _load_json(jf)
        if data:
            return data.get("metrics", {})

    return {}


def latest_eval_meta() -> dict:
    """返回最新评估 JSON 的元数据（split_method, audit hashes 等）。"""
    import glob as _glob

    result_dir = os.path.join(_SRC_DIR, "ml", "evaluation_results")
    json_files = sorted(
        _glob.glob(os.path.join(result_dir, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not json_files:
        return {}

    try:
        with open(json_files[0], encoding="utf-8") as f:
            data = json.load(f)
        return {
            "split_method": data.get("split_method", "unknown"),
            "timestamp": data.get("timestamp", ""),
            "data_hash": data.get("audit", {}).get("data_hash", "") if data.get("audit") else "",
            "model_hash": data.get("audit", {}).get("model_hash", "") if data.get("audit") else "",
        }
    except Exception:
        return {}


def auc() -> float:
    """当前模型 AUC：优先读评估 JSON，缺失时回退权威常量。"""
    return float(eval_metrics().get("roc_auc", DEFAULT_AUC) or DEFAULT_AUC)
