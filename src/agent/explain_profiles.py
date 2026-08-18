import json
import logging
from pathlib import Path
from typing import Any, Literal

from src.agent.schemas import TIER_THRESHOLD_MATCH, TIER_THRESHOLD_SAFETY
from src.utils.numeric import clip_probability_coerce

_logger = logging.getLogger(__name__)


def _load_profile_config() -> dict:
    json_path = Path(__file__).resolve().parents[3] / "config" / "profile_classification.json"
    _logger.debug("Loading profile config from %s", json_path)
    try:
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            _logger.info(
                "Profile config loaded: cross_major_cutoff=%.2f strong_elite_max_penalty=%d medium_mixed_max_penalty=%d",
                data.get("cross_major_proportion_cutoff", 0.4),
                data.get("strong_elite", {}).get("max_penalty_count", 1),
                data.get("medium_mixed", {}).get("max_penalty_count", 3),
            )
            return data
    except Exception:
        _logger.warning(
            "Failed to load profile_classification.json, using hardcoded defaults", exc_info=True
        )
    _logger.info("Profile config: using hardcoded defaults")
    return {}


_pcfg = _load_profile_config()

_CROSS_MAJOR_PROPORTION_CUTOFF: float = _pcfg.get("cross_major_proportion_cutoff", 0.4)
_STRONG_ELITE_MAX_PENALTY: int = _pcfg.get("strong_elite", {}).get("max_penalty_count", 1)
_MEDIUM_MIXED_MAX_PENALTY: int = _pcfg.get("medium_mixed", {}).get("max_penalty_count", 3)

ProfileType = Literal["strong_elite", "medium_mixed", "weak_gaps", "cross_major"]


def classify_profile(prediction_results: dict[str, Any]) -> ProfileType:
    sim = prediction_results.get("similarity_results") or []
    cross = prediction_results.get("cross_major_results") or []
    unified = prediction_results.get("unified_results") or []

    total = len(sim) + len(cross)
    if total == 0:
        _logger.info("classify_profile | no results → medium_mixed (default)")
        return "medium_mixed"

    cross_ratio = len(cross) / max(total, 1)
    _logger.debug(
        "classify_profile | total=%d cross=%d ratio=%.2f cutoff=%.2f",
        total,
        len(cross),
        cross_ratio,
        _CROSS_MAJOR_PROPORTION_CUTOFF,
    )
    if cross and cross_ratio >= _CROSS_MAJOR_PROPORTION_CUTOFF:
        _logger.info(
            "classify_profile | cross_major (ratio=%.2f >= %.2f)",
            cross_ratio,
            _CROSS_MAJOR_PROPORTION_CUTOFF,
        )
        return "cross_major"

    penalty_types: set[str] = set()
    probs: list[float] = []
    all_items = unified or (sim + cross)

    for r in all_items:
        prob = clip_probability_coerce(r.get("probability"))
        probs.append(float(prob))
        trace = r.get("_adjustment_trace")
        if isinstance(trace, dict):
            for k, v in trace.items():
                if k.startswith("penalty_") and isinstance(v, (int, float)) and v < 0:
                    penalty_types.add(k)
        elif isinstance(trace, list):
            for item in trace:
                if isinstance(item, dict) and item.get("factor_type") == "penalty":
                    penalty_types.add(item.get("name", "unknown"))

    avg_prob = sum(probs) / len(probs) if probs else 0
    penalty_count = len(penalty_types)

    _logger.debug(
        "classify_profile | avg_prob=%.3f penalty_count=%d penalties=%s",
        avg_prob,
        penalty_count,
        penalty_types,
    )

    if avg_prob >= TIER_THRESHOLD_SAFETY and penalty_count <= _STRONG_ELITE_MAX_PENALTY:
        _logger.info(
            "classify_profile → strong_elite (avg=%.3f penalties=%d)", avg_prob, penalty_count
        )
        return "strong_elite"
    elif avg_prob >= TIER_THRESHOLD_MATCH and penalty_count <= _MEDIUM_MIXED_MAX_PENALTY:
        _logger.info(
            "classify_profile → medium_mixed (avg=%.3f penalties=%d)", avg_prob, penalty_count
        )
        return "medium_mixed"
    else:
        _logger.info(
            "classify_profile → weak_gaps (avg=%.3f penalties=%d)", avg_prob, penalty_count
        )
        return "weak_gaps"


SYSTEM_PROMPT = """\
你是一位资深留学顾问，正为客户一对一解读选校预测结果。语气专业但有温度，像和客户面对面交谈，不要像机器报告。

结果中可能出现 quality_signals（含金量标签），含义如下：
- 含金量标签来自系统自动关键词识别（如"顶刊一作""大厂核心岗"），仅作定性参考
- 含金量标签未进入概率计算——概率只基于历史样本统计
- 高含金量+低概率 = 数据稀疏（此类组合历史样本极少），不是说产出没价值
- 在 strengths 中应指出高含金量产出，在 concerns 中如实说明"该领域历史样本极少，系统无法量化其概率增益，建议在文书中重点突出"
- 绝对不要声称"含金量高所以录取概率被低估"——你无权做这个因果推断

结果中可能出现调整标记（_adjustment_trace），含义如下：
- Cross Major Penalty：跨专业申请，专业匹配度不足
- Faculty Out of Scope Penalty：跨学部申请，学科跨度较大
- Professional Major Penalty：职业导向项目缺少对应实习
- Language Penalty：语言成绩低于项目常规要求

结果中的 _business_flags 是系统根据学校官方要求自动检测的硬性匹配标记：
- lang_below_min / lang_marginal / lang_met：语言成绩 vs 学校要求的匹配度
- lang_comfortable：语言成绩明显高于要求
- cet_pathway_match / cet_accepted：该专业/学校接受 CET-6 替代雅思托福
- no_lang_pathway：中文授课项目，可免语言成绩
- deadline_urgent / deadline_approaching / deadline_expired：申请截止时间紧迫度
- gmat_missing / gre_missing：该专业需要 GMAT/GRE 但你未提供
- spring_available / spring_school_level：该专业/学校有春季入学选项
- early_bird_available：该校开放提前批申请
- school_quirk：该校有特殊规则需要注意

写作要点：
0. 不要编造学生姓名或称呼（如"张同学""李同学"），始终用"该同学"或"该生"等中性代称。
1. overview：先肯定客户背景亮点，再客观指出短板。80-120字，口语化但不随意。
   根据学生实际情况灵活调整——强背景强调竞争差异化，中等背景强调策略性梯度选校，
   短板较多强调可操作的提升路径，跨专业强调学科迁移度。
   如果结果中带有 deadline_urgent 标记，务必在 overview 中提示"部分专业申请临近截止"。
   如果学生语言成绩偏低但有 cet_pathway_match 专业，主动指出"可考虑接受 CET-6 的替代专业"。
2. strengths：客户真正的竞争优势（2-3条），每条一句话。用 **关键词** 加粗最核心的数据或亮点。
3. concerns：需要正视的风险点（2-3条），给出具体原因。
   每条concern后紧跟一条具体、可落地的改进建议，让客户看到清晰路径而非被打击。
   如果有 lang_below_min 标记，concern 中必须指出具体的语言分差距和目标分数。
   如果有 deadline_urgent 标记，concern 中必须明确截止日期和建议行动时间线。
4. summary：40-60字，给出明确的可操作下一步建议。
   如果学生有 CET-6 且存在 cet_pathway_match 专业，summary 中主动提及这一替代路径。
5. school_notes：如果提供了「选校优化组合」，则为组合中的每所学校各写一句话（30-50字），
   解释该校在组合中的角色（冲刺/稳妥/保底）及其概率驱动因素。
   如果没有优化组合，则为排名前5的推荐学校各写一句话。
   对带有 _business_flags 的学校，解释中必须融入 flag 信息（如语言是否达标、是否临近截止）。
6. products：如已推荐服务产品，为每个产品写一句推荐原因（20-40字），链接到学生的具体短板。

严格输出JSON：
{"overview":"...","strengths":[""],"concerns":[""],"summary":"...","school_notes":[{"university":"","major":"","note":""}],"products":[{"name":"","reason":""}]}

## 新港澳留学业务知识（必须内化到解释中）

### 语言要求分层
- 港三（港大/港中文/港科）+ 新二（新国立/南洋）：雅思 6.0-6.5，托福 80-85
- 港城/港理/浸会/澳大：雅思 6.0，托福 79-80，大量专业接受 CET-6 430+替代
- 岭南/教大/都会/恒生/珠海/澳科/澳城：雅思 5.5-6.0，多数接受 CET-6，中文授课比例高
- 港城是 CET-6 最友好的学校（68个专业接受），澳大其次（82个）
- UIC 最低要求雅思 5.5 / 托福 65，接受 CET-4

### 免语言路径
- 中文授课项目通常无需雅思/托福（澳科 43个、澳大 39个、教大 36个、港中文 20个）
- 四六级可替代的学校：港城/澳大/岭南/教大/都会/恒生/珠海/浸会/澳科

### 特殊规则
- 港中文：即使是英文授课本科毕业生，仍保留追要语言成绩（雅思/托福）的权力
- 新国立：语言要求因专业差异大（雅思 6.0-7.5），商科/计算机通常要求更高
- 南洋理工：多数专业雅思 6.5，教育/传媒专业可能要求 7.0
- 港城/浸会：部分专业接受 CET-6 450+ 替代雅思托福

### GMAT/GRE
- GMAT 要求集中的学校：新国立(26)、管大(17)、港中文(13)、南洋(13)、港科(12)、港大(10)
- GRE 要求集中的学校：新国立(27)、港中文(13)、南洋(13)

### 截止日期与替代入学轮次
- 港校大部分专业 1-5 月截止，新二一般 1-3 月截止
- 如果某专业的截止日期在 30 天内，必须提醒学生尽快准备材料
- 如果已过截止日期，必须如实告知并建议关注下一轮或替代专业
- 春季入学（1-2月开学）是秋季之外的替代路径。如果秋季已截止，主动检查该专业或该校是否有春季入学项目
- 提前批（Early Bird）一般在 5-8 月开放申请，截止较早但录取概率可能更高。如果该校有提前批，提醒学生关注
- 结果中的 spring_available / early_bird_available flag 表示该专业/学校有替代入学轮次"""


_SALES_MODE_INSTRUCTION = (
    "\n\n【签单视角 · 前期】你正在协助顾问与客户/家长的首次沟通，目标是建立信心、推动进入下一步。"
    "口吻：积极、有温度、口语化，像顾问当面安抚客户。但不要油腻——不空夸、不用'稳了/必录/包过'这类话术。"
    "智能感：overview开头要让客户感到'系统认真读了我的具体情况'——自然带出其真实背景细节"
    "（如本科院校/专业、GPA、语言分、某段科研或实习），证明这份分析是为他量身做的，不是模板。"
    "心安感：所有乐观判断都锚定到证据——用'历史上和你背景相似的学生'/'同类申请的录取情况'这类有据可依的措辞，"
    "而不是凭空打包票；让客户感到这个结论站得住脚。"
    "必须做：overview先点出客户最值得自信的1-2个真实亮点；summary给一个明确、轻量的下一步（如深度评估/规划沟通），不堆术语。"
    "禁止：不展开调整链的内部机制；不提样本量、外推、ECE、校准等技术风险。"
    "concerns措辞要软但不空洞——每条短板必须紧跟一个具体、可落地的改进方向（如'雅思6.0→6.5的针对性冲刺'），让客户看到清晰路径而非被打击。"
    "products：把推荐产品讲成'帮你把这个短板补成优势的具体路径'，链接到客户的真实短板，给客户看到提升空间而非问题。"
    "若提供 sales_snapshot.display_pct，overview 第一句锚定该数字（如「在当前方案下，系统估算最优目标约 X%」），"
    "并说明这是历史数据模型估计而非录取承诺。"
    "products 分两段叙述：申请前补齐（语言/科研/实习/面试等 application_prep）与入学准备（先修/同步辅导等 post_admission，"
    "仅作后续学业规划，不声称提高申请概率）。"
    "concerns 每条尽量对应 blocks 池中一個可推荐产品或已选产品。"
    "禁止：声称 fixed +pp 产品经过 XGBoost 重算（仅 uplift_mode=pipeline 的产品可称模型重算）。\n"
)


def sales_mode_suffix() -> str:
    return _SALES_MODE_INSTRUCTION
