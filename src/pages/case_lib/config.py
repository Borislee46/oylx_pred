CATEGORIES_MAP = {"本科": ["undergrad"], "硕士": ["grad"], "博士": ["phd"]}

REVERSE_CATEGORIES_MAP = {
    english: chinese for chinese, english_list in CATEGORIES_MAP.items() for english in english_list
}

YEAR_COL = "year"
TARGET_UNI_COL = "申请院校"
TARGET_COUNTRY_COL = "申请国家"
SCHOOL_NAME_COL = "学校名称"
SOURCE_COUNTRY_COL = "国家"
DOMESTIC_UNI_CLASSIFICATION_COL = "国本院校分类"
OVERSEAS_UNI_QS_RANK_COL = "海本QS排名区间"
BACKGROUND_MAJOR_COL = "就读专业1"
ADMISSION_STATUS_COL = "录取状态"
IELTS_COL = "IELTS"
IELTS_SCORE_COL = "IELTS分数"
TOEFL_COL = "TOEFL"
TOEFL_SCORE_COL = "TOEFL分数"
CATEGORY_COL = "category"

BACKGROUND_UNI_COLS = ["就读院校1", "就读院校", "就读院校-原始", "就读/毕业学校修正"]

TARGET_MAJOR_COLS = ["申请专业", "申请专业-ERP选校协议", "专业英文名称修正"]

SYSTEM_COLS = ["申请体系"]

DISPLAY_COLS_CONFIG = {
    "本科": [
        ("year", "学年"),
        ("申请院校", "申请院校"),
        ("申请专业", "申请专业"),
        ("录取状态", "录取状态"),
        ("学生身份", "学生身份"),
        ("体系", "就读体系"),
        ("申请体系", "申请体系"),
        ("就读/毕业学校修正", "就读院校"),
        ("就读专业1", "就读专业"),
        ("GPA(GPA和GPA百分制任选其一填写即可）", "GPA"),
        ("高考成绩", "高考成绩"),
        ("SAT", "SAT"),
        ("ACT", "ACT"),
        ("A-LEVEL", "A-Level"),
        ("IB", "IB"),
        ("IELTS", "雅思"),
        ("TOEFL", "托福"),
    ],
    "博士": [
        ("year", "学年"),
        ("申请院校", "申请院校"),
        ("申请专业", "申请专业"),
        ("录取状态", "录取状态"),
        ("就读院校1", "就读院校"),
        ("就读专业1", "就读专业"),
        ("GPA1", "GPA"),
        ("GPA分制1", "GPA分制"),
        ("IELTS分数", "雅思分数"),
        ("TOEFL分数", "托福分数"),
        ("GRE分数", "GRE分数"),
        ("GMAT分数", "GMAT分数"),
        ("background_summary", "背提信息"),
    ],
    "硕士": [
        ("year", "学年"),
        ("申请院校", "申请院校"),
        ("专业英文名称修正", "申请专业"),
        ("录取状态", "录取状态"),
        ("就读院校1", "就读院校"),
        ("uni_classification_summary", "院校类型/QS"),
        ("就读专业1", "就读专业"),
        ("GPA1", "GPA"),
        ("GPA分制1", "GPA分制"),
        ("IELTS分数", "雅思分数"),
        ("TOEFL分数", "托福分数"),
        ("gre_gmat_score", "GRE/GMAT"),
        ("background_summary", "背提信息"),
    ],
}

INVALID_VALUES = ["空空空", "nan", "NaN", "null", "NULL", "None", "无", "", "0.0", "0"]

DOMESTIC_PREFIX = "国本-"
OVERSEAS_PREFIX = "海本-"

DATA_DIR = "src/machine_learning_models/data"
CASES_FILE_PATTERN = "cases_*.feather"
SCHOOL_BASE_PATH = "src/machine_learning_models/data/school_base.feather"

INITIAL_LOAD_COUNT = 100
LOAD_MORE_COUNT = 100
MAX_DISPLAY_COUNT = 500
GRID_HEIGHT = 1060

GPA_COLS_TO_FORMAT = ["GPA", "GPA分制", "GPA（百分制）"]
