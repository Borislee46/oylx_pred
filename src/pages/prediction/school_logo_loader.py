from pathlib import Path

_LOGO_DIR = Path(__file__).resolve().parents[3] / "assets" / "school_logos"
_FALLBACK = _LOGO_DIR.parent / "product_logo.png"

_CN_TO_KEY = {
    "香港大学": "hku",
    "香港中文大学": "cuhk",
    "香港中文大学 (深圳校区)": "cuhk_shenzhen",
    "香港科技大学": "hkust",
    "香港理工大学": "polyu",
    "香港城市大学": "cityu",
    "香港浸会大学": "hkbu",
    "香港岭南大学": "lingnan",
    "香港教育大学": "eduhk",
    "香港恒生大学": "hsuhk",
    "香港都会大学": "hkmu",
    "香港珠海学院": "chuhai",
    "澳门大学": "umacau",
    "澳门理工大学": "mpu",
    "澳门科技大学": "must",
    "澳门城市大学": "cityu_macau",
    "新加坡国立大学": "nus",
    "新加坡南洋理工大学": "ntu",
    "新加坡管理大学": "smu",
    "马来亚大学": "umalaya",
    "马来西亚国立大学": "ukm",
    "马来西亚理科大学": "usm",
    "马来西亚博特拉大学": "upm",
}

_VALID_KEYS = frozenset(_CN_TO_KEY.values())

_CN_TO_URL = {
    "香港大学": "https://www.hku.hk",
    "香港中文大学": "https://www.cuhk.edu.hk",
    "香港中文大学 (深圳校区)": "https://www.cuhk.edu.cn",
    "香港科技大学": "https://www.hkust.edu.hk",
    "香港理工大学": "https://www.polyu.edu.hk",
    "香港城市大学": "https://www.cityu.edu.hk",
    "香港浸会大学": "https://www.hkbu.edu.hk",
    "香港岭南大学": "https://www.ln.edu.hk",
    "香港教育大学": "https://www.eduhk.hk",
    "香港恒生大学": "https://www.hsu.edu.hk",
    "香港都会大学": "https://www.hkmu.edu.hk",
    "香港珠海学院": "https://www.chuhai.edu.hk",
    "澳门大学": "https://www.um.edu.mo",
    "澳门理工大学": "https://www.mpu.edu.mo",
    "澳门科技大学": "https://www.must.edu.mo",
    "澳门城市大学": "https://www.cityu.edu.mo",
    "新加坡国立大学": "https://www.nus.edu.sg",
    "新加坡南洋理工大学": "https://www.ntu.edu.sg",
    "新加坡管理大学": "https://www.smu.edu.sg",
    "马来亚大学": "https://www.um.edu.my",
    "马来西亚国立大学": "https://www.ukm.my",
    "马来西亚理科大学": "https://www.usm.my",
    "马来西亚博特拉大学": "https://www.upm.edu.my",
}


def get_school_url(school_name: str) -> str | None:
    return _CN_TO_URL.get(school_name)


def get_logo_path(school_name: str) -> str:
    key = _CN_TO_KEY.get(school_name, school_name)
    if key in _VALID_KEYS:
        logo = _LOGO_DIR / f"{key}.png"
        if logo.exists():
            return str(logo)
    return str(_FALLBACK)
