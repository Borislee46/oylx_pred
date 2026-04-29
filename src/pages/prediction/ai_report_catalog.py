from __future__ import annotations

from typing import Any

PRODUCTS: dict[str, dict[str, Any]] = {
    "high_end_app": {
        "name": "高端申请服务",
        "variant": "A/B/C计划",
        "scale": "8-10所",
        "price": "¥35,000-55,000",
        "dot": "#06b6d4",
    },
    "std_app": {
        "name": "标准申请服务",
        "variant": "港-A-4 / 新-A-6",
        "scale": "5-6所",
        "price": "¥12,000-20,000",
        "dot": "#94a3b8",
    },
    "bg_research": {
        "name": "背景提升 · 科研",
        "variant": "顶锋计划",
        "scale": "1个项目",
        "price": "¥25,000-37,000",
        "dot": "#8b5cf6",
    },
    "bg_intern": {
        "name": "背景提升 · 实习",
        "variant": "优选/威计划",
        "scale": "1段实习",
        "price": "¥28,000-37,000",
        "dot": "#f97316",
    },
    "tutoring": {
        "name": "学术辅导",
        "variant": "英领/博睿",
        "scale": "按需",
        "price": "¥12,000-30,000",
        "dot": "#ec4899",
    },
    "english": {
        "name": "英语实战训练",
        "variant": "海外英语实战营",
        "scale": "1期",
        "price": "¥6,000-15,000",
        "dot": "#f59e0b",
    },
}
