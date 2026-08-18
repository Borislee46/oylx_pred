"""Smart grep-like search engine for university/major fuzzy matching.

``smart_search()`` is the main entry point.  It accepts an optional LLM
expander that bridges Chinese→English gaps pure fuzz cannot cross.

Typical usage::

    from src.utils.search import smart_search, build_search_expander

    expander = build_search_expander()
    results = smart_search(
        "语言学",
        english_major_names,
        expander=expander,
        limit=15,
    )
    for name, score, idx in results:
        print(f"{name}  ({score:.0f})")
"""

from __future__ import annotations

from src.utils.search.engine import smart_search
from src.utils.search.expander import build_search_expander

__all__ = [
    "smart_search",
    "build_search_expander",
]
