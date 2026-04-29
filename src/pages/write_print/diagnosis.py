"""WritePrint — pattern detection, fix generation, and verdict logic."""

import json
import re
import statistics
from collections import Counter

import nltk

from src.pages.write_print.features import DATA_DIR

_keywords_cache: dict | None = None


def _load_ai_keywords() -> dict:
    global _keywords_cache
    if _keywords_cache is not None:
        return _keywords_cache
    path = DATA_DIR / "ai_keywords.json"
    try:
        with open(path, encoding="utf-8") as f:
            _keywords_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _keywords_cache = {}
    return _keywords_cache


AI_TRANSITIONS_SET = {
    "moreover",
    "furthermore",
    "additionally",
    "in addition",
    "however",
    "nevertheless",
    "nonetheless",
    "conversely",
    "consequently",
    "therefore",
    "thus",
    "hence",
    "importantly",
    "significantly",
    "notably",
    "subsequently",
    "meanwhile",
    "accordingly",
    "alternatively",
    "besides",
}

AI_TEMPLATES = [
    (
        r"\bThis experience (has\s+)?(made|taught|enabled|allowed)\s+me\b",
        '"This experience made me..."',
        'Narrate the impact directly: "After this, I..." or drop the moral entirely',
    ),
    (
        r"\bI believe that\b",
        '"I believe that..."',
        'Drop "I believe that" — just state your point. Or use "I think" for casual tone',
    ),
    (
        r"\bnot only\b.*\bbut also\b",
        '"not only...but also..."',
        "Split into two plain sentences. This structure is overused by AI",
    ),
    (
        r"\bIt is (important|essential|crucial|necessary|worth noting) (to|that)\b",
        '"It is important to..."',
        'Cut the filler: "We must..." / "The key point is..." / just state it',
    ),
    (
        r"\b(I am|I\'m) confident that\b",
        '"I am confident that..."',
        'Replace: "I know..." / "I will..." / or drop and just state the claim',
    ),
    (
        r"\bThrough (this|my|the)\b.{3,40}?\b(I (have )?(gained|learned|developed|acquired|enhanced))\b",
        '"Through X, I gained Y..."',
        'Reverse the order: "I learned Y while doing X" — more direct, less formulaic',
    ),
    (
        r"\b(I am|I\'m) particularly interested in\b",
        '"I am particularly interested in..."',
        '"I want to focus on..." or name the specific thing that excites you',
    ),
    (
        r"\bThis program will provide me with\b",
        '"This program will provide me with..."',
        '"In this program I can..." — active voice, your agency not theirs',
    ),
    (
        r"\bIn addition to\b.{3,50}?\b(I also|I have also|I)\b",
        '"In addition to X, I also Y..."',
        'Break into two sentences. "I did X. Separately, I..."',
    ),
    (
        r"\bAs a result\b.{3,40}?\b(I (have )?(decided|chosen|learned))\b",
        '"As a result, I decided..."',
        '"So I..." — one word instead of four. Simpler = more human',
    ),
    (
        r"\bI was responsible for\b",
        '"I was responsible for..."',
        '"I handled..." / "My job was to..." / "I took care of..."',
    ),
    (
        r"\bI am committed to\b",
        '"I am committed to..."',
        '"I plan to..." / "I will..." — more concrete, less ceremonial',
    ),
]


def _find_template_phrases(text: str, sentences: list) -> list:
    hits = []
    for pat, label, suggestion in AI_TEMPLATES:
        for m in re.finditer(pat, text, re.IGNORECASE):
            matched_text = m.group()
            pos = m.start()
            cum = 0
            sent_idx = None
            for i, s in enumerate(sentences):
                cum += len(s) + 1
                if pos < cum:
                    sent_idx = i
                    break
            hits.append(
                {
                    "match": matched_text,
                    "label": label,
                    "suggestion": suggestion,
                    "sentence_idx": sent_idx,
                }
            )
    return hits


def _find_transition_density(paragraphs: list) -> list:
    results = []
    for pi, para in enumerate(paragraphs):
        found = []
        for sent in nltk.sent_tokenize(para):
            parts = sent.strip().split()
            if parts:
                first = parts[0].lower().rstrip(",")
                if first in AI_TRANSITIONS_SET:
                    found.append(first)
        if found:
            results.append({"para_idx": pi, "count": len(found), "words": found})
    return results


def _find_ai_keywords(text: str, keywords_dict: dict) -> list:
    hits = []
    for m in re.finditer(r"\b\w+\b", text):
        lower = m.group().lower()
        if lower in keywords_dict:
            alts = keywords_dict[lower]
            pos = m.start()
            before = text[:pos]
            after = text[pos + len(m.group()) :]
            sent_start = before.rfind(".") + 1 if before.rfind(".") != -1 else 0
            sent_start = max(sent_start, before.rfind("!") + 1, before.rfind("?") + 1)
            sent_end = after.find(".")
            if sent_end == -1:
                sent_end = len(after)
            snippet = text[sent_start : pos + sent_end + 1].strip()[:120]
            snippet = snippet.replace("\n", " ")
            hits.append(
                {
                    "word": m.group(),
                    "alternatives": alts[:3],
                    "snippet": snippet,
                }
            )
    return hits


_STARTER_ALLOWLIST = {
    "i",
    "my",
    "we",
    "he",
    "she",
    "they",
    "it",
    "a",
    "an",
    "and",
    "but",
    "or",
    "so",
    "if",
    "as",
    "at",
    "in",
    "on",
    "to",
    "for",
    "with",
    "from",
    "when",
    "while",
    "since",
    "because",
    "although",
}


def _find_repetitive_starters(sentences: list) -> list:
    starters = []
    for i, s in enumerate(sentences):
        words = s.strip().split()
        if len(words) >= 2 and words[0].lower() not in _STARTER_ALLOWLIST:
            starters.append((words[0].lower(), i))
        if len(words) >= 4:
            phrase = f"{words[0].lower()} {words[1].lower()} {words[2].lower()}"
            if words[0].lower() not in _STARTER_ALLOWLIST:
                starters.append((phrase, i))

    starter_counts = Counter(s for s, _ in starters)
    issues = []
    for starter, count in starter_counts.items():
        words_in = starter.split()
        if len(words_in) == 1 and count >= 3:
            positions = [i for s, i in starters if s == starter]
            issues.append(
                {
                    "starter": starter.capitalize(),
                    "count": count,
                    "sentence_indices": positions,
                }
            )
        elif len(words_in) == 3 and count >= 2:
            positions = [i for s, i in starters if s == starter]
            issues.append(
                {
                    "starter": " ".join(w.capitalize() if len(w) > 3 else w for w in words_in),
                    "count": count,
                    "sentence_indices": positions,
                }
            )
    return issues


def _find_uniform_runs(sentences: list, sent_lengths: list) -> list:
    runs = []
    i = 0
    while i < len(sent_lengths) - 2:
        trio = sent_lengths[i : i + 3]
        if len(trio) == 3 and all(length > 8 for length in trio):
            m = statistics.mean(trio)
            if m > 0 and all(abs(length - m) / m < 0.20 for length in trio):
                j = i + 3
                while j < len(sent_lengths) and sent_lengths[j] > 8:
                    if abs(sent_lengths[j] - m) / m < 0.20:
                        j += 1
                    else:
                        break
                if j - i >= 3:
                    runs.append(
                        {
                            "start": i,
                            "end": j - 1,
                            "length": j - i,
                            "avg_words": round(m),
                            "sentences": sentences[i:j],
                        }
                    )
                i = j
                continue
        i += 1
    return runs


def _find_long_sentences(sentences: list, sent_lengths: list, threshold: int = 35) -> list:
    long_ones = []
    for i, (s, wc) in enumerate(zip(sentences, sent_lengths, strict=False)):
        if wc >= threshold:
            long_ones.append(
                {
                    "index": i,
                    "word_count": wc,
                    "text": s[:120] + ("..." if len(s) > 120 else ""),
                }
            )
    return long_ones


def _sentence_rhythm_chart(sent_lengths: list, width: int = 54) -> str:
    if not sent_lengths:
        return ""
    scale = width / max(max(sent_lengths), 1)
    lines = []
    for i, length in enumerate(sent_lengths):
        bar = "#" * max(1, int(length * scale))
        marker = ""
        if i >= 2 and all(
            abs(sent_lengths[j] - length) / max(length, 1) < 0.15 for j in range(i - 2, i)
        ):
            marker = " <-- uniform"
        lines.append(f"  [{i + 1:>2}] {bar} {length}w{marker}")
    return "\n".join(lines)


_DEFAULT_IMPACTS = {
    "keyword": 0.8,
    "template": 2.5,
    "transition": 1.5,
    "uniform": 3.0,
    "long_sent": 1.2,
    "starter": 3.0,
}


def _estimate_impact(n_fixes: int, category: str, current_score: float) -> float:
    try:
        weights = _get_impact_weights_cached()()
    except Exception:
        weights = None
    if weights is None:
        weights = _DEFAULT_IMPACTS
    per_fix = weights.get(category, _DEFAULT_IMPACTS.get(category, 1.0))
    return min(current_score * 0.3, n_fixes * per_fix)


_impact_weights_fn = None


def _get_impact_weights_cached():
    """Lazy-import & cache get_impact_weights to avoid circular import + hot-path import lock."""
    global _impact_weights_fn
    if _impact_weights_fn is None:
        from src.pages.write_print.model import get_impact_weights

        _impact_weights_fn = get_impact_weights
    return _impact_weights_fn


_DIFFICULTY = {
    "keyword": {"label": "Easy", "icon": "1"},
    "transition": {"label": "Easy", "icon": "1"},
    "template": {"label": "Medium", "icon": "2"},
    "long_sent": {"label": "Medium", "icon": "2"},
    "starter": {"label": "Medium", "icon": "2"},
    "uniform": {"label": "Harder", "icon": "3"},
}


def _verdict(score: float) -> dict:
    if score < 20:
        return {
            "label": "Likely human-written",
            "description": "Reads naturally. Minor formal phrasing is normal "
            "in academic English — not a red flag.",
        }
    elif score < 40:
        return {
            "label": "Mostly human",
            "description": "Some academic phrasing patterns detected, but "
            "this is common in polished personal statements. "
            "Review highlighted words if you want a more casual tone.",
        }
    elif score < 60:
        return {
            "label": "Mixed signals",
            "description": "Some sections read like AI-assisted writing. "
            "Not necessarily a problem — many strong applicants "
            "use AI for brainstorming. Focus on the LOCAL fixes below.",
        }
    elif score < 80:
        return {
            "label": "Significant AI patterns",
            "description": "Multiple indicators suggest AI involvement. "
            "Admissions readers may notice. Prioritise the "
            "template phrases and uniform sentence blocks.",
        }
    else:
        return {
            "label": "Strong AI patterns",
            "description": "Very likely AI-generated or heavily AI-edited. "
            "We recommend substantial rewriting — focus on GLOBAL "
            "fixes first, then LOCAL word swaps.",
        }
