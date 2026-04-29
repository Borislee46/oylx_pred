"""WritePrint — LLM-powered full-text rewriting via DeepSeek API."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

import requests

from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

_log = setup_logger("page3", "write_print")

# ── Thread-local session pool ──────────────────────────────
_tls = threading.local()
_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 256


def _get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


# ── API config ─────────────────────────────────────────────


def _get_api_config() -> dict[str, str]:
    cfg = load_app_config()
    return {
        "url": cfg.get("OPEN_AI_BASE_URL", ""),
        "key": cfg.get("OPEN_AI_API_KEY", ""),
        "model": cfg.get("OPEN_AI_MODEL", "deepseek-v4-flash"),
    }


def _cache_key(prompt: str, max_tokens: int) -> str:
    payload = json.dumps((prompt, max_tokens), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _call_llm(prompt: str, timeout: int = 60, max_tokens: int = 4000) -> str | None:
    cfg = _get_api_config()
    if not cfg["url"] or not cfg["key"]:
        _log.warning("LLM CALL | no config")
        return None

    # ── Cache check ──
    key = _cache_key(prompt, max_tokens)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _log.info(f"LLM CACHE HIT | len={len(_CACHE[key])}chars")
            return _CACHE[key]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}"}
    data: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    session = _get_session()
    last_error: str | None = None

    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = session.post(cfg["url"], headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices", [])
            elapsed = (time.perf_counter() - t0) * 1000
            if choices:
                output: str = choices[0].get("message", {}).get("content", "")
                _log.info(
                    f"LLM OK | attempt={attempt + 1} model={cfg['model']} "
                    f"output={len(output)}chars elapsed={elapsed:.0f}ms"
                )
                # Cache
                with _CACHE_LOCK:
                    _CACHE[key] = output
                    _CACHE.move_to_end(key)
                    while len(_CACHE) > _CACHE_MAX:
                        _CACHE.popitem(last=False)
                return output
            _log.warning(f"LLM EMPTY | attempt={attempt + 1} elapsed={elapsed:.0f}ms")
            return None
        except requests.Timeout:
            elapsed = (time.perf_counter() - t0) * 1000
            last_error = f"Timeout elapsed={elapsed:.0f}ms"
            _log.warning(f"LLM TIMEOUT | attempt={attempt + 1}/3 | {last_error}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        except requests.ConnectionError:
            elapsed = (time.perf_counter() - t0) * 1000
            last_error = f"ConnectionError elapsed={elapsed:.0f}ms"
            _log.warning(f"LLM CONN | attempt={attempt + 1}/3 | {last_error}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            last_error = f"{type(e).__name__} elapsed={elapsed:.0f}ms"
            _log.warning(f"LLM FAIL | attempt={attempt + 1}/3 | {last_error}")
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))

    _log.error(f"LLM EXHAUSTED | {last_error}")
    return None


# ── Prompts ────────────────────────────────────────────────

FULL_REWRITE_PROMPT = """You are an expert writing coach. Rewrite this personal statement to sound NATURALLY HUMAN. Current AI-detection score: {score}%.

Feature analysis — what makes it sound AI-generated:
{features}

RULES to reduce AI score while keeping academic quality:
1. Use FIRST-PERSON pronouns naturally (I, my, me, we) — this is the #1 signal
2. Mix SHORT sentences (3-8 words) with longer ones — avoid uniform length
3. Replace formal words with simpler ones: utilize→use, demonstrate→show, commence→start, furthermore→also, nevertheless→yet, consequently→so, facilitate→help, sufficient→enough, numerous→many, obtain→get, possess→have, regarding→about, additionally→also, implement→set up
4. Remove template phrases: "I believe that", "It is important to", "This experience taught me", "not only...but also", "I am confident that", "This program will provide me with", "In addition to...I also", "I am committed to"
5. Use ACTIVE voice, avoid "it is/was/has been ... that"
6. Vary paragraph structure — some single-sentence, some multi-sentence
7. PRESERVE all facts, achievements, names, institutions, dates — only change writing style
8. Keep academic tone — don't make it casual, just make it sound like a real person

Return ONLY the rewritten personal statement. No markdown, no explanation, no quotes, no preamble. Just the complete rewritten text.

ORIGINAL:
{text}"""

SENTENCE_REWRITE_PROMPT = """Rewrite this sentence to sound more human. Issues: {issues}

2 versions:
- conservative: minimal changes to fix the AI patterns
- bold: more natural phrasing, less formal

Return JSON: {{"rewrites": [{{"tone": "conservative", "text": "..."}}, {{"tone": "bold", "text": "..."}}]}}"""


# ── Public API ─────────────────────────────────────────────


def generate_full_rewrite(text: str, score: float, features: dict, timeout: int = 90) -> str | None:
    feature_lines = []
    for _fname, fb in features.items():
        if abs(fb["contribution"]) > 1.0:
            direction = "AI-like ↑" if fb["contribution"] > 0 else "human-like ↓"
            feature_lines.append(f"  {fb['label']}: {fb['value']:.3f} — {direction}")

    text_section = text[:6000] if len(text) > 6000 else text
    prompt = FULL_REWRITE_PROMPT.format(
        score=score,
        features="\n".join(feature_lines[:8]),
        text=text_section,
    )

    result = _call_llm(prompt, timeout=timeout, max_tokens=4000)
    if result:
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return result


def generate_sentence_rewrites(sentence: str, issues: list[str], timeout: int = 25) -> list[dict]:
    if not issues:
        return []

    prompt = (
        SENTENCE_REWRITE_PROMPT.format(
            issues="; ".join(issues[:3]),
        )
        + f"\n\nSENTENCE: {sentence}"
    )

    raw = _call_llm(prompt, timeout=timeout, max_tokens=600)
    if not raw:
        return []

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        return data.get("rewrites", [])
    except (json.JSONDecodeError, KeyError):
        return []
