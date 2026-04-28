"""WritePrint — LLM-powered full-text rewriting via DeepSeek API."""

import json
import time

import requests

from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

_log = setup_logger("page3", "write_print")


def _get_api_config() -> dict:
    cfg = load_app_config()
    return {
        "url": cfg.get("OPEN_AI_BASE_URL", ""),
        "key": cfg.get("OPEN_AI_API_KEY", ""),
        "model": cfg.get("OPEN_AI_MODEL", "deepseek-v4-flash"),
    }


def _call_llm(prompt: str, timeout: int = 60, max_tokens: int = 4000) -> str | None:
    cfg = _get_api_config()
    if not cfg["url"] or not cfg["key"]:
        _log.warning("LLM CALL | no config")
        return None

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}"}
    data = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(cfg["url"], headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices", [])
        elapsed = (time.perf_counter() - t0) * 1000
        if choices:
            output = choices[0].get("message", {}).get("content", "")
            _log.info(
                f"LLM OK | model={cfg['model']} tokens={max_tokens} "
                f"output={len(output) if output else 0}chars elapsed={elapsed:.0f}ms"
            )
            return output
        _log.warning(f"LLM EMPTY | elapsed={elapsed:.0f}ms")
        return None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _log.error(f"LLM FAIL | {type(e).__name__} elapsed={elapsed:.0f}ms")
        return None


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


def generate_full_rewrite(text: str, score: float, features: dict,
                          timeout: int = 90) -> str | None:
    """Generate a complete humanized rewrite with feature-level guidance."""
    feature_lines = []
    for fname, fb in features.items():
        if abs(fb['contribution']) > 1.0:
            direction = "AI-like ↑" if fb['contribution'] > 0 else "human-like ↓"
            feature_lines.append(
                f"  {fb['label']}: {fb['value']:.3f} — {direction}"
            )

    # Truncate text if needed to fit context
    text_section = text[:6000] if len(text) > 6000 else text

    prompt = FULL_REWRITE_PROMPT.format(
        score=score,
        features='\n'.join(feature_lines[:8]),
        text=text_section,
    )

    result = _call_llm(prompt, timeout=timeout, max_tokens=4000)
    if result:
        result = result.strip()
        # Strip markdown fences
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return result


def generate_sentence_rewrites(sentence: str, issues: list[str],
                               timeout: int = 25) -> list[dict]:
    """Generate per-sentence rewrites for display."""
    if not issues:
        return []

    prompt = SENTENCE_REWRITE_PROMPT.format(
        issues="; ".join(issues[:3]),
    ) + f"\n\nSENTENCE: {sentence}"

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
