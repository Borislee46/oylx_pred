"""
WritePrint — AI text detection for personal statements.

Ported from standalone tool. 6 surface features, no GPU/torch, ~120MB RAM.
Model: Ridge (L2-regularized), validated via Leave-One-Out CV.

Streamlit usage:
  from src.pages.write_print.engine import analyze
  from src.pages.write_print.model import get_model
  model, scaler = get_model()
  result = analyze(text, scaler, model)
"""

import statistics
import time

import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.utils.logger import setup_logger
from src.pages.write_print.features import (
    _tokenize,
    _compute_features_from_tokens,
    FEATURE_NAMES,
    FEATURE_LABELS,
)
from src.pages.write_print.diagnosis import (
    _load_ai_keywords,
    _find_template_phrases,
    _find_transition_density,
    _find_ai_keywords,
    _find_repetitive_starters,
    _find_uniform_runs,
    _find_long_sentences,
    _estimate_impact,
    _verdict,
    _DIFFICULTY,
)

_log = setup_logger("page3", "write_print")


def analyze(text: str, scaler: StandardScaler, model: Ridge) -> dict:
    """Return structured diagnosis dict for Streamlit / API consumption.

    Score is a point estimate. With n=9 training samples the LOO-CV MAE
    is ~18%, so treat scores as +/-18% approximate.
    """

    t0 = time.perf_counter()

    if len(text) < 50:
        return {'error': 'text too short', 'score': None, 'local_fixes': [],
                'global_fixes': [], 'quick_wins': []}

    words, sentences, sent_lengths, paragraphs, pos_tags = _tokenize(text)
    if not words or not sent_lengths:
        return {'error': 'no parseable text found', 'score': None,
                'local_fixes': [], 'global_fixes': [], 'quick_wins': []}

    feats = _compute_features_from_tokens(words, sent_lengths, paragraphs, text, pos_tags)
    feats_scaled = scaler.transform([feats])[0]
    raw_score = float(model.predict([feats_scaled])[0])
    score = max(0.0, min(100.0, raw_score))
    contributions = (model.coef_ * feats_scaled).tolist()
    intercept = float(model.intercept_)

    kw_dict = _load_ai_keywords()

    # ── feature breakdown ──
    feature_breakdown = {}
    for i, name in enumerate(FEATURE_NAMES):
        label, unit = FEATURE_LABELS[name]
        contrib = contributions[i]
        feature_breakdown[name] = {
            'label': label, 'unit': unit,
            'value': round(feats[i], 3),
            'contribution': round(contrib, 1),
            'direction': "AI" if contrib > 2 else ("human" if contrib < -2 else "neutral"),
        }

    # ── gather issues ──
    template_hits = _find_template_phrases(text, sentences)
    runs = _find_uniform_runs(sentences, sent_lengths)
    starter_issues = _find_repetitive_starters(sentences)
    trans_density = _find_transition_density(paragraphs)
    kw_hits = _find_ai_keywords(text, kw_dict)
    long_sents = _find_long_sentences(sentences, sent_lengths)

    local_fixes = []

    def _add_fix(category, label, detail, action, n=1, sent_idx=None):
        diff = _DIFFICULTY.get(category, {'label': '?', 'icon': '?'})
        local_fixes.append({
            'category': category, 'label': label, 'detail': detail,
            'action': action,
            'impact': _estimate_impact(n, category, score),
            'difficulty': diff['label'],
            'sentence_idx': sent_idx,
        })

    for t in template_hits:
        s_idx = t['sentence_idx']
        s = sentences[s_idx] if s_idx is not None else t['match']
        _add_fix('template', f"Template: {t['label']}",
                 s[:100] + ('...' if len(s) > 100 else ''),
                 t['suggestion'], sent_idx=s_idx)

    for run in runs[:5]:
        sents = run['sentences']
        if sents:
            mid = len(sents) // 2
            example = sents[mid][:90] + ('...' if len(sents[mid]) > 90 else '')
            _add_fix('uniform',
                     f"Uniform block: sentences {run['start']+1}-{run['end']+1} "
                     f"({run['length']} x ~{run['avg_words']}w)",
                     f'Pick one sentence to split. E.g.: "{example}"',
                     f"Split sentence {run['start'] + run['length']//2 + 1} "
                     f"into two: one short ({run['avg_words']//2}w), one medium.",
                     sent_idx=run['start'])

    for issue in starter_issues[:5]:
        indices = issue['sentence_indices']
        _add_fix('starter',
                 f"'{issue['starter']}' opens {issue['count']} sentences",
                 f"Sentences: {', '.join(str(i+1) for i in indices[:6])}",
                 "Flip the sentence order: put the second clause first. "
                 "Or start with 'What' / 'The' / a preposition.",
                 sent_idx=indices[0] if indices else None)

    for td in trans_density:
        if td['count'] >= 2:
            _add_fix('transition',
                     f"Paragraph {td['para_idx']+1}: {td['count']} AI transitions "
                     f"({', '.join(td['words'])})",
                     paragraphs[td['para_idx']][:120] + '...',
                     "Keep at most 1 transition in this paragraph. "
                     "Delete the rest — let the ideas connect naturally.",
                     n=td['count'], sent_idx=None)

    for h in kw_hits[:10]:
        snippet = h['snippet']
        s_idx = None
        for si, s in enumerate(sentences):
            if snippet[:30] in s or s[:30] in snippet:
                s_idx = si
                break
        _add_fix('keyword', f"AI keyword: \"{h['word']}\"",
                 f"Context: \"...{h['snippet']}...\"",
                 f"Replace \"{h['word']}\" with: {', '.join(h['alternatives'])}",
                 sent_idx=s_idx)

    for ls in long_sents[:5]:
        _add_fix('long_sent',
                 f"Sentence {ls['index']+1}: {ls['word_count']} words",
                 f'"{ls["text"]}"',
                 "Find a natural break point (after a comma or 'and') "
                 "and split into 2 sentences.",
                 sent_idx=ls['index'])

    local_fixes.sort(key=lambda x: -x['impact'])

    # ── global fixes ──
    global_fixes = []

    if len(sent_lengths) >= 3:
        scv = (statistics.stdev(sent_lengths) / statistics.mean(sent_lengths)
               if statistics.mean(sent_lengths) > 0 else 0)
        short_count = sum(1 for l in sent_lengths if l < 8)
        long_count = sum(1 for l in sent_lengths if l > 30)
        if scv < 0.45:
            msg = (f"Your sentences are similar in length throughout "
                   f"(only {short_count} very short, {long_count} very long). "
                   f"Good writing mixes punchy short sentences with longer ones. ")
            if short_count == 0:
                msg += ("Try adding a 3-5 word sentence right after a long one — "
                        "it creates rhythm. ")
            if long_count == 0:
                msg += ("Pick one key idea and elaborate with detail to "
                        "stretch it past 30 words. ")
            global_fixes.append(msg)

    if len(paragraphs) >= 3:
        para_sent_counts = [len(nltk.sent_tokenize(p)) for p in paragraphs]
        mean_ps = statistics.mean(para_sent_counts)
        if mean_ps > 0:
            pcv = statistics.stdev(para_sent_counts) / mean_ps
            if pcv < 0.55:
                global_fixes.append(
                    f"Your paragraphs are all similar length (~{mean_ps:.0f} "
                    f"sentences each). Try this: turn your strongest point into "
                    f"a single-sentence paragraph — it creates emphasis. Then "
                    f"merge two shorter paragraphs to restore balance."
                )

    long_ratio = sum(1 for w in words if len(w) >= 7) / len(words) if words else 0
    avg_wlen = sum(len(w) for w in words) / len(words) if words else 0
    if avg_wlen > 5.3 and long_ratio > 0.36:
        global_fixes.append(
            f"Your vocabulary leans academic (avg {avg_wlen:.1f} letters/word, "
            f"{long_ratio*100:.0f}% are 7+ letters). For a more natural tone, "
            f"try swapping a few long words for shorter ones — 'use' instead of "
            f"'utilize', 'show' instead of 'demonstrate'."
        )

    avg_sent = statistics.mean(sent_lengths) if sent_lengths else 0
    if avg_sent > 27:
        global_fixes.append(
            f"Your sentences average {avg_sent:.0f} words — on the long side "
            f"(most readable writing stays between 15-25). Find your longest "
            f"sentence; look for a comma or 'and' near the middle; replace "
            f"it with a period."
        )

    # ── summary ──
    local_impact = sum(f['impact'] for f in local_fixes[:8])
    est_new = max(0, score - local_impact)
    quick_wins = [f for f in local_fixes if f['category'] in ('keyword', 'transition')]

    elapsed = (time.perf_counter() - t0) * 1000
    fp = feature_breakdown.get("first_person", {}).get("value", 0)
    log_msg = (
        f"ANALYZE | score={score:.0f}% words={len(words)} "
        f"fixes={len(local_fixes)} first_person={fp:.3f} "
        f"elapsed={elapsed:.0f}ms"
    )
    if fp < 0.02 and score > 30:
        _log.warning(log_msg + " | LOW_FIRST_PERSON skew likely")
    else:
        _log.info(log_msg)

    return {
        'score': round(score, 1),
        'raw_score': round(raw_score, 1),
        'confidence': '±~18%',
        'verdict': _verdict(score),
        'features': feature_breakdown,
        'intercept': round(intercept, 1),
        'local_fixes': local_fixes,
        'global_fixes': global_fixes,
        'quick_wins': quick_wins[:3],
        'estimated_new_score': round(est_new, 1),
        'text_stats': {
            'words': len(words),
            'sentences': len(sentences),
            'paragraphs': len(paragraphs),
        },
        'sentence_lengths': sent_lengths,
    }
