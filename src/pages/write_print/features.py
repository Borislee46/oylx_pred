"""WritePrint — text I/O, tokenization, and 6 surface features."""

import re
import statistics
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from pypdf import PdfReader


def _ensure_nltk_data():
    for resource in ['punkt_tab', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
SCORE_PATTERN = re.compile(r"-(\d+)%\.pdf$", re.IGNORECASE)

FEATURE_NAMES = [
    "avg_word_len", "long_word_ratio", "avg_sent_len",
    "sentence_CV", "comma_density", "paragraph_CV",
    "passive_ratio", "ttr", "first_person", "burstiness",
]

FEATURE_LABELS = {
    "avg_word_len":    ("Word length",             "avg letters/word"),
    "long_word_ratio": ("Long words (7+ letters)", "fraction"),
    "avg_sent_len":    ("Sentence length",         "avg words"),
    "sentence_CV":     ("Sentence variety",        "length mix"),
    "comma_density":   ("Comma usage",             "per word"),
    "paragraph_CV":    ("Paragraph variety",       "length mix"),
    "passive_ratio":   ("Passive voice",           "fraction of verbs"),
    "ttr":             ("Lexical diversity",       "unique/total words"),
    "first_person":    ("First-person pronouns",   "per word"),
    "burstiness":      ("Sentence burstiness",     "local alternation"),
}


# ── text I/O ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        print(f"  [warn] PDF extract failed on '{pdf_path}': {e}")
        return ""


def clean_turnitin_report(raw: str) -> str:
    for pat in [
        r'^\s*\d+\s+\d+\s+-\s*(AI\s*)?\s*$',
        r'^\s*ID\s+trn:oid:::[\d:]+\s*$',
        r'^\d{7,8}\s+GMT[+-]\d+\s+\d+:\d+\s*$',
    ]:
        raw = re.sub(pat, '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\d+%\s+AI', '', raw)
    raw = re.sub(r'^[\w\s-]+\.(docx|pdf)\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^PS-\d+\s+\S+[\s-]+\S.*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\d+\s+\d+\s+AI\s+AI\s+AI\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\d{1,3}\s+\d{1,3}(,\d{3})?\s+\d{1,3}(,\d{3})?\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\d+\.\d+\s*KB\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    title_match = re.search(
        r'(Personal Statement|Motivation Letter|Statement of Purpose)',
        raw, re.IGNORECASE,
    )
    if title_match:
        raw = raw[title_match.start():]
    return raw.strip()


def read_input(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == '.pdf':
        raw = extract_text_from_pdf(str(p))
        return clean_turnitin_report(raw)
    elif p.suffix.lower() == '.txt':
        return p.read_text(encoding='utf-8', errors='replace').strip()
    else:
        return path.strip()


def parse_filename_score(filename: str) -> int:
    m = SCORE_PATTERN.search(filename)
    return int(m.group(1)) if m else None


# ── one-pass tokenization ───────────────────────────────────────────────────

_FIRST_PERSON = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}


def _tokenize(text: str):
    words = re.findall(r'\b\w+\b', text)
    raw_sentences = nltk.sent_tokenize(text)
    sent_lengths = [len(nltk.word_tokenize(s)) for s in raw_sentences]
    sent_lengths = [l for l in sent_lengths if l > 0]
    pos_tags = nltk.pos_tag(words)
    paras = [p.strip() for p in text.split('\n\n')]
    paras = [p for p in paras if len(p) > 30]
    return words, raw_sentences, sent_lengths, paras, pos_tags


# ── feature extractors ──────────────────────────────────────────────────────

def _feat_avg_word_len(words: list) -> float:
    if not words:
        return 4.0
    return sum(len(w) for w in words) / len(words)


def _feat_long_word_ratio(words: list) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) >= 7) / len(words)


def _feat_avg_sent_len(sent_lengths: list) -> float:
    if not sent_lengths:
        return 15.0
    return statistics.mean(sent_lengths)


def _feat_sentence_cv(sent_lengths: list) -> float:
    lengths = [l for l in sent_lengths if l > 1]
    if len(lengths) < 3:
        return 0.5
    m = statistics.mean(lengths)
    return statistics.stdev(lengths) / m if m > 0 else 0.5


def _feat_comma_density(text: str, words: list) -> float:
    if not words:
        return 0.0
    commas = text.count(',') + text.count(';') + text.count(':')
    return commas / len(words)


def _feat_paragraph_cv(paragraphs: list) -> float:
    if len(paragraphs) < 3:
        return 0.5
    para_lens = [len(nltk.sent_tokenize(p)) for p in paragraphs]
    para_lens = [l for l in para_lens if l > 0]
    if len(para_lens) < 2:
        return 0.5
    m = statistics.mean(para_lens)
    return statistics.stdev(para_lens) / m if m > 0 else 0.5


def _feat_passive_ratio(pos_tags: list) -> float:
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    verb_count = sum(1 for _, tag in pos_tags if tag in verb_tags)
    if verb_count < 3:
        return 0.15
    vbn_count = sum(1 for _, tag in pos_tags if tag == 'VBN')
    return vbn_count / verb_count


def _feat_ttr(words: list) -> float:
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)


def _feat_first_person_density(words: list) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if w.lower() in _FIRST_PERSON) / len(words)


def _feat_burstiness(sent_lengths: list) -> float:
    if len(sent_lengths) < 3:
        return 0.5
    diffs = [abs(sent_lengths[i] - sent_lengths[i - 1])
             for i in range(1, len(sent_lengths))]
    avg_diff = statistics.mean(diffs)
    avg_len = statistics.mean(sent_lengths)
    return avg_diff / avg_len if avg_len > 0 else 0.5


def _compute_features_from_tokens(words: list, sent_lengths: list,
                                   paragraphs: list, text: str,
                                   pos_tags: list = None) -> np.ndarray:
    feats = [
        _feat_avg_word_len(words),
        _feat_long_word_ratio(words),
        _feat_avg_sent_len(sent_lengths),
        _feat_sentence_cv(sent_lengths),
        _feat_comma_density(text, words),
        _feat_paragraph_cv(paragraphs),
    ]
    if pos_tags:
        feats += [
            _feat_passive_ratio(pos_tags),
            _feat_ttr(words),
            _feat_first_person_density(words),
            _feat_burstiness(sent_lengths),
        ]
    return np.array(feats)


def compute_features(text: str) -> np.ndarray:
    words, _, sent_lengths, paragraphs, pos_tags = _tokenize(text)
    return _compute_features_from_tokens(words, sent_lengths, paragraphs, text, pos_tags)
