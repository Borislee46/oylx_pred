# WritePrint — 文书AI检测

AI-generated text detection for personal statements and motivation letters. 6 stylometric features + Ridge regression, no GPU required.

## Directory Structure

```
src/pages/write_print/
├── __init__.py        # Public API: analyze, diagnose, get_model
├── engine.py          # Core: feature extraction + model fitting + analyze()
├── ui.py              # Streamlit page rendering
├── data/
│   ├── ai_keywords.json  # 95 AI-favored words → human alternatives
│   └── sample/           # 9 Turnitin-labeled PDFs (training data)
```

## Flow

```
User uploads/pastes text
    │
    ▼
engine.get_model()          # Fit once, cached via @st.cache_resource
    │
    ▼
engine.analyze(text)        # 6 features → Ridge → score 0-100%
    ├── feature_breakdown   # What pushed score up/down
    ├── local_fixes[]       # Specific words/sentences + fix actions
    ├── global_fixes[]      # Structure/pattern suggestions
    ├── quick_wins[]        # Easy 5-min fixes
    └── verdict             # Human-readable interpretation
    │
    ▼
ui.py renders:
    Score gauge → Feature metrics → Fix tabs (Easy/Medium/Harder) → Bottom line
```

## Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| `compute_features()` | `engine.py` | 6 surface features from text |
| `fit_model()` | `engine.py` | Ridge regression + LOO CV on labeled PDFs |
| `get_model()` | `engine.py` | Cached model loader (fit once, reuse) |
| `analyze()` | `engine.py` | Returns structured dict for UI rendering |
| `diagnose()` | `engine.py` | CLI print version (for debugging) |

## Features

| Feature | What it measures | Direction |
|---------|-----------------|-----------|
| Word length | Avg characters/word | Higher = more AI-like |
| Long words | Fraction ≥7 chars | Higher = more AI-like |
| Sentence length | Avg words/sentence | Higher = more AI-like |
| Sentence variety | CV of sentence lengths | Higher = more AI-like |
| Comma usage | Commas per word | Higher = more AI-like |
| Paragraph variety | CV of paragraph lengths | Higher = more AI-like |

Model: Ridge (L2). LOO-CV MAE ≈ 18.5% on 9 samples. Score shown as ±18%.

## Dependencies

- nltk (tokenization)
- numpy, scikit-learn (Ridge + StandardScaler)
- No GPU, no torch, ~120MB RAM
