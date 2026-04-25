# 02 — Data Acquisition & Preprocessing

**Last updated:** 2026-04-18

## Dataset Summary

| Dataset | Source | Size | Task | Format |
|---------|--------|------|------|--------|
| **MBTI** | Kaggle — Personality Café | ~8,600 users, 422K+ posts | 16-class MBTI + 4-dim binary | CSV |
| **Pandora** | Reddit (Gjurković & Šnajder 2021) | ~10K users, 1,568 OCEAN-labeled | 16-class MBTI + OCEAN binary | JSON |
| **Essays** | Pennebaker & King (1999) | 2,468 essays | Big Five (OCEAN) binary | CSV |
| **Personality Evd** | Sun et al. EMNLP 2024 | 1,846 dialogues, Chinese | OCEAN binary + evidence gold | JSON |

## Embeddings (optional, for LSTM baseline)

| Artifact | Size | Used by |
|----------|------|---------|
| GloVe 6B (300d) | ~1 GB | `lstm_baseline.py` with `glove_path` config |

Download via:
```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/download_embeddings.py --dim 300
```

---

## 1. MBTI Dataset (Kaggle)

### Download

```bash
# Option A: Kaggle CLI (recommended)
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d datasnaek/mbti-type -p data/raw/mbti/
unzip data/raw/mbti/mbti-type.zip -d data/raw/mbti/

# Option B: Manual
# https://www.kaggle.com/datasets/datasnaek/mbti-type
# Download mbti_1.csv → data/raw/mbti/
```

### Raw Format

```csv
type,posts
INFJ,"'http://www.youtube.com/watch?v=qsXHcwe3krw|||'http://41.media.tumblr.com/...|||'ENFPs..."
```

Each row = 1 user. `posts` column = last 50 posts joined by `|||`.

### Preprocessing (`src/data/mbti_parser.py`)

```python
"""
Pipeline:
1. Split posts by '|||' delimiter
2. Remove URLs (regex: https?://\S+)
3. Remove @mentions
4. Strip MBTI type mentions from text (avoid data leakage!)
   - Regex: r'\b(INFJ|INFP|INTJ|INTP|...)\b' (all 16 types)
5. Filter posts < 10 words
6. Aggregate: concatenate user's cleaned posts (max 512 tokens for transformers)
7. Stratified split: 70/15/15 train/val/test

Output: data/processed/mbti/{train,val,test}.jsonl
"""

MBTI_TYPES = [
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP",
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP"
]

# For 4-dimension binary classification (alternative to 16-class):
DIMENSIONS = {
    "IE": ("I", "E"),  # Introversion / Extraversion
    "SN": ("S", "N"),  # Sensing / Intuition
    "TF": ("T", "F"),  # Thinking / Feeling
    "JP": ("J", "P"),  # Judging / Perceiving
}
```

### Critical: Avoiding Data Leakage

Many MBTI forum posts explicitly mention MBTI types ("As an INTJ, I think..."). You **must** remove these mentions from training text — otherwise the model just learns string matching, not personality recognition.

```python
import re
MBTI_PATTERN = re.compile(r'\b(' + '|'.join(MBTI_TYPES) + r')\b', re.IGNORECASE)
text = MBTI_PATTERN.sub('[TYPE]', text)  # or remove entirely
```

---

## 2. Pandora Dataset (Reddit)

### Download

```bash
# Official source: https://psy.takelab.fer.hr/datasets/all/pandora/
# Registration required — fill the form to get download link

# After receiving link:
wget -P data/raw/pandora/ "<download_url>"
tar -xzf data/raw/pandora/pandora.tar.gz -d data/raw/pandora/

# Alternative mirror (check availability):
# https://zenodo.org/records/... (search "PANDORA personality Reddit")
```

### Raw Format

```json
// users.json — user metadata + Big Five scores
{
  "author": "username123",
  "mbti": "INTJ",
  "bigfive": {
    "openness": 4.2,
    "conscientiousness": 3.8,
    "extraversion": 2.1,
    "agreeableness": 3.5,
    "neuroticism": 3.9
  }
}

// comments/ — one file per user with all their Reddit comments
```

### Preprocessing (`src/data/pandora_parser.py`)

```python
"""
Pipeline:
1. Join user metadata with comments
2. For each user: sample up to 100 comments (balanced across subreddits)
3. Clean: remove Reddit markdown ([link](url)), quotes (> ...), bot text
4. Concatenate sampled comments per user
5. Binarize Big Five: score > 3.0 → High, else → Low (per trait)
   - This gives 5 binary classification tasks
6. Stratified split: 70/15/15

Output: data/processed/pandora/{train,val,test}.jsonl
"""
```

### Big Five Score → Label Mapping

```python
def binarize_ocean(score: float, threshold: float = 3.0) -> str:
    """Convert continuous Big Five score to binary label."""
    return "HIGH" if score > threshold else "LOW"

# Alternative: tercile split (LOW/MED/HIGH) for 3-class
def tercile_ocean(score: float) -> str:
    if score < 2.5: return "LOW"
    elif score < 3.5: return "MED"
    return "HIGH"
```

---

## 3. Essays Dataset (Pennebaker & King)

### Download

```bash
# Primary source (James Pennebaker's lab):
# https://web.archive.org/web/2024/https://personality-project.org/perproj/descriptions/data_files.html

# Kaggle mirror:
kaggle datasets download -d sameersingh612/essays-big5 -p data/raw/essays/
unzip data/raw/essays/essays-big5.zip -d data/raw/essays/

# Alternative: direct link
# https://github.com/jkwieser/personality-detection-text/tree/main/data
# Download essays.csv
```

### Raw Format

```csv
#AUTHID,cEXT,cNEU,cAGR,cCON,cOPN,TEXT
user001,y,n,y,y,n,"I am 22 years old and currently studying..."
```

Binary labels: `y` = HIGH, `n` = LOW for each Big Five trait.

### Preprocessing (`src/data/essays_parser.py`)

```python
"""
Pipeline:
1. Parse CSV with proper encoding (latin-1)
2. Map y/n → HIGH/LOW for each trait
3. Minimal cleaning: fix encoding artifacts, normalize whitespace
   (These are formal essays — preserve original language style)
4. Stratified split: 70/15/15

Output: data/processed/essays/{train,val,test}.jsonl
"""
```

---

## 4. Personality Evd Dataset

### Download

```bash
# Paper: "Revealing Personality Traits: A New Benchmark Dataset for
#         Explainable Personality Recognition on Dialogues" (2024)
# GitHub: https://github.com/Lei-Sun-RUC/Personality_Evd

git clone https://github.com/Lei-Sun-RUC/Personality_Evd.git data/raw/personality_evd/

# If repo unavailable, check:
# - Paper supplementary materials
# - HuggingFace Datasets: search "personality_evd" or "personality evidence"
```

### Raw Format

```json
{
  "dialogue": [
    {"speaker": "A", "utterance": "I always plan my days carefully."},
    {"speaker": "B", "utterance": "That's interesting, I'm more spontaneous."}
  ],
  "personality": {"A": "ISTJ", "B": "ENFP"},
  "evidence": [
    {
      "speaker": "A",
      "utterance_idx": 0,
      "trait": "Judging",
      "explanation": "Speaker A mentions planning days carefully, indicating a preference for structure."
    }
  ]
}
```

### Preprocessing (`src/data/personality_evd_parser.py`)

```python
"""
Pipeline:
1. Parse dialogue structure
2. For each speaker: concatenate their utterances as the "text"
3. Preserve evidence annotations (used for XAI evaluation ground truth)
4. Split per original paper's train/val/test if provided, else 70/15/15

Output:
  data/processed/personality_evd/{train,val,test}.jsonl
  data/processed/personality_evd/evidence_gold.jsonl  # ground truth evidence
"""
```

---

## Unified Data Format

All parsers produce the same JSONL schema:

```jsonl
{
  "id": "mbti_00001",
  "text": "cleaned concatenated text...",
  "label_mbti": "INTP",
  "label_mbti_dimensions": {"IE": "I", "SN": "N", "TF": "T", "JP": "P"},
  "label_ocean": {"O": "HIGH", "C": "LOW", "E": "LOW", "A": "HIGH", "N": "HIGH"},
  "source": "mbti",
  "split": "train",
  "metadata": {
    "user_id": "u123",
    "num_posts": 50,
    "avg_post_length": 87.3
  },
  "evidence_gold": null
}
```

Fields are `null` when not available from the source dataset.

## Running the Full Preprocessing Pipeline

```bash
# All datasets
python scripts/preprocess_data.py --config configs/data_config.yaml --all

# Individual dataset
python scripts/preprocess_data.py --dataset mbti
python scripts/preprocess_data.py --dataset pandora
python scripts/preprocess_data.py --dataset essays
python scripts/preprocess_data.py --dataset personality_evd

# Verify outputs
python scripts/preprocess_data.py --verify
# → prints row counts, label distributions, sample records
```

### `configs/data_config.yaml`

```yaml
datasets:
  mbti:
    raw_path: data/raw/mbti/mbti_1.csv
    output_dir: data/processed/mbti
    max_tokens: 512
    min_words: 10
    remove_type_mentions: true
    split_ratio: [0.70, 0.15, 0.15]
    seed: 42

  pandora:
    raw_path: data/raw/pandora/
    output_dir: data/processed/pandora
    max_comments_per_user: 100
    ocean_threshold: 3.0
    split_ratio: [0.70, 0.15, 0.15]
    seed: 42

  essays:
    raw_path: data/raw/essays/essays.csv
    output_dir: data/processed/essays
    encoding: latin-1
    split_ratio: [0.70, 0.15, 0.15]
    seed: 42

  personality_evd:
    raw_path: data/raw/personality_evd/
    output_dir: data/processed/personality_evd
    use_original_split: true
```

## Dataset Statistics (Expected)

| Dataset | Train | Val | Test | Classes |
|---------|-------|-----|------|---------|
| MBTI (16-class) | ~6,000 | ~1,300 | ~1,300 | 16 (imbalanced: INFP/INTP dominant) |
| MBTI (4-dim binary) | ~6,000 | ~1,300 | ~1,300 | 2 per dimension |
| Pandora (per trait) | ~7,000 | ~1,500 | ~1,500 | 2 (HIGH/LOW) per OCEAN trait |
| Essays (per trait) | ~1,728 | ~370 | ~370 | 2 (HIGH/LOW) per OCEAN trait |
| Personality Evd | varies | varies | varies | MBTI + evidence annotations |

### Class Imbalance Note

The MBTI dataset is heavily skewed toward IN** types (INFP ~21%, INTP ~14%). Use:
- Stratified splits (already specified)
- Class-weighted loss (`class_weight='balanced'` for sklearn, `weight` tensor for PyTorch)
- Report both macro and weighted F1
