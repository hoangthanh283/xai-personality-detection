# 08 — Data Analysis & Statistics

**Last updated:** 2026-04-18

Comprehensive analysis and statistics for all ingested datasets, including raw data characteristics, preprocessing decisions, label distributions, and quality checks.

---

## 1. Dataset Overview

| Dataset | Source | Raw Size | Processed Records | Label Types | Avg Words |
|---------|--------|----------|-------------------|-------------|-----------|
| MBTI | Kaggle (Personality Café) | ~50 MB | 8,673 (6071/1301/1301) | MBTI 16-class, 4-dim binary | 1,223 |
| Pandora | Reddit (Gjurković et al., 2020) | 5.26 GB | 10,258 (7180/1539/1539) | MBTI 16-class, 4-dim binary, OCEAN binary | 1,933 |
| Essays | Pennebaker & King (1999) | ~2 MB | 2,467 (1726/370/371) | OCEAN binary | 650 |
| Personality Evd | Lei Sun et al. (2024) | ~5 MB | 1,846 (1292/277/277) | OCEAN binary, evidence gold | 10 |

---

## 2. MBTI Dataset (Kaggle)

### Raw Format
- Single CSV: `type` (16 MBTI types) + `posts` (50 posts per user joined by `|||`)
- 8,673 users

### Preprocessing
- Posts split by `|||` delimiter
- URLs, @mentions, and **MBTI type mentions removed** (data leakage prevention)
- Posts < 10 words filtered out
- Concatenated per user, truncated to 2000 words max
- Stratified split on MBTI type

### Label Distribution (Train Split)

| MBTI Type | Count | Percentage |
|-----------|-------|------------|
| INFP | 1,282 | 21.1% |
| INFJ | 1,029 | 16.9% |
| INTP | 913 | 15.0% |
| INTJ | 764 | 12.6% |
| ENTP | 479 | 7.9% |
| ENFP | 472 | 7.8% |
| ISTP | 235 | 3.9% |
| ISFP | 190 | 3.1% |
| ENTJ | 162 | 2.7% |
| ISTJ | 144 | 2.4% |
| ENFJ | 133 | 2.2% |
| ISFJ | 116 | 1.9% |
| ESTP | 62 | 1.0% |
| ESFP | 34 | 0.6% |
| ESFJ | 29 | 0.5% |
| ESTJ | 27 | 0.4% |

### Key Observations
- **Severe class imbalance**: IN** types comprise ~71% of the dataset
- **I dimension dominant**: ~78% introverts vs ~22% extraverts
- **N dimension dominant**: ~79% intuitive vs ~21% sensing
- No OCEAN labels available

---

## 3. Pandora Dataset (Reddit) — Primary

### Raw Data Format
- **`author_profiles.csv`** (1.2 MB, 10,295 authors): Contains MBTI type, Big Five percentile scores (0–100), demographics
- **`all_comments_since_2015.csv`** (5.26 GB, ~17.6M comments): Reddit comments with `author`, `body`, `lang`, `subreddit`, `word_count`

### Author Profile Statistics

| Attribute | Available | Percentage |
|-----------|-----------|------------|
| Total authors | 10,295 | 100% |
| Valid MBTI type | 9,067 | 88.1% |
| All Big Five (percentile) | 1,568 | 15.2% |
| MBTI or Big Five | 10,258 | 99.6% |

### Big Five Score Distribution (percentile scale, 0–100)

| Trait | Count | Mean | Median | Std |
|-------|-------|------|--------|-----|
| Openness | 1,588 | 62.5 | 67.0 | 23.8 |
| Conscientiousness | 1,605 | 40.2 | 36.0 | 27.3 |
| Extraversion | 1,608 | 37.4 | 29.0 | 27.1 |
| Agreeableness | 1,606 | 42.4 | 41.0 | 27.6 |
| Neuroticism | 1,603 | 49.8 | 50.0 | 27.4 |

### Preprocessing Pipeline (`src/data/pandora_parser.py`)

1. **Author filtering**: Only authors with valid MBTI type (16 types) or complete Big Five scores
2. **Comment streaming**: Chunked reading (500K rows) of 17.6M comments, filtering for target authors only — reduces memory from 5.26 GB to manageable chunks
3. **English filtering**: Only comments with `lang == "en"` (eliminates ~48% of comments)
4. **Bot filtering**: Removes comments containing "i am a bot", "automoderator", "this action was performed automatically"
5. **Reddit markup cleaning**: Removes quote lines (`>`), markdown links (`[text](url)`), headings (`#`)
6. **Standard text cleaning**: URL removal, mention removal, MBTI type mention removal (leakage prevention), punctuation normalization
7. **Quality filter**: Comments < 5 words after cleaning are discarded
8. **Sampling**: Up to 100 comments per user (random sample), concatenated into single text
9. **Final validation**: Combined text must be ≥ 5 words; truncated to 2000 words max
10. **Binarization**: OCEAN percentiles → HIGH (>50) / LOW (≤50)
11. **Stratified split**: 70/15/15 on MBTI type

### Label Composition

| Label Type | Records | Percentage |
|------------|---------|------------|
| MBTI only | 8,690 | 84.7% |
| OCEAN only | 1,191 | 11.6% |
| Both MBTI + OCEAN | 377 | 3.7% |
| Neither | 0 | 0% |

### Split Statistics

| Split | Total | MBTI Labels | OCEAN Labels | Both | Avg Words |
|-------|-------|------------|--------------|------|-----------|
| train | 7,180 | 6,347 | 1,087 | 254 | 1,931 |
| val | 1,539 | 1,360 | 249 | 70 | 1,927 |
| test | 1,539 | 1,360 | 232 | 53 | 1,941 |

### MBTI Distribution (Train)

| Type | Count | % | | Type | Count | % |
|------|-------|---|-|------|-------|---|
| INTP | 1,635 | 25.8% | | ISTP | 285 | 4.5% |
| INTJ | 1,293 | 20.4% | | ENTJ | 224 | 3.5% |
| INFP | 752 | 11.8% | | ISTJ | 137 | 2.2% |
| INFJ | 736 | 11.6% | | ENFJ | 114 | 1.8% |
| ENTP | 442 | 7.0% | | ISFP | 86 | 1.4% |
| ENFP | 432 | 6.8% | | ISFJ | 76 | 1.2% |
| | | | | ESTP | 50 | 0.8% |
| | | | | ESFP | 35 | 0.6% |
| | | | | ESTJ | 30 | 0.5% |
| | | | | ESFJ | 20 | 0.3% |

### MBTI Dimension Distribution (All Splits)

| Dimension | Dominant | Count | Minority | Count | Ratio |
|-----------|----------|-------|----------|-------|-------|
| IE | I | 7,142 | E | 1,925 | 3.71:1 |
| SN | N | 8,039 | S | 1,028 | 7.82:1 |
| TF | T | 5,851 | F | 3,216 | 1.82:1 |
| JP | P | 5,310 | J | 3,757 | 1.41:1 |

### OCEAN Binary Distribution (Train)

| Trait | HIGH | LOW | HIGH% | LOW% | Imbalance Ratio |
|-------|------|-----|-------|------|-----------------|
| Openness (O) | 725 | 362 | 66.7% | 33.3% | 2.00:1 |
| Conscientiousness (C) | 371 | 716 | 34.1% | 65.9% | 0.52:1 |
| Extraversion (E) | 342 | 745 | 31.5% | 68.5% | 0.46:1 |
| Agreeableness (A) | 421 | 666 | 38.7% | 61.3% | 0.63:1 |
| Neuroticism (N) | 506 | 581 | 46.5% | 53.5% | 0.87:1 |

### Data Quality Checks

| Check | Result |
|-------|--------|
| User ID overlap across splits | **0 users** leak between train/val/test |
| MBTI type mentions in text | **0 records** contain leaked type strings |
| Empty/missing text | **0 records** |
| Consistent split tags | All records match their file's split |
| All records have source="pandora" | Verified |

## 4. Essays Dataset (Pennebaker & King)

### Raw Format
- CSV with columns: `#AUTHID`, `cEXT`, `cNEU`, `cAGR`, `cCON`, `cOPN`, `TEXT`
- Binary labels: `y`/`n` for each Big Five trait
- Formal essay text, minimal cleaning needed

### Split Statistics

| Split | Records | Avg Words |
|-------|---------|-----------|
| train | 1,726 | 649 |
| val | 370 | 666 |
| test | 371 | 651 |

### OCEAN Binary Distribution (Train)

| Trait | HIGH | LOW | HIGH% | LOW% | Imbalance Ratio |
|-------|------|-----|-------|------|-----------------|
| Openness (O) | 889 | 837 | 51.5% | 48.5% | 1.06:1 |
| Conscientiousness (C) | 870 | 856 | 50.4% | 49.6% | 1.02:1 |
| Extraversion (E) | 887 | 839 | 51.4% | 48.6% | 1.06:1 |
| Agreeableness (A) | 919 | 807 | 53.2% | 46.8% | 1.14:1 |
| Neuroticism (N) | 881 | 845 | 51.0% | 49.0% | 1.04:1 |

### Key Observations
- **Most balanced OCEAN dataset** — all traits near 50/50 split
- Small dataset (2,467 total) — risk of overfitting
- Formal essay style — very different from Reddit/social media text

---

## 5. Personality Evd Dataset

### Raw Format
- JSON/JSONL dialogue files with speaker turns, personality labels, and evidence annotations
- Very short texts (dialogue utterances, ~10 words average)

### Split Statistics

| Split | Records | Avg Words |
|-------|---------|-----------|
| train | 1,292 | 10 |
| val | 277 | 10 |
| test | 277 | 10 |

### OCEAN Binary Distribution (Train)

| Trait | HIGH | LOW | HIGH% | LOW% | Imbalance Ratio |
|-------|------|-----|-------|------|-----------------|
| Openness (O) | 1,157 | 135 | 89.5% | 10.5% | 8.57:1 |
| Conscientiousness (C) | 1,122 | 170 | 86.8% | 13.2% | 6.60:1 |
| Extraversion (E) | 1,262 | 30 | 97.7% | 2.3% | 42.07:1 |
| Agreeableness (A) | 887 | 405 | 68.7% | 31.3% | 2.19:1 |
| Neuroticism (N) | 795 | 497 | 61.5% | 38.5% | 1.60:1 |

### Key Observations
- **Extreme class imbalance**, especially Extraversion (42:1 ratio)
- Very short texts — may require different model architectures (e.g., no truncation needed)
- Contains gold evidence annotations for XAI evaluation
- Some texts are in Chinese (dialogue dataset from Chinese social media)

---

## 6. Cross-Dataset Comparison

### OCEAN Label Availability

| Dataset | OCEAN Records | Source Scale | Imbalance Severity |
|---------|---------------|--------------|-------------------|
| Pandora | 1,087 | Percentile (0–100) | Moderate (0.46–2.00) |
| Essays | 2,467 | Binary (y/n) | Low (1.02–1.14) |
| Personality Evd | 1,292 | Binary (HIGH/LOW) | Extreme (1.60–42.07) |

### Text Length Distribution

| Dataset | Avg Words | Min | Max | Typical Source |
|---------|-----------|-----|-----|------|---------------|
| MBTI | 1,226 | ~10 | ~2,000 | Forum posts (aggregated) |
| Pandora | 1,933 | ~5 | 2,000 | Reddit comments (aggregated) |
| Essays | 650 | ~10 | ~2,000 | Formal essays |
| Personality Evd | 10 | ~1 | ~50 | Dialogue utterances |

### MBTI Label Availability

| Dataset | MBTI Records | 16-class | 4-dim binary |
|---------|-------------|----------|---------------|
| MBTI | 8,673 | Yes | Yes (IE/SN/TF/JP) |
| Pandora | 9,067 | Yes | Yes (IE/SN/TF/JP) |
| Personality Evd | 0 | No | No |
| Essays | 0 | No | No |

---

## 7. Preprocessing Configuration

### Current `configs/data_config.yaml`

```yaml
datasets:
  mbti:
    raw_path: data/raw/mbti/mbti_1.csv
    output_dir: data/processed/mbti
    max_tokens: 512
    max_words: 2000
    max_total_words: 2000
    min_words: 10
    remove_type_mentions: true
    split_ratio: [0.70, 0.15, 0.15]
    seed: 42

  pandora:
    raw_path: data/raw/pandora/
    output_dir: data/processed/pandora
    max_comments_per_user: 100
    ocean_threshold: 50.0      # Percentile scale (0-100)
    min_words: 5
    max_words: 2000
    filter_english: true
    chunk_size: 500000
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
    use_original_split: false
    drop_unknown_ocean: true
    split_ratio: [0.70, 0.15, 0.15]
    seed: 42
```

### Recommended Baseline Config (`configs/baseline_config.yaml`)

Transformer dataset overrides already configured:

```yaml
transformer:
  dataset_overrides:
    essays:
      max_length: 384
      num_epochs: 10
      early_stopping_patience: 3
    pandora:
      max_length: 256
      num_epochs: 3
      early_stopping_patience: 2
      batch_size: 32
    personality_evd:
      max_length: 64
      num_epochs: 10
      early_stopping_patience: 3
```

---

## 8. Command Reference

```bash
# Preprocess individual datasets
make data-mbti
make data-pandora
python scripts/preprocess_data.py --dataset essays
python scripts/preprocess_data.py --dataset personality_evd

# Preprocess all datasets
make data-preprocess

# Verify processed outputs
make data-verify

# Train baselines on Pandora
python scripts/train_baseline.py --model all_ml --dataset pandora --task 16class
python scripts/train_baseline.py --model all_ml --dataset pandora --task ocean_binary
python scripts/train_baseline.py --model distilbert --dataset pandora --task 4dim
python scripts/train_baseline.py --model roberta --dataset pandora --task ocean_binary
```
