# 03 — Baseline Models

**Last updated:** 2026-04-18

## Overview

We train baselines across three model families to establish comparison benchmarks:

1. **Classical ML** (`src/baselines/ml_baselines.py`): TF-IDF + {LogReg, SVM, Naive Bayes, XGBoost, Random Forest}
2. **BiLSTM + Attention** (`src/baselines/lstm_baseline.py`): BiLSTM with attention pooling; supports GloVe 6B init
3. **Transformer** (`src/baselines/transformer_baseline.py`): DistilBERT, RoBERTa, XLM-R, multilingual-DistilBERT

All baselines are run on **4 datasets**:
- **MBTI** — 16-class + 4-dim binary (IE / SN / TF / JP)
- **Essays** — OCEAN binary (O / C / E / A / N)
- **Pandora** — OCEAN binary (5 traits)
- **personality_evd** — OCEAN binary (Chinese dialogues; multilingual models required)

> **Status:** All baselines complete with 110+ W&B runs.

> **Regime:** 100% leakage-free. MBTI type mentions stripped (0/8,675 users retain type keywords, verified). See `src/utils/text_utils.clean_text_pipeline`.

---

## 1. Classical ML Baselines

### Implementation: `src/baselines/ml_baselines.py`

```python
"""
Core class: MLBaselineTrainer

Usage:
    trainer = MLBaselineTrainer(config)
    trainer.fit(train_texts, train_labels)
    metrics = trainer.evaluate(test_texts, test_labels)
    trainer.save("outputs/models/tfidf_lr_mbti.pkl")
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs"
    ),
    "svm": LinearSVC(
        max_iter=5000, class_weight="balanced", C=1.0
    ),
    "naive_bayes": MultinomialNB(alpha=1.0),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced"
    ),
}

TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 3,
    "max_df": 0.95,
}
```

### Hyperparameter Search Space

```python
GRID = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "svm": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "naive_bayes": {"clf__alpha": [0.1, 0.5, 1.0, 2.0]},
    "xgboost": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1],
    },
    "random_forest": {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 20, 50],
    },
}
```

### Training Commands

```bash
# Single model
python scripts/train_baseline.py \
  --model logistic_regression \
  --dataset mbti \
  --task 16class \
  --config configs/baseline_config.yaml

# All ML baselines on all tasks
python scripts/train_baseline.py \
  --model all_ml \
  --dataset mbti \
  --task all \
  --grid_search

# Specific: SVM on Essays Big Five
python scripts/train_baseline.py \
  --model svm \
  --dataset essays \
  --task ocean_binary
```

### Ensemble Baseline

After training individual models, build an ensemble:

```python
"""
Ensemble: Soft voting over {LR, SVM, XGBoost, RF}
Implementation in ml_baselines.py: EnsembleClassifier

Uses sklearn VotingClassifier with 'soft' voting for models that
support predict_proba, falls back to 'hard' for LinearSVC.
"""

# Command:
python scripts/train_baseline.py \
  --model ensemble \
  --dataset mbti \
  --task 16class \
  --ensemble_members logistic_regression,xgboost,random_forest
```

### Actual Results (MBTI 16-class, cleaned data)

> **Important**: The MBTI Kaggle dataset has 47× class imbalance (INFP 21% vs ESTJ 0.4%). After removing type-mention leakage (the realistic/fair benchmark), 16-class accuracy for classical ML caps around 37%. Papers reporting 88–92% use leaky data — MbtiBench (2024) confirmed 31% of Kaggle MBTI posts contain type keywords. The 4-dim binary tasks (IE/SN/TF/JP) are the reliable benchmarks.

| Model | Accuracy | F1 (macro) | Notes |
|-------|:--------:|:----------:|-------|
| TF-IDF + LR | 32.1% | 21.8% | |
| TF-IDF + SVM | **37.0%** | 17.2% | best classical |
| TF-IDF + NB | 26.7% | 5.9% | majority-class collapse |
| TF-IDF + XGBoost | 33.6% | 11.1% | |
| TF-IDF + RF | 27.4% | 6.4% | |
| BiLSTM + Attn | 25.2% | 7.3% | random init; GloVe did not help (53% vocab coverage) |
| DistilBERT | 27.4% | 13.0% | |
| RoBERTa | **29.7%** | 9.9% | best transformer |

### Actual Results (MBTI 4-dim binary — primary benchmark)

| Model | IE | SN | TF | JP | Mean |
|-------|:--:|:--:|:--:|:--:|:----:|
| TF-IDF + LR | 73.5% | 81.6% | 77.8% | 65.0% | 74.5% |
| **TF-IDF + SVM** | **77.9%** | **86.9%** | **78.6%** | 65.6% | **77.2%** |
| TF-IDF + NB | 76.9% | 86.1% | 74.1% | 60.9% | 74.5% |
| TF-IDF + XGBoost | 77.8% | 86.2% | 76.2% | 66.5% | 76.7% |
| TF-IDF + RF | 76.9% | 86.1% | 71.9% | 61.0% | 74.0% |
| BiLSTM + Attn | 75.6% | 86.0% | 70.0% | 60.9% | 73.1% |
| DistilBERT | 76.6% | 86.1% | 73.2% | 61.8% | 74.4% |
| **RoBERTa** | 77.7% | 86.1% | 74.1% | **61.8%** | 74.9% |

### Actual Results (Essays OCEAN — 5 traits, mean accuracy)

| Model | O | C | E | A | N | Mean |
|-------|:-:|:-:|:-:|:-:|:-:|:----:|
| TF-IDF + LR | 61.2% | 57.7% | 53.1% | 57.4% | 55.5% | 57.0% |
| **TF-IDF + SVM** | 60.6% | 56.1% | 56.6% | 59.0% | 55.5% | **57.6%** |
| DistilBERT | 61.2% | 57.1% | 57.4% | 56.1% | 54.4% | 57.3% |
| BiLSTM + Attn (GloVe) | 64.7% | 56.9% | 51.2% | 53.4% | 52.3% | 55.7% |
| RoBERTa | 59.6% | 56.1% | 49.3% | 60.1% | 53.9% | 55.8% |

### Actual Results (Pandora OCEAN)

> **Majority-class baseline: 60.9%.** All models hit this ceiling — the 232-sample test set + 60–68% class skew leaves almost no room to beat the dummy.

| Model | Mean Acc | Mean F1 |
|-------|:--------:|:-------:|
| Majority dummy | 60.9% | ~38% |
| SVM | 60.8% | 49.2% |
| DistilBERT | 61.5% | 40.8% |
| RoBERTa | 60.9% | 37.8% |
| BiLSTM + Attn | **62.1%** | 44.8% |
| **Published SOTA (RoBERTa+MLP, arXiv:2406.16223)** | **74.8%** | **68.0%** |

Full per-dataset runs are recorded in W&B for metric-level traceability.

---

## 1.5 BiLSTM + Attention Baseline

### Implementation: `src/baselines/lstm_baseline.py`

Bridges the gap between classical ML (shallow) and Transformers (heavy). Provides a sequence-model baseline that's cheap to train and easy to understand.

**Architecture:**
- Word-level tokenizer (top-30K frequency vocab, pickle save/load)
- Optional GloVe 6B 300d embedding init (`glove_path` config; ~53% vocab coverage for MBTI)
- 2-layer BiLSTM, hidden_dim=256, bidirectional
- Attention pooling over LSTM outputs (masked by padding)
- Linear classification head

**Key training features:**
- `sqrt_balanced` class weighting: `w = sqrt(N / (K * n_c))`, clipped to [0.5, 2.0] — stabilises training on 47× imbalance without minority-class over-correction
- Per-epoch W&B logging (`train_loss`, `eval_accuracy`, `eval_f1_macro`)
- `ReduceLROnPlateau` scheduler + gradient clipping (max_norm=1.0)
- Early stopping (patience=5)

### Training command

```bash
# Download GloVe 6B embeddings (one-time)
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/download_embeddings.py --dim 300

# Train LSTM on any dataset
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/train_baseline.py \
    --model lstm \
    --dataset mbti \
    --task 4dim
```

### Observations

- GloVe helped Essays (+1.3 pp mean) and marginally helped 4-dim (+1.2 pp mean acc)
- GloVe did **not** help 16-class (24.2% vs 25.2% random init) — personality community-specific vocab not covered by GloVe
- On MBTI 4-dim, BiLSTM+Attn matches DistilBERT within 1.3 pp mean accuracy

---

## 2. Transformer Baselines

### Implementation: `src/baselines/transformer_baseline.py`

```python
"""
Uses HuggingFace Trainer API.

Supports two classification modes:
  - "16class": Single 16-way softmax (CrossEntropyLoss)
  - "4dim": Four binary classifiers (one per MBTI dimension)

Models: distilbert-base-uncased, roberta-base
"""

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
```

### Hyperparameters

```yaml
# configs/baseline_config.yaml
transformer:
  distilbert:
    model_name: distilbert-base-uncased
    max_length: 512
    batch_size: 32
    learning_rate: 2e-5
    weight_decay: 0.01
    num_epochs: 10
    warmup_ratio: 0.1
    lr_scheduler: linear
    early_stopping_patience: 3
    fp16: true          # Mixed precision on GPU
    gradient_accumulation_steps: 1

  roberta:
    model_name: roberta-base
    max_length: 512
    batch_size: 16      # Larger model → smaller batch
    learning_rate: 1e-5
    weight_decay: 0.01
    num_epochs: 8
    warmup_ratio: 0.1
    lr_scheduler: linear
    early_stopping_patience: 3
    fp16: true
    gradient_accumulation_steps: 2
```

### Training Commands

```bash
# DistilBERT on MBTI 16-class
python scripts/train_baseline.py \
  --model distilbert \
  --dataset mbti \
  --task 16class \
  --output_dir outputs/models/distilbert_mbti_16class \
  --wandb_project rag-xpr-baselines

# RoBERTa on MBTI 4-dimension binary
python scripts/train_baseline.py \
  --model roberta \
  --dataset mbti \
  --task 4dim \
  --output_dir outputs/models/roberta_mbti_4dim \
  --wandb_project rag-xpr-baselines

# DistilBERT on Essays (Big Five, 5 separate binary classifiers)
python scripts/train_baseline.py \
  --model distilbert \
  --dataset essays \
  --task ocean_binary \
  --output_dir outputs/models/distilbert_essays_ocean

# Resume from checkpoint
python scripts/train_baseline.py \
  --model distilbert \
  --dataset mbti \
  --task 16class \
  --resume_from outputs/models/distilbert_mbti_16class/checkpoint-5000
```

### 4-Dimension Binary Training

For MBTI 4-dim, train 4 independent binary classifiers:

```python
"""
Loop over dimensions: IE, SN, TF, JP
Each gets its own model, tokenizer, and training run.
Final MBTI type = concatenation of 4 dimension predictions.

Example: I + N + T + P → "INTP"
"""

DIMENSIONS = ["IE", "SN", "TF", "JP"]
for dim in DIMENSIONS:
    train_binary_classifier(
        model_name="distilbert-base-uncased",
        train_data=filter_dimension(train_data, dim),
        label_col=f"label_{dim}",
        output_dir=f"outputs/models/distilbert_mbti_{dim}",
    )
```

### Actual Results (Transformer, cleaned data)

| Model | Dataset | Task | Accuracy | F1 (macro) |
|-------|---------|------|:--------:|:----------:|
| DistilBERT | MBTI | 16-class | 27.4% | 13.0% |
| DistilBERT | MBTI | IE | 76.6% | 44.0% |
| DistilBERT | MBTI | SN | 86.1% | 46.3% |
| DistilBERT | MBTI | TF | 73.2% | 72.6% |
| DistilBERT | MBTI | JP | 61.8% | 58.6% |
| RoBERTa | MBTI | 16-class | **29.7%** | 9.9% |
| RoBERTa | MBTI | IE | **77.7%** | 50.4% |
| RoBERTa | MBTI | SN | 86.1% | 46.3% |
| RoBERTa | MBTI | TF | **74.1%** | 73.3% |
| RoBERTa | MBTI | JP | 61.8% | 53.5% |
| DistilBERT | Essays | O-N (mean) | 57.3% | 56.7% |
| RoBERTa | Essays | O-N (mean) | 55.8% | 54.0% |
| DistilBERT | Pandora | O-N (mean) | 61.5% | 40.8% |
| RoBERTa | Pandora | O-N (mean) | 60.9% | 37.8% |
| DistilBERT | personality_evd | O-N (mean) | 81.4% | 48.1% |
| RoBERTa (XLM-R) | personality_evd | O-N | _pending GPU rerun_ | _pending_ |

> **MBTI 16-class ceiling is data-limited:** The 8K-user Kaggle MBTI dataset with 47× class imbalance caps 16-class accuracy at ~30%. The 4-dim binary tasks are the primary benchmark where RoBERTa matches SVM within ~2 pp.

> **personality_evd accuracy is skew-inflated:** E-trait 97.5% HIGH labels produce misleading accuracy numbers. F1-macro (~48%) is the honest metric.

---

## 3. Training Infrastructure

### Hardware Requirements

| Model | GPU VRAM | Training Time (MBTI 16-class) |
|-------|----------|-------------------------------|
| TF-IDF + ML | CPU only | ~5–15 min per task |
| BiLSTM + Attn | 2 GB+ | ~3–5 min per task |
| DistilBERT | 4 GB+ (batch=16, grad_accum=2) | ~30 min – 1 h |
| RoBERTa | 5 GB+ (batch=2, grad_accum=8) | ~1–2 h |
| XLM-R (personality_evd) | 6 GB+; on 5.6 GB requires `gradient_checkpointing=true` + `batch_size=1` + `grad_accum=32` (see `scripts/rerun_roberta_personality_evd.sh`) | ~2 h |

### W&B Logging

All training scripts log to Weights & Biases:

```bash
# Set up W&B (one-time)
wandb login
export WANDB_PROJECT=rag-xpr-baselines

# Logged metrics:
# - train/loss, train/accuracy per step
# - eval/loss, eval/accuracy, eval/f1_macro per epoch
# - Best model checkpoint path
# - Hyperparameters
# - Confusion matrix (logged as W&B plot)
```

### Reproducibility Checklist

```python
# src/utils/seed.py
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

All scripts call `set_seed(42)` before any computation.

---

## 4. Post-Training: Extract Predictions for Comparison

After all baselines are trained, extract predictions on the test set:

```bash
# Generate predictions for all models on test set
python scripts/evaluate.py \
  --mode baseline_predictions \
  --models_dir outputs/models/ \
  --output outputs/predictions/baselines/

# Output format: outputs/predictions/baselines/{model}_{dataset}_{task}.jsonl
# Each line:
# {"id": "mbti_00001", "text": "...", "gold": "INTP", "pred": "INTJ", "prob": {...}}
```

These prediction files are used in [05_EXPERIMENT_PLAN.md](./05_EXPERIMENT_PLAN.md) for comparison against RAG-XPR.
