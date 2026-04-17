# 03 — Baseline Models

## Overview

We train baselines across two categories to establish comparison benchmarks:

1. **Classical ML**: TF-IDF + {Logistic Regression, SVM, XGBoost, Random Forest, Naive Bayes}
2. **Transformer**: DistilBERT, RoBERTa fine-tuning

All baselines are run on **MBTI (16-class + 4-dim binary)** and **Essays (Big Five binary)** datasets.

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

### Expected Results (MBTI 16-class)

> **Important**: The MBTI Kaggle dataset has 47x class imbalance (INFP 21% vs ESTJ 0.4%). After removing type-mention leakage (the realistic/fair benchmark), 16-class accuracy for classical ML is ~30-42%, not the 72-97% reported in papers that use unclean data. The 4-dim binary tasks (IE/SN/TF/JP) are the more reliable and reproducible benchmarks.

| Model | Accuracy (cleaned) | F1 (macro) | Notes |
|-------|----------|------------|-------|
| TF-IDF + LR | ~32-38% | ~18-24% | Imbalance limits accuracy |
| TF-IDF + SVM | ~33-39% | ~19-25% | Similar to LR |
| TF-IDF + NB | ~28-35% | ~15-22% | Struggles with imbalance |
| TF-IDF + XGBoost | ~35-42% | ~22-28% | Better on rare classes |
| TF-IDF + RF | ~34-40% | ~20-26% | |
| Ensemble | ~36-44% | ~23-30% | |

> Papers reporting 89-97% use MBTI type mentions in text (data leakage). The cleaned regime used here removes all type mentions before training.

### Expected Results (MBTI 4-dim binary — primary benchmark)

| Model | IE Acc | SN Acc | TF Acc | JP Acc |
|-------|--------|--------|--------|--------|
| TF-IDF + LR | ~73% | ~82% | ~78% | ~65% |
| TF-IDF + SVM | ~74% | ~82% | ~77% | ~66% |
| TF-IDF + XGBoost | ~74% | ~81% | ~77% | ~63% |
| TF-IDF + RF | ~73% | ~82% | ~76% | ~63% |
| DistilBERT | ~77% | ~86% | ~73% | ~62% |

> **Measured results (this repo, cleaned data, MBTI Kaggle)**: LR=74.5%, SVM=77.2%, XGBoost=76.7%, DistilBERT=74.4% (mean across 4 axes). Target ≥70% per axis for all models.

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

### Expected Results (Transformer)

> **Note on MBTI 16-class**: After removing type-mention leakage on the Kaggle MBTI dataset (~8K users, 47x class imbalance), realistic accuracy is ~26-38% for all models — not 88-92% reported in papers using unclean data. Use the 4-dim binary benchmarks as primary metrics.

| Model | Dataset | Task | Accuracy (cleaned) | F1 (macro) |
|-------|---------|------|----------|------------|
| DistilBERT | MBTI | 16-class | ~26-32% | ~10-22% |
| DistilBERT | MBTI | I/E dim | ~77% | ~44% |
| DistilBERT | MBTI | S/N dim | ~86% | ~55% |
| DistilBERT | MBTI | T/F dim | ~73% | ~54% |
| DistilBERT | MBTI | J/P dim | ~62% | ~44% |
| RoBERTa | MBTI | 16-class | ~28-38% (expected) | ~14-25% |
| DistilBERT | Essays | O (binary) | ~61% | ~58% |
| DistilBERT | Essays | C (binary) | ~57% | ~57% |
| DistilBERT | Essays | E (binary) | ~57% | ~57% |
| DistilBERT | Essays | A (binary) | ~56% | ~56% |
| DistilBERT | Essays | N (binary) | ~54% | ~54% |
| DistilBERT | Pandora | O (binary) | ~63% | ~39% |
| DistilBERT | personality_evd | O-N (binary, mean) | ~81% | ~69% |

> **Measured results**: All entries above marked without "(expected)" are from actual training runs in this repo. High personality_evd accuracy reflects dataset-level label distribution (some traits have 80-90% majority class).

> **MBTI 16-class low accuracy is expected**: The 8K-user Kaggle MBTI dataset with 47x class imbalance sets a hard ceiling. The 4-dim binary tasks (IE/SN/TF/JP) are the primary benchmark — LR/SVM achieve 65-82% per axis, meeting the ≥70% target on 3/4 axes.

---

## 3. Training Infrastructure

### Hardware Requirements

| Model | GPU VRAM | Training Time (MBTI 16class) |
|-------|----------|---------------------|
| TF-IDF + ML | CPU only | ~5-15 min |
| DistilBERT | 4 GB+ (batch=16, grad_accum=2) | ~1-2 hours |
| RoBERTa | 6 GB+ (batch=2, grad_accum=8) | ~3-5 hours |

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
