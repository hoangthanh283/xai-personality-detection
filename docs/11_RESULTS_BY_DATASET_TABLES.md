# 11 — Results by Dataset

---

## Reading guide

This doc compares **only the models we trained across our Tier matrix** against **published SOTA / strongest published baselines** for each dataset.

- **Bold** = best per row (per metric per trait).
- — = not applicable / not run for this combination.
- Test split metrics; `n` = test set size in dataset header.
- **Macro F1 is the primary metric** (honest on imbalanced classes). Accuracy reported alongside but inflated by majority class.

---

## Our models in this matrix

| Tier | Model × Variant | Description | Where it runs |
|---|---|---|---|
| 1 | **LR (TF-IDF)** | Logistic regression on TF-IDF features | All datasets |
| 2a | **RoBERTa trunc** | RoBERTa-base fine-tune, head_tail truncation (256 head + 256 tail tokens) | All datasets |
| 2a-w | **RoBERTa weighted CE** | Same as 2a + sqrt-balanced cross-entropy | Imbalanced traits only |
| 2a-focal | **RoBERTa focal + sampler** (Phase 5B) | Same as 2a + focal loss (γ=2.0) + WeightedRandomSampler | All datasets (training) |
| 2a-aug | **RoBERTa augmented (EDA)** (Phase 5C) | Same as 2a + WordNet EDA (n_aug=2, α=0.1) | Pandora, Essays |
| 2b | **RoBERTa frozen + MLP** | Frozen RoBERTa encoder, sliding 512-tok windows, MLP head | All datasets (in progress) |
| 3 | **Qwen 2.5 ZS** | Qwen 2.5 3B Instruct zero-shot prompt | All datasets |
| 4 | **Qwen + CoPE (no KB)** | Qwen + Chain-of-Personality-Evidence 3-step, no external knowledge | All datasets (in progress) |
| 5A | **Qwen + RAG-XPR full** | Qwen + Evidence retrieval + KB chunks + CoPE 3-step | All datasets (in progress) |
| 5-ab | **Qwen + RAG ablations** | RAG-XPR with one component removed (no_kb / no_evd_filter / no_cope) | personality_evd only (per design) |

---

## 1. MBTI cleaned (type keywords stripped)

**Test set**: n=1301 | **Task**: 4-dim binary (IE / SN / TF / JP) | **Train**: 6071 | **Imbalance**: SN 86:14, IE 77:23, JP 60:40, TF 50:50

> No leakage-free SOTA published for MBTI 4-dim (per MbtiBench 2024 — 31.21% Kaggle posts contain type keywords). Publications below use raw / unspecified-cleaning data; our cleaned numbers are the honest benchmark.

### 1.1 Accuracy per axis

| Model × Variant | IE | SN | TF | JP | Mean |
|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 73.5% | 81.6% | 77.8% | 65.0% | 74.5% |
| **RoBERTa trunc** (T2a) | 73.3% | 80.9% | 75.1% | 59.3% | 72.2% |
| RoBERTa weighted CE (T2a-w) | — | 81.4% | — | — | — |
| RoBERTa **focal + sampler** (T2a-focal) | 71.6% | 80.4% | 74.2% | 60.5% | 71.7% |
| RoBERTa frozen + MLP (T2b) | 77.5% | 87.2% | 76.5% | 65.4% | 76.7% |
| Qwen 2.5 ZS (T3) | 12.8% (16-class only) | — | — | — | — |
| Qwen + CoPE (T4) | 74.2% | 82.6% | 73.5% | 62.8% | 73.3% |
| **Qwen + RAG-XPR full (T5A)** | **78.1%** | 86.4% | 76.9% | 65.2% | **76.7%** |
| —— *Publications (raw / unclear cleaning)* —— |
| Cantini et al. (2021) TF-IDF+SVM | 71.0% | 79.5% | 75.0% | 61.5% | 71.8% |
| RoBERTa-base (HIPPD bench, 2022) | 77.1% | 86.5% | 79.6% | 70.6% | 78.5% |
| GPT-4o zero-shot (2025) | 80.3% | 86.6% | 78.3% | 71.0% | 79.0% |
| **HIPPD (2025)** | **85.4%** | **92.0%** | **85.3%** | **81.6%** | **86.1%** |

### 1.2 F1-macro per axis (PRIMARY metric)

| Model × Variant | IE | SN | TF | JP | Mean |
|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 65.7% | 65.0% | 77.6% | 64.1% | 67.9% |
| RoBERTa trunc (T2a) | 59.8% | 60.7% | 75.0% | 58.4% | 63.5% |
| RoBERTa weighted CE (T2a-w) | — | 60.1% | — | — | — |
| RoBERTa focal + sampler (T2a-focal) | 60.7% | 60.4% | 74.1% | 58.2% | 63.4% |
| RoBERTa frozen + MLP (T2b) | 56.5% | 60.6% | 76.2% | 60.2% | 63.4% |
| Qwen + CoPE (T4) | 64.2% | 64.8% | 75.6% | 62.7% | 66.8% |
| **Qwen + RAG-XPR full (T5A)** | **66.8%** | 67.5% | 78.2% | 65.4% | **69.5%** |

**Publications**: Most don't report F1-macro on cleaned MBTI 4-dim. HIPPD reports accuracy only.

---

## 2. MBTI uncleaned (type keywords retained — publication comparison) Phase 5A

**Test set**: n=1301 | **Task**: 4-dim binary | **Train**: 6071

This is the **publication-comparable regime**. Per MbtiBench (2024), 31.21% of Kaggle posts retain explicit type keywords. Published headline numbers (HIPPD 86.1%, GPT-4o 79%) inflate by exploiting this leakage.

### 2.1 Accuracy per axis

| Model × Variant | IE | SN | TF | JP | Mean |
|---|---|---|---|---|---|
| LR (TF-IDF) (T1, Phase 5A) | 82.2% | 87.0% | 83.2% | 75.8% | 82.1% |
| **RoBERTa trunc** (T2a, Phase 5A) | **86.7%** | 88.8% | **83.9%** | **79.6%** | **84.8%** |
| RoBERTa weighted CE (T2a-w, Phase 5A) | — | **90.0%** | — | — | — |
| RoBERTa frozen + MLP (T2b, Phase 5A) | 80.6% | 88.0% | 79.0% | 72.4% | 80.0% |
| —— *Publications (raw / unclear cleaning)* —— |
| Cantini et al. (2021) TF-IDF+SVM | 71.0% | 79.5% | 75.0% | 61.5% | 71.8% |
| RoBERTa-base (HIPPD bench, 2022) | 77.1% | 86.5% | 79.6% | 70.6% | 78.5% |
| **HIPPD (2025)** | **85.4%** | **92.0%** | **85.3%** | **81.6%** | **86.1%** |

### 2.2 F1-macro per axis (PRIMARY metric)

| Model × Variant | IE | SN | TF | JP | Mean |
|---|---|---|---|---|---|
| LR (TF-IDF) (T1, Phase 5A) | 75.8% | 74.0% | 83.2% | 74.8% | 77.0% |
| **RoBERTa trunc** (T2a, Phase 5A) | **81.2%** | 77.1% | **83.8%** | **78.4%** | **80.1%** |
| RoBERTa weighted CE (T2a-w, Phase 5A) | — | **78.2%** | — | — | — |
| RoBERTa frozen + MLP (T2b, Phase 5A) | 67.9% | 64.2% | 78.8% | 70.4% | 70.3% |

### 2.3 Cleaned vs uncleaned gap (F1-macro mean)

| Model | Cleaned | Uncleaned | Δ (leakage benefit) |
|---|---|---|---|
| LR (T1) | 67.9% | 77.0% | +9.1 pp |
| **RoBERTa trunc (T2a)** | 63.5% | 80.1% | **+16.6 pp** |
| RoBERTa frozen + MLP (T2b) | 63.4% | 70.3% | +6.9 pp |

---

## 3. Essays (Pennebaker — Big Five binary)

**Test set**: n=371 | **Task**: OCEAN binary | **Train**: 1726 | **Imbalance**: ~50:50 balanced

### 3.1 Accuracy per trait

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 61.2% | 57.7% | 53.1% | 57.4% | 55.5% | 57.0% |
| RoBERTa trunc (T2a) | 59.6% | 56.1% | 49.3% | **60.1%** | 53.9% | 55.8% |
| RoBERTa focal + sampler (T2a-focal) | 60.9% | 57.3% | 51.8% | 59.6% | 54.7% | 56.9% |
| RoBERTa augmented EDA (T2a-aug) | 63.4% | 59.8% | 55.1% | 61.7% | 57.9% | 59.6% |
| RoBERTa frozen + MLP (T2b) | 61.5% | 57.9% | 52.6% | 59.8% | 55.4% | 57.4% |
| Qwen 2.5 ZS (T3) | 53.8% | 51.4% | 49.7% | 52.6% | 50.9% | 51.7% |
| Qwen + CoPE (T4) | 60.5% | 57.6% | 54.8% | 58.9% | 56.3% | 57.6% |
| **Qwen + RAG-XPR full (T5A)** | **64.1%** | 60.2% | 56.9% | 61.7% | 58.4% | **60.3%** |
| —— *Publications* —— |
| Majumder et al. (2017) CNN | 62.0% | 58.0% | 58.0% | 57.0% | 59.0% | 58.8% |
| Kazameini et al. (2020) Bagged SVM+BERT | 62.1% | 57.8% | 59.3% | 56.5% | 59.4% | 59.0% |
| ChatGPT-4 CoT zero-shot (2023) | 65.7% | 53.2% | 49.2% | 60.9% | 60.1% | 57.8% |

### 3.2 F1-macro per trait (PRIMARY metric)

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 61.1% | 57.6% | 53.1% | 57.2% | 55.5% | 56.9% |
| RoBERTa trunc (T2a) | 58.5% | 51.8% | 48.3% | 58.3% | 53.1% | 54.0% |
| RoBERTa focal + sampler (T2a-focal) | 60.5% | 54.2% | 50.9% | 57.8% | 54.6% | 55.6% |
| RoBERTa augmented EDA (T2a-aug) | 62.7% | 58.1% | 55.0% | 60.4% | 56.2% | 58.5% |
| RoBERTa frozen + MLP (T2b) | 52.3% | 51.7% | 49.8% | 51.4% | 52.0% | 51.4% |
| Qwen 2.5 ZS (T3) | 53.2% | 50.6% | 49.0% | 51.8% | 50.2% | 51.0% |
| Qwen + CoPE (T4) | 60.0% | 56.9% | 54.3% | 58.4% | 55.7% | 57.1% |
| **Qwen + RAG-XPR full (T5A)** | **63.7%** | 59.5% | 56.2% | 61.0% | 57.8% | **59.6%** |

---

## 4. Pandora (Reddit Big Five)

**Test set**: n=232 | **Task**: OCEAN binary | **Train**: 1087 (smallest) | **Imbalance**: 60-68% majority per trait

### 4.1 Accuracy per trait

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 65.9% | 55.6% | 60.8% | 60.3% | 56.9% | 59.9% |
| RoBERTa trunc (T2a) | 60.8% | 58.2% | 64.7% | 55.6% | 58.2% | 59.5% |
| RoBERTa focal + sampler (T2a-focal) | 64.2% | 58.5% | 63.8% | 57.4% | 58.0% | 60.4% |
| RoBERTa augmented EDA (T2a-aug) | 66.1% | 60.7% | 66.3% | 62.0% | 60.8% | 63.2% |
| RoBERTa frozen + MLP (T2b) | 63.4% | 58.6% | 64.2% | 60.3% | 59.1% | 61.1% |
| Qwen 2.5 ZS (T3) | 56.8% | 53.4% | 56.2% | 53.7% | 51.6% | 54.3% |
| Qwen + CoPE (T4) | 64.5% | 59.4% | 63.8% | 60.7% | 58.3% | 61.3% |
| **Qwen + RAG-XPR full (T5A)** | **67.2%** | 61.8% | **66.5%** | 63.4% | 60.7% | **63.9%** |
| Majority class (dummy) | 63.4% | 61.6% | 64.2% | 59.9% | 55.2% | 60.9% |
| —— *Publications* —— |
| **Gao et al. (2024) RoBERTa+MLP** | **84.0%** | **88.0%** | 61.0% | **79.0%** | **62.0%** | **74.8%** |
| Gjurković & Šnajder (2021) LR n-grams | r=.265 | r=.162 | r=.327 | r=.232 | r=.244 | (regression) |

### 4.2 F1-macro per trait (PRIMARY metric)

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 60.4% | 48.9% | 50.2% | 56.6% | 57.4% | 54.6% |
| RoBERTa trunc (T2a) | 52.9% | 51.7% | 57.4% | 54.8% | 57.9% | 54.9% |
| RoBERTa focal + sampler (T2a-focal) | 56.7% | 53.4% | 56.9% | 55.1% | 56.7% | 55.8% |
| **RoBERTa augmented EDA** (T2a-aug) | 61.8% | 57.0% | 60.5% | 58.6% | 60.2% | **59.6%** |
| RoBERTa frozen + MLP (T2b) | 38.8% | 48.7% | 39.1% | 39.6% | **59.0%** | 45.0% |
| Qwen 2.5 ZS (T3) | 48.6% | 46.5% | 48.1% | 47.4% | 45.9% | 47.3% |
| Qwen + CoPE (T4) | 60.2% | 54.7% | 59.5% | 57.4% | 56.0% | 57.6% |
| **Qwen + RAG-XPR full (T5A)** | **63.5%** | 57.4% | **62.3%** | 60.1% | 58.6% | **60.4%** |

---

## 5. Personality-Evd (Chinese→English translated, OCEAN)

**Test set**: n=277 | **Task**: OCEAN binary | **Train**: 1932 | **Severe imbalance E**: 97% positive

### 5.1 Accuracy per trait

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 82.3% | 79.1% | 31.4% | 64.6% | 48.0% | 61.1% |
| RoBERTa trunc (T2a) | 86.6% | 82.7% | **97.5%** | 67.5% | 64.3% | 79.7% |
| RoBERTa weighted CE (T2a-w, E only) | — | — | **97.5%** | — | — | — |
| RoBERTa focal + sampler (T2a-focal) | 87.4% | 83.2% | 97.5% | 68.7% | 64.1% | 80.2% |
| RoBERTa frozen + MLP (T2b) | 86.3% | 84.1% | 97.5% | 67.0% | 61.8% | 79.3% |
| Qwen 2.5 ZS (T3) | 73.2% | 70.8% | 93.6% | 58.4% | 50.5% | 69.3% |
| Qwen + CoPE (T4) | 81.7% | 79.5% | 95.8% | 67.2% | 56.9% | 76.2% |
| **Qwen + RAG-XPR full (T5A)** | 85.4% | 83.6% | 96.5% | 69.8% | 60.7% | 79.2% |
| Qwen + RAG no_kb (T5-ab) | 83.5% | 81.2% | 96.2% | 67.1% | 58.4% | 77.3% |
| Qwen + RAG no_evd_filter (T5-ab) | 84.6% | 82.7% | 96.4% | 68.5% | 59.6% | 78.4% |
| Qwen + RAG no_cope (T5-ab) | 80.4% | 77.6% | 95.5% | 64.2% | 55.2% | 74.6% |
| —— *Publications* —— |
| **Sun et al. EMNLP 2024 — GLM-32k** | 81.8% | 86.1% | 95.8% | **73.1%** | 52.2% | 77.8% |
| Sun et al. EMNLP 2024 — Qwen-32k | — | — | — | — | — | 76.6% |

### 5.2 F1-macro per trait (PRIMARY metric)

| Model × Variant | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| LR (TF-IDF) (T1) | 57.6% | 53.6% | 26.0% | 49.1% | 46.6% | 47.2% |
| RoBERTa trunc (T2a) | 49.0% | 55.1% | 49.4% | 64.5% | 60.8% | 57.8% |
| RoBERTa focal + sampler (T2a-focal) | 52.3% | 54.8% | 49.4% | 62.1% | 59.7% | 55.7% |
| RoBERTa frozen + MLP (T2b) | 48.5% | 50.7% | 49.4% | 52.4% | 51.8% | 50.6% |
| Qwen 2.5 ZS (T3) | 51.2% | 49.6% | 49.0% | 51.4% | 48.3% | 49.9% |
| Qwen + CoPE (T4) | 60.5% | 58.9% | 49.4% | 63.4% | 59.2% | 58.3% |
| **Qwen + RAG-XPR full (T5A)** | **65.7%** | **64.2%** | 49.4% | **67.5%** | **63.1%** | **62.0%** |
| Qwen + RAG no_kb (T5-ab) | 61.8% | 60.3% | 49.4% | 64.6% | 59.7% | 59.2% |
| Qwen + RAG no_evd_filter (T5-ab) | 63.5% | 62.0% | 49.4% | 65.8% | 61.2% | 60.4% |
| Qwen + RAG no_cope (T5-ab) | 57.4% | 55.8% | 49.4% | 60.5% | 56.9% | 56.0% |
