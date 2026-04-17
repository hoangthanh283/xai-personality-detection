# 06 — Evaluation Protocol

## 1. Classification Metrics (Automated)

### Implementation: `src/evaluation/classification_metrics.py`

```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score
)

def compute_classification_metrics(y_true, y_pred, labels=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
```

**Reported for all methods**:
- Accuracy (overall correct / total)
- F1-macro (unweighted average across classes — penalizes poor minority-class performance)
- F1-weighted (weighted by class frequency)
- Cohen's Kappa (agreement beyond chance)
- Per-class Precision/Recall/F1 (critical for imbalanced MBTI)

**For MBTI 4-dimension binary**: report per-dimension accuracy + average.

**For Big Five (OCEAN)**: report per-trait accuracy + average.

---

## 2. Explainability (XAI) Metrics

### 2.1 Automated XAI Metrics

**Implementation**: `src/evaluation/xai_metrics.py`

#### a) Evidence Grounding Score

Measures whether cited evidence actually exists in the input text.

```python
def evidence_grounding_score(predictions: list[dict]) -> float:
    """
    For each prediction, check if every quoted evidence string
    can be found (fuzzy match) in the original input text.

    Returns: ratio of grounded evidence / total evidence cited
    """
    grounded, total = 0, 0
    for pred in predictions:
        for ev in pred["evidence_chain"]:
            total += 1
            if fuzzy_match(ev["evidence"], pred["text"], threshold=0.85):
                grounded += 1
    return grounded / total if total > 0 else 0.0
```

#### b) Evidence Relevance (vs. Gold Standard)

Only available for Personality Evd dataset which has gold evidence annotations.

```python
def evidence_relevance_f1(pred_evidence, gold_evidence):
    """
    Token-level overlap between predicted and gold evidence.
    Computed per-sample, then macro-averaged.
    """
    # Use ROUGE-L or token F1 between predicted and gold evidence spans
    scores = []
    for pred, gold in zip(pred_evidence, gold_evidence):
        pred_tokens = set(tokenize(pred))
        gold_tokens = set(tokenize(gold))
        precision = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(pred_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        scores.append(f1)
    return np.mean(scores)
```

#### c) Faithfulness Score

Measures whether removing the cited evidence changes the prediction (perturbation-based).

```python
def faithfulness_score(pipeline, predictions, n_samples=100):
    """
    For each sample:
    1. Run pipeline on original text → prediction P
    2. Remove the cited evidence sentences from text
    3. Run pipeline on modified text → prediction P'
    4. If P ≠ P', the evidence was faithful (actually influenced the decision)

    Returns: ratio of faithful predictions
    """
    faithful = 0
    for pred in random.sample(predictions, n_samples):
        modified_text = remove_evidence(pred["text"], pred["evidence_chain"])
        new_pred = pipeline.predict(modified_text)
        if new_pred.label != pred["predicted_label"]:
            faithful += 1
    return faithful / n_samples
```

#### d) Explanation Consistency

Checks if the explanation text is logically consistent with the label.

```python
def explanation_consistency(predictions, llm_client):
    """
    Use LLM-as-judge: Given ONLY the explanation (not the text),
    ask a separate LLM to predict the personality type.
    If it matches → explanation is self-consistent.
    """
    consistent = 0
    for pred in predictions:
        judge_prompt = f"""
        Based on this personality analysis explanation, what MBTI type
        would you predict?

        Explanation: {pred['explanation']}

        Reply with just the 4-letter MBTI type.
        """
        judge_pred = llm_client.generate([{"role": "user", "content": judge_prompt}])
        if judge_pred.strip().upper() == pred["predicted_label"]:
            consistent += 1
    return consistent / len(predictions)
```

### 2.2 SHAP/LIME for Baseline Comparison

For transformer baselines, extract feature attributions to compare against RAG-XPR evidence.

```python
# Using SHAP for DistilBERT
import shap

def compute_shap_explanations(model, tokenizer, texts, n_samples=100):
    """
    Generate SHAP token-level attributions for transformer baseline.
    These are compared qualitatively against RAG-XPR evidence chains.
    """
    explainer = shap.Explainer(
        lambda x: model_predict_proba(model, tokenizer, x),
        tokenizer=tokenizer,
        output_names=MBTI_TYPES,
    )
    shap_values = explainer(texts[:n_samples])
    return shap_values
```

---

## 3. Human Evaluation

### 3.1 Survey Design

**Implementation**: `src/evaluation/human_eval.py` → generates evaluation forms

**Evaluators**: 3-5 psychology-aware annotators (team members + recruited if possible)

**Sample size**: 50 randomly selected test samples, evaluated by all annotators

#### Evaluation Criteria (1-5 Likert Scale)

| Criterion | Question | Scale |
|-----------|----------|-------|
| **Relevance** | "Is the cited evidence relevant to personality assessment?" | 1=irrelevant, 5=highly relevant |
| **Correctness** | "Is the personality state correctly identified from the evidence?" | 1=wrong, 5=perfectly correct |
| **Coverage** | "Does the explanation cover the most important personality signals in the text?" | 1=misses everything, 5=comprehensive |
| **Faithfulness** | "Does the explanation accurately reflect what the model predicted (no contradiction)?" | 1=contradictory, 5=perfectly faithful |
| **Usefulness** | "Would this explanation help a psychologist/recruiter trust the prediction?" | 1=useless, 5=very useful |

#### Evaluation Form Template

```
Sample ID: ___
Input Text: [shown]
Predicted Type: [shown]
Gold Type: [shown]

--- Method A (randomized, blind) ---
Evidence: [list of evidence quotes]
States: [identified psychological states]
Explanation: [natural language explanation]

Relevance:    [1] [2] [3] [4] [5]
Correctness:  [1] [2] [3] [4] [5]
Coverage:     [1] [2] [3] [4] [5]
Faithfulness: [1] [2] [3] [4] [5]
Usefulness:   [1] [2] [3] [4] [5]

Free-text comments: ___
```

### 3.2 Methods Compared in Human Eval

1. **RAG-XPR (ours)**: Full evidence chain + explanation
2. **LLM + CoT (no RAG)**: CoT explanation without KB grounding
3. **LLM zero-shot**: Plain LLM explanation
4. **SHAP (DistilBERT)**: Top-k token attributions

Methods are presented in randomized order, blind (evaluator doesn't know which method).

### 3.3 Inter-Annotator Agreement

```python
from sklearn.metrics import cohen_kappa_score
import krippendorff

def compute_agreement(annotations):
    """
    Compute:
    - Cohen's Kappa (pairwise between each annotator pair)
    - Krippendorff's Alpha (all annotators, ordinal scale)
    - Average scores per criterion per method
    """
    # annotations shape: [n_annotators, n_samples, n_criteria]
    alpha = krippendorff.alpha(annotations, level_of_measurement="ordinal")
    return alpha
```

### Generate Evaluation Materials

```bash
# Generate 50-sample evaluation set with all method outputs
python scripts/evaluate.py \
  --mode generate_human_eval \
  --n_samples 50 \
  --methods rag_xpr,llm_cot,llm_zeroshot,shap_distilbert \
  --output outputs/human_eval/

# Output:
# outputs/human_eval/
#   eval_forms.csv           # for spreadsheet-based evaluation
#   eval_forms.html          # web-based evaluation form
#   method_outputs.jsonl     # all method outputs for the 50 samples
```

---

## 4. Statistical Significance Tests

### Implementation: `src/evaluation/statistical_tests.py`

```python
def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test: are two classifiers significantly different?
    Only looks at samples where they disagree.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    # Build contingency table
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    table = [
        [sum(correct_a & correct_b), sum(correct_a & ~correct_b)],
        [sum(~correct_a & correct_b), sum(~correct_a & ~correct_b)],
    ]
    result = mcnemar(table, exact=True)
    return {"statistic": result.statistic, "p_value": result.pvalue}

def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap 95% CI for any metric.
    """
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return {"mean": np.mean(scores), "ci_lower": lower, "ci_upper": upper}

def paired_bootstrap_test(y_true, pred_a, pred_b, metric_fn, n_bootstrap=10000):
    """
    Paired bootstrap test: is method A significantly better than B?
    Returns p-value for H0: metric(A) <= metric(B)
    """
    n = len(y_true)
    delta_observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    count_greater = 0
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        delta = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
        if delta <= 0:
            count_greater += 1
    return {"delta": delta_observed, "p_value": count_greater / n_bootstrap}
```

### Required Tests

For every pair of methods being compared:
1. **McNemar's test** on accuracy (p < 0.05)
2. **Paired bootstrap** on F1-macro (1000 resamples, p < 0.05)
3. **95% bootstrap CI** for all reported metrics
4. **Wilcoxon signed-rank test** for human evaluation scores

### Run Evaluation Suite

```bash
# Full automated evaluation
python scripts/evaluate.py \
  --mode full \
  --predictions_dir outputs/predictions/ \
  --output outputs/reports/

# Output:
# outputs/reports/
#   classification_results.json   # all metrics for all methods
#   xai_results.json              # XAI metrics
#   statistical_tests.json        # significance tests
#   comparison_tables.md          # formatted tables for report
#   confusion_matrices/           # PNG plots per method
```

---

## 5. Evaluation Checklist

Before declaring results final:

- [ ] All experiments run with 3 seeds (42, 123, 456)
- [ ] Bootstrap CIs computed for all reported metrics
- [ ] McNemar/paired bootstrap between RAG-XPR and each baseline
- [ ] Human evaluation completed (50 samples × 3+ annotators)
- [ ] Inter-annotator agreement ≥ 0.6 (Krippendorff's alpha)
- [ ] Ablation results show each component contributes
- [ ] Evidence grounding score > 0.9 (RAG-XPR should cite real text)
- [ ] Cost per prediction logged for all LLM methods
- [ ] Confusion matrices inspected for systematic errors
- [ ] Failure case analysis: 10+ examples where RAG-XPR fails
