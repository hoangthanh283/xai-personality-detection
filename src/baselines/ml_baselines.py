"""TF-IDF + classical ML baselines (LR, SVM, NB, XGBoost, RandomForest).

Usage:
    trainer = MLBaselineTrainer(config)
    trainer.fit(train_texts, train_labels)
    metrics = trainer.evaluate(test_texts, test_labels)
    trainer.save("outputs/models/tfidf_lr_mbti.pkl")
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, XGBClassifier will not be available")

TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 3,
    "max_df": 0.95,
}

GRID_SEARCH_PARAMS = {
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


def build_model(model_name: str, config: dict | None = None) -> Any:
    """Build a sklearn estimator from model name."""
    cfg = config or {}
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
        )
    elif model_name == "svm":
        return LinearSVC(
            max_iter=cfg.get("max_iter", 5000),
            class_weight=cfg.get("class_weight", "balanced"),
            C=cfg.get("C", 1.0),
        )
    elif model_name == "naive_bayes":
        return MultinomialNB(alpha=cfg.get("alpha", 1.0))
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.1),
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class MLBaselineTrainer:
    """TF-IDF + sklearn classifier pipeline."""

    def __init__(self, model_name: str = "logistic_regression", config: dict | None = None):
        self.model_name = model_name
        self.config = config or {}
        tfidf_cfg = {**TFIDF_PARAMS, **self.config.get("tfidf", {})}
        ngram_range = tfidf_cfg.get("ngram_range")
        if isinstance(ngram_range, list):
            tfidf_cfg["ngram_range"] = tuple(ngram_range)
        self.tfidf = TfidfVectorizer(**tfidf_cfg)
        classifier = build_model(model_name, self.config.get(model_name, {}))
        self.pipeline = Pipeline([("tfidf", self.tfidf), ("clf", classifier)])
        self.is_fitted = False
        self._label_encoder: LabelEncoder | None = None

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        use_grid_search: bool = False,
        cv: int = 5,
    ) -> "MLBaselineTrainer":
        logger.info(f"Training {self.model_name} on {len(train_texts)} samples...")
        if use_grid_search and self.model_name in GRID_SEARCH_PARAMS:
            param_grid = GRID_SEARCH_PARAMS[self.model_name]
            gs = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
            )
            gs.fit(train_texts, train_labels)
            self.pipeline = gs.best_estimator_
            logger.info(f"Best params: {gs.best_params_}")
        else:
            y = train_labels
            clf = self.pipeline.named_steps["clf"]
            if isinstance(clf, XGBClassifier):
                self._label_encoder = LabelEncoder().fit(train_labels)
                y = self._label_encoder.transform(train_labels)
            else:
                self._label_encoder = None
            self.pipeline.fit(train_texts, y)
        self.is_fitted = True
        logger.info("Training complete")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        preds = self.pipeline.predict(texts)
        if self._label_encoder is not None:
            preds = self._label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Return probability estimates (only for models that support it)."""
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(texts)
        return None

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        preds = self.predict(texts)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "classification_report": classification_report(labels, preds, zero_division=0),
        }
        logger.info(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1-macro: {metrics['f1_macro']:.4f} | "
            f"F1-weighted: {metrics['f1_weighted']:.4f}"
        )
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "unknown") -> "MLBaselineTrainer":
        trainer = cls(model_name=model_name)
        with open(path, "rb") as f:
            trainer.pipeline = pickle.load(f)
        trainer.is_fitted = True
        return trainer


class EnsembleClassifier:
    """Soft voting ensemble over multiple ML classifiers."""

    def __init__(self, members: list[str] | None = None, config: dict | None = None):
        self.config = config or {}
        members = members or ["logistic_regression", "xgboost", "random_forest"]
        tfidf_cfg = self.config.get("tfidf", TFIDF_PARAMS)
        estimators = []
        for name in members:
            clf = build_model(name, self.config.get(name, {}))
            estimators.append((name, clf))

        # Build individual pipelines
        self.pipelines = {
            name: Pipeline([("tfidf", TfidfVectorizer(**tfidf_cfg)), ("clf", clf)])
            for name, clf in estimators
        }
        self.member_names = [name for name, _ in estimators]
        self.labels_: list[str] = []

    def fit(self, train_texts: list[str], train_labels: list[str]) -> "EnsembleClassifier":
        self.labels_ = sorted(set(train_labels))
        for name, pipeline in self.pipelines.items():
            logger.info(f"Training ensemble member: {name}")
            pipeline.fit(train_texts, train_labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Majority vote across all member classifiers."""
        all_preds = np.array([p.predict(texts) for p in self.pipelines.values()])
        # Majority vote
        from scipy.stats import mode
        result, _ = mode(all_preds, axis=0)
        return result.flatten()

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }
