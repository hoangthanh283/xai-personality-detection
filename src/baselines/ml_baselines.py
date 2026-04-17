"""TF-IDF + classical ML baselines (LR, SVM, NB, XGBoost, RandomForest).

Usage:
    trainer = MLBaselineTrainer(config)
    trainer.fit(train_texts, train_labels)
    metrics = trainer.evaluate(test_texts, test_labels)
    trainer.save("outputs/models/tfidf_lr_mbti.pkl")
"""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    from sklearn.decomposition import TruncatedSVD

    HAS_SVD = True
except ImportError:
    HAS_SVD = False

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, XGBClassifier will not be available")

TFIDF_PARAMS = {
    "max_features": 30000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 3,
    "max_df": 0.95,
}

CHAR_TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "max_features": 20000,
    "sublinear_tf": True,
    "min_df": 3,
    "max_df": 0.95,
}

SVD_PARAMS = {
    "n_components": 300,
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

MODELS_USING_SVD = {"logistic_regression", "svm"}


def _coerce_ngram(params: dict[str, Any]) -> dict[str, Any]:
    ngram = params.get("ngram_range")
    if isinstance(ngram, list):
        params["ngram_range"] = tuple(ngram)
    return params


def _build_tfidf_vectorizer(config: Mapping[str, Any] | None) -> Any:
    """Build a word (and optionally char) TF-IDF vectorizer.

    The char-level TF-IDF is OFF by default because combining word + char
    TF-IDF with ~80K features on small datasets like MBTI 8k-user causes
    severe overfitting. Set ``tfidf.use_char_ngrams: true`` to opt in.
    """
    cfg = dict(config or {})
    tfidf_cfg = _coerce_ngram({**TFIDF_PARAMS, **cfg.get("tfidf", {})})
    use_char = bool(tfidf_cfg.pop("use_char_ngrams", False))
    char_cfg = _coerce_ngram({**CHAR_TFIDF_PARAMS, **cfg.get("tfidf_char", {})})

    word_tfidf = TfidfVectorizer(**tfidf_cfg)
    if not use_char:
        logger.info(
            f"TF-IDF (word-only): max_features={tfidf_cfg['max_features']}, "
            f"ngram_range={tfidf_cfg['ngram_range']}, min_df={tfidf_cfg['min_df']}"
        )
        return word_tfidf

    from sklearn.pipeline import FeatureUnion

    char_tfidf = TfidfVectorizer(**char_cfg)
    logger.info(
        "TF-IDF union: word "
        f"(max={tfidf_cfg['max_features']}, ngram={tfidf_cfg['ngram_range']}) + char "
        f"(max={char_cfg['max_features']}, ngram={char_cfg['ngram_range']})"
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


def _get_ml_model_config(config: Mapping[str, Any] | None, model_name: str) -> dict[str, Any]:
    if not config:
        return {}
    return dict(config.get("ml_models", {}).get(model_name, {}))


def _get_grid_search_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return dict(config.get("grid_search", {}))


def build_model(model_name: str, config: dict | None = None) -> Any:
    """Build a sklearn estimator from model name."""
    cfg = config or {}
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
            n_jobs=cfg.get("n_jobs"),
        )
    elif model_name == "svm":
        return LinearSVC(
            max_iter=cfg.get("max_iter", 5000),
            class_weight=cfg.get("class_weight", "balanced"),
            C=cfg.get("C", 1.0),
            dual=cfg.get("dual", "auto"),
        )
    elif model_name == "naive_bayes":
        return MultinomialNB(alpha=cfg.get("alpha", 1.0))
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.1),
            tree_method=cfg.get("tree_method", "hist"),
            n_jobs=cfg.get("n_jobs", 2),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            eval_metric=cfg.get("eval_metric", "mlogloss"),
            reg_alpha=cfg.get("reg_alpha", 0.0),
            reg_lambda=cfg.get("reg_lambda", 1.0),
            verbosity=0,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", None),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=cfg.get("n_jobs", 4),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class MLBaselineTrainer:
    """TF-IDF + sklearn classifier pipeline, with optional SVD for dense models."""

    def __init__(self, model_name: str = "logistic_regression", config: dict | None = None):
        self.model_name = model_name
        self.config = config or {}
        self.tfidf = _build_tfidf_vectorizer(self.config)
        classifier = build_model(model_name, _get_ml_model_config(self.config, model_name))

        use_svd = self.config.get("dimensionality_reduction", {}).get("enabled", False)
        use_svd = use_svd and model_name in MODELS_USING_SVD and HAS_SVD

        pipeline_steps = [("tfidf", self.tfidf)]
        if use_svd:
            svd_cfg = {**SVD_PARAMS, **self.config.get("dimensionality_reduction", {})}
            svd_cfg.pop("enabled", None)
            pipeline_steps.append(("svd", TruncatedSVD(**svd_cfg)))
            logger.info(f"Using TruncatedSVD with n_components={svd_cfg.get('n_components', 300)}")
        pipeline_steps.append(("clf", classifier))
        self.pipeline = Pipeline(pipeline_steps)
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
        y = train_labels
        clf = self.pipeline.named_steps["clf"]
        if isinstance(clf, XGBClassifier):
            self._label_encoder = LabelEncoder().fit(train_labels)
            y = self._label_encoder.transform(train_labels)
        else:
            self._label_encoder = None

        grid_cfg = _get_grid_search_config(self.config)
        if use_grid_search and self.model_name in GRID_SEARCH_PARAMS:
            param_grid = grid_cfg.get("param_grid", {}).get(
                self.model_name,
                GRID_SEARCH_PARAMS[self.model_name],
            )
            gs = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=grid_cfg.get("cv", cv),
                scoring=grid_cfg.get("scoring", "f1_macro"),
                n_jobs=grid_cfg.get("n_jobs", -1),
                verbose=1,
            )
            gs.fit(train_texts, y)
            self.pipeline = gs.best_estimator_
            logger.info(f"Best params: {gs.best_params_}")
        else:
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
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            if "svd" in self.pipeline.named_steps:
                X = self.pipeline.named_steps["tfidf"].transform(texts)
                X = self.pipeline.named_steps["svd"].transform(X)
                return clf.predict_proba(X)
            return self.pipeline.predict_proba(texts)
        return None

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        preds = self.predict(texts)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
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
        estimators = []
        for name in members:
            clf = build_model(name, _get_ml_model_config(self.config, name))
            estimators.append((name, clf))

        voting = "soft"
        for name, clf in estimators:
            if not hasattr(clf, "predict_proba"):
                logger.warning(
                    f"Model {name} does not support predict_proba. Falling back to 'hard' voting."
                )
                voting = "hard"
                break

        from sklearn.ensemble import VotingClassifier

        vc = VotingClassifier(estimators=estimators, voting=voting)
        vectorizer = _build_tfidf_vectorizer(self.config)

        self.pipeline = Pipeline([("tfidf", vectorizer), ("clf", vc)])
        self.member_names = members
        self._label_encoder = None

    def fit(self, train_texts: list[str], train_labels: list[str]) -> "EnsembleClassifier":
        y = train_labels
        from sklearn.preprocessing import LabelEncoder

        try:
            from xgboost import XGBClassifier

            has_xgb = any(
                isinstance(clf, XGBClassifier)
                for _, clf in self.pipeline.named_steps["clf"].estimators
            )
        except ImportError:
            has_xgb = False

        if has_xgb:
            self._label_encoder = LabelEncoder().fit(train_labels)
            y = self._label_encoder.transform(train_labels)

        logger.info(
            f"Training ensemble members: {self.member_names} (voting='{self.pipeline.named_steps['clf'].voting}')"
        )
        self.pipeline.fit(train_texts, y)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        preds = self.pipeline.predict(texts)
        if self._label_encoder is not None:
            preds = self._label_encoder.inverse_transform(preds)
        return preds

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        from sklearn.metrics import (accuracy_score, classification_report,
                                     f1_score, precision_score, recall_score)

        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "classification_report": classification_report(labels, preds, zero_division=0),
        }
