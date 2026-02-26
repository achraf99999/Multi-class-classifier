"""
Train TF-IDF + Logistic Regression pipeline on text features.
Supports optional hyperparameter tuning via GridSearchCV.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ml_classifier import config

logger = logging.getLogger(__name__)

DEFAULT_MAX_FEATURES = 50_000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MAX_ITER = 500

# Param grid for hyperparameter tuning (used when USE_HYPERPARAMETER_TUNING is True)
TUNING_PARAM_GRID = {
    "vec__max_features": [20_000, 50_000],
    "vec__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}


def _resolve_pipeline_params() -> tuple[int, tuple[int, int], int]:
    """Resolve max_features, ngram_range, max_iter: config overrides or defaults."""
    max_features = getattr(config, "MODEL_MAX_FEATURES", None)
    ngram_range = getattr(config, "MODEL_NGRAM_RANGE", None)
    max_iter = getattr(config, "MODEL_MAX_ITER", None)
    return (
        max_features if max_features is not None else DEFAULT_MAX_FEATURES,
        ngram_range if ngram_range is not None else DEFAULT_NGRAM_RANGE,
        max_iter if max_iter is not None else DEFAULT_MAX_ITER,
    )


def _make_pipeline() -> Pipeline:
    max_features, ngram_range, max_iter = _resolve_pipeline_params()
    return Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def _compute_val_metrics(
    pipeline: Pipeline,
    val_text: pd.Series,
    val_targets: pd.Series,
    classes: list[Any],
) -> dict[str, float]:
    """Compute and log accuracy, macro F1, log loss on validation set; return metrics dict."""
    val_pred = pipeline.predict(val_text)
    val_proba = pipeline.predict_proba(val_text)
    accuracy = float(accuracy_score(val_targets, val_pred))
    macro_f1 = float(f1_score(val_targets, val_pred, average="macro", zero_division=0))
    logloss = float(log_loss(val_targets, val_proba, labels=classes))
    logger.info(
        "Validation: accuracy=%.4f, macro_f1=%.4f, log_loss=%.4f",
        accuracy,
        macro_f1,
        logloss,
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "log_loss": logloss}


def train_pipeline(
    train_text: pd.Series,
    train_targets: pd.Series,
    val_text: pd.Series,
    val_targets: pd.Series,
) -> tuple[Pipeline, list[Any], dict[str, float]]:
    """
    Fit TF-IDF + Logistic Regression on training data with default hyperparameters.
    Use when USE_HYPERPARAMETER_TUNING is False for a fast run.
    Returns (pipeline, class_labels, validation_metrics).
    """
    pipeline = _make_pipeline()
    pipeline.fit(train_text, train_targets)
    classes = list(pipeline.classes_)
    metrics = _compute_val_metrics(pipeline, val_text, val_targets, classes)
    return pipeline, classes, metrics


def train_with_cv(
    train_text: pd.Series,
    train_targets: pd.Series,
    val_text: pd.Series,
    val_targets: pd.Series,
) -> tuple[Pipeline, list[Any], dict[str, Any]]:
    """
    Fit pipeline with hyperparameter tuning via GridSearchCV on the training set.
    Uses TUNING_PARAM_GRID and TUNING_CV from config; evaluates best estimator on val.
    Returns (pipeline, class_labels, metrics_dict with validation + best_params).
    """
    pipeline = _make_pipeline()
    search = GridSearchCV(
        pipeline,
        TUNING_PARAM_GRID,
        cv=config.TUNING_CV,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(train_text, train_targets)
    best = search.best_estimator_
    classes = list(best.classes_)
    logger.info("Best params: %s", search.best_params_)
    metrics = _compute_val_metrics(best, val_text, val_targets, classes)
    # best_params: make JSON-serializable (tuples -> lists)
    best_params = {k: list(v) if isinstance(v, tuple) else v for k, v in search.best_params_.items()}
    metrics["best_params"] = best_params
    return best, classes, metrics
