"""
Legal-BERT based text classifier: embed text with nlpaueb/legal-bert-base-uncased, then LogisticRegression.

See: https://huggingface.co/nlpaueb/legal-bert-base-uncased
     https://opensource.legal/projects/Legal_BERT
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from transformers import AutoModel, AutoTokenizer

from ml_classifier import config

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    """Use CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _texts_to_list(series_or_list: pd.Series | list[str]) -> list[str]:
    """Convert Series or list to list of strings."""
    if isinstance(series_or_list, pd.Series):
        return series_or_list.astype(str).tolist()
    return list(series_or_list)


class LegalBertClassifier:
    """
    Wrapper: embed text with Legal-BERT, classify with LogisticRegression.

    Exposes fit(text_series, targets) and predict_proba(text_series) so it can
    be used like the TF-IDF Pipeline in the rest of the codebase.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.LEGAL_BERT_MODEL
        self.max_length = max_length or config.BERT_MAX_LENGTH
        self.batch_size = batch_size or config.BERT_BATCH_SIZE
        self.device = _device()
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModel | None = None
        self.classifier: LogisticRegression | None = None
        self._classes_: list[Any] = []

    def _ensure_loaded(self) -> None:
        """Lazy-load tokenizer and model on first use."""
        if self.tokenizer is not None and self.model is not None:
            return
        logger.info("Loading Legal-BERT tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Compute [CLS] embeddings for a list of texts; returns (n, hidden_size)."""
        self._ensure_loaded()
        assert self.tokenizer is not None and self.model is not None
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        # [CLS] token (first token) as sentence representation
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_emb

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        """Embed all texts in batches."""
        chunks = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embs = [self._embed_batch(c) for c in chunks]
        return np.vstack(embs)

    def fit(
        self,
        X: pd.Series | list[str],
        y: pd.Series | np.ndarray,
    ) -> "LegalBertClassifier":
        """Compute Legal-BERT embeddings for X, then fit LogisticRegression on (embeddings, y)."""
        texts = _texts_to_list(X)
        logger.info("Computing Legal-BERT embeddings for %d training samples", len(texts))
        X_emb = self._embed_all(texts)
        y_flat = np.asarray(y).ravel()
        self.classifier = LogisticRegression(
            max_iter=getattr(config, "MODEL_MAX_ITER", None) or 500,
            class_weight="balanced",
            random_state=config.SEED,
        )
        self.classifier.fit(X_emb, y_flat)
        self._classes_ = list(self.classifier.classes_)
        return self

    def predict_proba(self, X: pd.Series | list[str]) -> np.ndarray:
        """Compute embeddings for X and return class probabilities from the classifier."""
        texts = _texts_to_list(X)
        X_emb = self._embed_all(texts)
        assert self.classifier is not None
        return self.classifier.predict_proba(X_emb)

    def predict(self, X: pd.Series | list[str]) -> np.ndarray:
        """Compute embeddings for X and return predicted class labels."""
        texts = _texts_to_list(X)
        X_emb = self._embed_all(texts)
        assert self.classifier is not None
        return self.classifier.predict(X_emb)

    @property
    def classes_(self) -> list[Any]:
        """Class labels (for compatibility with sklearn pipeline)."""
        return self._classes_


def _compute_val_metrics(
    estimator: LegalBertClassifier,
    val_text: pd.Series,
    val_targets: pd.Series,
    classes: list[Any],
) -> dict[str, float]:
    """Compute accuracy, macro F1, log loss on validation set."""
    val_pred = estimator.predict(val_text)
    val_proba = estimator.predict_proba(val_text)
    accuracy = float(accuracy_score(val_targets, val_pred))
    macro_f1 = float(f1_score(val_targets, val_pred, average="macro", zero_division=0))
    logloss = float(log_loss(val_targets, val_proba, labels=classes))
    logger.info(
        "Validation (Legal-BERT): accuracy=%.4f, macro_f1=%.4f, log_loss=%.4f",
        accuracy,
        macro_f1,
        logloss,
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "log_loss": logloss}


def train_legalbert(
    train_text: pd.Series,
    train_targets: pd.Series,
    val_text: pd.Series,
    val_targets: pd.Series,
) -> tuple[LegalBertClassifier, list[Any], dict[str, float]]:
    """
    Fit Legal-BERT embeddings + LogisticRegression on training data.
    Returns (estimator, class_labels, validation_metrics).
    """
    estimator = LegalBertClassifier()
    estimator.fit(train_text, train_targets)
    classes = list(estimator.classes_)
    metrics = _compute_val_metrics(estimator, val_text, val_targets, classes)
    return estimator, classes, metrics
