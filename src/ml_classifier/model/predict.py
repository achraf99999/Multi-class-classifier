"""
Predict class probabilities for records using the fitted pipeline.
Same feature builder as at train time (single code path).
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_classifier.features.build import build_text_features


def predict_proba_df(
    pipeline: Pipeline,
    records_df: pd.DataFrame,
    class_labels: list[int],
) -> pd.DataFrame:
    """
    Build text features, run predict_proba, return DataFrame with id and prob per class.

    Each row is one id; columns are id, prob_1, prob_2, ...; rows sum to 1.
    """
    text = build_text_features(records_df)
    proba = pipeline.predict_proba(text)
    out = pd.DataFrame(
        proba,
        columns=[f"prob_{c}" for c in class_labels],
        index=records_df.index,
    )
    out.insert(0, "id", records_df["id"].values)
    return out.reset_index(drop=True)
