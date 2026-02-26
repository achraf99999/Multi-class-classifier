"""
Build text features from raw records (title, abstract, categories).
Does not fit a vectorizer; only produces the text for the model layer.
"""

import numpy as np
import pandas as pd


def _safe_str(x) -> str:
    """Coerce value to string; use empty string for null/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def build_text_features(records_df: pd.DataFrame) -> pd.Series:
    """
    Build a single text feature per row: title + abstract + categories.

    Nulls are replaced with empty string. Categories are normalized
    (split on space and rejoined with spaces) so multi-category strings
    are consistent.

    Returns:
        Series of strings, one per row, index aligned with records_df.
    """
    title = records_df["title"].map(_safe_str)
    abstract = records_df["abstract"].map(_safe_str)
    raw_cat = records_df["categories"].map(_safe_str)
    categories = raw_cat.str.split().str.join(" ").replace("", "")

    combined = title + " " + abstract + " " + categories
    # Return as string Series so pipeline gets consistent input (single contract)
    return combined.str.strip().astype(str)


def build_advanced_features(records_df: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    """
    Stub for future advanced feature engineering.

    Intended extensions could include:
    - Separate TF-IDF (or counts) for title-only vs abstract-only, then concatenate.
    - One-hot or embedding of parsed categories.
    - Author count or other metadata-derived features.

    Returns:
        Not implemented; would return a feature matrix (DataFrame or ndarray).
    """
    raise NotImplementedError(
        "Advanced features (e.g. title-only TF-IDF, category encoding) not implemented."
    )
