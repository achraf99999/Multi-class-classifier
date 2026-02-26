"""
Stratified train/validation split.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_val_split(
    merged_df: pd.DataFrame,
    val_ratio: float = 0.2,
    stratify_col: str = "target",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split merged data into train and validation sets with stratification.

    Returns:
        (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        merged_df,
        test_size=val_ratio,
        stratify=merged_df[stratify_col],
        random_state=seed,
    )
    logger.info(
        "Split: train=%d, val=%d",
        len(train_df),
        len(val_df),
    )
    return train_df, val_df
