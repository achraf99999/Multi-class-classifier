"""
Load sample_data.json and sample_targets.csv, merge on id, and validate.
I/O and merging only; no feature construction or model code.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_merge(
    data_path: Path,
    targets_path: Path,
) -> pd.DataFrame:
    """
    Load JSON and CSV, merge on id (inner join), validate.

    Returns:
        DataFrame with columns from JSON plus 'target'. Only rows present
        in both sources are kept.
    """
    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)
    data_df = pd.DataFrame(records)

    targets_df = pd.read_csv(targets_path, dtype={"id": str, "target": int})

    target_ids = set(targets_df["id"])
    data_ids = set(data_df["id"])
    missing_in_json = target_ids - data_ids
    if missing_in_json:
        logger.warning(
            "Found %d target id(s) not present in JSON; they will be dropped in merge. "
            "First few: %s",
            len(missing_in_json),
            list(missing_in_json)[:5],
        )

    merged = data_df.merge(targets_df, on="id", how="inner")

    logger.info(
        "Merged dataset: %d rows (data: %d, targets: %d)",
        len(merged),
        len(data_df),
        len(targets_df),
    )
    return merged


def load_target_id_order(targets_path: Path) -> pd.DataFrame:
    """
    Load target CSV and return the id column in original row order.
    Use when you need the same order as in the targets file (e.g. for output).
    """
    return pd.read_csv(targets_path, usecols=["id"], dtype={"id": str})
