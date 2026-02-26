"""
CLI entry point: load -> split -> train -> predict -> save CSV.
Single entrypoint; all logic delegated to data, features, and model layers.

Behavior:
- Run with no args: use defaults from config (and built-in defaults for model params).
- Override params via CLI (--max-features, --ngram-range, --max-iter): use those for the pipeline.
- --tune: enable hyperparameter tuning (GridSearchCV) and use best params for prediction.
- --no-tune: disable tuning; use config overrides or defaults.
"""

import argparse
import json
import logging
from datetime import datetime

from ml_classifier import config
from ml_classifier.data.load import load_and_merge, load_target_id_order
from ml_classifier.data.split import train_val_split
from ml_classifier.features.build import build_text_features
from ml_classifier.model.predict import predict_proba_df
from ml_classifier.model.train import train_pipeline, train_with_cv


def _parse_ngram_range(s: str) -> tuple[int, int]:
    """Parse 'min,max' into (min, max) tuple, e.g. '1,2' -> (1, 2)."""
    parts = s.strip().split(",")
    if len(parts) != 2:
        raise ValueError("ngram_range must be two integers separated by comma, e.g. 1,2")
    return (int(parts[0].strip()), int(parts[1].strip()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multi-class classifier and write predictions to output/run_YYYYMMDD_HHMMSS/.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        metavar="N",
        help="TfidfVectorizer max_features (default: use config or 50000)",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        default=None,
        metavar="MIN,MAX",
        help="TfidfVectorizer ngram_range, e.g. 1,2 (default: use config or (1,2))",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="LogisticRegression max_iter (default: use config or 500)",
    )
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (GridSearchCV); use best params for prediction",
    )
    tuning.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning; use given or default params only",
    )
    parser.add_argument(
        "--tuning-cv",
        type=int,
        default=None,
        metavar="K",
        help="Number of CV folds when tuning (default: from config, e.g. 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Apply CLI overrides to config so rest of pipeline sees them
    if args.max_features is not None:
        config.MODEL_MAX_FEATURES = args.max_features
    if args.ngram_range is not None:
        config.MODEL_NGRAM_RANGE = _parse_ngram_range(args.ngram_range)
    if args.max_iter is not None:
        config.MODEL_MAX_ITER = args.max_iter
    if args.tune:
        config.USE_HYPERPARAMETER_TUNING = True
    if args.no_tune:
        config.USE_HYPERPARAMETER_TUNING = False
    if args.tuning_cv is not None:
        config.TUNING_CV = args.tuning_cv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    run_ts = datetime.now()
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")

    # 1. Load and merge
    merged = load_and_merge(config.SAMPLE_DATA_PATH, config.SAMPLE_TARGETS_PATH)
    merged_rows = len(merged)

    # 2. Split
    train_df, val_df = train_val_split(
        merged,
        val_ratio=config.VAL_RATIO,
        stratify_col=config.STRATIFY_COL,
        seed=config.SEED,
    )
    train_count, val_count = len(train_df), len(val_df)

    # 3. Build text features (same logic for train, val, and final predict)
    train_text = build_text_features(train_df)
    val_text = build_text_features(val_df)

    # 4. Train pipeline (with or without hyperparameter tuning)
    use_tuning = getattr(config, "USE_HYPERPARAMETER_TUNING", False)
    if use_tuning:
        pipeline, class_labels, metrics = train_with_cv(
            train_text,
            train_df[config.STRATIFY_COL],
            val_text,
            val_df[config.STRATIFY_COL],
        )
    else:
        pipeline, class_labels, metrics = train_pipeline(
            train_text,
            train_df[config.STRATIFY_COL],
            val_text,
            val_df[config.STRATIFY_COL],
        )

    # 5. Predict on full merged dataset (one row per id in merged)
    pred_df = predict_proba_df(pipeline, merged, class_labels)

    # Order rows to match sample_targets.csv id order (single place for target read)
    target_ids = load_target_id_order(config.SAMPLE_TARGETS_PATH)
    pred_df = target_ids.merge(pred_df, on="id", how="inner")
    num_predictions = len(pred_df)

    # 6. Save into a run-specific folder under output/ (no overwrite; keep all runs)
    run_dir = config.OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.csv"
    pred_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, num_predictions)

    # 7. Save run info and metrics as JSON
    validation = {k: v for k, v in metrics.items() if k != "best_params"}
    run_info = {
        "run_id": run_id,
        "timestamp": run_ts.isoformat(),
        "data": {
            "merged_rows": merged_rows,
            "train_count": train_count,
            "val_count": val_count,
            "predictions_count": num_predictions,
        },
        "config": {
            "val_ratio": config.VAL_RATIO,
            "seed": config.SEED,
            "use_hyperparameter_tuning": use_tuning,
            "tuning_cv": getattr(config, "TUNING_CV", None),
            "model_max_features": getattr(config, "MODEL_MAX_FEATURES", None),
            "model_ngram_range": getattr(config, "MODEL_NGRAM_RANGE", None),
            "model_max_iter": getattr(config, "MODEL_MAX_ITER", None),
        },
        "validation": validation,
        "output": {
            "run_dir": str(run_dir),
            "predictions_csv": str(out_path),
        },
    }
    if "best_params" in metrics:
        run_info["best_params"] = metrics["best_params"]
    run_info_path = run_dir / "run_info.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    logger.info("Wrote %s", run_info_path)


if __name__ == "__main__":
    main()
