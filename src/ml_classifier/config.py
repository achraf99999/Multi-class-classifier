"""
Configuration: paths, constants, and class names.
Single place to change data paths, random seed, or split ratio.
"""

from pathlib import Path

# Project root (directory containing pyproject.toml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Data files: expected in input_data/ inside the project (override by setting DATA_DIR)
DATA_DIR = PROJECT_ROOT / "input_data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.json"
SAMPLE_TARGETS_PATH = DATA_DIR / "sample_targets.csv"

# Reproducibility
SEED = 42

# Train/validation split
VAL_RATIO = 0.2
STRATIFY_COL = "target"

# Hyperparameter tuning: if True, use GridSearchCV; if False, fit with defaults only
USE_HYPERPARAMETER_TUNING = True
TUNING_CV = 3  # number of CV folds for grid search

# Model pipeline overrides (None = use built-in default)
# Set these to override TfidfVectorizer / LogisticRegression defaults when building the pipeline.
MODEL_MAX_FEATURES = None  # e.g. 30_000; default 50_000
MODEL_NGRAM_RANGE = None  # e.g. (1, 2); default (1, 2)
MODEL_MAX_ITER = None  # e.g. 1000; default 500

# Class labels (multi-class 1-4)
CLASS_LABELS = [1, 2, 3, 4]

# Output
OUTPUT_DIR = PROJECT_ROOT / "output"
