# Solution Description

This document describes what was built, how it works, and the main choices behind it—plus what we’d change with more time or in a production setting.

---

## What was built

A **multi-class classifier** that predicts one of four categories (targets 1–4) for arXiv paper metadata. It uses the provided `sample_data.json` (title, abstract, categories, etc.) and `sample_targets.csv` (id and target) to train a model, then outputs a **probability distribution** over the four classes for each paper id. Predictions are written to a timestamped CSV under `output/` so runs don’t overwrite each other.

---

## How the pipeline works

1. **Load and merge**  
   We load `sample_data.json` and `sample_targets.csv` and merge them on `id` (inner join). Only rows that appear in both files are kept. Any target ids missing from the JSON are logged so you can spot data issues.

2. **Train/validation split**  
   The merged data is split 80% train / 20% validation, with **stratification** on `target` so class proportions stay roughly the same in both sets. There’s no separate test set in this setup; with more time we’d add a held-out test split and report only on that.

3. **Text features**  
   For each record we build a single text field by concatenating `title`, `abstract`, and `categories` (nulls become empty strings). The same logic is used everywhere—training, validation, and final prediction—so we avoid feature drift and bugs.

4. **Model**  
   A scikit-learn **Pipeline** with:
   - **TfidfVectorizer**: `max_features=50_000`, `ngram_range=(1, 2)`, `sublinear_tf=True`
   - **LogisticRegression**: `max_iter=500`, `class_weight='balanced'`

   We fit the pipeline on the training text and targets. Validation metrics (accuracy, macro F1, log loss) are computed and logged for transparency; no hyperparameter tuning was done.

5. **Predictions**  
   For every id that appears in both input files, we build text features with the same function, run `predict_proba`, and write a CSV with columns `id`, `prob_1`, `prob_2`, `prob_3`, `prob_4`. Row order matches `sample_targets.csv`.

---

## Key decisions

- **Logistic Regression + TF-IDF**  
  We chose this as a strong, interpretable baseline for multi-class text classification. It gives proper probability outputs out of the box (`predict_proba`), is easy to maintain, and we didn’t need an extra calibration step. The trade-off was prioritizing clarity and maintainability over squeezing out the last bit of accuracy (e.g. no transformers or heavy tuning).

- **Single concatenated text**  
  Title, abstract, and categories are concatenated into one string before TF-IDF. Simple and effective. With more time we could give title vs abstract different weights or use separate vectorizers.

- **Stratified split**  
  Stratifying on `target` keeps the validation set representative of the class distribution, which makes the metrics more reliable.

- **One code path for features**  
  The same `build_text_features` is used for training, validation, and prediction. That keeps behavior consistent and reduces the chance of subtle bugs.

---

## What we’d do differently with more time

- **Hyperparameter tuning**  
  Run a grid or random search over the vectorizer (e.g. `max_features`, `ngram_range`) and classifier (e.g. `C`, solver), with cross-validation.

- **Richer feature engineering**  
  Separate TF-IDF for title vs abstract with different weights; one-hot or embedding for categories; explicit handling of missing abstract/categories.

- **Proper train/val/test split**  
  Reserve a test set (e.g. 70/15/15) and report only test metrics; use validation purely for model selection and tuning.

- **More robust evaluation**  
  Cross-validation for stability; per-class metrics and a confusion matrix in the write-up.

---

## Production-style improvements

If this were going into production, we’d add:

- **Config and logging**  
  Paths, seed, and split ratio already live in `config.py`. Next step: replace `print` with structured logging and optional config files (e.g. YAML) for different environments.

- **Model versioning**  
  Save the fitted pipeline (e.g. `joblib.dump`) with a version or timestamp, and document how to load and run inference.

- **Tests**  
  Unit tests for `load_and_merge` (expected columns, missing ids), `train_val_split` (stratification, sizes), and `build_text_features` (null handling, shape).

- **CI**  
  Run tests in CI and optionally a small pipeline run on a subset of data.

- **Containers**  
  A Dockerfile for reproducible runs.

- **Multi-label**  
  If the task were multi-label (multiple labels per sample), we’d use something like `OneVsRestClassifier` with a probabilistic base estimator and output one probability per label.

---

## Deliverables

- **Code**: `pyproject.toml`, `.python-version`, `run.py`, and `src/ml_classifier/` (config, cli, data, features, model), plus this README. Dependencies are in `pyproject.toml`; use `uv sync` to install.

- **Predictions**: Each run writes to `output/run_YYYYMMDD_HHMMSS/predictions.csv`. One row per id (aligned with `sample_targets.csv`), with columns `id`, `prob_1`, `prob_2`, `prob_3`, `prob_4` (probability distribution per row).

- **Description**: This document.
