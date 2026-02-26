# Solution Description

## What was built

A multi-class classifier that predicts one of four categories (target 1–4) for arXiv paper metadata. The pipeline uses the provided `sample_data.json` (title, abstract, categories, etc.) and `sample_targets.csv` (id and target) to train a model and output a probability distribution over the four classes for each id.

---

## Steps and pipeline

1. **Load and merge**  
   Load `sample_data.json` and `sample_targets.csv`, merge on `id` (inner join). Only rows present in both sources are kept. Missing target ids in the JSON are logged.

2. **Stratified split**  
   Split the merged data into train (80%) and validation (20%) with stratification on `target` so class proportions are preserved. No separate test set; with more time we would add a held-out test split and report only test metrics.

3. **Text features**  
   For each record, build a single text field: concatenation of `title`, `abstract`, and `categories` (nulls replaced with empty string). No vectorizer is fitted in this step; the same text is passed to the model layer.

4. **Model**  
   A scikit-learn `Pipeline` with:
   - **TfidfVectorizer**: `max_features=50_000`, `ngram_range=(1, 2)`, `sublinear_tf=True`
   - **LogisticRegression**: `max_iter=500`, `class_weight='balanced'`

   The pipeline is fitted on the training text and targets. Validation metrics (accuracy, macro F1, log loss) are computed and logged for transparency only; no hyperparameter tuning was performed.

5. **Predictions**  
   For the full merged dataset (every id that appears in both files), we build text features with the same function used at train time, run `predict_proba`, and write a CSV with columns `id`, `prob_1`, `prob_2`, `prob_3`, `prob_4`. Rows are ordered to match `sample_targets.csv`.

---

## Key decisions

- **Logistic Regression + TF-IDF**  
  Chosen as the best fit for this task: strong baseline for multi-class text classification, native probability outputs (`predict_proba`), interpretable and maintainable, and no extra calibration step. Design and maintainability were prioritized over squeezing the last bit of accuracy (e.g. no transformers or heavy tuning).

- **Single concatenated text**  
  Title, abstract, and categories are concatenated into one string before TF-IDF. Simple and effective; with more time we could add separate weighting or separate vectorizers for title vs abstract.

- **Stratified split**  
  Ensures validation set reflects class distribution; important for stable metric estimates.

- **No separate test file**  
  The task provides one dataset; we split it into train/val. With more time we would add a second split (e.g. 70/15/15) and report only on the held-out test set.

- **One code path for features**  
  The same `build_text_features` is used for training, validation evaluation, and final prediction, avoiding drift and bugs.

---

## What would be done differently with more time

- **Hyperparameter tuning**  
  Grid or random search over vectorizer (e.g. `max_features`, `ngram_range`) and classifier (e.g. `C`, solver) with cross-validation.

- **Richer feature engineering**  
  Separate TF-IDF for title vs abstract with different weights; one-hot or embedding for categories; handling of missing abstract/categories more explicitly.

- **Proper train/val/test split**  
  Reserve a test set and report only test metrics; use validation only for model selection and tuning.

- **More robust evaluation**  
  Cross-validation for stability; per-class metrics and confusion matrix in the write-up.

---

## Production-style improvements

- **Config and logging**  
  Paths, seed, and split ratio are already in `config.py`. Replace `print` with structured logging and optional config files (e.g. YAML) for different environments.

- **Model versioning**  
  Save the fitted pipeline (e.g. `joblib.dump`) with a version or timestamp; document how to load and run inference.

- **Tests**  
  Unit tests for `load_and_merge` (e.g. expected columns, handling of missing ids), `train_val_split` (stratification, sizes), and `build_text_features` (null handling, shape).

- **CI**  
  Run tests and optionally a small pipeline run on a subset of data in CI.

- **Containers**  
  Dockerfile for reproducible runs.

- **True multi-label**  
  If the task were multi-label (multiple labels per sample), we would use `OneVsRestClassifier` with a probabilistic base estimator or a neural multi-label head and output one probability per label.

---

## Deliverables

- **Code**: `pyproject.toml`, `.python-version`, `run.py`, `src/ml_classifier/` (config, cli, data, features, model), `README.md`. Dependencies are in `pyproject.toml`; use `uv sync` to install.
- **Predictions**: each run writes to `output/run_YYYYMMDD_HHMMSS/predictions.csv` so results are never overwritten; one row per id (aligned with `sample_targets.csv`), columns `id`, `prob_1`, `prob_2`, `prob_3`, `prob_4` (probability distribution per row).
- **Description**: This document.
