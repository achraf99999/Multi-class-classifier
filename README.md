# Multi-class classifier (arXiv metadata)

This project trains a **multi-class text classifier** on arXiv paper metadata and outputs a **probability distribution over four classes** for each paper. Records are linked by `id` between the metadata and the target labels; the model uses **title**, **abstract**, and **categories** to predict the label and writes one row per `id` with columns `prob_1`–`prob_4`. For a more detailed description of the solution approach, key decisions, and what we would do differently with more time or in production, see [SOLUTION.md](SOLUTION.md).

## Parameters and how the pipeline works

The pipeline uses **sample_data.json** (features: title, abstract, categories) and **sample_targets.csv** (labels: id, target), merged on `id`. See **Data** for where to place the files and how the merge works.

You can choose how text is represented before classification:

- **`--method tfidf`** (default) — Text is vectorized with **TfidfVectorizer**, then classified with **LogisticRegression**. Fast, no GPU required.
- **`--method legalbert`** — Text is embedded with [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) ([nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased)), then a **LogisticRegression** head is trained on the [CLS] embeddings. Better for legal/formal text; requires more memory and optionally a GPU. See [Legal-BERT (opensource.legal)](https://opensource.legal/projects/Legal_BERT).

**Pipeline parameters (TF-IDF only)**

| Parameter      | Role                            | Default | CLI override       |
|----------------|----------------------------------|---------|--------------------|
| `max_features` | TfidfVectorizer vocabulary size | 50,000  | `--max-features N` |
| `ngram_range`  | TfidfVectorizer n-grams         | (1, 2)  | `--ngram-range 1,2`|
| `max_iter`     | LogisticRegression iterations   | 500     | `--max-iter N`      |

**Run mode**

- **Default** — Uses config (and built-in defaults). With `--method tfidf`, if `USE_HYPERPARAMETER_TUNING` is true in config, GridSearchCV runs.
- **Override** — Pass `--max-features`, `--ngram-range`, and/or `--max-iter` to fix those for the run (TF-IDF only).
- **Tuning** — `--tune` / `--no-tune` and `--tuning-cv K` apply only to the TF-IDF pipeline. Legal-BERT uses fixed LogisticRegression on top of frozen embeddings.

## Requirements

- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.10+

## Setup

From the project root:

```bash
uv sync
```

This creates a virtual environment (`.venv`) and installs the package in editable mode. 

**Without uv:** `pip install -e ".[dev]"` (from the project root).

## Data

The pipeline reads **sample_data.json** and **sample_targets.csv** from **`input_data/`** inside the project. Put both files there so the default config finds them without changes.

**Required files**

- **sample_data.json** — Sample data for training: one JSON object per paper (`id`, `title`, `abstract`, `categories`, etc.). Used as features.
- **sample_targets.csv** — Labels to predict: one row per paper with `id` and `target` (integer 1–4). Used as the prediction target.

Both files are linked by their shared **`id`** feature. The pipeline **merges** them on `id` (inner join): each row then has metadata from the JSON and the label from the CSV, so the model can learn from the former and predict the latter. Predictions are produced for every `id` present in `sample_targets.csv` (that also exists in the JSON).

**Directory layout**

Place the data files in `input_data/` at the project root:

```
ml_classifier/
  input_data/
    sample_data.json
    sample_targets.csv
  pyproject.toml
  run.py
  src/
    ml_classifier/
  ...
```

To use a different location, set `DATA_DIR`, `SAMPLE_DATA_PATH`, or `SAMPLE_TARGETS_PATH` in `src/ml_classifier/config.py`.

## Run

From the project root:

```bash
uv run ml-classifier
```

Or:

```bash
uv run python run.py
```

This will load and merge the data, split into train/validation (80/20, stratified), build text features, train the pipeline, predict on all merged ids, and write results under `output/run_YYYYMMDD_HHMMSS/`.

### CLI behavior

- **Text representation** — `--method tfidf` (default) or `--method legalbert` to choose between TF-IDF and Legal-BERT embeddings.
- **Run with no args** — Uses defaults from config (and built-in defaults for model params). With TF-IDF, if `USE_HYPERPARAMETER_TUNING` is true in config, tuning runs.
- **Override params** — Pass `--max-features N`, `--ngram-range MIN,MAX` (e.g. `1,2`), and/or `--max-iter N` to override those pipeline parameters (TF-IDF only). Unset options fall back to config or built-in defaults.
- **Tuning mode** — `--tune` / `--no-tune` and `--tuning-cv K` apply only when `--method tfidf`.

Examples:

```bash
uv run ml-classifier --method tfidf --no-tune
uv run ml-classifier --method legalbert
uv run ml-classifier --max-features 30000 --ngram-range 1,2 --no-tune
uv run ml-classifier --tune --tuning-cv 5
uv run ml-classifier --help
```

## Outputs

- **output/run_YYYYMMDD_HHMMSS/** — Each run gets its own timestamped folder so results are never overwritten. Inside: **predictions.csv** (one row per id; columns `id`, `prob_1`, `prob_2`, `prob_3`, `prob_4`) and **run_info.json** (run metadata: data counts, config, validation metrics, best_params when tuning). The `output` folder is created if needed.
- **SOLUTION.md** — Description of the solution, decisions, and what would be done differently with more time or in production.

## Project structure

All source code lives under **`src/ml_classifier/`** (the installable package). Nothing else at the repo root except config files, the entry script, and generated output.

```
ml_classifier/
├── pyproject.toml          # Project metadata, deps, script entry point (ml-classifier)
├── .python-version        # Python 3.10 for uv
├── run.py                 # Thin wrapper: calls ml_classifier.cli.main()
├── README.md
├── SOLUTION.md
├── input_data/            # Input data (default location)
│   ├── sample_data.json
│   └── sample_targets.csv
├── output/
│   ├── run_20260226_143022/
│   │   ├── predictions.csv   # One run’s results (timestamped; no overwrite)
│   │   └── run_info.json     # Run metadata, validation metrics, best_params
│   └── run_20260226_150011/
│       ├── predictions.csv
│       └── run_info.json
│
└── src/ml_classifier/     # The package (import name: ml_classifier)
    ├── __init__.py
    ├── config.py          # Paths (data in input_data/), SEED, VAL_RATIO, class labels
    ├── cli.py             # Single entrypoint: load → split → train → predict → save CSV
    │
    ├── data/              # Data layer only (I/O, no features or model)
    │   ├── load.py        # load_and_merge(json, csv) → merged DataFrame
    │   └── split.py       # train_val_split(merged, stratify=target) → train_df, val_df
    │
    ├── features/          # Feature layer: build text for the model
    │   └── build.py       # build_text_features(df) → title + abstract + categories (no vectorizer)
    │
    └── model/             # Model layer: pipeline + predict
        ├── train.py       # TF-IDF: train_pipeline / train_with_cv
        ├── legalbert.py   # Legal-BERT: embeddings + LogisticRegression
        └── predict.py     # predict_proba_df(estimator, df, class_labels) → CSV-shaped DataFrame
```

**Flow:** `cli.py` imports from `config`, `data`, `features`, and `model`; runs the pipeline; writes to `output/run_YYYYMMDD_HHMMSS/predictions.csv` so every run is kept. Same feature builder is used for training and prediction (one code path).

**Linting/formatting:** `uv run ruff check src` and `uv run ruff format src`.

---



## References

- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) — scikit-learn: exhaustive search over specified parameter values for an estimator.
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — scikit-learn: logistic regression (logit, MaxEnt) classifier.
