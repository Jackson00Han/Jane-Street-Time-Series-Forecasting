# Jane Street Market TimeSeries Data Forecasting --LightGBM

This is a practical pipeline for the Jane Street time-series market forecasting competition ([Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)). It covers data cleaning → feature engineering → panelization → memmap building → LightGBM sliding-CV training → reporting, and scales to very large datasets (30M+ rows, ~1k features) with strict causal handling.


## Highlights

* Strong offline CV: weighted zero-mean R-squared score (WR2) ≈ 0.01435

* Fast pipeline with Polars: 20x faster than pandas for data preprocessing

* Robust outlier & missing-value handling: 3σ clipping + TTL/forward-fill

* Config-first reproducibility: a single config.yaml controls data ranges, FE, CV, and model hyperparameters.

* Multi-dimensional feature engineering: daily, same-time cross-day, and tick-history features.

* Memory-mapped matrices (memmap): fast reloads and lower memory during training.


## Repository Structure

```
JS/
├─ config/
│  ├─ config.yaml
├─ notebooks/
│  └─ eda.ipynb
├─ src/
│  └─ pipeline/
│     ├─ entrypoints/
│     │  ├─ clean.py
│     │  ├─ feature_selection.py
│     │  ├─ hp_tune.py
│     │  ├─ memmap_after_fs.py
│     │  ├─ memmap_fs.py
│     │  ├─ panel.py
│     │  └─ main_train.py
│     ├─ backtest.py
│     ├─ features.py
│     ├─ io.py
│     ├─ memmap.py
│     ├─ metrics.py
│     ├─ preprocess.py
│     ├─ validate.py
│     └─ __init__.py
├─ .env
├─ .gitignore
├─ pyproject.toml
├─ README.md
└─ uv.lock
```

## Environment & Reproducibility

- Using uv (recommended)
```bash
python -m venv .venv && source .venv/bin/activate
uv sync
uv pip install -e .
```


## How to run
Minimal edits: make sure config/config.yaml has correct azure.root / local.root
```
# 0) Raw → cleaned shards (optional if you already have clean data)
js-clean

# 1) Feature engineering (Stages A/B/C → fe_shards/)
js-fe

# 2) Panelization (join shards, causal ordering → panel_shards/)
js-panel

# 3) Memmap matrix (build small dataset with all features)
js-memmap-fs

# 4) train model to select important features on a small dataset
js-feature-selection

# 5) Memmap matrix (build full dataset with selected features)
js-memmap-after_fs

# 6) train model on full data with selected features 
js-train

```

## Results

Example outputs are available in the [`reports`](./reports/) folder. These are for reference only—you can improve them by further tuning through `config.yaml`.


