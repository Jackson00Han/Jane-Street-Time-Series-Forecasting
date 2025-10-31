# Jane Street Market TimeSeries Data Forecasting --GBM/Transformer

A practical pipeline for the Jane Street 2024-style intraday time-series task. It covers data cleaning → feature engineering → panelization → memmap building → LightGBM CV training → reporting. Designed for very large datasets (30M+ rows; 1k+ features) with strict causal handling. 

In addition, I experimented with a Temporal Fusion Transformer (TFT). Because training is computationally expensive at this scale, ongoing results are tracked in the Experiments section and will be updated over time. At this stage, TFT is not yet production-ready for this project. But you are welcome to reach out to me if you find any interesting results.

## Highlights

* Consistent top-tier offline CV performance: WR2 ≈ 0.01435

    - Sliding-window CV with a 7-day gap and 4:1 train-to-val split consistently achieves WR2 ≈ 0.01435 (mean across folds), with best iterations around ~1.1–1.3k. Using data ranging from 900 to 1698.

* Polars streaming pipeline: fast and lean
    - End-to-end data processing is implemented in Polars LazyFrame, delivering ~20× speedups vs. pandas on this workload while keeping memory modest for a large matrix (~30.9M rows × ~1k features). Runs comfortably on 16 CPU cores and ~100 GB RAM.

* Configuration-first, reproducible design

    - A single config.yaml governs data ranges, storage, winsorization, imputation, three-stage feature engineering, CV, and model hyperparameters. This makes experiments auditable, repeatable, and easy to scale. (No FE hyperparameter tuning yet—clear headroom remains.)

* Robust outlier & missing-value handling (causal)
    - Rolling 3-σ clipping per group.

    - A multi-step causal imputation chain: market-open TTL carry → intra-day forward-fill (with max tick gap) → same-time cross-day TTL → final intra-day propagate.

* Flexible, segment-aware feature engineering
    - Daily response features: previous-day close/mean/std, tails & tail-diffs, overnight gap, segmented rollings.
    - Same-time cross-day features: strict prev{k} at the same time_id, last-L mean/std, normalized slope, and cross-responder prev1 stats.
    - Tick-history features: multi-lag diffs/returns, rolling r-mean/r-std/r-z, EWMs, plus optional cross-sectional z-scores/ranks per group.

* Memmap training matrices for rapid iteration
    - Feature-reduced matrices are stored as memory-mapped arrays (X/y/w/date + feature list), which dramatically shortens reload and training cycles, enabling fast CV loops and feature selection.

* Trustworthy validation & reporting
    - Leakage-aware sliding CV (with gap days) and a custom metric(WR2) evaluator. Per-fold feature importance, best iterations, and summaries are exported for traceable model selection.

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
│     │  └─ train.py
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


## Quick Start

If you don‘t want to spend time on data preprocessing and prefer focusing on hyperparameter tuning, you can use my preprocessed memmap on Kaggle (link). All you need to do is go to config/config.yaml, and update the feature_list_path and train_mm by your path. 

For example, you may replace the item of the feature_list_path with the item of the quick_feature_list_path -- "/reports/features__fs__1400-1698__cv1-g7-r4__seed42__top1000__1761771286.txt"

Then, set a short window in the config and run 
```bash
js-train
```

## If you prefer running the complete process (CLI)
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
