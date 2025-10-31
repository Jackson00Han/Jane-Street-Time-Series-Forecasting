# Jane Street Market TimeSeries Data Forecasting --GBM/Transformer

A practical pipeline for the Jane Street 2024-style intraday time-series task. It covers data cleaning → feature engineering → panelization → memmap building → LightGBM CV training → reporting. Designed for very large datasets (30M+ rows; 1k+ features) with strict causal handling. 

Additionally, I also tried TFT model, since it was very much time consuming, any new progress about it will be updated in the experiments.

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
├─ .env #(not published)
├─ pyproject.toml
├─ README.md
└─ uv.lock
```

