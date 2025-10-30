# run_5_tune.py — Fixed-hyperparameter
from __future__ import annotations
import os, json, time, gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

from pipeline.io import cfg, ensure_dir_local
from pipeline.backtest import make_sliding_cv
from pipeline.metrics import lgb_wr2
from contextlib import redirect_stderr


# ---------- utils ----------
def load_mm(prefix: str):
    """
    Load memmap-backed arrays along with metadata.

    Expected files:
      - {prefix}_X.float32.mmap   : shape (N, F)
      - {prefix}_y.float32.mmap   : shape (N,)
      - {prefix}_w.float32.mmap   : shape (N,)
      - {prefix}_date.int32.mmap  : shape (N,)
      - {prefix}.meta.json        : contains 'n_rows', 'n_feat', 'features'
    """
    meta_path = f"{prefix}.meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    N, F = meta["n_rows"], meta["n_feat"]
    X = np.memmap(f"{prefix}_X.float32.mmap", dtype=np.float32, mode="r", shape=(N, F))
    y = np.memmap(f"{prefix}_y.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    w = np.memmap(f"{prefix}_w.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    d = np.memmap(f"{prefix}_date.int32.mmap", dtype=np.int32, mode="r", shape=(N,))
    return meta, X, y, w, d

def main():
    # =========================
    # 0) Paths / I/O setup
    # =========================
    local_root = cfg["local"]["root"]
    path_mm = cfg["derived_paths"].get("train_mm", None)
    report_root = f"{local_root}/{cfg['paths']['reports']['rel']}"; ensure_dir_local(report_root)
    # =========================
    # 1) Load memmap (already column-reduced)
    # =========================
    assert path_mm, "cfg['derived_paths']['train_mm'] is missing."
    assert os.path.exists(f"{path_mm}.meta.json"), f"memmap not found: {path_mm}.meta.json"

    meta, X, y, w, d = load_mm(path_mm)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap 'date' is not non-decreasing; check panel ordering."
    print(f"[fixed] loaded memmap: N={meta['n_rows']:,}, F={meta['n_feat']}, path_mm={path_mm}")

    # =========================
    # 2) Select training window by date
    # =========================
    win = cfg["dates"]["train"]
    lo, hi = int(win["date_lo"]), int(win["date_hi"])

    mask = (d >= lo) & (d <= hi)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows in window [{lo}, {hi}]")
    d_sub = d[idx]
    print(f"[fixed] using date range [{lo}, {hi}] -> {idx.size:,} rows")

    # =========================
    # 3) Build sliding CV folds
    # =========================
    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg["n_splits"])
    gap_days = int(cv_cfg["gap_days"])
    ratio = int(cv_cfg["train_to_val"])
    folds = make_sliding_cv(d_sub, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed.")
    print(f"[fixed] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")


    # =========================
    # 4) Build base Dataset (subsets will be derived from it)
    # =========================
    seed_val = int(cfg["seed"])
    ds_params = dict(
        max_bin=cfg["models"]["lgbm_train"]["params"].get("max_bin", 128),
        bin_construct_sample_cnt=200000,
        min_data_in_bin=cfg["models"]["lgbm_train"]["params"].get("min_data_in_bin", 1),
        data_random_seed=seed_val,
    )
    d_all = lgb.Dataset(X, label=y, weight=w, feature_name=feat_cols, free_raw_data=True, params=ds_params)
    d_base = d_all.subset(idx, params=ds_params)

    # =========================
    # 5) Fixed LightGBM hyperparameters
    # =========================
    lgbm_cfg = cfg["models"]["lgbm_train"]["params"]
    es_rounds = int(lgbm_cfg.get("early_stopping_rounds", 200))
    log_period = int(lgbm_cfg.get("log_period", 50))
    tune_rounds = int(lgbm_cfg.get("num_boost_round", 3000))

    device = lgbm_cfg.get("device_type", "cpu")
    fixed_params = dict(
        objective="regression",
        metric="None",
        verbosity=-1,
        device_type=device,
        seed=seed_val,
        learning_rate=lgbm_cfg.get("learning_rate", 0.01),
        num_leaves=lgbm_cfg.get("num_leaves", 63),
        max_depth=lgbm_cfg.get("max_depth", 8),
        feature_fraction=lgbm_cfg.get("feature_fraction", 0.8),
        bagging_fraction=lgbm_cfg.get("bagging_fraction", 0.8),
        bagging_freq=lgbm_cfg.get("bagging_freq", 1),
        min_data_in_leaf=lgbm_cfg.get("min_data_in_leaf", 200),
        min_sum_hessian_in_leaf=lgbm_cfg.get("min_sum_hessian_in_leaf", 1.0),
        min_gain_to_split=lgbm_cfg.get("min_gain_to_split", 0.00),
        lambda_l2=lgbm_cfg.get("lambda_l2", 10.0),
        lambda_l1=lgbm_cfg.get("lambda_l1", 0.1),
    )
    print(f"[fixed] params: {fixed_params}")


    # =========================
    # 6) Train and evaluate across folds
    # =========================
    fold_scores, best_iters = [], []

    # Per-fold feature importance (gain share)
    ranking_features = pd.DataFrame({"feature": feat_cols})

    for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="fixed_cv", leave=False), 1):
        assert np.all(np.diff(d_sub[tr_idx]) >= 0), f"train dates not sorted in fold {k}"
        assert np.all(np.diff(d_sub[va_idx]) >= 0), f"val dates not sorted in fold {k}"
        tr_lo, tr_hi = int(d_sub[tr_idx[0]]), int(d_sub[tr_idx[-1]])
        va_lo, va_hi = int(d_sub[va_idx[0]]), int(d_sub[va_idx[-1]])
        print(f"[cv] fold{k}: train_d=[{tr_lo}-{tr_hi}] n_tr={len(tr_idx):,} | "
              f"val_d=[{va_lo}-{va_hi}] n_va={len(va_idx):,}")

        dtrain = d_base.subset(tr_idx, params=ds_params)
        dvalid = d_base.subset(va_idx, params=ds_params)
        callbacks=[lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),lgb.log_evaluation(period=log_period)]
        
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            bst = lgb.train(
                {**fixed_params},
                dtrain,
                valid_sets=[dvalid, dtrain],
                valid_names=["val", "train"],
                feval=lgb_wr2,
                num_boost_round=tune_rounds,
                callbacks=callbacks,
            )
        wr2_k = float(bst.best_score["val"]["wr2"])
        it_k = int(bst.best_iteration)
        fold_scores.append(wr2_k)
        best_iters.append(it_k)

        g = bst.feature_importance(importance_type="gain", iteration=bst.best_iteration or tune_rounds).astype(float)
        ranking_features[f"fold_{k}_gain"] = (g / g.sum()) if g.sum() > 0 else np.zeros_like(g, dtype=float)

        bst.free_dataset(); del dtrain, dvalid, bst
        gc.collect()

    mean_wr2 = float(np.mean(fold_scores))
    print(f"[fixed] mean_wr2={mean_wr2:.6f} | per-fold={np.round(fold_scores, 6)}")

    ts = int(time.time())
    tag = f"train_range{lo}_{hi}__cv{n_splits}_g{gap_days}_r{ratio}__{ts}"
    
    # =========================
    # 7) Persist reports (feature importance, CV summary)
    # =========================
    fi_col = [c for c in ranking_features.columns if c.startswith("fold_")]
    ranking_features["mean_gain"] = ranking_features[fi_col].mean(axis=1)
    ranking_features = ranking_features.sort_values("mean_gain", ascending=False, ignore_index=True)

    fi_path = os.path.join(report_root, f"final_feature_importance_{tag}.csv")
    ranking_features[["feature", "mean_gain"]].to_csv(fi_path, index=False)
    print(f"[fixed][done] feature importance -> {fi_path}")

    out_path = os.path.join(report_root, f"summary_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": path_mm,
            "train_range": [int(lo), int(hi)],
            "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
            "ds_params": ds_params,
            "mean_wr2": mean_wr2,
            "fold_scores": fold_scores,
            "best_iterations": best_iters,
            "num_boost_round": tune_rounds,
            "early_stopping_rounds": es_rounds,
            "seed": seed_val,
            "fi_path": fi_path,
            "n_rows": int(meta["n_rows"]),
            "n_feat": int(meta["n_feat"]),
            "params": fixed_params,
            "tag": tag,
        }, f, indent=2)

    rows = []
    for i, (score, iters) in enumerate(zip(fold_scores, best_iters), 1):
        rows.append({"fold": i, "wr2": score, "best_iteration": iters})
    df = pd.DataFrame(rows).sort_values("wr2", ascending=False)
    rank_path = os.path.join(report_root, f"validation_report_{tag}.csv")
    df.to_csv(rank_path, index=False)

    print(f"[fixed][done] summary -> {out_path}")
    print(f"[fixed][done] validation report -> {rank_path}")


if __name__ == "__main__":
    main()
