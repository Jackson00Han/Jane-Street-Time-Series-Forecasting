# run_5_tune.py  —— 固定超参版（无 Optuna）
from __future__ import annotations
import os, json, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

from pipeline.io import cfg, P, ensure_dir_local
from pipeline.memmap import make_sliding_cv_fast

# ---------- utils ----------
def load_mm(prefix: str):
    meta_path = f"{prefix}.meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    N, F = meta["n_rows"], meta["n_feat"]
    X = np.memmap(f"{prefix}_X.float32.mmap", dtype=np.float32, mode="r", shape=(N, F))
    y = np.memmap(f"{prefix}_y.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    w = np.memmap(f"{prefix}_w.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    d = np.memmap(f"{prefix}_date.int32.mmap", dtype=np.int32,   mode="r", shape=(N,))
    return meta, X, y, w, d


def lgb_wr2_eval(y_pred: np.ndarray, dataset: lgb.Dataset):
    # Weighted R^2（higher is better）
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true, dtype=np.float64)
    w = w.astype(np.float64, copy=False)
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    w_sum = w.sum()
    y_bar = (w * y_true).sum() / max(w_sum, 1e-12)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_bar) ** 2).sum()
    wr2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    return "wr2", float(wr2), True


def _latest_mm_prefix(mm_root: str) -> str:
    metas = [os.path.join(mm_root, f) for f in os.listdir(mm_root) if f.endswith(".meta.json")]
    if not metas:
        raise FileNotFoundError(f"No *.meta.json found under {mm_root}")
    metas.sort(key=lambda p: os.path.getmtime(p))
    return metas[-1][:-len(".meta.json")]  # strip ".meta.json"


def main():
    # =========================
    # 0) 路径
    # =========================
    mm_root = P("local", cfg["paths"]["train_mm"])
    try:
        prefix = cfg["paths"]["train_mm_prefix"]   # 若在 yaml 固定了前缀，优先用这个
    except KeyError:
        prefix = _latest_mm_prefix(mm_root)

    models_dir = P("local", cfg["paths"]["models"])
    tune_dir   = os.path.join(models_dir, "tune")
    ensure_dir_local(models_dir); ensure_dir_local(tune_dir)

    # =========================
    # 1) 载入 memmap（已是“选列后”的瘦矩阵）
    # =========================
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap d 不是非降序；请检查 panel 分片或排序"
    print(f"[fixed] loaded memmap: N={meta['n_rows']:,}, F={meta['n_feat']}, prefix={prefix}")

    # =========================
    # 2) 选择“小样本日期窗”用于评测固定超参
    # =========================
    try:
        win = cfg["dates"]["tune_dates"]
        lo, hi = int(win["date_lo"]), int(win["date_hi"])
    except KeyError:
        raise RuntimeError("Please specify cfg['dates']['tune_dates'] with date_lo/date_hi.")

    mask = (d >= lo) & (d <= hi)
    idx  = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows in window [{lo}, {hi}]")
    d_sub = d[idx]

    print(f"[fixed] using date range [{lo}, {hi}] -> {idx.size:,} rows")

    # =========================
    # 3) CV 构造（在小窗内）
    # =========================
    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg["n_splits"])
    gap_days = int(cv_cfg["gap_days"])
    ratio    = int(cv_cfg["train_to_val"])
    folds = make_sliding_cv_fast(d_sub, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed.")
    print(f"[fixed] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")

    # =========================
    # 4) 数据集基座（先构建，用 subset 派生）
    # =========================
    seed_val = int(cfg["seed"])
    ds_params = dict(
        max_bin=63,
        bin_construct_sample_cnt=200000,
        min_data_in_bin=3,
        data_random_seed=seed_val,
    )
    d_all = lgb.Dataset(X, label=y, weight=w, feature_name=feat_cols, free_raw_data=True, params=ds_params)
    d_base = d_all.subset(idx, params=ds_params)

    # =========================
    # 5) 固定超参数（可由 cfg['lgb+fixed'] 覆盖）
    # =========================
    lgb_tune_cfg = cfg.get("lgb+tune", {})  # 只取 device_type、早停、rounds
    es_rounds = int(lgb_tune_cfg.get("early_stopping_rounds", 200))
    log_period = int(lgb_tune_cfg.get("log_period", 50))
    tune_rounds = int(lgb_tune_cfg.get("num_boost_round", 3000))

    device = lgb_tune_cfg.get("device_type", "cpu")
    fixed_default = dict(
        objective="regression",
        metric="None",
        device_type=device,
        seed=seed_val,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,
        min_data_in_leaf=200,
        # 需要可再加：lambda_l1=0.0, lambda_l2=0.0, min_gain_to_split=0.0
    )
    fixed_params = {**fixed_default, **cfg.get("lgb+fixed", {})}
    print(f"[fixed] params: {fixed_params}")

    # =========================
    # 6) 逐折训练与评估（无 Optuna）
    # =========================
    fold_scores, best_iters = [], []
    for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="fixed_cv", leave=False), 1):
        assert np.all(np.diff(d_sub[tr_idx]) >= 0), f"train dates not sorted in fold {k}"
        assert np.all(np.diff(d_sub[va_idx]) >= 0), f"val dates not sorted in fold {k}"
        tr_lo, tr_hi = int(d_sub[tr_idx[0]]), int(d_sub[tr_idx[-1]])
        va_lo, va_hi = int(d_sub[va_idx[0]]), int(d_sub[va_idx[-1]])
        print(f"[cv] fold{k}: train_d=[{tr_lo}-{tr_hi}] n_tr={len(tr_idx):,} | "
            f"val_d=[{va_lo}-{va_hi}] n_va={len(va_idx):,}")

        
        dtrain = d_base.subset(tr_idx, params=ds_params)
        dvalid = d_base.subset(va_idx, params=ds_params)

        bst = lgb.train(
            {**fixed_params, "verbosity": -1}, 
            dtrain,
            valid_sets=[dvalid],
            valid_names=["val"],
            feval=lgb_wr2_eval,
            num_boost_round=tune_rounds,
            callbacks=[
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
                lgb.log_evaluation(period=log_period),
            ],
        )
        wr2_k = float(bst.best_score["val"]["wr2"])
        it_k  = int(bst.best_iteration)
        fold_scores.append(wr2_k)
        best_iters.append(it_k)

        # 释放
        bst.free_dataset(); del dtrain, dvalid, bst
        gc.collect()

    mean_wr2 = float(np.mean(fold_scores))
    print(f"[fixed] mean_wr2={mean_wr2:.6f} | per-fold={np.round(fold_scores, 6)}")

    # =========================
    # 7) 落盘：固定参数的 CV 结果
    # =========================
    ts = int(time.time())
    tag = f"fixed__mm_{Path(prefix).name}__range{lo}-{hi}__cv{n_splits}-g{gap_days}-r{ratio}__{ts}"

    # summary json
    out_path = os.path.join(tune_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": prefix,
            "tune_range": [int(lo), int(hi)],
            "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
            "ds_params": ds_params,
            "mean_wr2": mean_wr2,
            "fold_scores": fold_scores,
            "best_iterations": best_iters,
            "num_boost_round": tune_rounds,
            "early_stopping_rounds": es_rounds,
            "seed": seed_val,
            "features": feat_cols,
            "n_rows": int(meta["n_rows"]),
            "n_feat": int(meta["n_feat"]),
            "params": fixed_params,
            "tag": tag,
        }, f, indent=2)

    # 每折排行榜（便于横向比较）
    rows = []
    for i, (score, iters) in enumerate(zip(fold_scores, best_iters), 1):
        rows.append({"fold": i, "wr2": score, "best_iteration": iters})
    df = pd.DataFrame(rows).sort_values("wr2", ascending=False)
    rank_path = os.path.join(tune_dir, f"folds__{tag}.csv")
    df.to_csv(rank_path, index=False)

    print(f"[fixed][done] summary -> {out_path}")
    print(f"[fixed][done] folds   -> {rank_path}")


if __name__ == "__main__":
    main()
