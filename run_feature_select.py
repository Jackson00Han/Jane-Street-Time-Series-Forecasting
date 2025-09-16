# run_feature_select.py
from __future__ import annotations
import os, json, time, gc, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb

from pipeline.io import cfg, P, ensure_dir_local
from pipeline.memmap import make_sliding_cv_fast

# -------- utils --------
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
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true, dtype=np.float64)
    # 加权 R²
    w = w.astype(np.float64, copy=False)
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    w_sum = w.sum()
    y_bar = (w * y_true).sum() / max(w_sum, 1e-12)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_bar) ** 2).sum()
    wr2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    return "wr2", float(wr2), True   # higher is better

def tag_for_fs(fs_lo:int, fs_hi:int, n_splits:int, gap:int, ratio:int, seed:int, top_k:int, ts:int):
    return f"fs__{fs_lo}-{fs_hi}__cv{n_splits}-g{gap}-r{ratio}__seed{seed}__top{top_k}__{ts}"

def main():
    # ---------- paths ----------
    mm_root   = P("local", cfg["paths"]["sample_mm"])
    prefix    = os.path.join(mm_root, "full_sample_v1")     # 与 run_memmap.py 保持一致
    rep_dir   = os.path.join(P("local", cfg["paths"]["reports"]), "fi")
    featset_dir   = os.path.join(P("local", cfg.get("paths", {}).get("models", "exp/v1/models")), "feature_set")
    ensure_dir_local(rep_dir); ensure_dir_local(featset_dir)

    # ---------- load memmap ----------
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap d 不是非降序；请检查 panel 分片或排序"
    print(f"[fs] N={meta['n_rows']:,}, F={meta['n_feat']}")



    # ---------- sample date range  ----------
    # 建议把范围放到 cfg 里：feature_select: { date_lo: 1000, date_hi: 1100 }
    fs_cfg = cfg['dates']["feature_select_dates"]
    fs_lo = int(fs_cfg.get("date_lo", 1000))
    fs_hi = int(fs_cfg.get("date_hi", 1100))

    mask = (d >= fs_lo) & (d <= fs_hi)
    subset_idx = np.flatnonzero(mask)
    if subset_idx.size == 0:
        raise ValueError(f"No rows in date range [{fs_lo}, {fs_hi}]")
    d_sub = d[subset_idx]
    assert np.all(np.diff(d_sub) >= 0), "子集 d 非单调；请检查 memmap"

    print(f"[fs] using date range [{fs_lo}, {fs_hi}] -> {subset_idx.size:,} rows")

    # ---------- CV folds ----------
    cv_cfg = cfg['cv']
    n_splits = int(cv_cfg.get("n_splits", 2))
    gap_days = int(cv_cfg.get("gap_days", 5))
    ratio    = int(cv_cfg.get("train_to_val", 9))
    
    seed_val = int(cfg["seed"])
    top_k    = int(cfg["lgb_select"].get("top_k", 632))
    train_lo, train_hi = cfg["dates"]["train_lo"], cfg["dates"]["train_hi"]
    
    
    folds = make_sliding_cv_fast(d_sub, n_splits = n_splits, gap_days = gap_days, train_to_val = ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed. Check cv settings and date order.")
    print(f"[fs] folds={len(folds)}, top_k={top_k}, seed={seed_val}")


    # ---------- LightGBM params ----------
    ds_params = dict(
        max_bin=63,
        bin_construct_sample_cnt=200000,
        min_data_in_bin=3,
        data_random_seed=int(cfg['seed']),
    )
    lgb_cfg = cfg['lgb_select']
    params = dict(
        objective="regression",
        metric="None",
        device_type=lgb_cfg.get("device_type", "gpu"),   # 如需 CPU 改为 "cpu"
        learning_rate=float(lgb_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgb_cfg.get("num_leaves", 63)),
        max_depth=int(lgb_cfg.get("max_depth", 8)),
        feature_fraction=float(lgb_cfg.get("feature_fraction", 0.80)),
        bagging_fraction=float(lgb_cfg.get("bagging_fraction", 0.80)),
        bagging_freq=int(lgb_cfg.get("bagging_freq", 1)),
        min_data_in_leaf=int(lgb_cfg.get("min_data_in_leaf", 200)),
        seed=seed_val
        # 若 CPU：可以加 num_threads=...
    )
    num_boost_round = int(lgb_cfg.get("num_boost_round", 4000))
    es_rounds       = int(lgb_cfg.get("early_stopping_rounds", 100))
    log_period      = int(lgb_cfg.get("log_period", 100))

    # ---------- dataset ----------
    d_all = lgb.Dataset(
        X, label=y, weight=w,
        feature_name=feat_cols,
        free_raw_data=True,       # 构好直方图后释放原始矩阵（节省内存）
        params=ds_params,
    )

    # ---------- training loop ----------
    scores = []
    fi = pd.DataFrame({"feature": feat_cols})

    for k, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"[model] fold {k}/{len(folds)}: train={tr_idx.size:,}, val={va_idx.size:,}")
        dtrain = d_all.subset(tr_idx, params=ds_params)
        dvalid = d_all.subset(va_idx, params=ds_params)

        bst = lgb.train(
            params, dtrain,
            valid_sets=[dvalid, dtrain],
            valid_names=["val", "train"],
            feval=lgb_wr2_eval,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=True),
                lgb.log_evaluation(period=log_period),
            ],
        )

        scores.append(float(bst.best_score["val"]["wr2"]))
        g = bst.feature_importance(importance_type="gain", iteration=bst.best_iteration or num_boost_round).astype(float)
        fi[f"fold{k}_gain_share"] = (g / g.sum()) if g.sum() > 0 else np.zeros_like(g, dtype=float)
        bst.free_dataset(); del dtrain, dvalid, bst; gc.collect()


    fi_cols = [c for c in fi.columns if c.startswith("fold")]
    fi["mean_gain_share"] = fi[fi_cols].mean(axis=1)
    fi = fi.sort_values("mean_gain_share", ascending=False, ignore_index=True)

    # ---------- select features ----------
    whitelist = list(cfg.get("white_list", []))
    fi_normal = fi[~fi["feature"].isin(whitelist)].reset_index(drop=True)
    final_feats = list(dict.fromkeys(whitelist + fi_normal["feature"].head(top_k).tolist()))
    print(f"[fs] selected {len(final_feats)} features (whitelist {len(whitelist)})")
    

    # 输出
    ts = int(time.time())
    tag = tag_for_fs(fs_lo, fs_hi, n_splits, gap_days, ratio, seed_val, top_k, ts)
    
    # 报告目录
    fi_path   = os.path.join(rep_dir, f"fi_gain_share__{tag}.csv")
    lst_path  = os.path.join(rep_dir, f"features__{tag}.txt")
    sum_path  = os.path.join(rep_dir, f"summary__{tag}.json")

    # 同步一份“可被训练脚本引用”的特征清单
    lst4train = os.path.join(featset_dir, f"{tag}.txt")

    # 写文件
    fi[["feature", "mean_gain_share"]].to_csv(fi_path, index=False)
    with open(lst_path, "w", encoding="utf-8") as f:
        for x in final_feats:
            f.write(f"{x}\n")
    with open(lst4train, "w", encoding="utf-8") as f:
        for x in final_feats:
            f.write(f"{x}\n")
    
    # 摘要
    summary = {
        "cv_scores": scores,
        "cv_mean_wr2": float(np.mean(scores)),
        "cv_std_wr2": float(np.std(scores)),
        "memmap_prefix": prefix,
        "n_rows": int(meta["n_rows"]),
        "n_feat": int(meta["n_feat"]),
        "selected_features": len(final_feats),
        "whitelist": whitelist,
        "fi_csv": fi_path,
        "features_txt": lst_path,
        "features_txt_for_training": lst4train,
        "train_range": [int(train_lo), int(train_hi)],
        "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
        "lgb_params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": es_rounds,
        "seed": seed_val,
        "tag": tag,
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[fs][done] mean_wr2={summary['cv_mean_wr2']:.6f}  (±{summary['cv_std_wr2']:.6f})")
    print(f"[fs][done] FI -> {fi_path}")
    print(f"[fs][done] features -> {lst_path} (and {lst4train})")
    print(f"[fs][done] summary -> {sum_path}")

if __name__ == "__main__":
    main()