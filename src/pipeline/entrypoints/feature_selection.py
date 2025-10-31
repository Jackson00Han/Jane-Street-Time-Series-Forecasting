# feature_select.py
from __future__ import annotations
import os, json, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm  

from pipeline.io import cfg, ensure_dir_local
from pipeline.backtest import make_sliding_cv
from pipeline.metrics import lgb_wr2


# ---------- logging helpers ----------
def _fmt_s(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:02d}m{s:02d}s" if m else f"{s:02d}s")

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

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


def tag_for_fs(fs_lo:int, fs_hi:int, n_splits:int, gap:int, ratio:int, seed:int, top_k:int, ts:int):
    return f"fs__{fs_lo}-{fs_hi}__cv{n_splits}-g{gap}-r{ratio}__seed{seed}__top{top_k}__{ts}"

def main():
    t0 = time.time()
    print(f"[{_now()}][fs] ===== Feature Selection started =====")

    # =========================
    # 0) Paths / prefix
    # =========================
    local_root = cfg["local"]["root"]
    mm_root = f"{local_root}/{cfg['paths']['fs_mm']['rel']}"
    DATE_LO = cfg["dates"]["mfs"]["date_lo"]
    DATE_HI = cfg["dates"]["mfs"]["date_hi"]
    prefix = os.path.join(mm_root, f"fs__{DATE_LO}-{DATE_HI}") 
    
    report_root = f"{local_root}/{cfg['paths']['reports']['rel']}"; ensure_dir_local(report_root)


    # =========================
    # 1) Load memmap
    # =========================
    print(f"[{_now()}][fs] Loading memmap...")
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap 'date' column is not non-decreasing; check panel ordering."
    print(f"[{_now()}][fs] Loaded. N={meta['n_rows']:,}, F={meta['n_feat']}, d_range=[{int(d.min())},{int(d.max())}]")

    # =========================
    # 2) Select FS date subset
    # =========================
    fs_cfg = cfg["dates"]["fs"]
    fs_lo = int(fs_cfg.get("date_lo", DATE_LO))
    fs_hi = int(fs_cfg.get("date_hi", DATE_HI))
    print(f"[{_now()}][fs] Selecting rows in [{fs_lo}, {fs_hi}] (inclusive)...")

    mask = (d >= fs_lo) & (d <= fs_hi)
    subset_idx = np.flatnonzero(mask)
    if subset_idx.size == 0:
        raise ValueError(f"No rows in date range [{fs_lo}, {fs_hi}]")
    d_sub = d[subset_idx]
    assert np.all(np.diff(d_sub) >= 0), "Subset 'date' column is not non-decreasing; check memmap."
    print(f"[{_now()}][fs] Subset ok: rows={subset_idx.size:,}, unique_days={np.unique(d_sub).size}, d_sub=[{int(d_sub.min())},{int(d_sub.max())}]")

    # =========================
    # 3) Build sliding CV folds
    # =========================
    print(f"[{_now()}][fs] Building sliding CV...")
    cv_cfg = cfg['cv']
    n_splits = int(cv_cfg.get("n_splits", 2))
    gap_days = int(cv_cfg.get("gap_days", 7))
    ratio    = int(cv_cfg.get("train_to_val", 5))

    seed_val = int(cfg["seed"])
    top_k    = int(cfg["models"]["lgbm_select"]["params"].get("select_top_k", 1000))

    folds = make_sliding_cv(d_sub, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed. Check cv settings and date order.")
    print(f"[{_now()}][fs] folds={len(folds)}, top_k={top_k}, seed={seed_val}")
    # Log date ranges and sample sizes per fold
    for k,(tr,va) in enumerate(folds,1):
        print(f"[{_now()}][cv] fold{k}: "
              f"train_d=[{int(d_sub[tr].min())}-{int(d_sub[tr].max())}] n_tr={tr.size:,} | "
              f"val_d=[{int(d_sub[va].min())}-{int(d_sub[va].max())}] n_va={va.size:,}")

    # =========================
    # 4) LightGBM params
    # =========================
    ds_params = dict(
        max_bin=31,
        bin_construct_sample_cnt=200000,
        min_data_in_bin=3,
        data_random_seed=int(cfg['seed']),
    )
    lgb_cfg = cfg["models"]["lgbm_select"]["params"]
    params = dict(
        objective="regression",
        metric="None",
        device_type=lgb_cfg.get("device_type", "cpu"), 
        learning_rate=float(lgb_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgb_cfg.get("num_leaves", 63)),
        max_depth=int(lgb_cfg.get("max_depth", 8)),
        feature_fraction=float(lgb_cfg.get("feature_fraction", 0.80)),
        bagging_fraction=float(lgb_cfg.get("bagging_fraction", 0.80)),
        bagging_freq=int(lgb_cfg.get("bagging_freq", 1)),
        min_data_in_leaf=int(lgb_cfg.get("min_data_in_leaf", 200)),
        seed=seed_val
    )
    num_boost_round = int(lgb_cfg.get("num_boost_round", 4000))
    es_rounds       = int(lgb_cfg.get("early_stopping_rounds", 100))
    log_period      = int(lgb_cfg.get("log_period", 100))
    print(f"[{_now()}][lgb] params: {params}")
    print(f"[{_now()}][lgb] num_boost_round={num_boost_round}, early_stopping_rounds={es_rounds}, log_period={log_period}")

    # =========================
    # 5) Build Dataset
    # =========================
    print(f"[{_now()}][lgb] Building Dataset...")
    t_ds = time.time()
    d_all = lgb.Dataset(
        X, label=y, weight=w,
        feature_name=feat_cols,
        free_raw_data=True,
        params=ds_params,
    )
    print(f"[{_now()}][lgb] Dataset ready in {_fmt_s(time.time()-t_ds)}")

    # =========================
    # 6) Train across folds
    # =========================
    scores = []
    fi = pd.DataFrame({"feature": feat_cols})
    print(f"[{_now()}][train] Start CV training...")

    for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="CV folds", unit="fold"), 1):
        t_fold = time.time()
        print(f"[{_now()}][train] >>> fold {k}/{len(folds)} "
              f"(train={tr_idx.size:,}, val={va_idx.size:,})")

        tr_g = subset_idx[tr_idx]
        va_g = subset_idx[va_idx]

        dtrain = d_all.subset(tr_g, params=ds_params)
        dvalid = d_all.subset(va_g, params=ds_params)

        bst = lgb.train(
            params, dtrain,
            valid_sets=[dvalid, dtrain],
            valid_names=["val", "train"],
            feval=lgb_wr2,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=True),
                lgb.log_evaluation(period=log_period),
            ],
        )

        val_wr2 = float(bst.best_score.get("val", {}).get("wr2", float("nan")))
        trn_wr2 = float(bst.best_score.get("train", {}).get("wr2", float("nan")))
        best_it = int(bst.best_iteration or 0)
        print(f"[{_now()}][train] <<< fold {k} done in {_fmt_s(time.time()-t_fold)} | "
              f"best_iter={best_it}, val_wr2={val_wr2:.6f}, train_wr2={trn_wr2:.6f}")

        scores.append(val_wr2)
        g = bst.feature_importance(importance_type="gain", iteration=bst.best_iteration or num_boost_round).astype(float)
        fi[f"fold{k}_gain_share"] = (g / g.sum()) if g.sum() > 0 else np.zeros_like(g, dtype=float)

        bst.free_dataset(); del dtrain, dvalid, bst; gc.collect()

    print(f"[{_now()}][train] All folds done. val_wr2 per fold = {[f'{s:.6f}' for s in scores]}")

    # =========================
    # 7) Aggregate FI & select features
    # =========================
    print(f"[{_now()}][fi] Aggregating feature importance & selecting top-k...")
    t_fi = time.time()
    fi_cols = [c for c in fi.columns if c.startswith("fold")]
    fi["mean_gain_share"] = fi[fi_cols].mean(axis=1)
    fi = fi.sort_values("mean_gain_share", ascending=False, ignore_index=True)

    allowlist = list(cfg["columns"].get("allowlist", []))
    fi_normal = fi[~fi["feature"].isin(allowlist)].reset_index(drop=True)
    top_k = int(cfg["models"]["lgbm_select"]["params"].get("select_top_k", 800))
    final_feats = list(dict.fromkeys(allowlist + fi_normal["feature"].head(top_k).tolist()))
    print(f"[{_now()}][fi] selected={len(final_feats)} (including allowlist {len(allowlist)}), "
          f"top mean_gain_share={fi['mean_gain_share'].iloc[:5].round(6).tolist()} "
          f"in {_fmt_s(time.time()-t_fi)}")

    # =========================
    # 8) Outputs
    # =========================

    ts  = int(time.time())
    tag = tag_for_fs(fs_lo, fs_hi, n_splits, gap_days, ratio, seed_val, top_k, ts)
    fi_path  = os.path.join(report_root,   f"fi_gain_share__{tag}.csv")
    lst_path = os.path.join(report_root,   f"features__{tag}.txt")
    sum_path = os.path.join(report_root,   f"summary__{tag}.json")

    print(f"[{_now()}][out] Saving feature importance -> {fi_path}")
    fi[["feature", "mean_gain_share"]].to_csv(fi_path, index=False)

    print(f"[{_now()}][out] Saving feature list -> {lst_path}")
    with open(lst_path, "w", encoding="utf-8") as f:
        for x in final_feats:
            f.write(f"{x}\n")

    summary = {
        "cv_scores": scores,
        "cv_mean_wr2": float(np.mean(scores)),
        "cv_std_wr2": float(np.std(scores)),
        "memmap_prefix": prefix,
        "n_rows": int(meta["n_rows"]),
        "n_feat": int(meta["n_feat"]),
        "selected_features": len(final_feats),
        "whitelist": allowlist,
        "fi_csv": fi_path,
        "features_txt": lst_path,
        "train_range": [int(fs_lo), int(fs_hi)],
        "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
        "lgb_params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": es_rounds,
        "seed": seed_val,
        "tag": tag,
    }
    print(f"[{_now()}][out] Writing summary -> {sum_path}")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{_now()}][fs][done] mean_wr2={summary['cv_mean_wr2']:.6f}  (±{summary['cv_std_wr2']:.6f})")
    print(f"[{_now()}][fs][done] FI -> {fi_path}")
    print(f"[{_now()}][fs][done] features -> {lst_path}")
    print(f"[{_now()}][fs][done] summary -> {sum_path}")
    print(f"[{_now()}][fs] ===== Finished in {_fmt_s(time.time()-t0)} =====")

if __name__ == "__main__":
    main()
