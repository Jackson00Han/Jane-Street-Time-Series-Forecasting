# run_model.py
from __future__ import annotations
import os, json, gc, time
from pathlib import Path
import numpy as np
import lightgbm as lgb

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
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true, dtype=np.float64)
    w = w.astype(np.float64, copy=False)
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    y_bar = (w * y_true).sum() / max(w.sum(), 1e-12)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_bar) ** 2).sum()
    wr2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    return "wr2", float(wr2), True

def _latest_feature_list():
    feat_dir = os.path.join(P("local", cfg["paths"]["models"]), "feature_set")
    ensure_dir_local(feat_dir)
    cands = [os.path.join(feat_dir, x) for x in os.listdir(feat_dir) if x.endswith(".txt")]
    if not cands:
        raise FileNotFoundError(f"No features.txt under {feat_dir}")
    return sorted(cands)[-1]

def _tag_from_features_path(p: str) -> str:
    return Path(p).stem  # 直接用 run_feature_select 的 tag

def main():
    # ---------- paths ----------
    mm_root   = P("local", cfg["paths"]["sample_mm"])
    prefix    = os.path.join(mm_root, "full_sample_v1")  # 与 run_memmap.py 保持一致
    models_dir = P("local", cfg["paths"]["models"])
    final_dir  = os.path.join(models_dir, "final")
    ensure_dir_local(models_dir); ensure_dir_local(final_dir)

    # ---------- load memmap ----------
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap d 不是非降序；请检查 panel 分片或排序"
    print(f"[model] loaded memmap: N={meta['n_rows']:,}, F={meta['n_feat']}")

    # ---------- read selected features ----------
    feature_list_path = cfg["paths"]["feature_list_path"]
    with open(feature_list_path, "r", encoding="utf-8") as f:
        selected_feats = [ln.strip() for ln in f if ln.strip()]
        
    tag = _tag_from_features_path(feature_list_path)
    print(f"[model] using features: {feature_list_path}  (tag={tag}, {len(selected_feats)} feats)")

    # map to indices
    feat2idx = {f: i for i, f in enumerate(feat_cols)}
    missing = [f for f in selected_feats if f not in feat2idx]
    if missing:
        print(f"[warn] {len(missing)} features not in memmap; ignored. e.g. {missing[:5]}")
    final_idx = np.array([feat2idx[f] for f in selected_feats if f in feat2idx], dtype=np.int32)
    if final_idx.size == 0:
        raise ValueError("No valid selected features found in memmap.")

    # ---------- CV folds ----------
    cv_cfg = cfg['cv']
    n_splits = int(cv_cfg.get("n_splits", 5))
    gap_days = int(cv_cfg.get("gap_days", 0))
    ratio    = int(cv_cfg.get("train_to_val", 9))
    folds = make_sliding_cv_fast(d, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed. Check cv settings and date order.")
    print(f"[model] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")

    # ---------- LightGBM params ----------
    seed_val = int(cfg["seed"])
    ds_params = dict(
        max_bin=63,
        bin_construct_sample_cnt=200000,
        min_data_in_bin=3,
        data_random_seed=seed_val,
    )
    lgb_cfg = cfg['lgb_train']
    params = dict(
        objective="regression",
        metric="None",
        device_type=lgb_cfg.get("device_type", "gpu"),  # 如需 CPU 改 "cpu"
        learning_rate=float(lgb_cfg.get("learning_rate", 0.05)),
        num_leaves=int(lgb_cfg.get("num_leaves", 63)),
        max_depth=int(lgb_cfg.get("max_depth", 8)),
        feature_fraction=float(lgb_cfg.get("feature_fraction", 0.80)),
        bagging_fraction=float(lgb_cfg.get("bagging_fraction", 0.80)),
        bagging_freq=int(lgb_cfg.get("bagging_freq", 1)),
        min_data_in_leaf=int(lgb_cfg.get("min_data_in_leaf", 200)),
        seed=seed_val,
    )
    num_boost_round = int(lgb_cfg.get("num_boost_round", 4000))
    es_rounds       = int(lgb_cfg.get("early_stopping_rounds", 100))
    log_period      = int(lgb_cfg.get("log_period", 100))
    final_margin    = float(lgb_cfg.get("final_boost_margin", 1.10))  # 中位数 × margin

    # ---------- Dataset（切列视图）----------
    feat_names_sel = [feat_cols[i] for i in final_idx]
    X_sel = X[:, final_idx] 
    d_all_sel = lgb.Dataset(
        X_sel, label=y, weight=w,
        feature_name=feat_names_sel,
        free_raw_data=True,
        params=ds_params,
    )

    # ---------- Stage 2: CV to pick final_num_boost_round ----------
    best_iters = []
    print(f"[stage2] start CV with {len(feat_names_sel)} features")
    for k, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"[stage2] fold {k}/{len(folds)}: train={tr_idx.size:,}, val={va_idx.size:,}")
        dtrain = d_all_sel.subset(tr_idx, params=ds_params)
        dvalid = d_all_sel.subset(va_idx, params=ds_params)
        bst = lgb.train(
            params, dtrain,
            valid_sets=[dvalid],
            valid_names=["val"],
            feval=lgb_wr2_eval,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=True),
                lgb.log_evaluation(period=log_period),
            ],
        )
        bi = int(bst.best_iteration or num_boost_round)
        best_iters.append(bi)
        print(f"[stage2] fold {k} best_it={bi}, val_wr2={bst.best_score['val']['wr2']:.6f}")
        bst.free_dataset(); del dtrain, dvalid, bst; gc.collect()

    final_num_boost_round = max(50, int(np.median(best_iters) * final_margin))
    print(f"[stage2] best_iters={best_iters} -> final_num_boost_round={final_num_boost_round}")

    # ---------- Final: train on ALL data with fixed rounds ----------
    print(f"[final] training with num_boost_round={final_num_boost_round}")
    bst_final = lgb.train(
        params, d_all_sel,
        num_boost_round=final_num_boost_round,
        feval=lgb_wr2_eval,
    )
    final_model_path = os.path.join(final_dir, f"lgb_final__{tag}.txt")
    bst_final.save_model(final_model_path, num_iteration=final_num_boost_round)

    # ---------- Summary ----------
    summary_path = os.path.join(final_dir, f"summary__{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_list_used": feature_list_path,
            "n_selected_features": len(feat_names_sel),
            "final_num_boost_round": int(final_num_boost_round),
            "stage2_best_iters": best_iters,
            "final_model_path": final_model_path,
            "seed": seed_val,
            "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
            "lgb_params": params,
            "num_boost_round_limit": num_boost_round,
            "early_stopping_rounds": es_rounds,
            "memmap_prefix": prefix,
            "tag": tag,
        }, f, indent=2)

    print(f"[final] saved model -> {final_model_path}")
    print(f"[final] summary -> {summary_path}")

if __name__ == "__main__":
    main()
