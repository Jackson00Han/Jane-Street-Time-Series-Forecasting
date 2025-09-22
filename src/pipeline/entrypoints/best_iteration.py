# run_6_best_iteration.py
from __future__ import annotations
import os, json, time, gc
from pathlib import Path

import numpy as np
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


def _latest_mm_prefix(mm_root: str) -> str:
    metas = [os.path.join(mm_root, f) for f in os.listdir(mm_root) if f.endswith(".meta.json")]
    if not metas:
        raise FileNotFoundError(f"No *.meta.json found under {mm_root}")
    metas.sort(key=lambda p: os.path.getmtime(p))
    return metas[-1][:-len(".meta.json")]  # strip ".meta.json"


def _latest_best_tune(models_dir: str) -> str:
    tune_dir = os.path.join(models_dir, "tune")
    files = []
    if os.path.isdir(tune_dir):
        files = [os.path.join(tune_dir, f) for f in os.listdir(tune_dir) if f.startswith("best__") and f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"No tune best__*.json under {tune_dir}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def main():
    # =========================
    # 0) 路径
    # =========================
    mm_root = P("local", cfg["paths"]["train_mm"])
    try:
        prefix = cfg["paths"]["train_mm_prefix"]    # 若 YAML 指定，优先用
    except KeyError:
        prefix = _latest_mm_prefix(mm_root)

    models_dir = P("local", cfg["paths"]["models"])
    out_dir    = os.path.join(models_dir, "best_iter")
    ensure_dir_local(models_dir); ensure_dir_local(out_dir)

    # 读取最近一次调参结果
    best_tune_path = _latest_best_tune(models_dir)
    with open(best_tune_path, "r", encoding="utf-8") as f:
        best_tune = json.load(f)
    best_params_from_tune = best_tune["best_params"]          # 只含被搜索的字段
    ds_params_from_tune   = best_tune.get("ds_params", None)  # 可能存在

    # =========================
    # 1) 载入 memmap（筛列后）
    # =========================
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap d 不是非降序；请检查 panel 分片或排序"
    print(f"[best-it] memmap loaded: N={meta['n_rows']:,}, F={meta['n_feat']}, prefix={prefix}")
    print(f"[best-it] using tuned params from: {best_tune_path}")

    # =========================
    # 2) 训练日期窗（全量）
    # =========================
    try:
        train_win = cfg["dates"]["final_train_dates"]
        lo = int(train_win["date_lo"]); hi = int(train_win["date_hi"])
    except KeyError:
        raise KeyError("请在 config/data.yaml 的 dates 下指定 final_train_dates")

    mask = (d >= lo) & (d <= hi)
    idx  = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows in date range [{lo}, {hi}]")
    d_sub = d[idx]
    assert np.all(np.diff(d_sub) >= 0), "子集 d 非单调；请检查 memmap"
    print(f"[best-it] date range [{lo}, {hi}] -> {idx.size:,} rows")

    # =========================
    # 3) CV 折
    # =========================
    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg["n_splits"])
    gap_days = int(cv_cfg["gap_days"])
    ratio    = int(cv_cfg["train_to_val"])
    folds = make_sliding_cv_fast(d_sub, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed.")
    print(f"[best-it] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")

    # =========================
    # 4) 训练参数（合并“调参最优 + 基础固定项”）
    # =========================
    seed_val = int(cfg["seed"])
    lgb_cfg  = cfg["lgb_best"]

    # dataset params
    if ds_params_from_tune is None:
        ds_params = dict(
            max_bin=63,
            bin_construct_sample_cnt=200000,
            min_data_in_bin=3,
            data_random_seed=seed_val,
        )
    else:
        ds_params = ds_params_from_tune

    # train params：基础固定 + 调参得到的字段
    params = dict(
        objective="regression",
        metric="None",
        device_type=lgb_cfg["device_type"],
        seed=seed_val,
    )
    params.update(best_params_from_tune)

    num_boost_round = int(lgb_cfg["num_boost_round"])         # 仅作上限，早停会截断
    es_rounds       = int(lgb_cfg["early_stopping_rounds"])
    log_period      = int(lgb_cfg["log_period"])
    final_margin    = float(lgb_cfg["final_boost_margin"])    # 中位数 × margin

    # =========================
    # 5) Dataset
    # =========================
    d_all  = lgb.Dataset(X, label=y, weight=w, feature_name=feat_cols, free_raw_data=True, params=ds_params)
    d_base = d_all.subset(idx, params=ds_params)

    # =========================
    # 6) CV 计算最佳轮数 + 分数
    # =========================
    best_iters = []
    fold_best_wr2 = []
    print(f"[best-it] start CV with {len(feat_cols)} features")

    for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="Best-Iter CV", unit="fold"), 1):
        print(f"[best-it] fold {k}/{len(folds)}: train={tr_idx.size:,}, val={va_idx.size:,}")
        dtrain = d_base.subset(tr_idx, params=ds_params)
        dvalid = d_base.subset(va_idx, params=ds_params)

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

        wr2_k = float(bst.best_score["val"]["wr2"])
        fold_best_wr2.append(wr2_k)
        print(f"[best-it] fold {k}: best_it={bi}, val_wr2={wr2_k:.6f}")

        bst.free_dataset(); del dtrain, dvalid, bst; gc.collect()

    cv_mean_wr2 = float(np.mean(fold_best_wr2))
    final_num_boost_round = max(50, int(np.median(best_iters) * final_margin))

    print(f"[best-it] CV wr2 mean={cv_mean_wr2:.6f} over {len(folds)} folds")
    print(f"[best-it] best_iters={best_iters} -> final_num_boost_round={final_num_boost_round}")

    # =========================
    # 7) 落盘结果
    # =========================
    ts  = int(time.time())
    tag = f"mm_{Path(prefix).name}__range{lo}-{hi}__cv{n_splits}-g{gap_days}-r{ratio}__seed{seed_val}__{ts}"
    out_json = os.path.join(out_dir, f"best_iter__{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": prefix,
            "train_range": [int(lo), int(hi)],
            "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
            "lgb_params": params,          # 已合并“调参最优”
            "ds_params": ds_params,
            "best_iters": best_iters,
            "cv_fold_best_wr2": fold_best_wr2,
            "cv_mean_wr2": cv_mean_wr2,
            "final_num_boost_round": int(final_num_boost_round),
            "features": feat_cols,
            "n_rows": int(meta["n_rows"]),
            "n_feat": int(meta["n_feat"]),
            "seed": seed_val,
            "tag": tag,
            "tune_source": best_tune_path,
        }, f, indent=2)

    print(f"[best-it][done] saved -> {out_json}")


if __name__ == "__main__":
    main()
