# run_5_tune.py
from __future__ import annotations
import os, json, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
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
        prefix = cfg["paths"]["train_mm_prefix"]   # 若你在 yaml 固定了前缀，优先用这个
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
    print(f"[tune] loaded memmap: N={meta['n_rows']:,}, F={meta['n_feat']}, prefix={prefix}")

    # =========================
    # 2) 选择“小样本日期窗”用于调参
    # =========================
    # 优先 dates.tune_dates -> dates.feature_select_dates -> [train_lo, train_hi]
    try:
        win = cfg["dates"]["tune_dates"]
        lo, hi = int(win["date_lo"]), int(win["date_hi"])
    except KeyError:
        raise RuntimeError("Please specify cfg['dates']['tune_dates'] with date_lo/date_hi for tuning.")

    mask = (d >= lo) & (d <= hi)
    idx  = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows in tune window [{lo}, {hi}]")
    d_sub = d[idx]
    assert np.all(np.diff(d_sub) >= 0), "子集 d 非单调；请检查 memmap"
    print(f"[tune] using date range [{lo}, {hi}] -> {idx.size:,} rows")

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
    print(f"[tune] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")

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
    # 5) Optuna：定义搜索空间 & 目标
    # =========================
    lgb_tune_cfg = cfg["lgb+tune"]  # 用来取 device_type、早停等“非搜索”参数
    es_rounds = int(lgb_tune_cfg["early_stopping_rounds"])
    log_period = int(lgb_tune_cfg["log_period"])
    # 调参时的上限轮数（早停会截断）
    try:
        tune_rounds = int(lgb_tune_cfg["num_boost_round"])
    except KeyError:
        tune_rounds = 3000  # 默认

    def suggest_params(trial: optuna.Trial) -> dict:
        # 基础固定参数
        p = dict(
            objective="regression",
            metric="None",
            device_type=lgb_tune_cfg["device_type"],
            seed=seed_val,
        )
        # 搜索空间（温和、收敛快）
        p.update(
            num_leaves=trial.suggest_int("num_leaves", 31, 255, step=16),
            max_depth=trial.suggest_int("max_depth", 6, 12),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.07),
            feature_fraction=trial.suggest_float("feature_fraction", 0.70, 0.95),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.70, 0.95),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 5),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 20.0),
            lambda_l1=trial.suggest_float("lambda_l1", 0.0, 5.0),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 0.1),
        )
        return p

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        fold_scores = []
        for k, (tr_idx, va_idx) in enumerate(tqdm(folds, desc=f"trial{trial.number}", leave=False), 1):
            dtrain = d_base.subset(tr_idx, params=ds_params)
            dvalid = d_base.subset(va_idx, params=ds_params)

            bst = lgb.train(
                params, dtrain,
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
            fold_scores.append(wr2_k)

            # 释放
            bst.free_dataset(); del dtrain, dvalid, bst
            gc.collect()

        mean_wr2 = float(np.mean(fold_scores))
        # 记录到 trial
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("mean_wr2", mean_wr2)
        return mean_wr2  # direction="maximize"

    # =========================
    # 6) 运行 Study
    # =========================
    n_trials = lgb_tune_cfg.get("n_trials", 30)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    
    sampler = optuna.samplers.TPESampler(seed=seed_val)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler, study_name="lgb_tune_wr2")
    print(f"[tune] starting study: n_trials={n_trials}, rounds={tune_rounds}, es={es_rounds}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best = study.best_trial
    print(f"[tune] BEST mean_wr2={best.value:.6f}")
    print(f"[tune] BEST params={best.params}")

    # =========================
    # 7) 落盘：最佳参数 + 排行
    # =========================
    ts = int(time.time())
    tag = f"tune__mm_{Path(prefix).name}__range{lo}-{hi}__cv{n_splits}-g{gap_days}-r{ratio}__trials{n_trials}__{ts}"

    # best.json
    best_path = os.path.join(tune_dir, f"best__{tag}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": prefix,
            "tune_range": [int(lo), int(hi)],
            "cv": {"n_splits": n_splits, "gap_days": gap_days, "train_to_val": ratio},
            "ds_params": ds_params,
            "best_mean_wr2": float(best.value),
            "best_params": best.params,
            "fold_scores": best.user_attrs.get("fold_scores", []),
            "n_trials": n_trials,
            "num_boost_round": tune_rounds,
            "early_stopping_rounds": es_rounds,
            "seed": seed_val,
            "features": feat_cols,
            "n_rows": int(meta["n_rows"]),
            "n_feat": int(meta["n_feat"]),
            "tag": tag,
        }, f, indent=2)

    # trials.csv（排行榜）
    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state.name not in ("COMPLETE", "PRUNED"):
            continue
        row = {
            "trial": t.number,
            "state": t.state.name,
            "value_mean_wr2": t.value if t.value is not None else float("nan"),
            "duration_sec": t.duration.total_seconds() if t.duration else float("nan"),
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("value_mean_wr2", ascending=False)
    rank_path = os.path.join(tune_dir, f"trials__{tag}.csv")
    df.to_csv(rank_path, index=False)

    print(f"[tune][done] best -> {best_path}")
    print(f"[tune][done] trials -> {rank_path}")


if __name__ == "__main__":
    main()
