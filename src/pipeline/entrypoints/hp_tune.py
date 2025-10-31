# This is an advanced option:
# If time allows, using Optuna hyperparameter search to improve LightGBM model performance. It is recommended to explore various hyperparameter configurations for optimal results.
# In this project, I didn't used it since it was time-consuming to run.


from __future__ import annotations
import os, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from tqdm.auto import tqdm
from pipeline.io import cfg, ensure_dir_local
from pipeline.backtest import make_sliding_cv            # ← unified CV helper
from pipeline.metrics import lgb_wr2                     # ← unified metric (feval)


# ---------- utils ----------
def load_mm(prefix: str):
    """
    Load memmap-backed arrays and metadata.

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
    tune_dir = os.path.join(local_root, cfg["paths"]["models"]["rel"], "tune")
    ensure_dir_local(tune_dir)

    # Prefer explicit memmap prefix from config (produced by memmap_after_fs step)
    path_mm = cfg["derived_paths"].get("train_mm", None)
    assert path_mm, "cfg['derived_paths']['train_mm'] is missing."
    assert os.path.exists(f"{path_mm}.meta.json"), f"memmap not found: {path_mm}.meta.json"

    # =========================
    # 1) Load memmap (already column-reduced)
    # =========================
    meta, X, y, w, d = load_mm(path_mm)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap 'date' is not non-decreasing; check panel ordering."
    print(f"[tune] loaded memmap: N={meta['n_rows']:,}, F={meta['n_feat']}, prefix={path_mm}")

    # =========================
    # 2) Pick a smaller date window for tuning
    # =========================

    win = cfg["dates"]["fs"]  # e.g. fs window for faster tuning
    lo, hi = int(win["date_lo"]), int(win["date_hi"])


    mask = (d >= lo) & (d <= hi)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No rows in tune window [{lo}, {hi}]")
    d_sub = d[idx]
    assert np.all(np.diff(d_sub) >= 0), "subset 'date' is not non-decreasing."
    print(f"[tune] using date range [{lo}, {hi}] -> {idx.size:,} rows")

    # =========================
    # 3) Build CV folds inside the tuning window
    # =========================
    cv_cfg = cfg["cv"]
    n_splits = int(cv_cfg["n_splits"])
    gap_days = int(cv_cfg["gap_days"])
    ratio = int(cv_cfg["train_to_val"])
    folds = make_sliding_cv(d_sub, n_splits=n_splits, gap_days=gap_days, train_to_val=ratio)
    if not folds:
        raise RuntimeError("No CV folds constructed.")
    print(f"[tune] CV folds: {len(folds)} (cv={n_splits}, gap={gap_days}, r={ratio})")

    # =========================
    # 4) Base Dataset (derive subsets per fold)
    # =========================
    seed_val = int(cfg["seed"])
    # Slightly larger max_bin can sometimes help during tuning; keep moderate to avoid memory spikes
    ds_params = dict(
        max_bin=63,
        bin_construct_sample_cnt=200000,
        min_data_in_bin=3,
        data_random_seed=seed_val,
    )
    d_all = lgb.Dataset(X, label=y, weight=w, feature_name=feat_cols, free_raw_data=True, params=ds_params)
    d_base = d_all.subset(idx, params=ds_params)

    # =========================
    # 5) Optuna: search space & objective
    # =========================
    # Use the same node as training params (device_type, rounds, ES, logging) to match your config layout.
    lgb_cfg = cfg["models"]["lgbm_train"]["params"]
    device_type = lgb_cfg.get("device_type", "cpu")
    es_rounds = int(lgb_cfg.get("early_stopping_rounds", 100))
    log_period = int(lgb_cfg.get("log_period", 100))
    tune_rounds = int(lgb_cfg.get("num_boost_round", 2000))  # early-stopping will cap this

    def suggest_params(trial: optuna.Trial) -> dict:
        """Param search space (compact, converges quickly)."""
        p = dict(
            objective="regression",
            metric="None",
            device_type=device_type,
            seed=seed_val,
        )
        p.update(
            num_leaves=trial.suggest_int("num_leaves", 31, 127, step=16),
            max_depth=trial.suggest_int("max_depth", 6, 12),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08),
            feature_fraction=trial.suggest_float("feature_fraction", 0.70, 0.95),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.70, 0.95),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 3),
            lambda_l2=trial.suggest_float("lambda_l2", 0.0, 20.0),
            lambda_l1=trial.suggest_float("lambda_l1", 0.0, 10.0),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 0.2),
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
                feval=lgb_wr2,
                num_boost_round=tune_rounds,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
                    lgb.log_evaluation(period=log_period),
                ],
            )
            wr2_k = float(bst.best_score["val"]["wr2"])
            fold_scores.append(wr2_k)

            bst.free_dataset(); del dtrain, dvalid, bst
            gc.collect()

        mean_wr2 = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("mean_wr2", mean_wr2)
        return mean_wr2  # direction="maximize"

    # =========================
    # 6) Run study
    # =========================
    n_trials = int(lgb_cfg.get("n_trials", 30))     # allow configuring in the same params block
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=seed_val)
    study_name = f"lgb_tune_wr2__{Path(path_mm).name}__{lo}-{hi}"
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler, study_name=study_name)
    print(f"[tune] starting study: n_trials={n_trials}, rounds={tune_rounds}, es={es_rounds}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best = study.best_trial
    print(f"[tune] BEST mean_wr2={best.value:.6f}")
    print(f"[tune] BEST params={best.params}")

    # =========================
    # 7) Persist results (best.json + trials.csv)
    # =========================
    ts = int(time.time())
    tag = f"tune__mm_{Path(path_mm).name}__range{lo}-{hi}__cv{n_splits}-g{gap_days}-r{ratio}__trials{n_trials}__{ts}"

    best_path = os.path.join(tune_dir, f"best__{tag}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": path_mm,
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
