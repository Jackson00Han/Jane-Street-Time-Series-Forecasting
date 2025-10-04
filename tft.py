# -*- coding: utf-8 -*-
from __future__ import annotations

# ── 标准库 ──────────────────────────────────────────────────────────────────
import os
import time
import random
from pathlib import Path
from collections import defaultdict

# ── 第三方 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds

import torch
import torch.backends.cudnn as cudnn
import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# 你的工程工具
from pipeline.io import cfg, P, fs, storage_options, ensure_dir_local, ensure_dir_az
from pipeline.stream_input import ShardedBatchStream  # 你已实现好的 IterableDataset

# ---- 性能/兼容开关（仅一次）----
os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, os.cpu_count() // 2)))
pl.enable_string_cache()
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

    
# ───────────────────────────────────────────────────────────────────────────
# 仅保留这一个小工具函数：滑动窗划分（和你之前一致）
def make_sliding_cv_by_days(all_days: np.ndarray, *, n_splits: int, gap_days: int, train_to_val: int):
    all_days = np.asarray(all_days).ravel()
    K, R, G = n_splits, train_to_val, gap_days
    usable = len(all_days) - G
    if usable <= 0 or K <= 0 or R <= 0:
        return []
    V_base, rem = divmod(usable, R + K)
    if V_base <= 0:
        return []
    T = R * V_base
    v_lens = [V_base + 1 if i < rem else V_base for i in range(K)]
    folds, v_lo = [], T + G
    for V_i in v_lens:
        v_hi, tr_hi, tr_lo = v_lo + V_i, v_lo - G, v_lo - G - T
        if tr_lo < 0 or v_hi > len(all_days):
            break
        folds.append((all_days[tr_lo:tr_hi], all_days[v_lo:v_hi]))
        v_lo = v_hi
    return folds


# ───────────────────────────────────────────────────────────────────────────
def main():
    print(f"[{_now()}] imports ok")

    # ========== 1) 统一配置 ==========
    # 键/目标/权重列
    G_SYM, G_DATE, G_TIME = cfg["keys"]          # e.g. ("symbol_id","date_id","time_id")
    TARGET_COL = cfg["target"]                   # e.g. "responder_6"
    WEIGHT_COL = cfg["weight"]                   # 允许为 None

    # 时间特征
    TIME_FEATURES = ["time_pos", "time_sin", "time_cos", "time_bucket"]

    # 原始连续特征（先用一列试跑；后续替换为 TopK）
    RAW_FEATURES = ["feature_36"]

    # 训练 & CV 超参
    N_SPLITS     = 1
    GAP_DAYS     = 0
    TRAIN_TO_VAL = 2
    ENC_LEN      = 10
    PRED_LEN     = 1
    BATCH_SIZE   = 1024
    LR           = 1e-2
    HIDDEN       = 16
    HEADS        = 1
    DROPOUT      = 0.1
    MAX_EPOCHS   = 2
    CHUNK_DAYS   = 2   

    # 数据路径
    PANEL_DIR_AZ   = P("az", cfg["paths"].get("panel_shards", "panel_shards"))
    TFT_AZ_ROOT    = P("az", "tft")
    TFT_LOCAL_ROOT = P("local", "tft")

    # 目录准备
    ensure_dir_az(TFT_AZ_ROOT)
    CLEAN_DIR_AZ = f"{TFT_AZ_ROOT}/clean"; ensure_dir_az(CLEAN_DIR_AZ)
    ensure_dir_local(TFT_LOCAL_ROOT)
    CKPTS_DIR = Path(TFT_LOCAL_ROOT) / "ckpts"; ensure_dir_local(CKPTS_DIR.as_posix())
    LOGS_DIR  = Path(TFT_LOCAL_ROOT) / "logs";  ensure_dir_local(LOGS_DIR.as_posix())

    print("[config] ready")

    # ========== 2) 读取原始面板 & 建全局 time_idx ==========
    data_paths = fs.glob(f"{PANEL_DIR_AZ}/*.parquet")
    data_paths = [p if p.startswith("az://") else f"az://{p}" for p in data_paths]
    lf_data = pl.scan_parquet(data_paths, storage_options=storage_options)
    lf_data = lf_data.filter(pl.col(G_DATE).is_between(1650, 1655, closed="both"))   # 含端点

    lf_grid = (
        lf_data.select([G_DATE, G_TIME]).unique()
               .sort([G_DATE, G_TIME])
               .with_row_index("time_idx")
               .with_columns(pl.col("time_idx").cast(pl.Int64))
    )
    grid_path_local = P("local", "tft/panel/grid_timeidx.parquet")
    ensure_dir_local(Path(grid_path_local).parent.as_posix())
    lf_grid.collect(streaming=True).write_parquet(grid_path_local, compression="zstd")
    grid_lazy = pl.scan_parquet(grid_path_local)

    NEED_COLS = list(dict.fromkeys([G_SYM, G_DATE, G_TIME, WEIGHT_COL, TARGET_COL] + TIME_FEATURES + RAW_FEATURES))
    lf0 = (
        lf_data.join(grid_lazy, on=[G_DATE, G_TIME], how="left")
               .select(NEED_COLS + ["time_idx"])
               .sort([G_DATE, G_TIME, G_SYM])
    )
    print(f"[{_now()}] lazyframe ready")

    # ========== 3) CV 划分 ==========
    all_days = (
        lf0.select(pl.col(G_DATE)).unique().sort(by=G_DATE)
           .collect(streaming=True).get_column(G_DATE).to_numpy()
    )
    folds_by_day = make_sliding_cv_by_days(all_days, n_splits=N_SPLITS, gap_days=GAP_DAYS, train_to_val=TRAIN_TO_VAL)
    assert len(folds_by_day) > 0, "no CV folds constructed"
    stats_hi = int(folds_by_day[0][0][-1])
    print(f"[stats] upper bound day for z-score = {stats_hi}")

    # ========== 4) 连续特征清洗 + Z-score ==========
    inf2null_exprs  = [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in RAW_FEATURES]
    isna_flag_exprs = [pl.col(c).is_null().cast(pl.Int8).alias(f"{c}__isna") for c in RAW_FEATURES]
    ffill_exprs     = [pl.col(c).forward_fill().over(G_SYM).fill_null(0.0).alias(c) for c in RAW_FEATURES]

    lf_clean = (
        lf0.with_columns(inf2null_exprs)
           .with_columns(isna_flag_exprs)
           .with_columns(ffill_exprs)
    )

    lf_stats_sym = (
        lf_clean.filter(pl.col(G_DATE) <= stats_hi)
                .group_by(G_SYM)
                .agg([pl.col(c).mean().alias(f"mu_{c}") for c in RAW_FEATURES] +
                     [pl.col(c).std(ddof=0).alias(f"std_{c}") for c in RAW_FEATURES])
    )
    lf_stats_glb = (
        lf_clean.filter(pl.col(G_DATE) <= stats_hi)
                .select([pl.col(c).mean().alias(f"mu_{c}_glb") for c in RAW_FEATURES] +
                        [pl.col(c).std(ddof=0).alias(f"std_{c}_glb") for c in RAW_FEATURES])
    )

    lf_z = lf_clean.join(lf_stats_glb, how="cross").join(lf_stats_sym, on=G_SYM, how="left")

    eps = 1e-6
    Z_COLS, NAMARK_COLS = [], [f"{c}__isna" for c in RAW_FEATURES]
    for c in RAW_FEATURES:
        mu_sym, std_sym = f"mu_{c}", f"std_{c}"
        mu_glb, std_glb = f"mu_{c}_glb", f"std_{c}_glb"
        mu_use, std_use = f"{c}_mu_use", f"{c}_std_use"
        z_name = f"{c}_z"

        lf_z = lf_z.with_columns(
            pl.when(pl.col(mu_sym).is_null()).then(pl.col(mu_glb)).otherwise(pl.col(mu_sym)).alias(mu_use),
            pl.when(pl.col(std_sym).is_null() | (pl.col(std_sym) == 0)).then(pl.col(std_glb)).otherwise(pl.col(std_sym)).alias(std_use),
        ).with_columns(
            ((pl.col(c) - pl.col(mu_use)) / (pl.col(std_use) + eps)).alias(z_name)
        ).drop([mu_glb, std_glb, mu_sym, std_sym, mu_use, std_use])

        Z_COLS.append(z_name)

    OUT_COLS = [G_SYM, G_DATE, G_TIME, "time_idx", WEIGHT_COL, TARGET_COL] + TIME_FEATURES + Z_COLS + NAMARK_COLS
    lf_out = lf_z.select(OUT_COLS).sort([G_DATE, G_TIME, G_SYM])

    # ========== 5) 按 30 天分组落地（chunk_XXXX_YYYY） ==========
    day_list   = list(map(int, all_days))
    day_chunks = [day_list[i:i + CHUNK_DAYS] for i in range(0, len(day_list), CHUNK_DAYS)]

    for ci, chunk in enumerate(day_chunks, 1):
        df_chunk = lf_out.filter(pl.col(G_DATE).is_in(chunk)).collect()
        table = df_chunk.to_arrow()

        chunk_dir = f"{CLEAN_DIR_AZ}/chunk_{chunk[0]:04d}_{chunk[-1]:04d}"
        ds.write_dataset(
            data=table,
            base_dir=chunk_dir,
            filesystem=fs,
            format="parquet",
            partitioning=None,
            existing_data_behavior="overwrite_or_ignore",
            basename_template="data-{i}.parquet",
            max_rows_per_file=50_000_000,
        )
        print(f"[{_now()}] chunk {ci}/{len(day_chunks)} -> days {chunk[0]}..{chunk[-1]} written")

    print(fs.ls(CLEAN_DIR_AZ)[:5])

    # ========== 6) 归并 chunk → 路径清单 ==========
    entries = fs.ls(CLEAN_DIR_AZ)
    chunk_dirs: list[str] = []
    for e in entries:
        path = e if isinstance(e, str) else (e.get("name") or e.get("path") or e.get("Key") or str(e))
        if path.rstrip("/").split("/")[-1].startswith("chunk_"):
            chunk_dirs.append(path if path.startswith("az://") else f"az://{path}")
    chunk_dirs = sorted(chunk_dirs)

    chunk2paths: dict[str, list[str]] = defaultdict(list)
    for cdir in chunk_dirs:
        paths = fs.glob(f"{cdir}/*.parquet")
        paths = [p if p.startswith("az://") else f"az://{p}" for p in paths]
        if not paths:
            print(f"[WARN] empty chunk: {cdir}")
        chunk2paths[cdir] = paths

    all_paths = [p for plist in chunk2paths.values() for p in plist]
    print(f"[prep] {len(chunk_dirs)} chunks; {len(all_paths)} files total")

    # ========== 7) 训练列选择 ==========
    UNKNOWN_REALS = Z_COLS + NAMARK_COLS + TIME_FEATURES
    TRAIN_COLS    = [G_SYM, "time_idx", WEIGHT_COL, TARGET_COL, *UNKNOWN_REALS]

    # ========== 8) 训练（按 CV 折） ==========
    best_ckpt_paths, fold_metrics = [], []

    for fold_id, (train_days, val_days) in enumerate(folds_by_day, start=1):
        print(f"[fold {fold_id}] train {train_days[0]}..{train_days[-1]} ({len(train_days)} days), "
              f"val {val_days[0]}..{val_days[-1]} ({len(val_days)} days)")

        # 8.1 Template（用训练起始前 N 天固化编码/缩放配置）
        days_sorted = np.sort(train_days)
        TEMPLATE_DAYS = min(1, len(days_sorted))
        tmpl_days = list(map(int, days_sorted[:TEMPLATE_DAYS]))

        pdf_tmpl = (
            pl.scan_parquet(all_paths, storage_options=storage_options)
              .filter(pl.col(G_DATE).is_in(tmpl_days))
              .select(TRAIN_COLS)
              .collect(streaming=True)
              .to_pandas()
        )
        pdf_tmpl[G_SYM] = pdf_tmpl[G_SYM].astype("str")
        pdf_tmpl.sort_values([G_SYM, "time_idx"], inplace=True)
        print(f"[fold {fold_id}] template days={tmpl_days}, template shape={pdf_tmpl.shape}")

        # 8.2 验证集
        pdf_val = (
            pl.scan_parquet(all_paths, storage_options=storage_options)
              .filter(pl.col(G_DATE).is_in(list(map(int, val_days))))
              .select(TRAIN_COLS)
              .collect(streaming=True)
              .to_pandas()
        )
        pdf_val[G_SYM] = pdf_val[G_SYM].astype("str")
        pdf_val.sort_values([G_SYM, "time_idx"], inplace=True)
        print(f"template {pdf_tmpl.shape}, val {pdf_val.shape}")

        # 8.3 TimeSeries 模板
        identity_scalers = {name: None for name in UNKNOWN_REALS}
        template = TimeSeriesDataSet(
            pdf_tmpl,
            time_idx="time_idx",
            target=TARGET_COL,
            group_ids=[G_SYM],
            weight=WEIGHT_COL,
            max_encoder_length=ENC_LEN,
            max_prediction_length=PRED_LEN,
            static_categoricals=[G_SYM],
            time_varying_unknown_reals=UNKNOWN_REALS,
            lags=None,
            categorical_encoders={G_SYM: NaNLabelEncoder(add_nan=True)},
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            target_normalizer=None,
            scalers=identity_scalers,
        )

        # 8.4 验证 Loader
        validation = TimeSeriesDataSet.from_dataset(template, data=pdf_val, stop_randomization=True)
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=BATCH_SIZE * 2,
            num_workers=0, #min(8, max(1, os.cpu_count() - 2)),
            pin_memory=True,
            persistent_workers=False, #True,
            #prefetch_factor=4,
        )

        # 8.5 训练流（按 chunk 一次读取；若需严格时间顺序，在类里关闭 shuffle）
        train_stream = ShardedBatchStream(
            template_tsd=template,
            chunk_dirs=chunk_dirs,
            chunk2paths=chunk2paths,
            g_sym=G_SYM,
            batch_size=max(256, BATCH_SIZE),
            buffer_batches=0,
            seed=42,
            cols=TRAIN_COLS,
            print_every_chunks=1,
        )
        train_loader = DataLoader(
            train_stream,
            batch_size=None,
            num_workers=0, #min(8, max(1, os.cpu_count() - 2)),
            persistent_workers=False, #True,
            #prefetch_factor=2,
            pin_memory=True,
            #multiprocessing_context="spawn",
        )

        # 8.6 callbacks/logger/trainer
        ckpt_dir_fold = Path(CKPTS_DIR) / f"fold_{fold_id}"
        ensure_dir_local(ckpt_dir_fold.as_posix())

        callbacks = [
            EarlyStopping(monitor="val_RMSE", mode="min", patience=1),
            ModelCheckpoint(
                monitor="val_RMSE",
                mode="min",
                save_top_k=1,
                dirpath=ckpt_dir_fold.as_posix(),
                filename=f"fold{fold_id}-tft-best-{{epoch:02d}}-{{val_RMSE:.5f}}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        logger = TensorBoardLogger(save_dir=LOGS_DIR.as_posix(), name=f"tft_f{fold_id}", default_hp_metric=False)

        trainer = L.Trainer(
            accelerator="gpu", devices=1, precision="bf16-mixed",
            max_epochs=MAX_EPOCHS,
            num_sanity_val_steps=0,
            gradient_clip_val=0.5,
            log_every_n_steps=20,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=callbacks,
            logger=logger,
            default_root_dir=CKPTS_DIR.as_posix(),
            
            limit_train_batches=200,
        )

        # 8.7 模型并训练
        tft = TemporalFusionTransformer.from_dataset(
            template,
            loss=MAE(),
            logging_metrics=[RMSE()],
            learning_rate=float(cfg.get("tft", {}).get("lr", LR)),
            hidden_size=int(cfg.get("tft", {}).get("hidden_size", HIDDEN)),
            attention_head_size=int(cfg.get("tft", {}).get("heads", HEADS)),
            dropout=float(cfg.get("tft", {}).get("dropout", DROPOUT)),
            reduce_on_plateau_patience=4,
        )

        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # 8.8 结果
        es_cb, ckpt_cb = callbacks[0], callbacks[1]
        print("epoch_end_at   :", trainer.current_epoch)
        print("global_step    :", trainer.global_step)
        print("val_best_score :", float(ckpt_cb.best_model_score))
        print("es_stopped_ep  :", getattr(es_cb, "stopped_epoch", None))
        print("es_wait_count  :", getattr(es_cb, "wait_count", None))

        best_ckpt_paths = getattr(main, "_best_ckpt_paths", [])
        best_ckpt_paths.append(ckpt_cb.best_model_path)
        setattr(main, "_best_ckpt_paths", best_ckpt_paths)

        fold_metrics = getattr(main, "_fold_metrics", [])
        fold_metrics.append(float(ckpt_cb.best_model_score))
        setattr(main, "_fold_metrics", fold_metrics)

        cv_rmse = np.mean(fold_metrics)
        print(f"[CV] mean val_RMSE = {cv_rmse:.6f}".replace("cv_rmSE", "cv_rmse"))

    print("[done] best_ckpts =", getattr(main, "_best_ckpt_paths", []))


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
