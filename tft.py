# -*- coding: utf-8 -*-
from __future__ import annotations

# ── 标准库 ──────────────────────────────────────────────────────────────────
import os
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── 第三方 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.backends.cudnn as cudnn
import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.data import TorchNormalizer


# 你的工程工具
from pipeline.io import cfg, P, fs, storage_options, ensure_dir_local
from pipeline.stream_input_local import ShardedBatchStream  # 使用下方给你的新版类
from pipeline.wr2 import WR2

# ---- 性能/兼容开关（仅一次）----
os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, os.cpu_count() // 2)))
pl.enable_string_cache()
cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _now() -> str:
    import time as _t
    return _t.strftime("%Y-%m-%d %H:%M:%S")

import time, torch, lightning as L

def _infer_bsz(batch):
    if isinstance(batch, dict):
        for v in batch.values():
            if torch.is_tensor(v) and v.dim() > 0:
                return v.size(0)
    if isinstance(batch, (list, tuple)) and batch:
        return _infer_bsz(batch[0])
    return None

class SamplesPerSec(L.Callback):
    def __init__(self, log_every_n_steps=50, ema=0.9):
        self.log_every = log_every_n_steps
        self.ema = ema
        self._t0 = None
        self._step0 = None
        self._count = 0
        self._ema_sps = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._step0 = trainer.global_step

        bsz = _infer_bsz(batch) or 0
        world = getattr(trainer, "world_size", 1) or 1
        self._count += bsz * world

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 每 log_every 步记录一次
        if (trainer.global_step - self._step0) >= self.log_every:
            now = time.perf_counter()
            dt = max(1e-6, now - self._t0)
            sps = self._count / dt
            its = (trainer.global_step - self._step0) / dt

            # EMA 平滑
            self._ema_sps = sps if self._ema_sps is None else self.ema*self._ema_sps + (1-self.ema)*sps

            trainer.logger.log_metrics({
                "train/samples_per_sec": sps,
                "train/samples_per_sec_ema": self._ema_sps,
                "train/it_per_sec": its,
            }, step=trainer.global_step)

            # 窗口复位
            self._t0 = now
            self._step0 = trainer.global_step
            self._count = 0

from lightning.pytorch.callbacks import TQDMProgressBar

class ShowKeysProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        want = ["val_WR2"]
        for k in want:
            v = trainer.callback_metrics.get(k, None)
            if v is not None:
                try:
                    items[k] = f"{float(v):.4f}"
                except Exception:
                    items[k] = str(v)
        return items



# ───────────────────────────────────────────────────────────────────────────
# 滑动窗划分
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
    G_SYM, G_DATE, G_TIME = cfg["keys"]          # e.g. ("symbol_id","date_id","time_id")
    TARGET_COL = cfg["target"]                   # e.g. "responder_6"
    WEIGHT_COL = cfg["weight"]                   # 允许为 None

    # 已知时间特征（给 encoder+decoder）
    TIME_FEATURES = ["time_pos", "time_sin", "time_cos", "time_bucket"]

    # 连续特征（示例）
    BASIC_FEATURES = ["feature_36", 
                    "feature_06", 
                    "feature_04", 
                    "feature_16", 
                    "feature_69", 
                    "feature_22",
                    "feature_20", 
                    "feature_58", 
                    "feature_24", 
                    "feature_27",
                    "feature_37"]
    ADDED_COLS = ["feature_08__ewm5", 
                "responder_5_prevday_std", 
                "responder_3_prevday_std", 
                "responder_4_prev_tail_d1", 
                "feature_53__rstd3", 
                "feature_16__ewm5", 
                "feature_01__ewm5", 
                "responder_7_prevday_std", 
                "responder_8_prevday_mean", 
                "feature_38__ewm5", 
                "feature_05__ewm5", 
                "responder_1_close_roll3_std",
                "responder_3_prevday_mean",
                "feature_37__ewm5",
                "responder_4_prevday_std",
                "responder_3_prev_tail_d1",
                "responder_6_prevday_std",
                "responder_2_prevday_mean",
                "responder_0_prevday_std",
                "responder_3_prev2day_close",
                "responder_8_prevday_std",
                "responder_2_prev_tail_d1",
                "responder_4_prevday_mean"
            ]
    RAW_FEATURES = BASIC_FEATURES + ADDED_COLS

    # 训练 & CV 超参
    N_SPLITS     = 1
    GAP_DAYS     = 7
    TRAIN_TO_VAL = 5
    ENC_LEN      = 50
    DEC_LEN      = 1
    PRED_LEN     = DEC_LEN
    BATCH_SIZE   = 512   
    LR           = 1e-3
    HIDDEN       = 16
    HEADS        = 1
    DROPOUT      = 0.1
    MAX_EPOCHS   = 20
    CHUNK_DAYS   = 30

    # 数据路径
    PANEL_DIR_AZ   = P("az", cfg["paths"].get("panel_shards", "panel_shards"))
    TFT_LOCAL_ROOT = P("local", "tft")

    # 目录准备
    ensure_dir_local(TFT_LOCAL_ROOT)
    LOCAL_CLEAN_DIR = f"{TFT_LOCAL_ROOT}/clean"; ensure_dir_local(LOCAL_CLEAN_DIR)
    CKPTS_DIR = Path(TFT_LOCAL_ROOT) / "ckpts"; ensure_dir_local(CKPTS_DIR.as_posix())
    LOGS_DIR  = Path(TFT_LOCAL_ROOT) / "logs";  ensure_dir_local(LOGS_DIR.as_posix())

    print("[config] ready")

    # ========== 2) 读取原始面板 & 建全局 time_idx ==========
    data_paths = fs.glob(f"{PANEL_DIR_AZ}/*.parquet")
    data_paths = [p if p.startswith("az://") else f"az://{p}" for p in data_paths]
    lf_data = pl.scan_parquet(data_paths, storage_options=storage_options)
    lf_data = lf_data.filter(pl.col(G_DATE).is_between(1610, 1690, closed="both"))

    lf_grid = (
        lf_data.select([G_DATE, G_TIME]).unique()
            .sort([G_DATE, G_TIME])
            .with_row_index("time_idx")
            .with_columns(pl.col("time_idx").cast(pl.Int64))
    )
    grid_path_local = P("local", "tft/panel/grid_timeidx.parquet")
    Path(grid_path_local).parent.mkdir(parents=True, exist_ok=True)
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

    # ========== 5) 按 30 天分组落地（Feather，单文件/块） ==========
    day_list   = list(map(int, all_days))
    day_chunks = [day_list[i:i + CHUNK_DAYS] for i in range(0, len(day_list), CHUNK_DAYS)]

    for ci, chunk in enumerate(day_chunks, 1):
        df_chunk = lf_out.filter(pl.col(G_DATE).is_in(chunk)).collect()
        local_chunk_dir = f"{LOCAL_CLEAN_DIR}/chunk_{chunk[0]:04d}_{chunk[-1]:04d}"
        Path(local_chunk_dir).mkdir(parents=True, exist_ok=True)
        df_chunk.write_ipc(f"{local_chunk_dir}/data.feather")
        print(f"[{_now()}] chunk {ci}/{len(day_chunks)} -> days {chunk[0]}..{chunk[-1]} written (local)")

    # ========== 6) 归并 chunk → 路径清单（Feather） ==========
    chunk_dirs = sorted(p.as_posix() for p in Path(LOCAL_CLEAN_DIR).glob("chunk_*") if p.is_dir())

    chunk2paths: dict[str, list[str]] = {}
    for cdir in chunk_dirs:
        paths = sorted(p.as_posix() for p in Path(cdir).glob("*.feather"))
        if not paths:
            print(f"[WARN] empty chunk: {cdir}")
        chunk2paths[cdir] = paths

    all_paths = [p for plist in chunk2paths.values() for p in plist]
    print(f"[prep] {len(chunk_dirs)} local chunks; {len(all_paths)} files total")

    # ========== 7) 训练列（known/unknown 分开） ==========
    KNOWN_REALS   = TIME_FEATURES
    UNKNOWN_REALS = Z_COLS + NAMARK_COLS
    TRAIN_COLS    = [G_SYM, "time_idx", WEIGHT_COL, TARGET_COL, *KNOWN_REALS, *UNKNOWN_REALS]

    # ========== 8) 训练（按 CV 折） ==========
    best_ckpt_paths, fold_metrics = [], []

    for fold_id, (train_days, val_days) in enumerate(folds_by_day, start=1):
        print(f"[fold {fold_id}] train {train_days[0]}..{train_days[-1]} ({len(train_days)} days), "
            f"val {val_days[0]}..{val_days[-1]} ({len(val_days)} days)")

        # 8.1 Template（用训练起始前 N 天固化编码/缩放配置）
        days_sorted = np.sort(train_days)
        TEMPLATE_DAYS = min(3, len(days_sorted))
        tmpl_days = list(map(int, days_sorted[:TEMPLATE_DAYS]))

        # Feather 懒加载
        pdf_tmpl = (
            pl.scan_ipc(all_paths)
            .filter(pl.col(G_DATE).is_in(tmpl_days))
            .select(TRAIN_COLS)
            .collect(streaming=True)
            .to_pandas()
        )
        pdf_tmpl[G_SYM] = pdf_tmpl[G_SYM].astype("str")
        pdf_tmpl.sort_values([G_SYM, "time_idx"], inplace=True)
        print(f"[fold {fold_id}] template days={tmpl_days}, template shape={pdf_tmpl.shape}")


        # 8.2 TimeSeries 模板（known/unknown 正确配置）
        identity_scalers = {name: None for name in (KNOWN_REALS + UNKNOWN_REALS)}
        template = TimeSeriesDataSet(
            pdf_tmpl,
            time_idx="time_idx",
            target=TARGET_COL,
            group_ids=[G_SYM],
            weight=WEIGHT_COL,
            max_encoder_length=ENC_LEN, 
            min_encoder_length=ENC_LEN,
            max_prediction_length=PRED_LEN, 
            min_prediction_length=PRED_LEN,
            static_categoricals=[G_SYM],
            time_varying_known_reals=KNOWN_REALS,
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

        # 8.3 验证集
        
        val_lo,val_hi = int(val_days[0]), int(val_days[-1])
        need_days = int(np.ceil((ENC_LEN + 100) / 968))
        extra_days = max(need_days, 2)  # 至少多取2天，保证连续
        cut_lo = val_lo - extra_days # 往前推2天，保证 encoder context 连续
        
        pdf_val = (
            pl.scan_ipc(all_paths)
            .filter(pl.col(G_DATE).is_between(cut_lo, val_hi, closed="both"))
            .collect(streaming=True)
            .to_pandas()
        )
        pdf_val[G_SYM] = pdf_val[G_SYM].astype("str")
        pdf_val.sort_values([G_SYM, "time_idx"], inplace=True)
        
        val_start_idx = pdf_val.loc[pdf_val[G_DATE] == val_lo, "time_idx"].min()
        assert pd.notna(val_start_idx), f"No rows found for val_lo={val_lo}"
        val_start_idx = int(val_start_idx)

        pdf_val = pdf_val[TRAIN_COLS]
        print(f"validation set get previous {extra_days} days, from {cut_lo} to {val_hi}, "
            f"shape={pdf_val.shape}, val_start_idx={val_start_idx}")
        
        # 8.4 验证 Loader
        validation = TimeSeriesDataSet.from_dataset(
            template, 
            data=pdf_val, 
            stop_randomization=True,
            min_prediction_idx=val_start_idx
        )
        
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        
        n_val_batches = len(val_loader)
        print(f"[debug] val_loader batches = {n_val_batches}")
        assert n_val_batches > 0, "Empty val dataloader. Check min_prediction_idx/ENC_LEN/date windows."

        # 8.5 训练流（Feather + 流式）
        start_date = int(train_days[0])
        end_date   = int(train_days[-1])
        train_period = (start_date, end_date)

        train_stream = ShardedBatchStream(
            template_tsd=template,
            chunk_dirs=chunk_dirs,
            chunk2paths=chunk2paths,
            train_period=train_period,
            g_sym=G_SYM,
            g_date=G_DATE,
            batch_size=BATCH_SIZE*2,             
            buffer_batches=16,
            seed=42,
            cols=TRAIN_COLS,
            print_every_chunks=0,
            file_format="feather",      # 关键
        )
        train_loader = DataLoader(
            train_stream,
            batch_size=None,
            num_workers=14,
            persistent_workers=True,    # 外层 persistent，避免反复 spawn
            prefetch_factor=8,
            pin_memory=True,
            multiprocessing_context="spawn",
        )

        # 8.6 callbacks/logger/trainer
        ckpt_dir_fold = Path(CKPTS_DIR) / f"fold_{fold_id}"
        ckpt_dir_fold.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_WR2", 
                mode="max", 
                patience=3, 
                check_on_train_epoch_end=False
            ),
            ModelCheckpoint(
                monitor="val_WR2",
                mode="max",
                save_top_k=1,
                dirpath=ckpt_dir_fold.as_posix(),
                filename=f"fold{fold_id}-tft-best-{{epoch:02d}}-{{val_WR2:.5f}}",
                save_on_train_epoch_end=False
            ),
            LearningRateMonitor(logging_interval="step"),
            SamplesPerSec(log_every_n_steps=200),   # ← 记录吞吐
            DeviceStatsMonitor(),                  # ← 记录 GPU util/mem 曲线
            ShowKeysProgressBar(),
            
        ]
        RUN_NAME = (
            f"f{fold_id}"
            f"_E{MAX_EPOCHS}"
            f"_lr{LR}"
            f"_bs{BATCH_SIZE}"
            f"_enc{ENC_LEN}_dec{DEC_LEN}"
            f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        logger = TensorBoardLogger(
            save_dir=LOGS_DIR.as_posix(),
            name="tft",
            version=RUN_NAME,
            default_hp_metric=False
        )

        print(f"[run] name={logger.name}  version={logger.version}  log_dir={logger.log_dir}")

        trainer = L.Trainer(
            accelerator="gpu", devices=1, precision="bf16-mixed",
            max_epochs=MAX_EPOCHS,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            gradient_clip_val=0.5,
            log_every_n_steps=200,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=callbacks,
            logger=logger,
            default_root_dir=CKPTS_DIR.as_posix(),
            #limit_train_batches=100,   # 先快速验证吞吐；确认后可去掉
            # profiler="simple",        # 如需定位瓶颈，打开
        )

        trainer.logger.log_hyperparams({"run_name": RUN_NAME})

        # 8.7 模型并训练
        tft = TemporalFusionTransformer.from_dataset(
            template,
            loss=RMSE(),
            logging_metrics=[WR2()],
            learning_rate=LR,
            hidden_size=HIDDEN,
            attention_head_size=HEADS,
            dropout=DROPOUT,
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

        best_ckpt_paths.append(ckpt_cb.best_model_path)
        fold_metrics.append(float(ckpt_cb.best_model_score))

        cv_wr2 = np.mean(fold_metrics)
        print(f"[CV] mean val_wr2 = {cv_wr2:.6f}")

    print("[done] best_ckpts =", best_ckpt_paths)

if __name__ == "__main__":
    main()
