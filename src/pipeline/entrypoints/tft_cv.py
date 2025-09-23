# tft_cv.py  —— 统一 lightning.pytorch，按 symbol 标准化 + 缺失标记

from __future__ import annotations
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE

from pipeline.io import cfg, P, fs, storage_options, ensure_dir_local

def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")

# ---------- 工具函数 ----------
def add_missing_flags_and_fill(df: pd.DataFrame, group_col: str, cont_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """连续特征：添加 __isna 标记，组内 ffill，兜底 0 填充。"""
    if not cont_cols:
        return [], df
    df[cont_cols] = df[cont_cols].replace([np.inf, -np.inf], np.nan)
    flags = []
    # grouped ffill
    g = df.groupby(group_col, observed=False)
    for c in cont_cols:
        flag = f"{c}__isna"
        flags.append(flag)
        df[flag] = df[c].isna().astype("int8")
        df[c] = g[c].ffill()
        df[c] = df[c].fillna(0.0)
    return flags, df

def standardize_by_symbol(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        group_col: str,
                        cont_cols: list[str],
                        eps: float = 1e-6):
    """按 symbol 对连续特征做标准化；val 用 train 的统计量，新 symbol 回退到 train 的全局统计。"""
    if not cont_cols:
        # 没有需要标准化的列，直接返回
        return train_df, val_df, pd.DataFrame(), pd.DataFrame()

    mu_df  = train_df.groupby(group_col, observed=False)[cont_cols].mean()
    std_df = train_df.groupby(group_col, observed=False)[cont_cols].std(ddof=1).clip(lower=eps)

    # train
    train_mu  = mu_df.loc[train_df[group_col]].to_numpy()
    train_std = std_df.loc[train_df[group_col]].to_numpy()
    X_train   = train_df[cont_cols].to_numpy(dtype=np.float32)
    train_df[cont_cols] = ((X_train - train_mu) / train_std).astype(np.float32)

    # val
    val_syms = val_df[group_col]
    if (~val_syms.isin(mu_df.index)).any():
        g_mu  = train_df[cont_cols].mean()
        g_std = train_df[cont_cols].std(ddof=1).clip(lower=eps)
        val_mu  = mu_df.reindex(val_syms).fillna(g_mu).to_numpy()
        val_std = std_df.reindex(val_syms).fillna(g_std).to_numpy()
    else:
        val_mu  = mu_df.loc[val_syms].to_numpy()
        val_std = std_df.loc[val_syms].to_numpy()
    X_val = val_df[cont_cols].to_numpy(dtype=np.float32)
    val_df[cont_cols] = ((X_val - val_mu) / val_std).astype(np.float32)
    return train_df, val_df, mu_df, std_df

def build_trainer():
    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger(save_dir=P("local", "tft/logs"), name="tft", default_hp_metric=False)
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="tft-best-{epoch:02d}-{val_loss:.5f}"),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = L.Trainer(
        max_epochs=int(cfg.get("tft", {}).get("max_epochs", 2)),
        accelerator="auto",
        precision=precision,
        gradient_clip_val=float(cfg.get("tft", {}).get("grad_clip", 1.0)),
        log_every_n_steps=50,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=P("local", "tft/ckpts"),
    )
    return trainer

# ---------- 主流程 ----------
def main():
    print(f"[{_now()}][tft] ===== start =====")
    target_col = cfg["target"]                 # 如 "responder_6"
    g_sym, g_date, g_time = cfg["keys"]        # 如 ("symbol_id","date_id","time_id")
    TIME_SORT = cfg["sorts"].get("time_major", [g_date, g_time, g_sym])

    # 1) 选择特征列（示例：你后续替换为真实列表）
    base_features   = ["feature_01"]                 # TODO: 放入你的 79 原始列子集
    resp_his_feats  = ["responder_6_prevday_close"]  # 示例
    feat_his_feats  = ["feature_00__lag1"]           # 示例
    feature_cols = list(dict.fromkeys(base_features + resp_his_feats + feat_his_feats))

    # TSD 的 unknown_reals = 解码期未知的连续变量（不含 target）
    need_cols = list(dict.fromkeys(cfg["keys"] + [target_col] + feature_cols))

    # 2) 读 panel（Lazy） & 构 grid
    panel_dir = P("az", cfg["paths"].get("panel_shards", "panel_shards"))
    glob_pat  = f"{panel_dir}/*.parquet"
    if not fs.glob(glob_pat.replace("az://", "")):
        raise FileNotFoundError(f"No parquet shards under: {glob_pat}")
    lf = pl.scan_parquet(glob_pat, storage_options=storage_options)

    grid_path = P("local", "tft/panel/grid_timeidx.parquet")
    if not Path(grid_path).exists():
        lf_grid = (
            lf.select([g_date, g_time]).unique()
            .sort([g_date, g_time])
            .with_row_index("time_idx")
            .with_columns(pl.col("time_idx").cast(pl.Int64))
        )
        ensure_dir_local(Path(grid_path).parent.as_posix())
        lf_grid.collect(streaming=True).write_parquet(grid_path, compression="zstd")
        print(f"[{_now()}][tft] grid saved -> {grid_path}")
    grid_lazy = pl.scan_parquet(grid_path)

    # 全局 time_idx 连续性（安全检查）
    grid_df = grid_lazy.select([g_date, g_time, "time_idx"]).collect()
    ti = grid_df["time_idx"]
    assert grid_df.select(pl.col("time_idx").is_duplicated().any()).item() is False
    assert ti.max() - ti.min() + 1 == len(ti), "全局 time_idx 不连续"

    # 3) 时间窗 + join time_idx + 选列
    lo = cfg["dates"]["tft_dates"]["date_lo"]; hi = cfg["dates"]["tft_dates"]["date_hi"]
    lw = lf.filter(pl.col(g_date).is_between(lo, hi, closed="both"))
    lw_with_idx = (
        lw.join(grid_lazy, on=[g_date, g_time], how="left")
        .select(need_cols + ["time_idx"])
        .sort(TIME_SORT)
    )
    print(f"[{_now()}][tft] schema -> {lw_with_idx.collect_schema().names()}")

    # 4) 小窗 demo → pandas
    demo_lo, demo_hi = 1600, 1630
    df = (
        lw_with_idx
        .filter(pl.col(g_date).is_between(demo_lo, demo_hi, closed="both"))
        .collect(streaming=True)
        .to_pandas()
    ).sort_values([g_sym, "time_idx"])

    # 类型
    df[g_sym] = df[g_sym].astype("string").astype("category")
    df["time_idx"] = df["time_idx"].astype("int64")

    # 缺失处理（只作用于连续特征，不动 target）
    miss_flags, df = add_missing_flags_and_fill(df, g_sym, feature_cols)
    unknown_reals = list(dict.fromkeys(feature_cols + miss_flags))  # 特征 + 标记（不含 target）

    # 降精度
    for c in unknown_reals:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
    df[target_col] = pd.to_numeric(df[target_col], downcast="float")

    # 因果切分
    cutoff = int(df["time_idx"].quantile(0.9))
    train_df = df[df["time_idx"] <= cutoff].copy()
    val_df   = df[df["time_idx"] >  cutoff].copy()

    # 5) 按 symbol 标准化（仅连续特征；不含 missing flags / target）
    cont_cols = [c for c in feature_cols if c in train_df.columns]
    train_df, val_df, mu_df, std_df = standardize_by_symbol(train_df, val_df, g_sym, cont_cols)
    for c in cont_cols:
        train_df[c] = train_df[c].astype("float32")
        val_df[c]   = val_df[c].astype("float32")
    for f in miss_flags:
        if f in train_df: train_df[f] = train_df[f].astype("int8")
        if f in val_df:   val_df[f]   = val_df[f].astype("int8")

    print(f"[{_now()}][tft] standardize done, cont_cols={len(cont_cols)}")

    # 6) TimeSeriesDataSet
    training = TimeSeriesDataSet(
        train_df.sort_values([g_sym, "time_idx"]),
        time_idx="time_idx",
        target=target_col,
        group_ids=[g_sym],
        static_categoricals=[g_sym],        # == ["symbol_id"]
        static_reals=[],
        time_varying_known_categoricals=[], # 后续可加入交易日历等“未来可知”变量
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=unknown_reals,  # 只放特征与标记（不含 target）
        max_encoder_length=int(cfg.get("tft",{}).get("enc_len", 30)),
        max_prediction_length=1,
        target_normalizer=None,             # 先不对 target 做归一化
        categorical_encoders={g_sym: NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

    train_loader = training.to_dataloader(
        train=True, batch_size=int(cfg.get("tft",{}).get("batch_size", 1024)), num_workers=4
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=int(cfg.get("tft",{}).get("batch_size", 1024)), num_workers=4
    )

    # 7) Trainer + Model
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    trainer = build_trainer()

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(cfg.get("tft",{}).get("lr", 1e-3)),
        hidden_size=int(cfg.get("tft",{}).get("hidden_size",128)),
        attention_head_size=int(cfg.get("tft",{}).get("heads",4)),
        dropout=float(cfg.get("tft",{}).get("dropout",0.2)),
        loss=SMAPE(),
        reduce_on_plateau_patience=4,
    )

    # 保险：模型基类一致性（防止再遇到 “model must be LightningModule”）
    assert isinstance(tft, L.LightningModule), f"type={type(tft)}"

    trainer.fit(tft, train_loader, val_loader)
    print(f"[{_now()}][tft] ===== finished =====")

if __name__ == "__main__":
    main()
