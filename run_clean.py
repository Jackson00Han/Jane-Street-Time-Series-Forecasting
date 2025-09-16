# run_clean.py
from __future__ import annotations
from pathlib import Path
import polars as pl

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local, ensure_dir_az
from pipeline.preprocess import clip_upper, rolling_sigma_clip, causal_impute



def main():
    # ---- 1) 发现原始分片（Azure）----
    raw_paths = fs.glob(f"az://jackson/js_exp/raw/train.parquet/partition_id=[4-6]/*.parquet")
    
    paths = [f"az://{p}" for p in raw_paths]
    assert raw_paths, f"No raw parquet files under {raw_root}"

    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    RESP_COLS = [f"responder_{i}" for i in range(9)]
    batch_size = cfg['fill']['batch_size']
    TIME_SORT = cfg['sorts']['time_major']
    KEYS, WEIGHT = cfg["keys"], cfg["weight"]

    # ---- 2) 读取 + 时间桶 ----
    lb = pl.scan_parquet(paths, storage_options=storage_options)
    if "date_lo" in cfg['dates'] and "date_hi" in cfg['dates']:
        lb = lb.filter(pl.col("date_id").is_between(int(cfg["dates"]["date_lo"]), int(cfg["dates"]["date_hi"]), closed="both"))

    B = int(cfg['trading']['bucket_size']); T = cfg['trading']['ticks'] 
    
    # 添加时间桶列
    lb = (
        lb.with_columns(bucket_raw = pl.col('time_id') * pl.lit(B) // pl.lit(T))
        .with_columns(time_bucket = clip_upper(pl.col('bucket_raw'), B - 1).cast(pl.UInt8))
        .drop('bucket_raw')
        .sort(KEYS)
    )
    
    # Clipping
    lf_clip = rolling_sigma_clip(
        lf=lb, 
        clip_features=FEATURE_ALL, 
        over_cols=cfg['winsorization']['groupby'],
        is_sorted=True, 
        window=cfg['winsorization']['window'], 
        k=cfg['winsorization']['z_k'],
        ddof=cfg['winsorization']['ddof'], 
        min_valid=cfg['winsorization']['min_valid'],
        cast_float32=cfg['winsorization']['cast_float32'], 
        sanitize=cfg['winsorization'].get('sanitize', True)
    )
    clip_out = Path(P("local", cfg["paths"]["cache"])) / "sample_clipped.parquet"; ensure_dir_local(clip_out.parent)

    lf_clip.collect().write_parquet(str(clip_out), compression="zstd")
    
    # Imputing
    lf_imp = (
        causal_impute(
            lf=pl.scan_parquet(clip_out.as_posix()).sort(KEYS),
            impute_cols=FEATURE_ALL,
            open_tick_window=cfg['fill']['open_tick_window'],
            ttl_days_open=cfg['fill']['ttl_days_open'],
            intra_ffill_max_gap_ticks=cfg['fill']['intra_ffill_max_gap_ticks'],
            ttl_days_same_tick=cfg['fill']['ttl_days_same_tick'],
            is_sorted=True,
        )
        .with_columns([pl.col(c).fill_null(0.0).alias(c) for c in FEATURE_ALL])
        .with_columns([pl.col(k).cast(pl.Int32).alias(k) for k in KEYS])
    )
    imp_path = Path(P("local", cfg["paths"]["cache"])) / "sample_imputed.parquet"; ensure_dir_local(imp_path.parent)
    lf_imp.collect(streaming=True).write_parquet(str(imp_path), compression="zstd", use_pyarrow=True)  # 可加 use_pyarrow=True

    # 合并响应变量
    

    # 右表：去重 + 对齐类型
    rhs = (
        lb.select([*KEYS, WEIGHT, 'time_bucket', *RESP_COLS])
        .with_columns([pl.col(k).cast(pl.Int32) for k in KEYS])
    )
    # 左表
    imp_path = Path(P("local", cfg["paths"]["cache"])) / "sample_imputed.parquet"
    lf_imp = pl.scan_parquet(str(imp_path)).with_columns([pl.col(k).cast(pl.Int32) for k in KEYS])

    dmin, dmax = (
        lf_imp.select(
            pl.col('date_id').min().alias('dmin'),
            pl.col('date_id').max().alias('dmax')
            )
        .collect()
        .row(0)
    )

    path = P('az', cfg['paths']['clean_shards']); ensure_dir_az(path)
    print(f"Processing date range: {dmin} to {dmax}")
    for lo in range(dmin, dmax + 1, batch_size):
        hi = min(lo + batch_size, dmax + 1)

        left = (
            lf_imp
            .filter(pl.col('date_id').is_between(lo, hi, closed='left'))
        )
        right = rhs.filter(pl.col('date_id').is_between(lo, hi, closed='left')).sort(TIME_SORT).unique(subset=KEYS, keep='last')

        part = (left.join(right, on=KEYS, how='left')).sort(TIME_SORT)
        
        feature_cols = [c for c in part.collect_schema().names() if c not in set([*KEYS, WEIGHT, 'time_bucket', *RESP_COLS])]
        part = part.select([*KEYS, WEIGHT, 'time_bucket', *feature_cols,  *RESP_COLS])


        # 命名时注意 hi 是排他的，所以文件名用 hi-1
        out_lo, out_hi = lo, hi - 1
        (
            part.sink_parquet(
                f"{path}/clean_{out_lo:04d}_{out_hi:04d}.parquet",
                compression="zstd",
                statistics=True,                 # 写入页/列统计划出更快
                storage_options=storage_options,
            )
        )

if __name__ == "__main__":
    main()
    
    
    