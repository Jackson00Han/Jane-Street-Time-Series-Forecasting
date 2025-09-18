# run_clean.py
from __future__ import annotations
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local, ensure_dir_az
from pipeline.preprocess import clip_upper, rolling_sigma_clip, causal_impute

def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def main():
    # ===============================
    # 1) 发现原始分片（Azure）
    # ===============================
    pattern = "az://jackson/js_exp/raw/train.parquet/partition_id=*/*.parquet"
    raw_paths = fs.glob(pattern)            # 可能返回带或不带 az://
    paths = sorted(azify(p) for p in raw_paths)

    # ===============================
    # 2) 常量/列名与通用配置
    # ===============================
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    RESP_COLS   = [f"responder_{i}" for i in range(9)]
    batch_size  = cfg["fill"]["batch_size"] # 每批次含多少天
    TIME_SORT   = cfg["sorts"]["time_major"]
    KEYS, WEIGHT = cfg["keys"], cfg["weight"]

    # ===============================
    # 3) 读取 + 时间桶
    # ===============================
    lb = pl.scan_parquet(paths, storage_options=storage_options)
    if "date_lo" in cfg["dates"] and "date_hi" in cfg["dates"]:
        lb = lb.filter(
            pl.col("date_id").is_between(
                int(cfg["dates"]["date_lo"]),
                int(cfg["dates"]["date_hi"]),
                closed="both",
            )
        )

    B = int(cfg["trading"]["bucket_size"])
    T = int(cfg["trading"]["ticks"])

    # 添加时间桶列，并按 KEY 排序（为后续时序操作做准备）
    lb = (
        lb.with_columns(bucket_raw = pl.col("time_id") * pl.lit(B) // pl.lit(T))
          .with_columns(time_bucket = clip_upper(pl.col("bucket_raw"), B - 1).cast(pl.UInt8))
          .drop("bucket_raw")
          .sort(KEYS)
    )

    # ===============================
    # 4) Clipping（滚动 z-score 截断）
    # ===============================
    lf_clip = rolling_sigma_clip(
        lf=lb,
        clip_features=FEATURE_ALL,
        over_cols=cfg["winsorization"]["groupby"],
        is_sorted=True,
        window=cfg["winsorization"]["window"],
        k=cfg["winsorization"]["z_k"],
        ddof=cfg["winsorization"]["ddof"],
        min_valid=cfg["winsorization"]["min_valid"],
        cast_float32=cfg["winsorization"]["cast_float32"],
        sanitize=cfg["winsorization"].get("sanitize", True),
    )
    clip_out = Path(P("local", cfg["paths"]["cache"])) / "sample_clipped.parquet"
    ensure_dir_local(clip_out.parent)
    lf_clip.collect(streaming=True).write_parquet(str(clip_out), compression="zstd")

    # ===============================
    # 5) Imputing（因果填补）
    # ===============================
    lf_imp = (
        causal_impute(
            lf=pl.scan_parquet(clip_out.as_posix()).sort(KEYS),
            impute_cols=FEATURE_ALL,
            open_tick_window=cfg["fill"]["open_tick_window"],
            ttl_days_open=cfg["fill"]["ttl_days_open"],
            intra_ffill_max_gap_ticks=cfg["fill"]["intra_ffill_max_gap_ticks"],
            ttl_days_same_tick=cfg["fill"]["ttl_days_same_tick"],
            is_sorted=True,
        )
        .with_columns([pl.col(c).fill_null(0.0).alias(c) for c in FEATURE_ALL])
        .with_columns([pl.col(k).cast(pl.Int32).alias(k) for k in KEYS])
    )
    imp_path = Path(P("local", cfg["paths"]["cache"])) / "sample_imputed.parquet"
    ensure_dir_local(imp_path.parent)
    lf_imp.collect(streaming=True).write_parquet(str(imp_path), compression="zstd", use_pyarrow=True)

    # ===============================
    # 6) 合并响应变量（右表：权重+响应+time_bucket）
    # ===============================

    # 右表：
    rhs = (
        lb.select([*KEYS, WEIGHT, 'time_bucket', *RESP_COLS])
        .with_columns([pl.col(k).cast(pl.Int32) for k in KEYS]).sort(TIME_SORT)
    )
    print("Right table schema:", rhs.collect_schema())
    print("row count:", rhs.select(pl.count()).collect())

    # 左表
    imp_path = Path(P("local", cfg["paths"]["cache"])) / "sample_imputed.parquet"
    lf_imp = pl.scan_parquet(str(imp_path)).with_columns([pl.col(k).cast(pl.Int32) for k in KEYS])
    lf_imp = lf_imp.sort(TIME_SORT)

    print("Left table schema:", lf_imp.collect_schema())
    print("row count:", lf_imp.select(pl.count()).collect())


    dmin, dmax = (
        lf_imp.select(
            pl.col('date_id').min().alias('dmin'),
            pl.col('date_id').max().alias('dmax')
            )
        .collect()
        .row(0)
    )
    print(f"Date range: {dmin} to {dmax}, total {dmax - dmin + 1} days")

    path = P('az', cfg['paths']['clean_shards'])
    fs.makedirs(path, exist_ok=True)
    print(f"Processing date range: {dmin} to {dmax}")
    
    # ===============================
    # 7) 按日批输出（进度条）
    # ===============================

    total_batches = (dmax - dmin + 1 + batch_size - 1) // batch_size
    for lo in tqdm(range(dmin, dmax + 1, batch_size), total=total_batches, desc="clean shards"):
        hi = min(lo + batch_size, dmax + 1)

        left = (
            lf_imp
            .filter(pl.col('date_id').is_between(lo, hi, closed='left'))
        )
        
        right = rhs.filter(pl.col('date_id').is_between(lo, hi, closed='left'))

        part = (left.join(right, on=TIME_SORT, how='left')).sort(TIME_SORT)

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
