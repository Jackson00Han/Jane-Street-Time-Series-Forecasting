# run_clean.py
from __future__ import annotations
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, ensure_dir_local, ensure_dir_az
from pipeline.preprocess import clip_upper, rolling_sigma_clip, causal_impute

def main():
    # ===============================
    # 1) IO roots
    # ===============================
    azure_root = cfg["azure"]["root"]
    ensure_dir_az(azure_root)

    local_root = cfg["local"]["root"]
    ensure_dir_local(local_root)

    # Build raw-partitions listing (ensure every path has az:// scheme)
    raw_dir = f"az://{azure_root}/{cfg['paths']['raw_partitions']['rel']}"
    raw_matches = fs.glob(f"{raw_dir}/partition_id=*/*.parquet")
    raw_az_paths = [f"az://{p}" for p in raw_matches]
    raw_data_paths = sorted(raw_az_paths)

    # Local cache files for intermediates
    path_clip = Path(local_root) / cfg["paths"]["cache"]["rel"] / "clipped.parquet"
    ensure_dir_local(path_clip.parent)

    path_imp = Path(local_root) / cfg["paths"]["cache"]["rel"] / "imputed.parquet"
    ensure_dir_local(path_imp.parent)

    clean_root = f"az://{azure_root}/{cfg['paths']['clean_shards']['rel']}"; ensure_dir_az(clean_root)
    
    # ===============================
    # 2) Constants / columns
    # ===============================
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    RESP_COLS   = [f"responder_{i}" for i in range(9)]
    batch_size  = int(cfg["fill"]["batch_size"])
    TIME_SORT   = cfg["sorts"]["time_major"]
    SYMBOL_SORT = cfg["sorts"].get("symbol_major", TIME_SORT)
    KEYS, WEIGHT = cfg["columns"]["keys"], cfg["columns"]["weight"]
    
    
    # ===============================
    # 3) Load raw + create time buckets
    # ===============================
    lf_raw = pl.scan_parquet(raw_data_paths, storage_options=storage_options)
    lf_raw = lf_raw.filter(
        pl.col("date_id").is_between(
            int(cfg["dates"]["clean"]["date_lo"]),
            int(cfg["dates"]["clean"]["date_hi"]),
            closed="both",
        )
    )

    B = int(cfg["trading"]["bucket_size"])
    T = int(cfg["trading"]["daily_ticks"])

    lf_raw = (
        lf_raw.with_columns(bucket_raw = pl.col("time_id") * pl.lit(B) // pl.lit(T))
              .with_columns(time_bucket = clip_upper(pl.col("bucket_raw"), B - 1).cast(pl.UInt8))
              .drop("bucket_raw")
              .sort(SYMBOL_SORT)
    )

    # ===============================
    # 4) Robust clipping (rolling z-score)
    # ===============================
    lf_clip = rolling_sigma_clip(
        lf=lf_raw,
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
    lf_clip.collect(streaming=True).write_parquet(str(path_clip), compression="zstd")
    
    # ===============================
    # 5) Causal imputation
    # ===============================
    lf_imp = (
        causal_impute(
            lf=pl.scan_parquet(path_clip).sort(SYMBOL_SORT),
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
    lf_imp.collect(streaming=True).write_parquet(str(path_imp), compression="zstd")

    # ===============================
    # 6) Join targets/weights/time_bucket (right table)
    # ===============================
    rhs = (
        lf_raw.select([*KEYS, WEIGHT, "time_bucket", *RESP_COLS])
              .with_columns([pl.col(k).cast(pl.Int32) for k in KEYS])
              .sort(TIME_SORT)
    )

    # Left table (imputed features)
    lf_imp2 = pl.scan_parquet(str(path_imp)).with_columns([pl.col(k).cast(pl.Int32) for k in KEYS]).sort(TIME_SORT)

    dmin, dmax = (
        lf_imp2.select(pl.col("date_id").min().alias("dmin"), pl.col("date_id").max().alias("dmax"))
               .collect()
               .row(0)
    )
    
    # ===============================
    # 7) Write by day (batched)
    # ===============================
    total_batches = (dmax - dmin + 1 + batch_size - 1) // batch_size
    for lo in tqdm(range(dmin, dmax + 1, batch_size), total=total_batches, desc="clean shards"):
        hi = min(lo + batch_size, dmax + 1)

        left  = lf_imp2.filter(pl.col("date_id").is_between(lo, hi, closed="left"))
        right = rhs   .filter(pl.col("date_id").is_between(lo, hi, closed="left"))

        part = left.join(right, on=TIME_SORT, how="left").sort(TIME_SORT)

        # hi is exclusive → name file with hi-1
        out_lo, out_hi = lo, hi - 1
        out_path = f"{clean_root}/clean_{out_lo:04d}_{out_hi:04d}.parquet"

        part.collect(streaming=True).write_parquet(
            out_path,
            compression="zstd",
            statistics=True,
            storage_options=storage_options,
        )

if __name__ == "__main__":
    main()
