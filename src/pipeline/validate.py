# pipeline/validate.py
from __future__ import annotations
import polars as pl
from pipeline.io import storage_options
import numpy as np
    
def assert_time_monotone(path, *, date_col="date_id", time_col="time_id"):
    s = (
        pl.scan_parquet(path, storage_options=storage_options)
        .select([
            (pl.col(date_col).diff().fill_null(0) < 0).any().alias('date_drop'),
            ((pl.col(date_col).diff().fill_null(0) == 0) &
             (pl.col(time_col).diff().fill_null(0) < 0)).any().alias('time_drop')
        ])
        .collect(streaming=True)
    )
    assert not s['date_drop'][0] and not s['time_drop'][0]

def assert_panel_shard(path, lo, hi, *, date_col="date_id", time_col="time_id"):
    s = (
        pl.scan_parquet(path, storage_options=storage_options)
        .select([
            pl.col(date_col).min().alias("dmin"),
            pl.col(date_col).max().alias("dmax"),
            (pl.col(date_col).diff().fill_null(0) < 0).any().alias('date_drop'),
            ((pl.col(date_col).diff().fill_null(0) == 0) &
             (pl.col(time_col).diff().fill_null(0) < 0)).any().alias('time_drop'),
        ])
        .collect(streaming=True)
    )
    dmin, dmax = int(s["dmin"][0]), int(s["dmax"][0])
    assert dmin == lo and dmax == hi, f"date range mismatch: got [{dmin},{dmax}] expect [{lo},{hi}]"
    assert not s['date_drop'][0] and not s['time_drop'][0]
