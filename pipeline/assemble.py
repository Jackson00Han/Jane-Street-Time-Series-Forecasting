# pipeline/assemble.py
from __future__ import annotations
from typing import Sequence
import numpy as np
import polars as pl

def add_time_features(lf: pl.LazyFrame, T: int) -> pl.LazyFrame:
    T32 = np.float32(T)
    twopi_over_T = np.float32(2.0 * np.pi) / T32
    return (
        lf.with_columns([
            pl.col("time_id").cast(pl.Float32).alias("time_pos"),
            (pl.col("time_id").cast(pl.Float32) * pl.lit(twopi_over_T, dtype=pl.Float32)).alias("_phase_"),
        ])
        .with_columns([
            pl.col("_phase_").sin().cast(pl.Float32).alias("time_sin"),
            pl.col("_phase_").cos().cast(pl.Float32).alias("time_cos"),
        ])
        .drop(["_phase_"])
    )

def join_panel(
    *,
    clean_scan: pl.LazyFrame,
    fe_stage_a: pl.LazyFrame,
    fe_stage_b: pl.LazyFrame,
    fe_stage_c_scans: list[pl.LazyFrame],
    keys: Sequence[str],
    time_sort: Sequence[str],
) -> pl.LazyFrame:
    panel = clean_scan.join(fe_stage_a, on=list(keys), how="left", suffix="_A")
    panel = panel.join(fe_stage_b, on=list(keys), how="left", suffix="_B")
    for C in fe_stage_c_scans:
        panel = panel.join(C, on=list(keys), how="left", suffix="_C")
    return panel.sort(time_sort)
