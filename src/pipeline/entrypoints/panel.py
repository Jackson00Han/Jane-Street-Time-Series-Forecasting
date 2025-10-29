# run_2_panel.py
from __future__ import annotations
import re, gc
import polars as pl
import numpy as np
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, ensure_dir_az
from pipeline.validate import assert_panel_shard

def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def main():
    # ---- Columns & config ----
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    KEYS = tuple(cfg["columns"]["keys"])              # ["symbol_id","date_id","time_id"]
    g_sym, g_date, g_time = KEYS
    TARGET, WEIGHT = cfg["columns"]["target"], cfg["columns"]["weight"]

    # helper columns
    TB = "time_bucket"                                # column name (listed in columns.allowlist)
    TIME_SORT = cfg["sorts"]["time_major"]
    
    # ---- Paths ----
    azure_root = cfg["azure"]["root"]
    clean_root = azify(f"{azure_root}/{cfg['paths']['clean_shards']['rel']}")
    fe_root    = azify(f"{azure_root}/{cfg['paths']['fe_shards']['rel']}")
    panel_root = azify(f"{azure_root}/{cfg['paths']['panel_shards']['rel']}"); ensure_dir_az(panel_root)

    # ---- Date range for panel ----
    DATE_LO = int(cfg["dates"]["panel"]["date_lo"])
    DATE_HI = int(cfg["dates"]["panel"]["date_hi"])
    print(f"[panel] target date range: {DATE_LO}..{DATE_HI}")

    # ---- Enumerate FE shard windows to stitch ----
    shard_dirs = [azify(p) for p in sorted(fs.glob(f"{fe_root}/*"))]
    wins: list[tuple[int,int]] = []
    for p in shard_dirs:
        bn = p.rstrip("/").split("/")[-1]  # e.g. fe_1030_1059
        m = re.match(r"fe_(\d+)_(\d+)$", bn)
        if not m:
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:
            wins.append((lo, hi))
    wins = sorted(set(wins))
    print(f"[panel] windows to stitch: {len(wins)} (first 5: {wins[:5]})")

    # ---- Base table (clean) as the join backbone ----
    cast_keys = [pl.col(k).cast(pl.Int32).alias(k) for k in KEYS]
    lc_base = (
        pl.scan_parquet(f"{clean_root}/*.parquet", storage_options=storage_options)
          .with_columns(cast_keys)
    )
    lc_base = lc_base.sort(TIME_SORT)

    # time features
    T = np.float32(cfg["trading"]["daily_ticks"])
    twopi_over_T = np.float32(2.0*np.pi) / T
    ti_f = pl.col(g_time).cast(pl.Float32)

    # ---- Stitching windows ----
    for (lo, hi) in tqdm(wins, desc="Stitch panel shards"):
        w_lo, w_hi = max(lo, DATE_LO), min(hi, DATE_HI)
        shard_name = f"fe_{lo:04d}_{hi:04d}"
        fe_dir = azify(f"{fe_root}/{shard_name}")

        # Base table filtering + time features
        lf = (
            lc_base.filter(pl.col(g_date).is_between(w_lo, w_hi))
                   .select([*KEYS, TB, TARGET, WEIGHT, *FEATURE_ALL])
                   .with_columns([
                       ti_f.alias("time_pos"),
                       (ti_f * pl.lit(twopi_over_T, dtype=pl.Float32)).alias("_phase_"),
                   ])
                   .with_columns([
                       pl.col("_phase_").sin().cast(pl.Float32).alias("time_sin"),
                       pl.col("_phase_").cos().cast(pl.Float32).alias("time_cos"),
                   ])
                   .drop(["_phase_"])
        )

        # A/B/C scanning
        A_scan = pl.scan_parquet(f"{fe_dir}/stage_a.parquet", storage_options=storage_options).with_columns(cast_keys)
        B_scan = pl.scan_parquet(f"{fe_dir}/stage_b.parquet", storage_options=storage_options).with_columns(cast_keys)
        C_paths = [azify(p) for p in sorted(fs.glob(f"{fe_dir}/stage_c_*.parquet"))]
        C_scans = [pl.scan_parquet(p, storage_options=storage_options).with_columns(cast_keys) for p in C_paths]

        # Stitching each join
        panel = lf.join(A_scan, on=list(KEYS), how="left", suffix="_A")
        panel = panel.join(B_scan, on=list(KEYS), how="left", suffix="_B")
        for Cscan in tqdm(C_scans, desc=f"Join C {shard_name}", leave=False):
            panel = panel.join(Cscan, on=list(KEYS), how="left", suffix="_C")

        # Sorting + writing to disk
        panel = panel.sort(TIME_SORT)
        df_out = panel.collect(streaming=True)

        out_path = f"{panel_root}/panel_{w_lo:04d}_{w_hi:04d}.parquet"
        with fs.open(out_path, "wb") as f:
            df_out.write_parquet(f, compression="zstd")
        print(f"[panel] wrote {out_path} with {df_out.shape[0]:,} rows")

        # Time monotonicity check within the shard (can be controlled by cfg.debug.check_time_monotone)
        if cfg.get("debug", {}).get("check_time_monotone", True):
            assert_panel_shard(out_path, w_lo, w_hi, date_col=g_date, time_col=g_time)

        del df_out
        gc.collect()

    print("[panel] all shards stitched.")

if __name__ == "__main__":
    main()
