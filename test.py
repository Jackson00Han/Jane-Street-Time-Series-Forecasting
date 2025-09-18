import re, os, polars as pl
from pipeline.io import storage_options, cfg, fs, P

panel_root = P("az", cfg["paths"]["panel_shards"])
date_col = cfg["keys"][1]   # 'date_id'
wcol     = cfg["weight"]    # 'weight'
fs_lo    = int(cfg['dates']["feature_select_dates"]["date_lo"])
fs_hi    = int(cfg['dates']["feature_select_dates"]["date_hi"])

def key(p):
    m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)

paths = [f"az://{p}" for p in sorted(fs.glob(f"{panel_root}/panel_*.parquet"), key=key)]

bad_panels = []
for p in paths:
    lb = (pl.scan_parquet(p, storage_options=storage_options)
            .select([date_col, wcol])
            .filter((pl.col(date_col) >= fs_lo) & (pl.col(date_col) <= fs_hi)))
    out = (lb.select([
              pl.len().alias("n"),
              pl.col(wcol).is_null().sum().alias("nulls"),
              pl.col(wcol).is_nan().sum().alias("nans"),
              pl.col(wcol).dtype.str_repr().alias("dtype"),
           ]).collect(streaming=True))
    n, nulls, nans, = int(out["n"][0]), int(out["nulls"][0]), int(out["nans"][0])
    if n and (nulls or nans):
        bad_panels.append((os.path.basename(p), n, nulls, nans))

print("[panel] bad panels:", len(bad_panels))
for row in bad_panels[:10]:
    print("[panel] ", row)
