# run_1_fe.py
from __future__ import annotations
from pathlib import Path
import gc, re
import polars as pl
import numpy as np
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_az
from pipeline.features import run_staged_engineering, StageA, StageB, StageC

def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def main():
    # ---- 常量/列名 ----
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    # 3 static features
    SYMBOL_STATIC_FEATURES = ["feature_09", "feature_10", "feature_11"]
    FEATURES_DY = [f for f in FEATURE_ALL if f not in SYMBOL_STATIC_FEATURES]

    
    RESP_COLS   = [f"responder_{i}" for i in range(9)]
    KEYS        = tuple(cfg["keys"])
    g_sym, g_date, g_time = KEYS
    TB = cfg['time_bucket']
    DATA_LO, DATA_HI = cfg["dates"]["fe_dates"]["date_lo"], cfg["dates"]["fe_dates"]["date_hi"]
    # ---- I/O ----
    clean_root = azify(P("az", cfg["paths"]["clean_shards"]))
    fe_root    = azify(P("az", cfg["paths"]["fe_shards"]))
    ensure_dir_az(fe_root)

    clean_paths = [azify(p) for p in sorted(fs.glob(f"{clean_root}/*.parquet"))]
    if not clean_paths:
        raise FileNotFoundError(f"No clean shards under {clean_root}")
    
    
    lc = pl.scan_parquet(clean_paths, storage_options=storage_options)
    lc = (
        lc.filter(pl.col(g_date).is_between(DATA_LO, DATA_HI))
    )
    
    days = (lc.select(pl.col("date_id").unique().sort())
              .collect(streaming=True)["date_id"].to_list())
    print(f"[FE] total unique days: {len(days)} in {clean_root}")

    # ---- 阶段配置 ----
    fea = cfg.get("feature_eng", {})
    A_cfg, B_cfg, C_cfg = fea.get("A", {}), fea.get("B", {}), fea.get("C", {})
    A = StageA(
        tail_lags=A_cfg.get("tail_lags", [1]),
        tail_diffs=A_cfg.get("tail_diffs", [1]),
        rolling_windows=A_cfg.get("rolling_windows", [3]),
        is_sorted=A_cfg.get("is_sorted", False),
        cast_f32=A_cfg.get("cast_f32", True),
    ) if A_cfg.get("enabled", True) else None
    B = StageB(
        lags=B_cfg.get("lags", [1,2,3,4,5,6,7,14,21,30]),
        stats_rep_cols=B_cfg.get("stats_rep_cols", None),
        add_prev1_multirep=B_cfg.get("add_prev1_multirep", True),
        batch_size=B_cfg.get("batch_size", 5),
        is_sorted=B_cfg.get("is_sorted", False),
        cast_f32=B_cfg.get("cast_f32", True),
    ) if B_cfg.get("enabled", True) else None
    C = StageC(
        lags=C_cfg.get("lags", [1,3]),
        ret_periods=C_cfg.get("ret_periods", [1]),
        diff_periods=C_cfg.get("diff_periods", [1]),
        rz_windows=C_cfg.get("rz_windows", [5]),
        ewm_spans=C_cfg.get("ewm_spans", [10]),
        keep_rmean_rstd=C_cfg.get("keep_rmean_rstd", True),
        cs_cols=C_cfg.get("cs_cols", None),
        batch_size=C_cfg.get("batch_size", 10),
        is_sorted=C_cfg.get("is_sorted", False),
        cast_f32=C_cfg.get("cast_f32", True),
    ) if C_cfg.get("enabled", True) else None

    # ---- 分片窗口 ----
    PAD_DAYS  = int(fea.get('fe_pad_days', 30))
    CORE_DAYS = int(fea.get('fe_core_days', 30))
    total_fe_batches = max(0, ((len(days) - PAD_DAYS) + CORE_DAYS - 1) // CORE_DAYS)

    for start in tqdm(range(PAD_DAYS, len(days), CORE_DAYS),
                      total=total_fe_batches, desc="FE shards (A/B/C)"):
        core_lo_idx = start
        core_hi_idx = min(start + CORE_DAYS - 1, len(days) - 1)
        pad_lo_idx  = core_lo_idx - PAD_DAYS

        core_lo, core_hi = days[core_lo_idx], days[core_hi_idx]
        pad_lo = days[pad_lo_idx]


        lf_shard = (
            lc.filter(pl.col(g_date).is_between(pad_lo, core_hi))
              .select([*cfg['keys'], cfg['weight'], TB, *RESP_COLS, *FEATURES_DY])
        )

        out_dir = azify(f"{fe_root}/fe_{core_lo:04d}_{core_hi:04d}")
        ensure_dir_az(out_dir)     

        print(f"[FE] build {out_dir}  (pad:{pad_lo} → core:{core_lo}..{core_hi})")
        run_staged_engineering(
            lf_base=lf_shard,
            keys=cfg['keys'],
            rep_cols=RESP_COLS,
            feature_cols=FEATURES_DY,
            out_dir=out_dir,
            A=A, B=B, C=C,
            write_date_between=(core_lo, core_hi),
        )

    print("[FE] all shards done.")

if __name__ == "__main__":
    main()
