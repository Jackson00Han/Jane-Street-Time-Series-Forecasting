# run_fe.py
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np
import gc
from pipeline.io import cfg, fs, storage_options, P, ensure_dir_az
from pipeline.features import run_staged_engineering, StageA, StageB, StageC
from pipeline.validate import assert_panel_shard
import re

# ---------- small utils ----------
def azify(p: str) -> str:
    """Ensure an Azure path has 'az://' prefix exactly once."""
    return p if p.startswith("az://") else f"az://{p}"



def main():
    # ----- constants & columns -----
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    RESP_COLS   = [f"responder_{i}" for i in range(9)]
    KEYS        = tuple(cfg["keys"])
    g_sym, g_date, g_time = KEYS
    TARGET, WEIGHT = cfg["target"], cfg["weight"]
    TB = cfg['time_bucket']
    TIME_SORT = cfg['sorts']['time_major']
    
    # ticks 读取
    T = np.float32(cfg["trading"]["ticks"])
    TWOPI_over_T = np.float32(2.0*np.pi) / T     # 全是 float32
    twopi_over_T_lit = pl.lit(TWOPI_over_T, dtype=pl.Float32)
    
    # ----- read shards -----
    clean_root = azify(P("az", cfg["paths"]["clean_shards"]))
    fe_root    = azify(P("az", cfg["paths"]["fe_shards"]))
    panel_root = azify(P("az", cfg["paths"]["panel_shards"]))
    ensure_dir_az(fe_root)
    ensure_dir_az(panel_root)
    
    clean_paths = [azify(p) for p in sorted(fs.glob(f"{clean_root}/*.parquet"))]
    if not clean_paths:
        raise FileNotFoundError(f"No clean shards under {clean_root}")  
    lc = pl.scan_parquet(clean_paths, storage_options=storage_options)
    # 如有需要，可以筛选日期范围 lc = lc.filter(pl.col(g_date).is_between(...))
    
    days = lc.select(pl.col(g_date).unique().sort()).collect(streaming=True)[g_date].to_list()
        
    # ----- build stages from cfg -----
    fea = cfg.get("feature_eng", {})
    A_cfg = fea.get("A", {})
    B_cfg = fea.get("B", {})
    C_cfg = fea.get("C", {})
    A_enabled = A_cfg.get("enabled", True)
    B_enabled = B_cfg.get("enabled", True)
    C_enabled = C_cfg.get("enabled", True)

    A = (StageA(
            tail_lags=A_cfg.get("tail_lags", [1]),
            tail_diffs=A_cfg.get("tail_diffs", [1]),
            rolling_windows=A_cfg.get("rolling_windows", [3]),
            prev_soft_days=A_cfg.get("prev_soft_days", 7),
            is_sorted=A_cfg.get("is_sorted", False),
            cast_f32=A_cfg.get("cast_f32", True),
        ) if A_enabled else None)

    B = (StageB(
            ndays=B_cfg.get("ndays", 5),
            stats_rep_cols=B_cfg.get("stats_rep_cols", None),
            add_prev1_multirep=B_cfg.get("add_prev1_multirep", True),
            batch_size=B_cfg.get("batch_size", 5),
            prev_soft_days=B_cfg.get("prev_soft_days", 7),
            is_sorted=B_cfg.get("is_sorted", False),
            cast_f32=B_cfg.get("cast_f32", True),
        ) if B_enabled else None)

    C = (StageC(
            lags=C_cfg.get("lags", [1,3]),
            ret_periods=C_cfg.get("ret_periods", [1]),
            diff_periods=C_cfg.get("diff_periods", [1]),
            rz_windows=C_cfg.get("rz_windows", [5]),
            ewm_spans=C_cfg.get("ewm_spans", [10]),
            keep_rmean_rstd=C_cfg.get("keep_rmean_rstd", True),
            cs_cols=C_cfg.get("cs_cols", None),
            prev_soft_days=C_cfg.get("prev_soft_days", 7),
            batch_size=C_cfg.get("batch_size", 10),
            is_sorted=C_cfg.get("is_sorted", False),
            cast_f32=C_cfg.get("cast_f32", True),
        ) if C_enabled else None)
    
    
    # -------- FE shard loop: read [pad_lo..core_hi], write only [core_lo..core_hi] --------
    PAD_DAYS = int(fea.get("fe_pad_days", 30))
    CORE_DAYS = int(fea.get("fe_core_days", 30))

    for start in range(PAD_DAYS, len(days), CORE_DAYS):
        core_lo_idx = start
        core_hi_idx = min(start + CORE_DAYS - 1, len(days) - 1) # 闭区间
        pad_lo_idx = core_lo_idx - PAD_DAYS
        
        core_lo, core_hi = days[core_lo_idx], days[core_hi_idx]
        pad_lo = days[pad_lo_idx]
        
        # 仅读本片+pad的输入 （懒加载）
        lf_shard = (lc.filter(pl.col(g_date).is_between(pad_lo, core_hi))
                    .select([*cfg['keys'], cfg['weight'], TB, *RESP_COLS, *FEATURE_ALL]))
        # 输出目录
        out_dir = azify(f"{fe_root}/fe_{core_lo:04d}_{core_hi:04d}")
        ensure_dir_az(out_dir)
        
        run_staged_engineering(
            lf_base = lf_shard,
            keys = cfg['keys'],
            rep_cols = RESP_COLS,
            feature_cols = FEATURE_ALL,
            out_dir = out_dir,
            A = A,
            B = B,
            C = C,
            write_date_between=(core_lo, core_hi)
            )
        print(f"[FE] days {core_lo}..{core_hi} (pad from {pad_lo}) -> {out_dir}")
        
    # -------- stitch A/B/C into panel shards --------

    DATE_LO, DATE_HI = cfg['dates']['train_lo'], cfg['dates']['train_hi']
    print(f"DATE_LO: {DATE_LO}, DATE_HI: {DATE_HI}")

    # 列出 fe 分片目录，并解析窗口
    shard_dirs = [p for p in fs.glob(f"{fe_root}/*")]
    shard_dirs = [azify(p) for p in sorted(shard_dirs)]
    wins: list[tuple[int,int]] = []
    for p in shard_dirs:
        base = p.rstrip("/").split("/")[-1]  # e.g. fe_1030_1059
        m = re.match(r"fe_(\d+)_(\d+)$", base)
        if not m:
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:
            wins.append((lo, hi))
    wins = sorted(set(wins))
    print(f"windows in range: {wins[:5]} ... (total {len(wins)})")
    
    # 预备：把 clean 作为基表源（一次 cast 即可）
    cast_keys = [pl.col(k).cast(pl.Int32).alias(k) for k in KEYS]
    lc_base = pl.scan_parquet(clean_paths, storage_options=storage_options).with_columns(cast_keys)
    
    # 构造 base 选择器（一次性加时间特征
    ti_f = pl.col(g_time).cast(pl.Float32)
    
    for (lo, hi) in wins:
        # 与全局区间取交集，防止边缘窗口越界
        w_lo, w_hi = max(lo, DATE_LO), min(hi, DATE_HI)
        
        shard_name = f"fe_{lo:04d}_{hi:04d}"
        fe_dir = azify(f"{fe_root}/{shard_name}")
        # 基表（筛行 + 选列 + 时间三件套）
        lf = (
            lc_base.filter(pl.col("date_id").is_between(w_lo, w_hi))
            .select([*KEYS, TB, TARGET, WEIGHT, *FEATURE_ALL])
            .with_columns([
                ti_f.alias("time_pos"),
                (ti_f * twopi_over_T_lit).alias("_phase_").cast(pl.Float32),
            ])
            .with_columns([
                # 兼容旧版：对表达式调用 .sin() / .cos()
                pl.col("_phase_").sin().cast(pl.Float32).alias("time_sin"),
                pl.col("_phase_").cos().cast(pl.Float32).alias("time_cos"),
            ])
            .drop(["_phase_"])
        )
        

        # A/B
        A = pl.scan_parquet(f"{fe_dir}/stage_a.parquet", storage_options=storage_options).with_columns(cast_keys)
        B = pl.scan_parquet(f"{fe_dir}/stage_b.parquet", storage_options=storage_options).with_columns(cast_keys)
        
        # C（多文件）
        C_paths = [azify(p) for p in sorted(fs.glob(f"{fe_dir}/stage_c_*.parquet"))]
        C_scans = [
            pl.scan_parquet(p, storage_options=storage_options).with_columns(cast_keys)
            for p in C_paths
        ]
        
        # 逐个 join 
        panel = lf.join(A, on=list(KEYS), how="left", suffix="_A")
        panel = panel.join(B, on=list(KEYS), how="left", suffix="_B")
        for C in C_scans:
            panel = panel.join(C, on=list(KEYS), how="left", suffix="_C")
        
        # 排序
        panel = panel.sort(TIME_SORT)
        
        # 收集并写盘
        df_out = panel.collect(streaming=True)
        out_path = f"{panel_root}/panel_{w_lo:04d}_{w_hi:04d}.parquet"
        with fs.open(out_path, "wb") as f:
            df_out.write_parquet(f, compression="zstd")
            
        print(f"[panel] wrote {out_path} with {df_out.shape[0]} rows")
        
        if cfg.get("debug", {}).get("check_time_monotone", True):
            assert_panel_shard(out_path, w_lo, w_hi, date_col=g_date, time_col=g_time)
            
        del df_out
        gc.collect()

if __name__ == "__main__":
    main()
        