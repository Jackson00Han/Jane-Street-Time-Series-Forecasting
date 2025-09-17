# run_fe.py
from __future__ import annotations
from pathlib import Path
import re
import gc

import polars as pl
import numpy as np
from tqdm.auto import tqdm  # ✅ 进度条

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_az
from pipeline.features import run_staged_engineering, StageA, StageB, StageC
from pipeline.validate import assert_panel_shard


# ---------- small utils ----------
def azify(p: str) -> str:
    """Ensure an Azure path has 'az://' prefix exactly once."""
    return p if p.startswith("az://") else f"az://{p}"


def main():
    # ==============================
    # 0) 常量与列名
    # ==============================
    FEATURE_ALL = [f"feature_{i:02d}" for i in range(79)]
    RESP_COLS   = [f"responder_{i}" for i in range(9)]
    KEYS        = tuple(cfg["keys"])
    g_sym, g_date, g_time = KEYS
    TARGET, WEIGHT = cfg["target"], cfg["weight"]
    TB = cfg['time_bucket']
    TIME_SORT = cfg['sorts']['time_major']

    # ticks
    T = np.float32(cfg["trading"]["ticks"])
    TWOPI_over_T = np.float32(2.0*np.pi) / T
    twopi_over_T_lit = pl.lit(TWOPI_over_T, dtype=pl.Float32)

    # ==============================
    # 1) 路径与读取 clean shards
    # ==============================
    clean_root = azify(P("az", cfg["paths"]["clean_shards"]))
    fe_root    = azify(P("az", cfg["paths"]["fe_shards"]))
    panel_root = azify(P("az", cfg["paths"]["panel_shards"]))
    ensure_dir_az(fe_root)
    ensure_dir_az(panel_root)

    clean_paths = [azify(p) for p in sorted(fs.glob(f"{clean_root}/*.parquet"))]
    if not clean_paths:
        raise FileNotFoundError(f"No clean shards under {clean_root}")

    lc = pl.scan_parquet(clean_paths, storage_options=storage_options)
    days = (
        lc.select(pl.col("date_id").unique().sort())
          .collect(streaming=True)["date_id"]
          .to_list()
    )

    # ==============================
    # 2) 读取/构建 A/B/C 阶段配置
    # ==============================
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

    # ==============================
    # 3) FE 按片生成：读 [pad_lo..core_hi]，仅写 [core_lo..core_hi]
    # ==============================
    ensure_dir_az(fe_root)
    PAD_DAYS  = int(fea.get('fe_pad_days', 30))
    CORE_DAYS = int(fea.get('fe_core_days', 30))

    total_fe_batches = max(0, ((len(days) - PAD_DAYS) + CORE_DAYS - 1) // CORE_DAYS)
    for start in tqdm(range(PAD_DAYS, len(days), CORE_DAYS),
                      total=total_fe_batches, desc="FE shards (A/B/C)"):
        core_lo_idx = start
        core_hi_idx = min(start + CORE_DAYS - 1, len(days) - 1)  # 闭区间
        pad_lo_idx  = core_lo_idx - PAD_DAYS

        core_lo, core_hi = days[core_lo_idx], days[core_hi_idx]
        pad_lo = days[pad_lo_idx]

        # 仅读本片+pad 的输入（懒加载）
        lf_shard = (
            lc.filter(pl.col(g_date).is_between(pad_lo, core_hi))
              .select([*cfg['keys'], cfg['weight'], TB, *RESP_COLS, *FEATURE_ALL])
        )

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
            write_date_between=(core_lo, core_hi),
        )
        print(f"[FE] days {core_lo}..{core_hi} (pad from {pad_lo}) -> {out_dir}")

    # ==============================
    # 4) 拼接 A/B/C → Panel 分片
    # ==============================
    DATE_LO, DATE_HI = cfg['dates']['train_lo'], cfg['dates']['train_hi']
    print(f"DATE_LO: {DATE_LO}, DATE_HI: {DATE_HI}")

    # 解析 fe 分片窗口
    shard_dirs = [azify(p) for p in sorted(fs.glob(f"{fe_root}/*"))]
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

    # 基表（clean）作为 join 基础，统一 cast key
    cast_keys = [pl.col(k).cast(pl.Int32).alias(k) for k in KEYS]
    lc_base = (
        pl.scan_parquet(f"{clean_root}/*.parquet", storage_options=storage_options)
          .with_columns(cast_keys)
    )
    ti_f = pl.col(g_time).cast(pl.Float32)

    for (lo, hi) in tqdm(wins, desc="Stitch panel shards"):
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
                       pl.col("_phase_").sin().cast(pl.Float32).alias("time_sin"),
                       pl.col("_phase_").cos().cast(pl.Float32).alias("time_cos"),
                   ])
                   .drop(["_phase_"])
        )

        # A/B
        A_scan = pl.scan_parquet(f"{fe_dir}/stage_a.parquet", storage_options=storage_options).with_columns(cast_keys)
        B_scan = pl.scan_parquet(f"{fe_dir}/stage_b.parquet", storage_options=storage_options).with_columns(cast_keys)

        # C（多文件）
        C_paths = [azify(p) for p in sorted(fs.glob(f"{fe_dir}/stage_c_*.parquet"))]
        C_scans = [pl.scan_parquet(p, storage_options=storage_options).with_columns(cast_keys) for p in C_paths]

        # 逐个 join
        panel = lf.join(A_scan, on=list(KEYS), how="left", suffix="_A")
        panel = panel.join(B_scan, on=list(KEYS), how="left", suffix="_B")
        for C in tqdm(C_scans, desc=f"Join C {shard_name}", leave=False):
            panel = panel.join(C, on=list(KEYS), how="left", suffix="_C")

        panel = panel.sort(TIME_SORT)

        # 收集并写盘（Azure）
        df_out = panel.collect(streaming=True)
        out_path = f"{panel_root}/panel_{w_lo:04d}_{w_hi:04d}.parquet"
        with fs.open(out_path, "wb") as f:
            df_out.write_parquet(f, compression="zstd")
        print(f"[panel] wrote {out_path} with {df_out.shape[0]} rows")

        # 片内时间单调性检查（可开关）
        if cfg.get("debug", {}).get("check_time_monotone", True):
            assert_panel_shard(out_path, w_lo, w_hi, date_col=g_date, time_col=g_time)

        del df_out
        gc.collect()


if __name__ == "__main__":
    main()
