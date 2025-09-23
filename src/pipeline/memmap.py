# pipeline/memmap.py
from __future__ import annotations
from typing import List, Dict
import os, json, time, gc
import numpy as np
import polars as pl
from pipeline.io import storage_options  # 用于远端读取

def shard2memmap(
    sorted_paths: List[str],
    feat_cols: List[str],
    prefix: str,
    *,
    date_col: str = "date_id",
    target_col: str = "responder_0",
    weight_col: str = "weight",
) -> Dict[str, str]:
    """将若干已排序的 panel 分片堆叠成本地 memmap。"""

    # 预扫行数
    counts = []
    for p in sorted_paths:
        k = pl.scan_parquet(p, storage_options=storage_options).select(pl.len()).collect(streaming=True).item()
        counts.append(int(k))

    n_rows, n_feat = int(sum(counts)), len(feat_cols)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    X = np.memmap(f"{prefix}_X.float32.mmap", dtype=np.float32, mode="w+", shape=(n_rows, n_feat))
    y = np.memmap(f"{prefix}_y.float32.mmap", dtype=np.float32, mode="w+", shape=(n_rows,))
    w = np.memmap(f"{prefix}_w.float32.mmap", dtype=np.float32, mode="w+", shape=(n_rows,))
    d = np.memmap(f"{prefix}_date.int32.mmap", dtype=np.int32,   mode="w+", shape=(n_rows,))

    need_cols = [date_col, target_col, weight_col, *feat_cols]

    ofs = 0
    for p, k in zip(sorted_paths, counts):
        df = pl.scan_parquet(p, storage_options=storage_options).select(need_cols).collect(streaming=True)
        

        # 转 numpy（零拷贝尽可能）
        X_block = df.select(feat_cols).to_numpy().astype(np.float32, copy=False)
        y_block = df.get_column(target_col).to_numpy().astype(np.float32, copy=False).ravel()
        w_block = df.get_column(weight_col).to_numpy().astype(np.float32, copy=False).ravel()
        d_block = df.get_column(date_col).to_numpy().astype(np.int32,   copy=False).ravel()

        X[ofs:ofs+k, :] = X_block
        y[ofs:ofs+k]    = y_block
        w[ofs:ofs+k]    = w_block
        d[ofs:ofs+k]    = d_block

        ofs += k
        del df, X_block, y_block, w_block, d_block
        gc.collect()

    X.flush(); y.flush(); w.flush(); d.flush()

    meta = {
        "n_rows": int(n_rows),
        "n_feat": int(n_feat),
        "dtype": {"X": "float32", "y": "float32", "w": "float32", "date_id": "int32"},
        "features": list(feat_cols),
        "target": target_col,
        "weight": weight_col,
        "date_col": date_col,
        "files": sorted_paths,
        "ts": time.time(),
    }
    with open(f"{prefix}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "X": f"{prefix}_X.float32.mmap",
        "y": f"{prefix}_y.float32.mmap",
        "w": f"{prefix}_w.float32.mmap",
        "date": f"{prefix}_date.int32.mmap",
        "meta": f"{prefix}.meta.json",
    }

def day_ptrs_from_sorted_dates(d: np.ndarray):
    d = d.ravel()
    starts = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1]
    days   = d[starts]
    ends   = np.r_[starts[1:], d.size]
    return days, starts, ends

def make_sliding_cv_fast(date_ids: np.ndarray, *, n_splits: int, gap_days: int = 5, train_to_val: int = 9):
    days, starts, ends = day_ptrs_from_sorted_dates(date_ids)
    N, R, K, G = len(days), int(train_to_val), int(n_splits), int(gap_days)
    usable = N - G
    if usable <= 0 or K <= 0 or R <= 0:
        return []
    V_base, rem = divmod(usable, R + K)
    if V_base <= 0:
        return []
    T = R * V_base
    v_lens = [V_base + 1 if i < rem else V_base for i in range(K)]
    v_lo = T + G
    folds = []
    for V_i in v_lens:
        v_hi  = v_lo + V_i
        tr_hi = v_lo - G
        tr_lo = tr_hi - T
        if tr_lo < 0 or v_hi > N:
            break
        tr_idx = np.arange(starts[tr_lo], ends[tr_hi-1])
        va_idx = np.arange(starts[v_lo],   ends[v_hi-1])
        folds.append((tr_idx, va_idx))
        v_lo = v_hi
    return folds
