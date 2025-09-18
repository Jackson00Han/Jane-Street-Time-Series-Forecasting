# run_feature_select.py
from __future__ import annotations
import os, json, time, gc, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm  # ✅ 进度条

from pipeline.io import cfg, P, ensure_dir_local
from pipeline.memmap import make_sliding_cv_fast

# ---------- helpers for logging ----------
def _fmt_s(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:02d}m{s:02d}s" if m else f"{s:02d}s")

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# -------- utils --------
def load_mm(prefix: str):
    meta_path = f"{prefix}.meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    N, F = meta["n_rows"], meta["n_feat"]
    X = np.memmap(f"{prefix}_X.float32.mmap", dtype=np.float32, mode="r", shape=(N, F))
    y = np.memmap(f"{prefix}_y.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    w = np.memmap(f"{prefix}_w.float32.mmap", dtype=np.float32, mode="r", shape=(N,))
    d = np.memmap(f"{prefix}_date.int32.mmap", dtype=np.int32,   mode="r", shape=(N,))
    return meta, X, y, w, d

def lgb_wr2_eval(y_pred: np.ndarray, dataset: lgb.Dataset):
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true, dtype=np.float64)
    # 加权 R²
    w = w.astype(np.float64, copy=False)
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    w_sum = w.sum()
    y_bar = (w * y_true).sum() / max(w_sum, 1e-12)
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - y_bar) ** 2).sum()
    wr2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    return "wr2", float(wr2), True   # higher is better

def tag_for_fs(fs_lo:int, fs_hi:int, n_splits:int, gap:int, ratio:int, seed:int, top_k:int, ts:int):
    return f"fs__{fs_lo}-{fs_hi}__cv{n_splits}-g{gap}-r{ratio}__seed{seed}__top{top_k}__{ts}"

def main():
    t0 = time.time()
    print(f"[{_now()}][fs] ===== Feature Selection started =====")

    # =========================
    # 0) 路径
    # =========================
    mm_root   = P("local", cfg["paths"]["fs_mm"])
    prefix    = os.path.join(mm_root, "full_sample_v1")     # 与 run_memmap.py 保持一致
    rep_dir   = os.path.join(P("local", cfg["paths"]["reports"]), "fi")
    featset_dir = os.path.join(P("local", cfg.get("paths", {}).get("models", "exp/v1/models")), "feature_set")
    ensure_dir_local(rep_dir); ensure_dir_local(featset_dir)
    print(f"[{_now()}][fs] IO ready. memmap_prefix={prefix}")
    print(f"[{_now()}][fs] reports -> {rep_dir}")
    print(f"[{_now()}][fs] feature_set -> {featset_dir}")

    # =========================
    # 1) 载入 memmap
    # =========================
    print(f"[{_now()}][fs] Loading memmap...")
    meta, X, y, w, d = load_mm(prefix)
    feat_cols = meta["features"]
    assert np.all(np.diff(d) >= 0), "memmap d 不是非降序；请检查 panel 分片或排序"
    print(f"[{_now()}][fs] Loaded. N={meta['n_rows']:,}, F={meta['n_feat']}, d_range=[{int(d.min())},{int(d.max())}]")
    print(w)
    
if __name__ == "__main__":
    main()
