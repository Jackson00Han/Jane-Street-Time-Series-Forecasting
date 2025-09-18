from __future__ import annotations
import os, re, json, time, math, signal
import polars as pl
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local
from pipeline.memmap import shard2memmap

# ---------- helpers ----------
def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def to_int(x, name):
    try:
        return int(str(x).strip())
    except Exception as e:
        raise ValueError(f"{name} 不是整数，可解析的值：{x!r}") from e

def shard_key(path: str):
    m = re.search(r"panel_(\d+)_(\d+)\.parquet$", path)
    if not m:
        return (10**12, 10**12, path)
    lo, hi = map(int, m.groups())
    return (lo, hi, path)

# ---------- main ----------
def main():
    # 0) 配置 & 路径
    date_cfg = cfg["dates"]["feature_select_dates"]
    DATE_LO = to_int(date_cfg.get("date_lo", 1100), "date_lo")
    DATE_HI = to_int(date_cfg.get("date_hi", 1200), "date_hi")
    if DATE_LO > DATE_HI:
        raise ValueError(f"date_lo({DATE_LO}) > date_hi({DATE_HI})，请检查配置")

    mm_root = P("local", cfg["paths"]["fs_mm"])
    ensure_dir_local(mm_root)
    prefix = os.path.join(mm_root, "full_sample_v1")

    panel_root = azify(P("az", cfg["paths"]["panel_shards"]))
    print(f"[memmap] panel_root = {panel_root}")
    print(f"[memmap] target date range (inclusive) = [{DATE_LO}, {DATE_HI}]")

    # 1) 列出需要的分片（与区间相交）
    all_panels = sorted([azify(p) for p in fs.glob(f"{panel_root}/panel_*.parquet")], key=shard_key)
    panel_paths: list[str] = []
    skipped = 0
    for p in all_panels:
        m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
        if not m:
            skipped += 1
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:
            panel_paths.append(p)

    if not panel_paths:
        raise FileNotFoundError(
            f"No panel shards overlapping [{DATE_LO}, {DATE_HI}] under {panel_root} "
            f"(checked {len(all_panels)} files, skipped {skipped} non-matching names)"
        )

    print(f"[memmap] shards selected = {len(panel_paths)} / {len(all_panels)}")
    print(f"[memmap] first shard = {os.path.basename(panel_paths[0])}, last shard = {os.path.basename(panel_paths[-1])}")

    # 2) 抽样读取 schema（确保列集一致）
    sample_path = panel_paths[0]
    schema = pl.scan_parquet(sample_path, storage_options=storage_options).collect_schema()
    names = schema.names()
    keys   = tuple(cfg["keys"])      # e.g. ('symbol_id','date_id','time_id')
    target = cfg["target"]           # e.g. 'responder_6'
    weight = cfg["weight"]           # e.g. 'weight'

    # 校验关键列存在
    missing_core = [c for c in [*keys, target, weight] if c not in names]
    if missing_core:
        raise RuntimeError(f"核心列缺失: {missing_core} in {sample_path}")

    # 特征列全集（保持原顺序，排除 keys/target/weight）
    feat_cols = [c for c in names if c not in (*keys, target, weight)]
    if not feat_cols:
        raise RuntimeError("未检测到任何特征列（除去 keys/target/weight 之后为空）")

    print(f"[memmap] features detected = {len(feat_cols)} (excluding keys/target/weight)")
    print(f"[memmap] keys={keys}, target={target}, weight={weight}")

    # 3) 统计每个分片的行数 & 日期范围（可视化进度）
    shard_stats = []
    t0 = time.time()
    print("[memmap] scanning shard stats...")
    for p in tqdm(panel_paths, desc="scan shards", unit="shard"):
        # 仅取必要列加速
        lb = pl.scan_parquet(p, storage_options=storage_options).select([keys[1]])  # date_id
        cnt = lb.select(pl.len()).collect(streaming=True).item()
        # 取 min/max date（两次聚合合并成一次）
        s = lb.select(pl.min(keys[1]).alias("dmin"), pl.max(keys[1]).alias("dmax")).collect(streaming=True)
        dmin, dmax = int(s["dmin"][0]), int(s["dmax"][0])
        shard_stats.append((p, cnt, dmin, dmax))
    print(f"[memmap] shard stats done in {time.time()-t0:.1f}s")
    total_rows = int(sum(c for _, c, _, _ in shard_stats))
    d_min = min(d0 for _, _, d0, _ in shard_stats)
    d_max = max(d1 for _, _, _, d1 in shard_stats)
    print(f"[memmap] total rows (raw, all selected shards) = {total_rows:,}")
    print(f"[memmap] available date range from selected shards = [{d_min}, {d_max}]")

    # 4) 写 memmap（带“优雅终止”）
    stop = {"flag": False}
    def _stop(sig, frame):
        stop["flag"] = True
        print("\n[memmap] Caught signal; will stop after current shard and finalize partial outputs.")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[memmap] writing memmap prefix = {prefix}")
    mm_paths = shard2memmap(
        sorted_paths=[p for p, *_ in shard_stats],
        feat_cols=feat_cols,
        prefix=prefix,
        date_col=keys[1],     # 'date_id'
        target_col=target,
        weight_col=weight,
        # 如果你的 shard2memmap 支持以下可选参数，可解开以进一步可视化：
        # progress=True, on_shard_done=callback, stop_flag=stop,
    )
    print("[memmap] files written:", mm_paths)

    # 5) 写 meta.json（更丰富，便于后续自检）
    meta = {
        "n_rows_raw": total_rows,
        "n_feat": len(feat_cols),
        "features": feat_cols,
        "keys": list(keys),
        "target": target,
        "weight": weight,
        "date_lo_requested": DATE_LO,
        "date_hi_requested": DATE_HI,
        "date_lo_available": d_min,
        "date_hi_available": d_max,
        "panel_root": panel_root,
        "shards": [{"path": p, "rows": int(c), "dmin": int(a), "dmax": int(b)} for p, c, a, b in shard_stats],
        "created_at": int(time.time()),
    }
    meta_path = f"{prefix}.build_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[memmap] build meta -> {meta_path}")

    # 6) 事后快速自检：打开你 downstream 的 meta 看“真值”
    try:
        with open(f"{prefix}.meta.json", "r", encoding="utf-8") as f:
            real = json.load(f)
        print(f"[memmap][check] n_rows (final) = {real.get('n_rows'):,}, n_feat = {real.get('n_feat')}")
        print(f"[memmap][check] d_min = {real.get('date_min')}, d_max = {real.get('date_max')}" if "date_min" in real else "[memmap][check] (date_min/date_max not in .meta.json)")
    except Exception as e:
        print(f"[memmap][warn] cannot read final meta ({e}); ensure shard2memmap writes .meta.json")

if __name__ == "__main__":
    main()
