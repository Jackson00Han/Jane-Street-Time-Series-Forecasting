from __future__ import annotations
import os, re, json, time, signal
import polars as pl
from tqdm.auto import tqdm

from pipeline.io import cfg, fs, storage_options, ensure_dir_local
from pipeline.memmap import shard2memmap

# ---------- helpers ----------
def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def shard_key(path: str):
    m = re.search(r"panel_(\d+)_(\d+)\.parquet$", path)
    if not m:
        return (10**12, 10**12, path)
    lo, hi = map(int, m.groups())
    return (lo, hi, path)

# ---------- main ----------
def main():
    # 0) Config & paths
    DATE_LO = cfg["dates"]["mfs"]["date_lo"]
    DATE_HI = cfg["dates"]["mfs"]["date_hi"]
    print(f"[memmap] target date range (inclusive) = [{DATE_LO}, {DATE_HI}]")

    local_root = cfg["local"]["root"]
    mm_root = f"{local_root}/{cfg['paths']['fs_mm']['rel']}"
    ensure_dir_local(mm_root)

    # panels live in Azure: azure.root + paths.panel_shards.rel
    panel_root = azify(f"{cfg['azure']['root']}/{cfg['paths']['panel_shards']['rel']}")

    # Prefix for output files (e.g., .../fs_mm/v1)
    prefix = os.path.join(mm_root, f"fs__{DATE_LO}-{DATE_HI}")
    
    

    # 1) List shards overlapping the target date range
    all_panels = sorted([azify(p) for p in fs.glob(f"{panel_root}/panel_*.parquet")], key=shard_key)
    panel_paths: list[str] = []
    skipped = 0
    for p in all_panels:
        m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
        if not m:
            skipped += 1
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:  # overlap check
            panel_paths.append(p)

    if not panel_paths:
        raise FileNotFoundError(
            f"No panel shards overlapping [{DATE_LO}, {DATE_HI}] under {panel_root} "
            f"(checked {len(all_panels)} files, skipped {skipped} non-matching names)"
        )

    print(f"[memmap] shards selected = {len(panel_paths)} / {len(all_panels)}")
    print(f"[memmap] first shard = {os.path.basename(panel_paths[0])}, last shard = {os.path.basename(panel_paths[-1])}")

    # 2) Sample schema to determine core/feature columns
    sample_path = panel_paths[0]
    print(f"sample_path: {sample_path}")
    schema = pl.scan_parquet(sample_path, storage_options=storage_options).collect_schema()
    names = schema.names()
    print(f"feat number: {len(names)}")
    
    # Read keys/target/weight from config.columns
    keys   = tuple(cfg["columns"]["keys"])      # e.g. ('symbol_id','date_id','time_id')
    target = cfg["columns"]["target"]          # e.g. 'responder_6'
    weight = cfg["columns"]["weight"]           # e.g. 'weight'

    # Validate presence of core columns
    missing_core = [c for c in [*keys, target, weight] if c not in names]
    if missing_core:
        raise RuntimeError(f"Missing core columns: {missing_core} in {sample_path}")

    # Feature columns = all minus keys/target/weight (preserve original order)
    feat_cols = [c for c in names if c not in (*keys, target, weight)]
    if not feat_cols:
        raise RuntimeError("No feature columns detected (empty after excluding keys/target/weight)")

    print(f"[memmap] features detected = {len(feat_cols)} (excluding keys/target/weight)")
    print(f"[memmap] keys={keys}, target={target}, weight={weight}")

    # 3) Scan shard stats (row count and date range) for logging/meta
    shard_stats = []
    print("[memmap] scanning shard stats...")
    for p in tqdm(panel_paths, desc="scan shards", unit="shard"):
        lb = pl.scan_parquet(p, storage_options=storage_options).select([keys[1]])  # date_id only
        cnt = lb.select(pl.len()).collect(streaming=True).item()
        s = lb.select(pl.min(keys[1]).alias("dmin"), pl.max(keys[1]).alias("dmax")).collect(streaming=True)
        dmin, dmax = int(s["dmin"][0]), int(s["dmax"][0])
        shard_stats.append((p, cnt, dmin, dmax))
    total_rows = int(sum(c for _, c, _, _ in shard_stats))
    d_min = min(d0 for _, _, d0, _ in shard_stats)
    d_max = max(d1 for _, _, _, d1 in shard_stats)
    print(f"[memmap] total rows across selected shards = {total_rows:,}")
    print(f"[memmap] available date range from selected shards = [{d_min}, {d_max}]")

    # 4) Write memmap
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
    )
    print("[memmap] files written:", mm_paths)

    # 5) Build-time meta (for quick self-checks)
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

    # 6) Post-check: read the memmap's own meta if available
    try:
        with open(f"{prefix}.meta.json", "r", encoding="utf-8") as f:
            real = json.load(f)
        print(f"[memmap][check] n_rows (final) = {real.get('n_rows'):,}, n_feat = {real.get('n_feat')}")
        print(f"[memmap][check] d_min = {real.get('date_min')}, d_max = {real.get('date_max')}" if "date_min" in real else "[memmap][check] (date_min/date_max not in .meta.json)")
    except Exception as e:
        print(f"[memmap][warn] cannot read final meta ({e}); ensure shard2memmap writes .meta.json")

if __name__ == "__main__":
    main()
