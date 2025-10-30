# run_memmap_after_fs.py
from __future__ import annotations
import os, re, time, json
import polars as pl

from pipeline.io import cfg, fs, storage_options, ensure_dir_local
from pipeline.memmap import shard2memmap

def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def _tag_from_features_path(p: str) -> str:
    # e.g., ".../features__fs__1400-1698__cv2-...txt" -> "features__fs__1400-1698__cv2-..."
    return os.path.splitext(os.path.basename(p))[0]


def main():
    # ============ 0) Config & paths ============
    DATE_LO = int(cfg["dates"]["mtrain"]["date_lo"])
    DATE_HI = int(cfg["dates"]["mtrain"]["date_hi"])

    local_root = cfg["local"]["root"]
    azure_root = cfg["azure"]["root"]

    # Local output root for train memmap
    mm_root = f"{local_root}/{cfg['paths']['train_mm']['rel']}"
    ensure_dir_local(mm_root)

    # Azure panel root (must be az:// for remote scan/glob)
    panel_root = azify(f"{azure_root}/{cfg['paths']['panel_shards']['rel']}")

    # Feature list produced by feature selection
    feat_list_path = cfg["derived_paths"]["feature_list_path"]
    tag = _tag_from_features_path(feat_list_path)

    print(f"[cfg] full train range: [{DATE_LO}, {DATE_HI}]")
    print(f"[cfg] panel_root: {panel_root}")
    print(f"[cfg] feature list: {feat_list_path} (tag={tag})")
    print(f"[io ] train_mm root: {mm_root}")

    # ============ 1) Match panel shards within the date range ============
    panel_all = [azify(p) for p in fs.glob(f"{panel_root}/panel_*.parquet")]
    panel_paths: list[str] = []
    for p in sorted(panel_all):
        m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
        if not m:
            continue
        lo, hi = map(int, m.groups())
        # interval overlap
        if hi >= DATE_LO and lo <= DATE_HI:
            panel_paths.append(p)

    if not panel_paths:
        raise FileNotFoundError(f"No panel shards in [{DATE_LO},{DATE_HI}] under {panel_root}")

    print(f"[scan] matched {len(panel_paths)} shards")
    for s in panel_paths[:3]:
        print(f"       - {s}")
    if len(panel_paths) > 3:
        print(f"       ... (+{len(panel_paths)-3} more)")
        

    # ============ 2) Read schema & build feature columns ============
    names = pl.scan_parquet(panel_paths[0], storage_options=storage_options).collect_schema().names()
    schema_set = set(names)

    # Core columns (from config.columns)
    keys   = tuple(cfg["columns"]["keys"])
    target = cfg["columns"]["target"]
    weight = cfg["columns"]["weight"]

    # Selected features + allowlist (config.columns.allowlist)
    with open(feat_list_path, "r", encoding="utf-8") as f:
        selected_feats = [ln.strip() for ln in f if ln.strip()]
    allowlist = list(cfg["columns"].get("allowlist", []))

    # Preserve order: allowlist first, then selected; de-duplicate while keeping order.
    # Intersect with schema and exclude keys/target/weight.
    want = list(dict.fromkeys(allowlist + selected_feats))
    feat_cols = [c for c in want if (c in schema_set) and (c not in (*keys, target, weight))]

    missing = [c for c in want if c not in schema_set]
    if missing:
        print(f"[warn] {len(missing)} features not found in schema; ignored. e.g. {missing[:5]}")

    if not feat_cols:
        raise RuntimeError("No valid selected features found in panel schema.")

    print(f"[schema] keys={keys}, target={target}, weight={weight}")
    print(f"[schema] selected features: {len(feat_cols)} (e.g. {feat_cols[:5]})")
    

    # ============ 2.5) Count rows & estimate disk footprint ============
    counts = []
    t_count = time.time()
    for p in panel_paths:
        k = pl.scan_parquet(p, storage_options=storage_options).select(pl.len()).collect(streaming=True).item()
        counts.append(int(k))
    n_rows = int(sum(counts))
    n_feat = len(feat_cols)
    est_gb = (n_rows * (n_feat + 3) * 4) / 1e9  # X + y + w + date (float32/int32)
    print(f"[plan] rows={n_rows:,}, feats={n_feat} -> estimated disk ~{est_gb:.1f} GB "
          f"(counted in {time.time()-t_count:.1f}s)")

    # ============ 3) Write memmap ============
    # Prefer a config-provided prefix when present to keep runs consistent.
    prefix =os.path.join(mm_root, f"full_train__features__{tag}__range{DATE_LO}-{DATE_HI}"); ensure_dir_local(prefix)

    print(f"[write] start -> prefix: {prefix}")
    t0 = time.time()

    mm_paths = shard2memmap(
        sorted_paths=panel_paths,
        feat_cols=feat_cols,
        prefix=prefix,
        date_col=keys[1],   # e.g., 'date_id'
        target_col=target,  # e.g., 'responder_6'
        weight_col=weight,  # e.g., 'weight'
    )

    print(f"[done] memmap written in {(time.time()-t0)/60:.1f} min")
    print(f"[done] shards:   {len(panel_paths)}")
    print(f"[done] features: {len(feat_cols)}")
    print(f"[done] files: {json.dumps(mm_paths, indent=2)}")

if __name__ == "__main__":
    main()
