# run_memmap.py
from __future__ import annotations
import os, re
import polars as pl

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local
from pipeline.memmap import shard2memmap

def azify(p: str) -> str:
    return p if p.startswith("az://") else f"az://{p}"

def main():
    DATE_LO, DATE_HI = cfg["dates"]["train_lo"], cfg["dates"]["train_hi"]

    # 本地 memmap 根目录
    mm_root = P("local", cfg["paths"]["sample_mm"])
    ensure_dir_local(mm_root)

    # 远端 panel 分片目录
    panel_root = azify(P("az", cfg["paths"]["panel_shards"]))

    # 挑选落在区间内的 panel_XXXX_YYYY.parquet
    panel_all = [azify(p) for p in fs.glob(f"{panel_root}/panel_*.parquet")]
    panel_paths = []
    for p in sorted(panel_all):
        m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
        if not m:
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:
            panel_paths.append(p)

    if not panel_paths:
        raise FileNotFoundError(f"No panel shards found in [{DATE_LO}, {DATE_HI}] under {panel_root}")

    # 任选一个分片拿列名（包含 base+engineered）
    sample_path = panel_paths[0]
    names = pl.scan_parquet(sample_path, storage_options=storage_options).collect_schema().names()

    # 组装特征列（排除 keys/target/weight）
    keys = tuple(cfg["keys"])
    target = cfg["target"]
    weight = cfg["weight"]
    feat_cols = [c for c in names if c not in (*keys, target, weight)]
    if not feat_cols:
        raise RuntimeError("No feature columns detected for memmap.")

    # 输出前缀
    prefix = os.path.join(mm_root, "full_sample_v1")

    # 写 memmap
    mm_paths = shard2memmap(
        sorted_paths=panel_paths,
        feat_cols=feat_cols,
        prefix=prefix,
        date_col=keys[1],           # 通常是 'date_id'
        target_col=target,          # 如 'responder_0'
        weight_col=weight,          # 如 'weight'
    )
    print(mm_paths)

if __name__ == "__main__":
    main()
