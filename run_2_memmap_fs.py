# run_memmap_fs.py
from __future__ import annotations

import os
import re
import polars as pl

from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local
from pipeline.memmap import shard2memmap


# -----------------------------
# helpers
# -----------------------------
def azify(p: str) -> str:
    """Ensure Azure path has 'az://' exactly once."""
    return p if p.startswith("az://") else f"az://{p}"


def main():
    # =============================
    # 0) 读取配置 & I/O 根路径
    # =============================
    DATE_LO, DATE_HI = cfg["dates"]["feature_select_dates"].get("date_lo", 1300), cfg["dates"]["feature_select_dates"].get("date_hi", 1500)

    mm_root = P("local", cfg["paths"]["sample_mm"])         # 本地 memmap 根目录
    ensure_dir_local(mm_root)

    panel_root = azify(P("az", cfg["paths"]["panel_shards"]))  # 远端 panel 分片目录

    # =============================
    # 1) 发现需要的 panel 分片
    # =============================
    # 只挑选与 [DATE_LO, DATE_HI] 区间相交的 panel_XXXX_YYYY.parquet
    panel_all = [azify(p) for p in fs.glob(f"{panel_root}/panel_*.parquet")]
    panel_paths: list[str] = []
    for p in sorted(panel_all):
        m = re.search(r"panel_(\d+)_(\d+)\.parquet$", p)
        if not m:
            continue
        lo, hi = map(int, m.groups())
        if hi >= DATE_LO and lo <= DATE_HI:
            panel_paths.append(p)

    if not panel_paths:
        raise FileNotFoundError(
            f"No panel shards found in [{DATE_LO}, {DATE_HI}] under {panel_root}"
        )

    # =============================
    # 2) 抽样读取列名（作为特征全集模板）
    # =============================
    sample_path = panel_paths[0]
    names = (
        pl.scan_parquet(sample_path, storage_options=storage_options)
          .collect_schema()
          .names()
    )

    keys   = tuple(cfg["keys"])
    target = cfg["target"]
    weight = cfg["weight"]

    # 排除 keys/target/weight，剩下即为特征列（base+engineered）
    feat_cols = [c for c in names if c not in (*keys, target, weight)]
    if not feat_cols:
        raise RuntimeError("No feature columns detected for memmap.")

    # =============================
    # 3) 写 memmap
    # =============================
    prefix = os.path.join(mm_root, "full_sample_v1")  # 输出前缀（文件名基）
    mm_paths = shard2memmap(
        sorted_paths=panel_paths,
        feat_cols=feat_cols,
        prefix=prefix,
        date_col=keys[1],   # 通常是 'date_id'
        target_col=target,  # 如 'responder_6'
        weight_col=weight,  # 如 'weight'
    )

    # 小结
    print("[memmap] shards:", len(panel_paths))
    print("[memmap] features:", len(feat_cols))
    print("[memmap] prefix:", prefix)
    print("[memmap] files:", mm_paths)


if __name__ == "__main__":
    main()
