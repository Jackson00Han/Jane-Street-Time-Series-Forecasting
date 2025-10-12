# --- 必需的标准库 ---
from __future__ import annotations
from collections import deque
import random
from typing import List, Dict, Tuple, Optional

# --- 第三方 ---
import polars as pl
from torch.utils.data import IterableDataset, get_worker_info
from pytorch_forecasting import TimeSeriesDataSet

# 你工程里的工具（按你的项目路径）
from pipeline.io import storage_options



class ShardedBatchStream(IterableDataset):
    def __init__(
        self,
        template_tsd: TimeSeriesDataSet,
        chunk_dirs: list[str],
        chunk2paths: dict[str, list[str]],
        train_period: tuple[int, int] | None,
        g_sym: str,
        g_date: str,
        batch_size: int = 128,
        buffer_batches: int = 128,
        seed: int = 42,
        cols: list[str] | None = None,
        print_every_chunks: int = 1,
    ):
        super().__init__()
        self.template = template_tsd
        self.chunk_dirs = list(chunk_dirs)
        self.chunk2paths = chunk2paths
        self.train_period = train_period  # (lo_tr, hi_tr), 含端点
        self.g_sym = g_sym
        self.g_date = g_date
        self.batch_size = batch_size
        self.buffer_batches = buffer_batches
        self.seed = seed
        self.cols = cols
        self.print_every_chunks = print_every_chunks

        # 先按折过滤 chunk 目录（粗筛）
        if self.train_period is not None:
            self.chunk_dirs = [c for c in self.chunk_dirs if self.keep_chunk_for_train(c)]
        print(f"[ShardedBatchStream] 使用的chunk: {self.chunk_dirs}")
            
    @staticmethod
    def parse_chunk_span(cdir: str) -> tuple[int, int]:
        # az://.../chunk_1652_1653 -> (1652, 1653)
        name = cdir.rstrip("/").split("/")[-1]
        lo, hi = name.split("_")[-2:]
        return int(lo), int(hi)

    def keep_chunk_for_train(self, cdir: str) -> bool:
        lo, hi = self.parse_chunk_span(cdir)
        lo_tr, hi_tr = self.train_period  # 含端点
        # 只排除完全不相交： [lo,hi] 与 [lo_tr,hi_tr]
        return not (hi <= lo_tr or lo >= hi_tr)

    def __iter__(self):
        rng = random.Random(self.seed)
        chunk_dirs = self.chunk_dirs[:]

        wi = get_worker_info()
        if wi is not None:
            chunk_dirs = chunk_dirs[wi.id::wi.num_workers]

        buf = deque()
        total_glb = len(self.chunk_dirs)
        dir2gid = {d: i + 1 for i, d in enumerate(sorted(self.chunk_dirs))}

        for idx, cdir in enumerate(chunk_dirs, 1):
            paths = self.chunk2paths.get(cdir, [])
            if not paths:
                continue

            lf = pl.scan_parquet(paths, storage_options=storage_options)

            # 行级二次过滤（精筛，防 chunk 覆盖 train+val）
            if self.train_period is not None:
                lo_tr, hi_tr = self.train_period
                lf = lf.filter(pl.col(self.g_date).is_between(lo_tr, hi_tr, closed="both"))

            if self.cols is not None:
                lf = lf.select(self.cols)

            pdf = lf.collect(streaming=True).to_pandas()
            if pdf.empty:
                # 不抛异常，直接跳过（可能 chunk 与train仅边界重合，过滤后为空）
                continue

            if (idx % self.print_every_chunks) == 0 and (wi is None or wi.id == 0):
                gid = dir2gid.get(cdir, idx)
                print(f"[loader] chunk {gid}/{total_glb}: {cdir} -> rows={len(pdf):,}")

            pdf[self.g_sym] = pdf[self.g_sym].astype(str)
            pdf.sort_values([self.g_sym, "time_idx"], inplace=True)

            tsds = TimeSeriesDataSet.from_dataset(self.template, data=pdf, stop_randomization=True)
            dl = tsds.to_dataloader(
                train=True,
                batch_size=self.batch_size,
                num_workers=0,  # 内层别并行
                shuffle=False,
                pin_memory=True,
            )

            if self.buffer_batches > 0:
                for batch in dl:
                    buf.append(batch)
                    if len(buf) >= self.buffer_batches:
                        k = rng.randrange(len(buf))
                        if k:
                            buf.rotate(-k)
                        yield buf.popleft()
            else:
                for batch in dl:
                    yield batch

        while buf:
            yield buf.popleft()
