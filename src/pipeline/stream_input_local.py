# --- 必需的标准库 ---
from __future__ import annotations
from collections import deque
import random
from typing import List, Dict, Tuple, Optional, Iterable

# --- 第三方 ---
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as ft
from torch.utils.data import IterableDataset, get_worker_info
from pytorch_forecasting import TimeSeriesDataSet

class ShardedBatchStream(IterableDataset):
    def __init__(self, template_tsd, chunk_dirs, chunk2paths, train_period,
                 g_sym, g_date, batch_size=512, buffer_batches=8, seed=42,
                 cols=None, print_every_chunks=1, file_format="parquet"):
        super().__init__()
        self.template = template_tsd
        self.chunk_dirs  = list(chunk_dirs)
        self.chunk2paths = chunk2paths
        self.train_period = train_period
        self.g_sym  = g_sym
        self.g_date = g_date
        self.bs     = int(batch_size)
        self.buf_n  = int(buffer_batches)
        self.seed   = int(seed)
        self.cols   = list(cols) if cols is not None else None
        self.print_every_chunks = int(print_every_chunks)
        self.file_format = file_format.lower()

        # ✅ 读表列：一定包含 date、time_idx、group，避免后续 KeyError
        must_cols = [self.g_date, "time_idx", self.g_sym]
        self.read_cols = sorted(set((self.cols or []) + must_cols))

        # 先按折粗筛目录
        if self.train_period is not None:
            self.chunk_dirs = [c for c in self.chunk_dirs if self._keep_chunk_for_train(c)]

        print(f"[ShardedBatchStream] 使用的chunk: {self.chunk_dirs}")


        def __len__(self) -> int:
            # 仅用 date 列粗估，避免读全表；按 batch_size 折算步数
            try:
                total_rows = 0
                for cdir in self.chunk_dirs:
                    paths = self.chunk2paths.get(cdir, [])
                    if not paths:
                        continue
                    if self.file_format == "feather":
                        # feather 单文件
                        t = ft.read_table(paths[0], columns=[self.g_date], memory_map=True)
                    else:
                        t = pq.read_table(paths, columns=[self.g_date], memory_map=True)
                    d = t[self.g_date].to_numpy()
                    if self.train_period is not None:
                        lo, hi = self.train_period
                        total_rows += int(((d >= lo) & (d <= hi)).sum())
                    else:
                        total_rows += len(d)
                return max(1, total_rows // max(1, self.bs))
            except Exception:
                return 1
            
            
    @staticmethod
    def _parse_chunk_span(cdir: str):
        name = cdir.rstrip("/").split("/")[-1]
        lo, hi = name.split("_")[-2:]
        return int(lo), int(hi)

    def _keep_chunk_for_train(self, cdir: str) -> bool:
        lo, hi = self._parse_chunk_span(cdir)
        lo_tr, hi_tr = self.train_period
        return not (hi < lo_tr or lo > hi_tr)

    def _read_chunk(self, paths: list[str]) -> pa.Table:
        if self.file_format == "feather":
            assert len(paths) == 1
            return ft.read_table(paths[0], columns=self.read_cols, memory_map=True)
        return pq.read_table(paths, columns=self.read_cols, memory_map=True)

    def __iter__(self):
        import random
        from collections import deque
        from torch.utils.data import DataLoader

        rng = random.Random(self.seed)
        chunk_dirs = self.chunk_dirs[:]
        wi = get_worker_info()
        if wi is not None:
            chunk_dirs = chunk_dirs[wi.id::wi.num_workers]

        buf = deque(maxlen=max(1, self.buf_n))
        total = len(self.chunk_dirs)

        for i, cdir in enumerate(chunk_dirs, 1):
            paths = self.chunk2paths.get(cdir, [])
            if not paths:
                continue

            # 只读一次到内存
            table = self._read_chunk(paths)

            # 行级过滤（必须先有 g_date 列，已在 read_cols 中确保）
            if self.train_period is not None:
                lo_tr, hi_tr = self.train_period
                d = table[self.g_date].to_numpy()
                mask = (d >= lo_tr) & (d <= hi_tr)
                if not mask.any():
                    continue
                table = table.filter(pa.array(mask))

            # 转 pandas，并做排序 & 类型
            pdf = table.to_pandas()
            if pdf.empty:
                continue
            pdf[self.g_sym] = pdf[self.g_sym].astype("str")
            pdf.sort_values([self.g_sym, "time_idx"], inplace=True)

            # 可选：训练真正用的列（丢掉 g_date 等只为过滤而读的列）
            if self.cols is not None:
                pdf = pdf[self.cols]

            if (i % self.print_every_chunks) == 0 and (wi is None or wi.id == 0):
                print(f"[loader] chunk {i}/{total}: {cdir} -> rows={len(pdf):,}")

            # 在内存里构建 tsds，并一次性建 dataloader（⚠️ 不要在每个 batch 重建）
            tsds = TimeSeriesDataSet.from_dataset(self.template, data=pdf, stop_randomization=False)
            dl = tsds.to_dataloader(train=True, batch_size=self.bs, num_workers=0,
                                    shuffle=False, pin_memory=True)

            if self.buf_n <= 1:
                for b in dl:
                    yield b
            else:
                for b in dl:
                    buf.append(b)
                    if len(buf) >= buf.maxlen:
                        k = rng.randrange(len(buf))
                        if k:
                            buf.rotate(-k)
                        yield buf.popleft()

        while buf:
            yield buf.popleft()
