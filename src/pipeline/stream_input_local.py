# --- 必需的标准库 ---
from __future__ import annotations
from collections import deque
import os
import random
from typing import List, Dict, Tuple, Optional, Iterable

# --- 第三方 ---
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as ft
from torch.utils.data import IterableDataset, get_worker_info
from pytorch_forecasting import TimeSeriesDataSet


class ShardedBatchStream(IterableDataset):
    """
    每个 chunk 只读一次 → 内存构建 TSDS → 内层 DataLoader(并行打包/切窗) → 产出 batch。
    外层 DataLoader 再用 persistent_workers + prefetch 做搬运。
    """
    def __init__(self, template_tsd, chunk_dirs, chunk2paths, train_period,
                 g_sym, g_date, batch_size=512, buffer_batches=8, seed=42,
                 cols=None, print_every_chunks=1, file_format="feather"):
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

        # 读表列：一定包含 date、time_idx、group，避免后续 KeyError
        must_cols = [self.g_date, "time_idx", self.g_sym]
        self.read_cols = sorted(set((self.cols or []) + must_cols))

        # 先按折粗筛目录
        if self.train_period is not None:
            self.chunk_dirs = [c for c in self.chunk_dirs if self._keep_chunk_for_train(c)]

        print(f"[ShardedBatchStream] 使用的chunk: {self.chunk_dirs}")


    @staticmethod
    def _parse_chunk_span(cdir: str):
        # .../chunk_1650_1669 -> (1650, 1669)
        name = cdir.rstrip("/").split("/")[-1]
        lo, hi = name.split("_")[-2:]
        return int(lo), int(hi)

    def _keep_chunk_for_train(self, cdir: str) -> bool:
        lo, hi = self._parse_chunk_span(cdir)
        lo_tr, hi_tr = self.train_period
        return not (hi < lo_tr or lo > hi_tr)

    def _read_chunk(self, paths: list[str]) -> pa.Table:
        if self.file_format == "feather":
            assert len(paths) == 1, "feather 每块单文件写入，paths 应为长度 1"
            return ft.read_table(paths[0], columns=self.read_cols, memory_map=True)
        return pq.read_table(paths, columns=self.read_cols, memory_map=True)

    def __iter__(self):
        from torch.utils.data import DataLoader

        rng = random.Random(self.seed)
        chunk_dirs = self.chunk_dirs[:]

        wi = get_worker_info()
        if wi is not None:
            # 多 worker：按 worker_id 切分块，避免重复
            chunk_dirs = chunk_dirs[wi.id::wi.num_workers]

        buf = deque(maxlen=max(1, self.buf_n))
        total = len(self.chunk_dirs)

        for i, cdir in enumerate(chunk_dirs, 1):
            paths = self.chunk2paths.get(cdir, [])
            if not paths:
                continue

            # 只读一次到内存（Arrow Table）
            table = self._read_chunk(paths)

            # 行级过滤（train 期间）
            if self.train_period is not None:
                lo_tr, hi_tr = self.train_period
                d = table[self.g_date].to_numpy()
                mask = (d >= lo_tr) & (d <= hi_tr)
                if not mask.any():
                    continue
                table = table.filter(pa.array(mask))

            # 转 pandas（一次），排序 & 类型
            pdf = table.to_pandas()
            if pdf.empty:
                continue
            pdf[self.g_sym] = pdf[self.g_sym].astype("str")
            pdf.sort_values([self.g_sym, "time_idx"], inplace=True)

            # 训练真正用的列（丢掉仅过滤所需列）
            if self.cols is not None:
                pdf = pdf[self.cols]

            if (self.print_every_chunks is not None 
                and self.print_every_chunks > 0 
                and (i % self.print_every_chunks) == 0
                and (wi is None or wi.id == 0)):
                print(f"[loader] chunk {i+1}/{n}: {path} -> rows={rows}")

            # 在内存里构建 TSDS，并一次性建内层 DataLoader（并行打包/切窗）
            tsds = TimeSeriesDataSet.from_dataset(self.template, data=pdf, stop_randomization=False)
            dl = tsds.to_dataloader(
                train=True,
                batch_size=self.bs,
                num_workers=0,
                persistent_workers=False,   # 每个 chunk 生命周期短
                prefetch_factor=None,
                shuffle=False,
                pin_memory=True,
            )

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
