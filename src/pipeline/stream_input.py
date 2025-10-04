from typing import List, Dict
from collections import deque
import random
import polars as pl
from torch.utils.data import IterableDataset, get_worker_info
from pytorch_forecasting import TimeSeriesDataSet
from pipeline.io import storage_options

# ========= IterableDataset：按 chunk 一次性读 + 一次性构建 =========
class ShardedBatchStream(IterableDataset):
    def __init__(
        self,
        template_tsd,
        chunk_dirs: list[str],
        chunk2paths: dict[str, list[str]],
        g_sym: str,
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
        self.g_sym = g_sym
        self.batch_size = batch_size
        self.buffer_batches = buffer_batches
        self.seed = seed
        self.cols = cols
        self.print_every_chunks = print_every_chunks

    def __iter__(self):
        rng = random.Random(self.seed)
        chunk_dirs = self.chunk_dirs[:]
        
        dir2gid = {d: i+1 for i, d in enumerate(sorted(chunk_dirs))}
        total_glb = len(dir2gid)
        
        #rng.shuffle(chunk_dirs)   # 若需严格时间顺序，可注释

        wi = get_worker_info()
        if wi is not None:
            chunk_dirs = chunk_dirs[wi.id::wi.num_workers]

        buf = deque()
        total = len(chunk_dirs)

        for idx, cdir in enumerate(chunk_dirs, 1):
            paths = self.chunk2paths.get(cdir, [])
            if not paths:
                continue
            lf = pl.scan_parquet(paths, storage_options=storage_options)
            if self.cols is not None:
                lf = lf.select(self.cols)
            pdf = lf.collect(streaming=True).to_pandas()
            if pdf.empty:
                raise ValueError(f"empty chunk: {cdir}")

            if (idx % self.print_every_chunks) == 0 and (wi is None or wi.id == 0):
                gid = dir2gid.get(cdir, idx)
                print(f"[loader] chunk {gid}/{total_glb}: {cdir} -> rows={len(pdf):,}")

            pdf[self.g_sym] = pdf[self.g_sym].astype(str)
            pdf.sort_values([self.g_sym, "time_idx"], inplace=True)

            tsds = TimeSeriesDataSet.from_dataset(self.template, data=pdf, stop_randomization=True)
            dl = tsds.to_dataloader(
                train=True,
                batch_size=self.batch_size,
                num_workers=0,     # 内层别并行
                shuffle=False,
                pin_memory=True,
            )

            for batch in dl:
                if self.buffer_batches > 0:
                    buf.append(batch)
                    if len(buf) >= self.buffer_batches:
                        k = rng.randrange(len(buf))
                        if k: buf.rotate(-k)
                        yield buf.popleft()
                else:
                    yield batch

        while buf:
            yield buf.popleft()


