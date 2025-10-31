from __future__ import annotations
from functools import cached_property
from typing import Sequence, Dict, Any, List, Tuple, Optional
import random


class MiniWindowDataset:
    """
    一个极简的“滑动窗口”时间序列数据集（单序列版）。

    - data: 原始序列（list/tuple）
    - enc_len, dec_len: 编码器与解码器的默认长度
    - min_enc_len, max_dec_len: 允许的最小/最大长度（用于随机长度时的边界）
    - randomize_length: 若为 (alpha, beta)，则使用 Beta(alpha, beta) 随机缩短 encoder，
                        并把多出来的长度分配给 decoder（不超过 max_dec_len）
    - random_seed: 若设置，则随机长度是可复现的（只影响 __getitem__ 内部的随机）
    """

    def __init__(
        self,
        data: Sequence[float | int],
        enc_len: int,
        dec_len: int,
        min_enc_len: Optional[int] = None,
        max_dec_len: Optional[int] = None,
        randomize_length: Optional[Tuple[float, float]] = None,
        random_seed: Optional[int] = None,
    ):
        self.data = list(data)
        self.enc_len = int(enc_len)
        self.dec_len = int(dec_len)
        self.min_enc_len = int(min_enc_len) if min_enc_len is not None else self.enc_len
        self.max_dec_len = int(max_dec_len) if max_dec_len is not None else self.dec_len
        self.randomize_length = randomize_length
        self._rng = random.Random(random_seed) if random_seed is not None else None

        # 参数校验
        if self.enc_len <= 0 or self.dec_len <= 0:
            raise ValueError("enc_len/dec_len must be positive.")
        if self.min_enc_len <= 0 or self.min_enc_len > self.enc_len:
            raise ValueError("min_enc_len must be > 0 and <= enc_len.")
        if self.dec_len > self.max_dec_len:
            raise ValueError("dec_len must be <= max_dec_len.")

    @cached_property
    def starts(self) -> List[int]:
        """
        所有合法的起始下标（按“最大窗口” enc_len + max_dec_len 计算）。
        """
        n = len(self.data) - self.enc_len - self.max_dec_len + 1
        return list(range(max(n, 0)))

    @classmethod
    def with_defaults(cls, data: Sequence[float | int]) -> "MiniWindowDataset":
        """一个方便的构造器：默认 enc_len=3, dec_len=2。"""
        return cls(data, enc_len=3, dec_len=2)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        start = self.starts[idx]
        enc_len = self.enc_len
        dec_len = self.dec_len

        # 随机长度（可复现）
        if self.randomize_length is not None:
            alpha, beta = self.randomize_length
            p = (self._rng or random).betavariate(alpha, beta)  # 0~1

            # 按 p 在 [min_enc_len, enc_len] 内随机 encoder 长度
            enc_range = enc_len - self.min_enc_len
            new_enc = self.min_enc_len + round(p * enc_range)

            # 多出来的长度给 decoder（上限 max_dec_len）
            surplus = enc_len - new_enc
            new_dec = min(self.max_dec_len, dec_len + surplus)

            enc_len, dec_len = new_enc, new_dec

        # 实际切片
        enc_slice = slice(start, start + enc_len)
        dec_slice = slice(start + enc_len, start + enc_len + dec_len)
        encoder = self.data[enc_slice]
        decoder = self.data[dec_slice]

        # 简单的 target_scale：encoder 的平均绝对值
        target_scale = (sum(abs(x) for x in encoder) / len(encoder)) if encoder else 1.0

        return {
            "start": start,
            "encoder": encoder,
            "decoder": decoder,
            "target_scale": target_scale,
            "enc_len": enc_len,
            "dec_len": dec_len,
        }


def mini_collate_pad_plus(
    batch: List[Dict[str, Any]],
    pad_value: float | int = 0,
) -> Dict[str, Any]:
    """
    把若干个样本对齐（padding）成一个 batch，并返回 mask、decoder_time_idx 等。
    """
    starts = [item["start"] for item in batch]
    enc_lens = [item["enc_len"] for item in batch]
    dec_lens = [item["dec_len"] for item in batch]

    max_enc_len = max(enc_lens) if enc_lens else 0
    max_dec_len = max(dec_lens) if dec_lens else 0

    # padding 到各自最大长度
    enc_padded = [
        item["encoder"] + [pad_value] * (max_enc_len - item["enc_len"]) for item in batch
    ]
    dec_padded = [
        item["decoder"] + [pad_value] * (max_dec_len - item["dec_len"]) for item in batch
    ]

    # 0/1 mask
    enc_mask = [[1] * l + [0] * (max_enc_len - l) for l in enc_lens]
    dec_mask = [[1] * l + [0] * (max_dec_len - l) for l in dec_lens]

    # decoder 的时间索引（方便对齐/可视化）
    decoder_time_idx = []
    for s, el in zip(starts, enc_lens):
        decoder_time_idx.append([s + el + j for j in range(max_dec_len)])

    return {
        "starts": starts,
        "enc_padded": enc_padded,
        "dec_padded": dec_padded,
        "enc_lens": enc_lens,
        "dec_lens": dec_lens,
        "encoder_mask": enc_mask,
        "decoder_mask": dec_mask,
        "decoder_time_idx": decoder_time_idx,
    }


def to_loader(
    ds: MiniWindowDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: Optional[int] = None,
):
    """
    极简 DataLoader 生成器（单序列）。
    - shuffle 可用 seed 固定顺序（仅对抽样索引）
    - 每个样本在 __getitem__ 内部仍可以走“随机长度”（如果设置了 randomize_length）
    """
    idxs = list(range(len(ds)))
    if shuffle:
        rng = random.Random(seed)   # 只使用这一个 RNG，确保可复现
        rng.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        chunk = idxs[i:i + batch_size]
        if drop_last and len(chunk) < batch_size:
            continue
        batch = [ds[j] for j in chunk]
        yield mini_collate_pad_plus(batch)

def main():
    data = list(range(1, 11))  # 1..10
    ds = MiniWindowDataset(
        data,
        enc_len=5,
        dec_len=2,
        min_enc_len=3,
        max_dec_len=3,
        randomize_length=(0.5, 0.5),  # 开启随机长度
        random_seed=42,               # 让随机长度可复现
    )

    # 单个样本看看
    a = ds[0]
    b = ds[1]
    print("a:", a)
    print("b:", b)

    # 手动拼 batch
    collated = mini_collate_pad_plus([a, b])
    print("collated:", collated, "\n")

    # 用 loader 批量取（shuffle+seed 可复现抽样顺序）
    print("--- loader ---")
    for i, batch in enumerate(to_loader(ds, batch_size=2, shuffle=True, drop_last=False, seed=0)):
        print(f"batch {i}:")
        print("  enc_lens:", batch["enc_lens"])
        print("  dec_lens:", batch["dec_lens"])
        print("  enc_padded:", batch["enc_padded"])
        print("  dec_padded:", batch["dec_padded"])
        print("  decoder_time_idx:", batch["decoder_time_idx"])

# ------------------ Demo ------------------
if __name__ == "__main__":
    main()
