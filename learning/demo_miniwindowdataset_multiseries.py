from __future__ import annotations
from functools import cached_property
from typing import Sequence, Dict, Any, List, Tuple, Optional
import random

def _per_item_rng(seed: Optional[int], group: str, start: int) -> random.Random | None:
    if seed is None:
        return None
    h = hash((seed, group, start)) & 0xFFFFFFFF
    return random.Random(h)
            

class MultiWindowDataset_Step2:
    def __init__(
        self,
        series_map: Dict[str, Sequence[float | int]],
        enc_len: int,
        dec_len: int,
        min_enc_len: Optional[int] = None,
        max_dec_len: Optional[int] = None,
        randomize_length: Optional[Tuple[float, float]] = None,
        random_seed: Optional[int] = None,
        static_map: Optional[Dict[str, Dict[str, Any]]] = None,
        past_cov_map: Optional[Dict[str, Sequence[Sequence[float | int]]]] = None,
        future_cov_map: Optional[Dict[str, Sequence[Sequence[float | int]]]] = None,
    ):
        # ---- 基本参数（复制你单序列版的校验风格即可）----
        self.enc_len = int(enc_len)
        self.dec_len = int(dec_len)
        self.min_enc_len = int(min_enc_len) if min_enc_len is not None else self.enc_len
        self.max_dec_len = int(max_dec_len) if max_dec_len is not None else self.dec_len
        self.randomize_length = randomize_length
        self.random_seed = random_seed
        self.static_map = static_map or {}
        self.past_cov_map = past_cov_map or {}
        self.future_cov_map = future_cov_map or {}

        # TODO: 参数校验（与单序列版一致：enc/dec 正数，min_enc_len 范围，dec_len<=max_dec_len）
        if self.enc_len <= 0 or self.dec_len <= 0:
            raise ValueError("enc_len/dec_len must be positive.")
        if self.min_enc_len <= 0 or self.min_enc_len > self.enc_len:
            raise ValueError("min_enc_len must be > 0 and <= enc_len.")
        if self.dec_len > self.max_dec_len:
            raise ValueError("dec_len must be <= max_dec_len.")
        
        # ---- 多序列容器 ----
        # TODO: 保存 group 名称列表与序列列表（转成 list）
        # self.groups = ...
        # self.series = ...
        self.groups = list(series_map.keys())
        self.series = [list(series_map[g]) for g in self.groups]


    @cached_property
    def starts_per_group(self) -> List[List[int]]:
        # TODO: 对每条序列算合法起点 (enc_len + max_dec_len) 窗口   
        res: List[List[int]] = []
        for seq in self.series:
            L = len(seq)
            W = self.enc_len + self.max_dec_len
            n = max(L - W + 1, 0)
            res.append(list(range(n)))
        return res

    @cached_property
    def index(self) -> List[Tuple[int, int]]:
        # TODO: 展开为全局索引 [(group_idx, start), ...]
        out: List[Tuple[int, int]] = []
        for gi, starts in enumerate(self.starts_per_group):
            for s in starts:
                out.append((gi, s))
        return out

    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        group_idx, start = self.index[idx]
        group = self.groups[group_idx]
        seq = self.series[group_idx]
        
        enc_len = self.enc_len
        dec_len = self.dec_len
        
        if self.randomize_length is not None:
            alpha, beta = self.randomize_length
            rng = _per_item_rng(self.random_seed, group, start)
            R = rng or random
            p = R.betavariate(alpha, beta)
            
            enc_range = enc_len - self.min_enc_len
            new_enc = self.min_enc_len + round(p * enc_range)
            
            surplus = enc_len - new_enc
            new_dec = min(self.max_dec_len, dec_len + surplus)

            enc_len = new_enc
            dec_len = new_dec

        enc = seq[start : start + enc_len]
        dec = seq[start + enc_len : start + enc_len + dec_len]
        
        past_enc = None
        if group in self.past_cov_map:
            past_full = self.past_cov_map[group]
            past_enc = past_full[start : start + enc_len]
        fut_dec = None
        if group in self.future_cov_map:
            fut_full = self.future_cov_map[group]
            fut_dec = fut_full[start + enc_len : start + enc_len + dec_len]
        
        if enc:
            m = sum(enc) / len(enc)
            v = sum((x - m) ** 2 for x in enc) / len(enc)
            std = ( v ** 0.5 )
            eps = 1e-6
            target_scale = (m, std + eps)
        else:
            target_scale = (0.0, 1.0)
            
        return {
            "group": self.groups[group_idx],
            "group_idx": group_idx,
            "start": start,
            "encoder": enc,
            "decoder": dec,
            "enc_len": len(enc),
            "dec_len": len(dec),
            "target_scale": target_scale,
            "static": self.static_map.get(group, None),
            "past_cov_enc": past_enc,
            "future_cov_dec": fut_dec,
        }

def to_loader_multi(
    ds,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: Optional[int] = None,
):
    idxs = list(range(len(ds)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        chunk = idxs[i:i+batch_size]
        if drop_last and len(chunk) < batch_size:
            continue
        batch_items = [ds[j] for j in chunk]
        yield multi_collate_pad_plus(batch_items)

from typing import List, Dict, Any

def _pad_2d_rows(rows: List[List[float]], pad_rows: int) -> List[List[float]]:
    feat_dim = len(rows[0]) if rows else 0
    out = [r[:] for r in rows]
    for _ in range(pad_rows - len(out)):
        out.append([0.0]*feat_dim)
    return out

def multi_collate_pad_plus(
    batch: List[Dict[str, Any]],
    pad_value: float | int = 0,
) -> Dict[str, Any]:
    if not batch:
        return {}
    
    has_past = any(b.get("past_cov_enc") is not None for b in batch)
    has_fut = any(b.get("future_cov_dec") is not None for b in batch)
    
    starts      = [b["start"] for b in batch]
    groups      = [b["group"] for b in batch]
    group_idxs  = [b["group_idx"] for b in batch]
    enc_lens    = [b["enc_len"] for b in batch]
    dec_lens    = [b["dec_len"] for b in batch]
    scales      = [b["target_scale"] for b in batch]
    statics     = [b.get("static") for b in batch]  # 现在可能全是 None

    max_enc_len = max(enc_lens) if enc_lens else 0
    max_dec_len = max(dec_lens) if dec_lens else 0

    # padding
    enc_padded = [b["encoder"] + [pad_value]*(max_enc_len - b["enc_len"]) for b in batch]
    dec_padded = [b["decoder"] + [pad_value]*(max_dec_len - b["dec_len"]) for b in batch]

    # masks
    encoder_mask = [[1]*l + [0]*(max_enc_len - l) for l in enc_lens]
    decoder_mask = [[1]*l + [0]*(max_dec_len - l) for l in dec_lens]

    # time idx（对齐到统一长度）
    encoder_time_idx = [[s + j for j in range(max_enc_len)] for s in starts]
    decoder_time_idx = [[s + el + j for j in range(max_dec_len)] for s, el in zip(starts, enc_lens)]

    result = {
        "groups": groups,
        "group_idxs": group_idxs,
        "starts": starts,
        "enc_padded": enc_padded,
        "dec_padded": dec_padded,
        "enc_lens": enc_lens,
        "dec_lens": dec_lens,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "encoder_time_idx": encoder_time_idx,
        "decoder_time_idx": decoder_time_idx,
        "target_scale": scales,
        "statics": statics,
    }

    if has_past:
        result["past_cov_enc_padded"] = [
            _pad_2d_rows(b.get("past_cov_enc") or [], max_enc_len) for b in batch
        ]  # B × max_enc_len × Fp

    if has_fut:
        result["future_cov_dec_padded"] = [
            _pad_2d_rows(b.get("future_cov_dec") or [], max_dec_len) for b in batch
        ]
    return result

def to_loader_multi(
    ds,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: Optional[int] = None,
):
    idxs = list(range(len(ds)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        chunk = idxs[i:i+batch_size]
        if drop_last and len(chunk) < batch_size:
            continue
        batch_items = [ds[j] for j in chunk]
        yield multi_collate_pad_plus(batch_items)


def main():
    # 造两种协变量：Fp=2（过去），Ff=1（未来）
    def make_rows(L, F, start=0):
        return [[(t+start)*10 + f for f in range(F)] for t in range(L)]

    series_map = {
        "A": list(range(1, 13)),     # L=12
        "B": [10,9,8,7,6,5,4,3,2],   # L=9
    }
    past_cov_map = {
        "A": make_rows(12, 2),
        "B": make_rows(9,  2, start=100),
    }
    future_cov_map = {
        "A": make_rows(12, 1, start=200),
        "B": make_rows(9,  1, start=300),
    }

    ds2 = MultiWindowDataset_Step2(
        series_map,
        enc_len=5, dec_len=2,
        min_enc_len=3, max_dec_len=4,
        randomize_length=(0.7, 0.7),
        random_seed=42,
        static_map=None,
        past_cov_map=past_cov_map,
        future_cov_map=future_cov_map,
    )

    x0 = ds2[0]
    print("x0 enc_len/dec_len:", x0["enc_len"], x0["dec_len"])
    print("x0 target_scale:", x0["target_scale"])

# ------------------ Demo ------------------
if __name__ == "__main__":
    main()
