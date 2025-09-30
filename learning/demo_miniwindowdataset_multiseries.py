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
    ):
        # ---- 基本参数（复制你单序列版的校验风格即可）----
        self.enc_len = int(enc_len)
        self.dec_len = int(dec_len)
        self.min_enc_len = int(min_enc_len) if min_enc_len is not None else self.enc_len
        self.max_dec_len = int(max_dec_len) if max_dec_len is not None else self.dec_len
        self.randomize_length = randomize_length
        self.random_seed = random_seed

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
        
        target_scale = (sum(abs(x) for x in enc) / len(enc)) if enc else 1.0
        return {
            "group": self.groups[group_idx],
            "group_idx": group_idx,
            "start": start,
            "encoder": enc,
            "decoder": dec,
            "enc_len": len(enc),
            "dec_len": len(dec),
            "target_scale": target_scale,
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

def multi_collate_pad_plus(
    batch: List[Dict[str, Any]],
    pad_value: float | int = 0,
) -> Dict[str, Any]:
    if not batch:
        return {}

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

    return {
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

    series_map = {
        "A": list(range(1, 13)),  # 长一点
        "B": [10, 9, 8, 7, 6, 5, 4, 3, 2],
    }
    ds2 = MultiWindowDataset_Step2(
        series_map,
        enc_len=5, dec_len=2,
        min_enc_len=3, max_dec_len=4,
        randomize_length=(0.7, 0.7),
        random_seed=42,
    )
    
    
    print("\n--- loader ---")
    for i, batch in enumerate(to_loader_multi(ds2, batch_size=2, shuffle=True, seed=0, drop_last=False)):
        print(f"batch {i}: groups={batch['groups']}, enc_lens={batch['enc_lens']}, dec_lens={batch['dec_lens']}")
        if i >= 1:
            break

    # 先确保 __getitem__ 里的切片在 if 外统一执行（无论是否随机）
    # enc = seq[start:start+enc_len]; dec = seq[start+enc_len : start+enc_len+dec_len]

    print("\n--- time_idx + mask sanity ---")
    for i, batch in enumerate(to_loader_multi(ds2, batch_size=2, shuffle=True, seed=0, drop_last=False)):
        print(f"batch {i}:")
        print("  groups:", batch["groups"])
        print("  enc_lens:", batch["enc_lens"], "dec_lens:", batch["dec_lens"])
        print("  encoder_time_idx[0]:", batch["encoder_time_idx"][0])
        print("  decoder_time_idx[0]:", batch["decoder_time_idx"][0])
        print("  encoder_mask[0]:", batch["encoder_mask"][0])
        print("  decoder_mask[0]:", batch["decoder_mask"][0])
        break


    print("\n--- mini tests ---")
    # 取一个 batch
    b = next(to_loader_multi(ds2, batch_size=3, shuffle=True, seed=123, drop_last=False))

    # 1) mask 的 1 的个数应等于各自长度
    for ml, m in zip(b["enc_lens"], b["encoder_mask"]):
        assert sum(m) == ml, "encoder_mask sum != enc_len"
    for ml, m in zip(b["dec_lens"], b["decoder_mask"]):
        assert sum(m) == ml, "decoder_mask sum != dec_len"

    # 2) padding 区域确实填了 pad_value（默认 0）
    pad = 0
    max_enc = len(b["enc_padded"][0])
    max_dec = len(b["dec_padded"][0])
    for row, l in zip(b["enc_padded"], b["enc_lens"]):
        assert all(v == pad for v in row[l:max_enc]), "encoder padding not filled properly"
    for row, l in zip(b["dec_padded"], b["dec_lens"]):
        assert all(v == pad for v in row[l:max_dec]), "decoder padding not filled properly"

    # 3) 时间索引单调 + 无穿组（简单校验）
    for row in b["encoder_time_idx"]:
        assert all(row[i] + 1 == row[i+1] for i in range(len(row)-1)), "encoder_time_idx not contiguous"
    for row in b["decoder_time_idx"]:
        assert all(row[i] + 1 == row[i+1] for i in range(len(row)-1)), "decoder_time_idx not contiguous"

    # 4) 可复现性：相同样本（同 group, start）长度应一致
    #   取数据集第 0 个样本，再次取
    x0a = ds2[0]
    x0b = ds2[0]
    assert (x0a["enc_len"], x0a["dec_len"]) == (x0b["enc_len"], x0b["dec_len"]), "per-item RNG not deterministic"

    print("mini tests: OK")


# ------------------ Demo ------------------
if __name__ == "__main__":
    main()
