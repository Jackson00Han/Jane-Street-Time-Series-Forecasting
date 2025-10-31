from __future__ import annotations
import numpy as np

def day_ptrs_from_sorted_dates(d: np.ndarray):
    d = d.ravel()
    starts = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1]
    days   = d[starts]
    ends   = np.r_[starts[1:], d.size]
    return days, starts, ends


def make_sliding_cv(date_ids: np.ndarray, *, n_splits: int, gap_days: int = 5, train_to_val: int = 9):
    days, starts, ends = day_ptrs_from_sorted_dates(date_ids)
    N, R, K, G = len(days), int(train_to_val), int(n_splits), int(gap_days)
    usable = N - G
    if usable <= 0 or K <= 0 or R <= 0:
        return []
    V_base, rem = divmod(usable, R + K)
    if V_base <= 0:
        return []
    T = R * V_base
    v_lens = [V_base + 1 if i < rem else V_base for i in range(K)]
    v_lo = T + G
    folds = []
    for V_i in v_lens:
        v_hi  = v_lo + V_i
        tr_hi = v_lo - G
        tr_lo = tr_hi - T
        if tr_lo < 0 or v_hi > N:
            break
        tr_idx = np.arange(starts[tr_lo], ends[tr_hi-1])
        va_idx = np.arange(starts[v_lo],   ends[v_hi-1])
        folds.append((tr_idx, va_idx))
        v_lo = v_hi
    return folds