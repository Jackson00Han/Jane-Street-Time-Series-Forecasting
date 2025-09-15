# pipeline/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable, Optional, Tuple
import polars as pl


# 特征工程

# A：响应列的“上一日尾部/日度摘要”
def fe_resp_daily(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str, str, str] = ("symbol_id","date_id","time_id"),
    rep_cols: Sequence[str],
    is_sorted: bool = False,
    prev_soft_days: Optional[int] = None,
    cast_f32: bool = True,
    tail_lags: Sequence[int] = (1,),
    tail_diffs: Sequence[int] = (1,),
    rolling_windows: Sequence[int] | None = (3,),
) -> pl.LazyFrame:
    """一次日频聚合得到昨日尾部与日级摘要 → 统一 TTL 到“对 d 生效的历史值” → 回拼到 tick 级。"""
    g_symbol, g_date, g_time = keys

    # 若未保证排序，这里补一次（只影响 lf；日频表仍会再按 (symbol,date) 排）
    if not is_sorted:
        lf = lf.sort([g_symbol, g_date, g_time])

    # --- 一次性日频聚合 ---
    need_L = sorted(set(tail_lags) | {k+1 for k in tail_diffs} | {1})
    agg_exprs: list[pl.Expr] = []
    for r in rep_cols:
        # 尾部倒数第 L（长度不足 L → null）
        for L in need_L:
            agg_exprs.append(
                pl.when(pl.len() >= L)
                .then(pl.col(r).sort_by(pl.col(g_time)).tail(L).first())
                .otherwise(None)
                .alias(f"{r}_prev_tail_lag{L}")
            )
        # 当日统计（显式补上 prevday_close）
        agg_exprs += [
            pl.col(r).sort_by(pl.col(g_time)).last().alias(f"{r}_prevday_close"),
            pl.col(r).mean().alias(f"{r}_prevday_mean"),
            pl.col(r).std(ddof=1).alias(f"{r}_prevday_std"),
        ]

    daily = (
        lf.group_by([g_symbol, g_date])
        .agg(agg_exprs)
        .sort([g_symbol, g_date])                # 供下面 shift/ffill 正确运行
    )

    # 派生（当日）dK：last - (K+1 from end)
    daily = daily.with_columns([
        (pl.col(f"{r}_prev_tail_lag1") - pl.col(f"{r}_prev_tail_lag{K+1}")).alias(f"{r}_prev_tail_d{K}")
        for r in rep_cols for K in tail_diffs
        if f"{r}_prev_tail_lag{K+1}" in daily.collect_schema().names()
    ])

    # prev2day/overnight/rolling（仍是“当日相对”的量）
    daily = daily.with_columns([
        pl.col(f"{r}_prevday_close").shift(1).over(g_symbol).alias(f"{r}_prev2day_close")
        for r in rep_cols
    ]).with_columns(
        [
            (pl.col(f"{r}_prevday_close") - pl.col(f"{r}_prevday_mean")).alias(f"{r}_prevday_close_minus_mean")
            for r in rep_cols
        ] + [
            (pl.col(f"{r}_prevday_close") - pl.col(f"{r}_prev2day_close")).alias(f"{r}_overnight_gap")
            for r in rep_cols
        ]
    )

    if rolling_windows:
        wins = sorted({int(w) for w in rolling_windows if int(w) > 1})
        roll_exprs: list[pl.Expr] = []
        for r in rep_cols:
            base = pl.col(f"{r}_prevday_close")
            for w in wins:
                roll_exprs += [
                    base.rolling_mean(window_size=w, min_samples=1).over(g_symbol)
                        .alias(f"{r}_close_roll{w}_mean"),
                    base.rolling_std(window_size=w, ddof=1, min_samples=2).over(g_symbol)
                        .alias(f"{r}_close_roll{w}_std"),
                ]
        daily = daily.with_columns(roll_exprs)

    # === 核心：将上面的“当日统计/尾部衍生列”转换为“对 d 生效的历史 TTL 值” ===
    prev_cols = [c for c in daily.collect_schema().names() if c not in (g_symbol, g_date)]
    exprs: list[pl.Expr] = []
    for c in prev_cols:
        # 最近一次（发生在当前日之前）的非空日期与值
        last_non_null_day = (
            pl.when(pl.col(c).is_not_null()).then(pl.col(g_date)).otherwise(None)
            .forward_fill().over(g_symbol)
            .shift(1)
        )
        last_non_null_val = pl.col(c).forward_fill().over(g_symbol).shift(1)

        if prev_soft_days is None:
            resolved = last_non_null_val  # 无限 TTL：总取最近一次历史非空
        else:
            gap_days = (pl.col(g_date) - last_non_null_day).cast(pl.Int32)
            resolved = pl.when(gap_days.is_not_null() & (gap_days <= int(prev_soft_days))) \
                        .then(last_non_null_val) \
                        .otherwise(None)

        if cast_f32:
            resolved = resolved.cast(pl.Float32)
        exprs.append(resolved.alias(c))    # 列名不变，语义已是“对 d 生效的历史值”

    daily_prev = daily.with_columns(exprs)

    # 回拼到 tick 级（左连），并固定顺序（可选）
    out = lf.join(daily_prev, on=[g_symbol, g_date], how="left")
    out = out.sort([g_symbol, g_date, g_time])
    return out



# B：同 time_id 跨日的 prev{k} + 统计
def fe_resp_same_tick_xday(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str,str,str] = ("symbol_id","date_id","time_id"),
    rep_cols: Sequence[str],
    is_sorted: bool = False,
    prev_soft_days: Optional[int] = None,   # None=严格d-k；整数=TTL
    cast_f32: bool = True,
    ndays: int = 5,
    stats_rep_cols: Optional[Sequence[str]] = None,
    add_prev1_multirep: bool = True,
    batch_size: int = 5,
) -> pl.LazyFrame:
    
    g_symbol, g_date, g_time = keys

    # 保证 (symbol,time) 组内按 date 递增（shift(k).over([symbol,time]) 的因果顺序）
    if not is_sorted:
        lf = lf.sort([g_symbol, g_time, g_date]) # 注意不是date, time

    if stats_rep_cols is None:
        stats_rep_cols = list(rep_cols)

    def _chunks(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    lf_cur = lf

    # 1) prev{k} with strict / TTL
    for batch in _chunks(list(rep_cols), batch_size):
        exprs = []
        for r in batch:
            for k in range(1, ndays + 1):
                val_k  = pl.col(r).shift(k).over([g_symbol, g_time])
                day_k  = pl.col(g_date).shift(k).over([g_symbol, g_time])
                gap_k  = (pl.col(g_date) - day_k).cast(pl.Int32)

                if prev_soft_days is None:
                    # 严格 d-k：gap==k
                    keep = gap_k.is_not_null() & (gap_k == k)
                else:
                    # TTL：只要在当前日之前，且 gap<=K
                    keep = gap_k.is_not_null() & (gap_k > 0) & (gap_k <= int(prev_soft_days))

                val_k = pl.when(keep).then(val_k).otherwise(None)
                if cast_f32:
                    val_k = val_k.cast(pl.Float32)
                exprs.append(val_k.alias(f"{r}_same_t_prev{k}"))
        lf_cur = lf_cur.with_columns(exprs)

    # 2) mean/std（忽略 null）
    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in range(1, ndays + 1)]
            vals = pl.concat_list([pl.col(c) for c in cols]).list.drop_nulls()
            m = vals.list.mean()
            s = vals.list.std(ddof=1)   # 和全局统计一致
            if cast_f32:
                m = m.cast(pl.Float32); s = s.cast(pl.Float32)
            exprs += [
                m.alias(f"{r}_same_t_last{ndays}_mean"),
                s.alias(f"{r}_same_t_last{ndays}_std"),
            ]
        lf_cur = lf_cur.with_columns(exprs)

    # 3) slope：时间方向设为“最近为正、久远为负”（正=近期上升）
    x = np.arange(ndays, 0, -1, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    x_lits = [pl.lit(float(v)) for v in x]

    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in range(1, ndays + 1)]
            mean_ref = pl.col(f"{r}_same_t_last{ndays}_mean")
            std_ref  = pl.col(f"{r}_same_t_last{ndays}_std")
            terms = [((pl.col(c) - mean_ref) / (std_ref + 1e-9)) * x_lits[i]
                    for i, c in enumerate(cols)]
            # ——更稳：对 null 显式置 0，避免某些版本 sum_horizontal 因 null 变 null
            terms = [pl.when(pl.col(c).is_not_null() & mean_ref.is_not_null() & std_ref.is_not_null())
                    .then(t).otherwise(pl.lit(0.0)) for t, c in zip(terms, cols)]

            n_eff = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in cols]).cast(pl.Float32)
            den   = pl.when(n_eff > 0).then(n_eff).otherwise(pl.lit(1.0))
            slope = pl.sum_horizontal(terms) / den
            if cast_f32:
                slope = slope.cast(pl.Float32)
            exprs.append(slope.alias(f"{r}_same_t_last{ndays}_slope"))
        lf_cur = lf_cur.with_columns(exprs)

    # 4) 跨 responder 的 prev1 行内统计（可选）
    if add_prev1_multirep and len(rep_cols) > 0:
        n_rep = len(rep_cols)  
        prev1_cols = [f"{r}_same_t_prev1" for r in rep_cols]
        prev1_list = pl.concat_list([pl.col(c) for c in prev1_cols]).list.drop_nulls()
        m1 = prev1_list.list.mean()
        s1 = prev1_list.list.std(ddof=1)
        if cast_f32:
            m1 = m1.cast(pl.Float32); s1 = s1.cast(pl.Float32)
        lf_cur = lf_cur.with_columns([
            m1.alias(f"prev1_same_t_mean_{n_rep}rep"),
            s1.alias(f"prev1_same_t_std_{n_rep}rep"),
        ])

    # 出口保持有序，便于后续 C 阶段 shift/rolling
    lf_cur = lf_cur.sort([g_symbol, g_date, g_time])
    return lf_cur




# C 系列：

def fe_feat_history(
    *,
    lf: pl.LazyFrame,
    keys: Tuple[str,str,str] = ("symbol_id","date_id","time_id"),
    feature_cols: Sequence[str],
    is_sorted: bool = False,
    prev_soft_days: Optional[int] = None,
    cast_f32: bool = True,
    batch_size: int = 10,
    lags: Iterable[int] = (1, 3),
    ret_periods: Iterable[int] = (1,),
    diff_periods: Iterable[int] = (1,),
    rz_windows: Iterable[int] = (5,),
    ewm_spans: Iterable[int] = (10,),
    keep_rmean_rstd: bool = True,
    cs_cols: Optional[Sequence[str]] = None,
) -> pl.LazyFrame:
    
    g_sym, g_date, g_time = keys
    
    by_grp = [g_sym]
    by_cs  = [g_date, g_time]

    need_cols = [*keys, *feature_cols]
    schema = lf.collect_schema().names()
    miss = [c for c in need_cols if c not in schema]
    if miss:
        raise KeyError(f"Columns not found: {miss}")

    lf_out = lf.select(need_cols)
    if not is_sorted:
        lf_out = lf_out.sort(list(keys))

    def _chunks(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    # ---- 规范化参数：None/[] -> 空元组；并去重/转 int/保正数 ----
    def _clean_pos_sorted_unique(x):
        if not x:
            return tuple()
        return tuple(sorted({int(v) for v in x if int(v) >= 1}))

    LAGS   = _clean_pos_sorted_unique(lags)
    K_RET  = _clean_pos_sorted_unique(ret_periods)
    K_DIFF = _clean_pos_sorted_unique(diff_periods)
    RZW    = _clean_pos_sorted_unique(rz_windows)
    SPANS  = _clean_pos_sorted_unique(ewm_spans)

    # C1 lags
    if LAGS:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for L in LAGS:
                last_date_L = pl.col(g_date).shift(L).over(by_grp)
                gap_L = (pl.col(g_date) - last_date_L).cast(pl.Int32)
                if prev_soft_days is not None:
                    keep_L = gap_L.is_not_null() & (gap_L > 0) & (gap_L <= pl.lit(int(prev_soft_days)))
                for c in batch:
                    e = pl.col(c).shift(L).over(by_grp)
                    if prev_soft_days is not None:
                        e = pl.when(keep_L).then(e).otherwise(None)
                    if cast_f32:
                        e = e.cast(pl.Float32)
                    exprs.append(e.alias(f"{c}__lag{L}"))
            lf_out = lf_out.with_columns(exprs)

    # C2 returns（可选）
    if K_RET:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                cur = pl.col(c)
                for k in K_RET:
                    if k in LAGS:
                        prev = pl.col(f"{c}__lag{k}")  # 已含 TTL
                    else:
                        prev = pl.col(c).shift(k).over(by_grp)
                        if prev_soft_days is not None:
                            last_date_k = pl.col(g_date).shift(k).over(by_grp)
                            gap_k = (pl.col(g_date) - last_date_k).cast(pl.Int32)
                            keep_k = gap_k.is_not_null() & (gap_k > 0) & (gap_k <= pl.lit(int(prev_soft_days)))
                            prev = pl.when(keep_k).then(prev).otherwise(None)
                    ret = pl.when(prev.is_not_null() & (prev.abs() > 1e-12)).then(cur / prev - 1.0).otherwise(None)
                    if cast_f32:
                        ret = ret.cast(pl.Float32)
                    exprs.append(ret.alias(f"{c}__ret{k}"))
            lf_out = lf_out.with_columns(exprs)


    # C3 diffs（可选）
    if K_DIFF:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                cur = pl.col(c)
                for k in K_DIFF:
                    if k in LAGS:
                        prevk = pl.col(f"{c}__lag{k}")  # 已含 TTL
                    else:
                        prevk = pl.col(c).shift(k).over(by_grp)
                        if prev_soft_days is not None:
                            last_date_k = pl.col(g_date).shift(k).over(by_grp)
                            gap_k = (pl.col(g_date) - last_date_k).cast(pl.Int32)
                            keep_k = gap_k.is_not_null() & (gap_k > 0) & (gap_k <= pl.lit(int(prev_soft_days)))
                            prevk = pl.when(keep_k).then(prevk).otherwise(None)
                    d = pl.when(prevk.is_not_null()).then(cur - prevk).otherwise(None)
                    if cast_f32:
                        d = d.cast(pl.Float32)
                    exprs.append(d.alias(f"{c}__diff{k}"))
            lf_out = lf_out.with_columns(exprs)



    # C4 rolling r-z
    if RZW:
        for batch in _chunks(feature_cols, batch_size):
            exprs_base = []
            # 统一构造 t-1 的基准值（含 TTL 掩码）
            if prev_soft_days is not None:
                last_date_1 = pl.col(g_date).shift(1).over(by_grp)
                gap_1 = (pl.col(g_date) - last_date_1).cast(pl.Int32)
                keep_1 = gap_1.is_not_null() & (gap_1 > 0) & (gap_1 <= pl.lit(int(prev_soft_days)))

            for c in batch:
                # 若之前已在 C1 产出 __lag1，可直接用： base = pl.col(f"{c}__lag1")
                base = pl.col(c).shift(1).over(by_grp)
                if prev_soft_days is not None:
                    base = pl.when(keep_1).then(base).otherwise(None)
                exprs_base.append(base.alias(f"{c}__tminus1_base"))
            lf_out = lf_out.with_columns(exprs_base)

            # 真正的 rolling r-z
            roll_exprs = []
            for c in batch:
                base = pl.col(f"{c}__tminus1_base")
                for w in RZW:
                    m  = base.rolling_mean(window_size=w, min_samples=1).over(by_grp)
                    s  = base.rolling_std(window_size=w, ddof=1, min_samples=2).over(by_grp)  # 统一 ddof=1
                    den = (s.fill_null(0.0) + 1e-9)
                    rz = (base - m) / den
                    if cast_f32:
                        m = m.cast(pl.Float32); s = s.cast(pl.Float32); rz = rz.cast(pl.Float32)
                    if keep_rmean_rstd:
                        roll_exprs += [
                            m.alias(f"{c}__rmean{w}"),
                            s.alias(f"{c}__rstd{w}"),
                            rz.alias(f"{c}__rz{w}"),
                        ]
                    else:
                        roll_exprs.append(rz.alias(f"{c}__rz{w}"))
            lf_out = lf_out.with_columns(roll_exprs)
            lf_out = lf_out.drop([f"{c}__tminus1_base" for c in batch])


    # C5 EWM（可选）
    if SPANS:
        for batch in _chunks(feature_cols, batch_size):
            exprs_base = []

            # TTL 掩码（t-1）
            if prev_soft_days is not None:
                last_date_1 = pl.col(g_date).shift(1).over(by_grp)
                gap_1 = (pl.col(g_date) - last_date_1).cast(pl.Int32)
                keep_1 = gap_1.is_not_null() & (gap_1 > 0) & (gap_1 <= pl.lit(int(prev_soft_days)))

            # 构造 t-1 基准（若你已在 C1 产出 __lag1，可以直接用它替代下面两行）
            for c in batch:
                base = pl.col(c).shift(1).over(by_grp)
                if prev_soft_days is not None:
                    base = pl.when(keep_1).then(base).otherwise(None)
                exprs_base.append(base.alias(f"{c}__tminus1_base"))
            lf_out = lf_out.with_columns(exprs_base)

            # 计算 EWM
            ewm_exprs = []
            for c in batch:
                base = pl.col(f"{c}__tminus1_base")
                for s in SPANS:
                    ema = base.ewm_mean(span=int(s), adjust=False, ignore_nulls=True).over(by_grp)
                    if cast_f32:
                        ema = ema.cast(pl.Float32)
                    ewm_exprs.append(ema.alias(f"{c}__ewm{s}"))
            lf_out = lf_out.with_columns(ewm_exprs)

            # 清理临时列
            lf_out = lf_out.drop([f"{c}__tminus1_base" for c in batch])


    # C6 cross-section rank（可选）
    if cs_cols:
        cs_cols = [c for c in cs_cols if c in feature_cols]
        if cs_cols:

            # TTL 掩码（t-1）
            if prev_soft_days is not None:
                last_date_1 = pl.col(g_date).shift(1).over(by_grp)
                gap_1 = (pl.col(g_date) - last_date_1).cast(pl.Int32)
                keep_1 = gap_1.is_not_null() & (gap_1 > 0) & (gap_1 <= pl.lit(int(prev_soft_days)))

            # 先构造每列的 t-1 基准（含 TTL）
            exprs_base = []
            for c in cs_cols:
                base = pl.col(c).shift(1).over(by_grp)
                if prev_soft_days is not None:
                    base = pl.when(keep_1).then(base).otherwise(None)
                exprs_base.append(base.alias(f"{c}__tminus1_base"))
            lf_out = lf_out.with_columns(exprs_base)

            # 基于 t-1：截面 z 与 rank(0..1)
            cs_exprs = []
            for c in cs_cols:
                base = pl.col(f"{c}__tminus1_base")

                # 截面统计（只用该列的 t-1）
                n_valid = base.is_not_null().cast(pl.Int32).sum().over(by_cs)
                mu      = base.mean().over(by_cs)
                sig     = base.std(ddof=1).over(by_cs)

                # z-score（数值更稳：sig.fill_null(0)+eps）
                z = ((base - mu) / (sig.fill_null(0.0) + 1e-9)) \
                        .cast(pl.Float32 if cast_f32 else pl.Float64)

                # 百分位排名：空→None；n=1→0.5
                rank_raw = base.rank(method="average").over(by_cs)
                csrank = pl.when(base.is_null()).then(None).otherwise(
                    pl.when(n_valid > 1)
                    .then((rank_raw - 0.5) / n_valid.cast(pl.Float32))
                    .otherwise(pl.lit(0.5))
                ).cast(pl.Float32 if cast_f32 else pl.Float64)

                cs_exprs += [z.alias(f"{c}__cs_z"), csrank.alias(f"{c}__csrank")]

            lf_out = lf_out.with_columns(cs_exprs)

            # 清理临时列
            lf_out = lf_out.drop([f"{c}__tminus1_base" for c in cs_cols])
    return lf_out

   
@dataclass
class StageA:
    tail_lags: Sequence[int]
    tail_diffs: Sequence[int]
    rolling_windows: Optional[Sequence[int]]
    prev_soft_days: Optional[int] = None
    is_sorted: bool = False
    cast_f32: bool = True

@dataclass
class StageB:
    ndays: int
    stats_rep_cols: Optional[Sequence[str]] = None
    add_prev1_multirep: bool = True
    batch_size: int = 5
    prev_soft_days: Optional[int] = None
    is_sorted: bool = False
    cast_f32: bool = True

# C 的每个操作可选；None / [] 表示跳过该操作
@dataclass
class StageC:
    lags: Optional[Iterable[int]] = None
    ret_periods: Optional[Iterable[int]] = None
    diff_periods: Optional[Iterable[int]] = None
    rz_windows: Optional[Iterable[int]] = None
    ewm_spans: Optional[Iterable[int]] = None
    cs_cols: Optional[Sequence[str]] = None
    keep_rmean_rstd: bool = True
    prev_soft_days: Optional[int] = None
    batch_size: Optional[int] = 10
    is_sorted: bool = False
    cast_f32: bool = True
    

def assert_time_monotone(path, *, date_col="date_id", time_col="time_id"):
    s = (pl.scan_parquet(path, storage_options=storage_options)
           .select([
               (pl.col(date_col).diff().fill_null(0) < 0).any().alias('date_drop'),
               ((pl.col(date_col).diff().fill_null(0) == 0) &
                (pl.col(time_col).diff().fill_null(0) < 0)).any().alias('time_drop')
           ])
           .collect(streaming=True))
    assert not s['date_drop'][0] and not s['time_drop'][0]


def run_staged_engineering(
    lf_base: pl.LazyFrame,
    *,
    keys: Sequence[str],
    rep_cols: Sequence[str],
    feature_cols: Sequence[str],
    out_dir: str,
    A: StageA | None = None,
    B: StageB | None = None,
    C: StageC | None = None,
    write_date_between: tuple[int, int] | None = None,   # 新增：只写核心区间
):
    g_symbol, g_date, g_time = keys

    def _save(lf_out: pl.LazyFrame, path: str):
        if write_date_between is None:
            raise ValueError("write_date_between must be specified to avoid date overlap")
        lo, hi = write_date_between
        
        sk = [g_date, g_time, g_symbol]
        
        df = lf_out.filter(pl.col(g_date).is_between(lo, hi)).sort(sk).collect()
        with fs.open(path, "wb") as f:   # 复用你上面构好的 fs (fsspec)
            df.write_parquet(f, compression="zstd")
        if cfg.get("debug", {}).get("check_time_monotone", True):
            assert_time_monotone(path, date_col=g_date, time_col=g_time)


        
    # ---------- A ----------
    if A is not None:
        lf_resp = lf_base.select([*keys, *rep_cols])
        lf_a_full = fe_resp_daily(
            lf_resp,
            keys=tuple(keys),
            rep_cols=rep_cols,
            is_sorted=A.is_sorted,
            prev_soft_days=A.prev_soft_days,
            cast_f32=A.cast_f32,
            tail_lags=A.tail_lags,
            tail_diffs=A.tail_diffs,
            rolling_windows=A.rolling_windows,
        )
        drop = set(keys) | set(rep_cols)
        a_cols = [c for c in lf_a_full.collect_schema().names() if c not in drop]
        _save(lf_a_full.select([*keys, *a_cols]), f"{out_dir}/stage_a.parquet")
        

    # ---------- B ----------
    if B is not None:
        lf_resp = lf_base.select([*keys, *rep_cols])
        lf_b_full = fe_resp_same_tick_xday(
            lf_resp,
            keys=tuple(keys),
            rep_cols=rep_cols,
            is_sorted=B.is_sorted,
            prev_soft_days=B.prev_soft_days,
            cast_f32=B.cast_f32,
            ndays=B.ndays,
            stats_rep_cols=B.stats_rep_cols,
            add_prev1_multirep=B.add_prev1_multirep,
            batch_size=B.batch_size,
        )
        drop = set(keys) | set(rep_cols)
        b_cols = [c for c in lf_b_full.collect_schema().names() if c not in drop]
        _save(lf_b_full.select([*keys, *b_cols]), f"{out_dir}/stage_b.parquet")

    # ---------- C（按操作分别输出） ----------
    if C is not None:
        def _do_op(op_name: str, **op_flags):
            lf_src = lf_base.select([*keys, *feature_cols])
            lf_c = fe_feat_history(
                lf=lf_src,
                keys=tuple(keys),
                feature_cols=feature_cols,
                is_sorted=C.is_sorted,
                prev_soft_days=C.prev_soft_days,
                cast_f32=C.cast_f32,
                batch_size=C.batch_size,
                lags=op_flags.get("lags"),
                ret_periods=op_flags.get("ret_periods"),
                diff_periods=op_flags.get("diff_periods"),
                rz_windows=op_flags.get("rz_windows"),
                ewm_spans=op_flags.get("ewm_spans"),
                keep_rmean_rstd=C.keep_rmean_rstd,
                cs_cols=op_flags.get("cs_cols"),
            ).drop(feature_cols)
            cols = [c for c in lf_c.collect_schema().names() if c not in keys]
            _save(lf_c.select([*keys, *cols]), f"{out_dir}/stage_c_{op_name}.parquet")

        if C.lags:         _do_op("lags",   lags=C.lags)
        if C.ret_periods:  _do_op("ret",    ret_periods=C.ret_periods)
        if C.diff_periods: _do_op("diff",   diff_periods=C.diff_periods)
        if C.rz_windows:   _do_op("rz",     rz_windows=C.rz_windows)
        if C.ewm_spans:    _do_op("ewm",    ewm_spans=C.ewm_spans)
        if C.cs_cols:      _do_op("csrank", cs_cols=C.cs_cols)
        