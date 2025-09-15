from __future__ import annotations
from typing import Sequence, Optional, Tuple
import polars as pl


def clip_upper(expr: pl.Expr, ub: int) -> pl.Expr:
    return pl.when(expr > pl.lit(ub)).then(pl.lit(ub)).otherwise(expr)


def rolling_sigma_clip(
    lf: pl.LazyFrame,
    clip_features: Sequence[str],
    over_cols: Sequence[str],
    *,
    is_sorted: bool = False,
    window: int = 50,
    k: float = 3.0,
    ddof: int = 1,
    min_valid: int = 10,
    cast_float32: bool = True,
    sanitize: bool = True,
) -> pl.LazyFrame:
    if not is_sorted:
        raise ValueError("Input LazyFrame must be pre-sorted by ['symbol_id','date_id','time_id']")

    required = {"symbol_id","date_id","time_id","time_bucket"} | set(clip_features)
    names = set(lf.collect_schema().names())
    missing = list(required - names)
    if missing:
        raise KeyError(f"Missing columns: {missing}")


    base = lf.select(pl.col(["symbol_id","date_id","time_id","time_bucket"] + list(clip_features)))
    min_need = max(min_valid, ddof + 1)
    min_samp = ddof + 1

    exprs = []
    for c in clip_features:
        x = pl.col(c)
        if cast_float32:
            x = x.cast(pl.Float32)
        if sanitize:
            x = pl.when(x.is_finite()).then(x).otherwise(None)

        # 注意：这里不要 over
        xlag = x.shift(1)

        # 只在 rolling 结果上 over（组内历史）
        cnt = (
            xlag.is_not_null()
                .cast(pl.Int32)
                .rolling_sum(window_size=window, min_samples=ddof + 1)
        ).over(over_cols)

        mu = (
            xlag.rolling_mean(window_size=window, min_samples=ddof + 1)
        ).over(over_cols)

        sd = (
            xlag.rolling_std(window_size=window, ddof=ddof, min_samples=ddof + 1)
        ).over(over_cols)

        lo, hi = mu - k * sd, mu + k * sd
        exprs.append(
            pl.when(cnt >= max(min_valid, ddof + 1))
            .then(x.clip(lo, hi))
            .otherwise(x)
            .alias(c)
        )

    return base.with_columns(exprs)



def causal_impute(
    lf: pl.LazyFrame,
    impute_cols: Sequence[str],
    *,
    open_tick_window: Tuple[int, int] = (0, 10),
    ttl_days_open: int = 5,
    intra_ffill_max_gap_ticks: Optional[int] = 100,
    ttl_days_same_tick: Optional[int] = 5,
    is_sorted: bool = False,
) -> pl.LazyFrame:
    if not is_sorted:
        raise ValueError("Input LazyFrame must be pre-sorted by ['symbol_id','date_id','time_id']")

    # 参数合法性
    assert intra_ffill_max_gap_ticks is None or intra_ffill_max_gap_ticks >= 0
    assert ttl_days_same_tick is None or ttl_days_same_tick >= 0

    # 统一 dtype（可选，但更稳）
    lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in impute_cols])
    
    
    t0, t1 = open_tick_window
    is_open = pl.col("time_id").is_between(t0, t1, closed="left")  # [t0, t1)

    # ---- 1) 开盘：跨日承接（TTL）----
    open_exprs = []
    for c in impute_cols:
        last_date = (
            pl.when(pl.col(c).is_not_null()).then(pl.col("date_id"))
            .forward_fill().over("symbol_id")
        )
        cand = pl.col(c).forward_fill().over("symbol_id")
        gap  = (pl.col("date_id") - last_date).cast(pl.Int32)
        open_exprs.append(
            pl.when(is_open 
                    & pl.col(c).is_null() 
                    & (gap.fill_null(ttl_days_open + 1) <= ttl_days_open))
            .then(cand)
            .otherwise(pl.col(c))
            .alias(c)
        )
    lf1 = lf.with_columns(open_exprs)

    # ---- 2) 日内 ffill（(symbol,date)），可限步数 ----
    if intra_ffill_max_gap_ticks is None:
        lf2 = lf1.with_columns([pl.col(c).forward_fill().over(["symbol_id","date_id"]).alias(c) for c in impute_cols])
    else:
        k = intra_ffill_max_gap_ticks
        exprs = []
        for c in impute_cols:
            last_t = (
                pl.when(pl.col(c).is_not_null()).then(pl.col("time_id"))
                .forward_fill().over(["symbol_id","date_id"])
            )
            cand = pl.col(c).forward_fill().over(["symbol_id","date_id"])
            gap  = (pl.col("time_id") - last_t).cast(pl.Int32)
            exprs.append(
                pl.when(pl.col(c).is_null() & (gap.fill_null(k + 1) <= k))
                .then(cand)
                .otherwise(pl.col(c))
                .alias(c)
            )
        lf2 = lf1.with_columns(exprs)

    # ---- 3) 同一 time_id 跨日承接（TTL，可选）----
    lf3 = lf2
    if ttl_days_same_tick is not None:
        d = ttl_days_same_tick
        exprs = []
        for c in impute_cols:
            last_date_same = (
                pl.when(pl.col(c).is_not_null()).then(pl.col("date_id"))
                .forward_fill().over(["symbol_id","time_id"])
            )
            cand_same = pl.col(c).forward_fill().over(["symbol_id","time_id"])
            gap2 = (pl.col("date_id") - last_date_same).cast(pl.Int32)
            exprs.append(
                pl.when(pl.col(c).is_null() & (gap2.fill_null(d + 1) <= d))
                .then(cand_same)
                .otherwise(pl.col(c))
                .alias(c)
            )
        lf3 = lf2.with_columns(exprs)

    # ---- 4) 再日内 ffill 传播（与步骤2同逻辑）----
    if intra_ffill_max_gap_ticks is None:
        lf4 = lf3.with_columns([pl.col(c).forward_fill().over(["symbol_id","date_id"]).alias(c) for c in impute_cols])
    else:
        k = intra_ffill_max_gap_ticks
        exprs = []
        for c in impute_cols:
            last_t = (
                pl.when(pl.col(c).is_not_null()).then(pl.col("time_id"))
                .forward_fill().over(["symbol_id","date_id"])
            )
            cand = pl.col(c).forward_fill().over(["symbol_id","date_id"])
            gap  = (pl.col("time_id") - last_t).cast(pl.Int32)
            exprs.append(
                pl.when(pl.col(c).is_null() & (gap.fill_null(k + 1) <= k))
                .then(cand)
                .otherwise(pl.col(c))
                .alias(c)
            )
        lf4 = lf3.with_columns(exprs)

    KEYS = ["symbol_id","date_id","time_id"]
    return lf4.select([*KEYS, *impute_cols])

