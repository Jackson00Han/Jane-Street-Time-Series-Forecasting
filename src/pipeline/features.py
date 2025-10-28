# pipeline/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl

from pipeline.io import cfg, fs, storage_options  # removed unused P
from pipeline.validate import assert_time_monotone


# =========================
# A) Daily response features
# =========================
def fe_resp_daily(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str, str, str] = ("symbol_id", "date_id", "time_id"),
    rep_cols: Sequence[str],
    is_sorted: bool = False,
    cast_f32: bool = True,
    tail_lags: Sequence[int] = (1,),
    tail_diffs: Sequence[int] = (1,),
    rolling_windows: Sequence[int] | None = (3,),
) -> pl.LazyFrame:
    """
    Build daily response features (causal).
    - Aggregate intraday ticks to day level in Polars.
    - Compute gap/streak and segmented rolling stats in pandas.
    - Join back to ticks and broadcast historical values to current day.
    """
    g_symbol, g_date, g_time = keys

    # 0) Ensure causal sort on ticks
    if not is_sorted:
        lf = lf.sort([g_symbol, g_date, g_time])

    # 1) Intraday → daily aggregates (Polars)
    need_L = sorted(set(tail_lags) | {k + 1 for k in tail_diffs} | {1})
    agg_exprs: list[pl.Expr] = []
    for r in rep_cols:
        # last-L intraday value after sorting by time
        for L in need_L:
            agg_exprs.append(
                pl.when(pl.len() >= L)
                .then(pl.col(r).sort_by(pl.col(g_time)).tail(L).first())
                .otherwise(None)
                .alias(f"{r}_prev_tail_lag{L}")
            )
        # day stats
        agg_exprs += [
            pl.col(r).sort_by(pl.col(g_time)).last().alias(f"{r}_prevday_close"),
            pl.col(r).mean().alias(f"{r}_prevday_mean"),
            pl.col(r).std(ddof=1).alias(f"{r}_prevday_std"),
        ]

    daily_pl = (
        lf.group_by([g_symbol, g_date])
        .agg(agg_exprs)
        .sort([g_symbol, g_date])
        .collect(streaming=False)  # small daily table
    )

    # 2) Gap/streak + rolling (pandas; stable segmented ops)
    daily = daily_pl.to_pandas()
    daily.sort_values([g_symbol, g_date], inplace=True)

    gap1 = daily.groupby(g_symbol, sort=False)[g_date].diff().fillna(1).astype(np.int32)
    daily["__gap1"] = gap1
    daily["__streak_id"] = (gap1 != 1).groupby(daily[g_symbol]).cumsum().astype(np.int32)

    add_cols: dict[str, pd.Series] = {}

    # 2.1 tail diffs: lag1 - lag(K+1)
    for r in rep_cols:
        for K in tail_diffs:
            col = (daily[f"{r}_prev_tail_lag1"] - daily[f"{r}_prev_tail_lag{K+1}"])
            add_cols[f"{r}_prev_tail_d{K}"] = col.astype(np.float32 if cast_f32 else np.float64)

    # 2.2 previous-day close, overnight gap, close-minus-mean
    for r in rep_cols:
        prev = daily.groupby(g_symbol, sort=False)[f"{r}_prevday_close"].shift(1)
        prev2 = np.where(daily["__gap1"].values == 1, prev, np.nan)
        add_cols[f"{r}_prev2day_close"] = pd.Series(prev2, index=daily.index).astype(
            np.float32 if cast_f32 else np.float64
        )
        add_cols[f"{r}_overnight_gap"] = (
            daily[f"{r}_prevday_close"] - add_cols[f"{r}_prev2day_close"]
        ).astype(np.float32 if cast_f32 else np.float64)
        add_cols[f"{r}_prevday_close_minus_mean"] = (
            daily[f"{r}_prevday_close"] - daily[f"{r}_prevday_mean"]
        ).astype(np.float32 if cast_f32 else np.float64)

    # 2.3 segmented rolling on (symbol, __streak_id)
    wins = sorted({int(w) for w in (rolling_windows or []) if int(w) > 1})
    if wins:
        grp = daily.groupby([g_symbol, "__streak_id"], sort=False)
        for r in rep_cols:
            sgb = grp[f"{r}_prevday_close"]
            for w in wins:
                mean_w = sgb.rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                std_w = sgb.rolling(window=w, min_periods=2).std(ddof=1).reset_index(level=[0, 1], drop=True)
                if cast_f32:
                    mean_w = mean_w.astype(np.float32)
                    std_w = std_w.astype(np.float32)
                add_cols[f"{r}_close_roll{w}_mean"] = mean_w
                add_cols[f"{r}_close_roll{w}_std"] = std_w

    if add_cols:
        new_df = pd.DataFrame(add_cols, index=daily.index)
        if cast_f32:
            new_df = new_df.astype(np.float32, copy=False)
        daily = pd.concat([daily, new_df], axis=1, copy=False)

    # 3) Back to Polars; broadcast prior-day values to all ticks of day d
    daily_pl2 = pl.from_pandas(daily, include_index=False).sort([g_symbol, g_date])
    exclude = {g_symbol, g_date, "__gap1", "__streak_id"}
    prev_cols = [c for c in daily_pl2.columns if c not in exclude]

    hist_exprs: list[pl.Expr] = []
    for c in prev_cols:
        prev_val = pl.col(c).shift(1).over(g_symbol)
        resolved = pl.when(pl.col("__gap1") == 1).then(prev_val).otherwise(None)
        if cast_f32:
            resolved = resolved.cast(pl.Float32)
        hist_exprs.append(resolved.alias(c))

    daily_prev = daily_pl2.lazy().with_columns(hist_exprs).drop(["__gap1", "__streak_id"])

    # 4) Join back to ticks
    out = lf.join(daily_prev, on=[g_symbol, g_date], how="left").sort([g_symbol, g_date, g_time])
    return out


# =========================================
# B) Same-time (time_id) cross-day features
# =========================================
def fe_resp_same_tick_xday(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str, str, str] = ("symbol_id", "date_id", "time_id"),
    rep_cols: Sequence[str],
    lags: Sequence[int],
    is_sorted: bool = False,
    cast_f32: bool = True,
    stats_rep_cols: Optional[Sequence[str]] = None,
    add_prev1_multirep: bool = True,
    batch_size: int = 5,
) -> pl.LazyFrame:
    g_symbol, g_date, g_time = keys
    if not lags:
        raise ValueError("`lags` cannot be empty.")
    use_lags = sorted({int(x) for x in lags if int(x) > 0})
    if not use_lags:
        raise ValueError("`lags` must contain positive integers.")

    if stats_rep_cols is None:
        stats_rep_cols = list(rep_cols)

    # 1) Ensure causal sort by (symbol, time, date)
    if not is_sorted:
        lf = lf.sort([g_symbol, g_time, g_date])

    def _chunks(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i : i + k]

    lf_cur = lf

    # 2) Strict prev{k} with exact day gap check
    for batch in _chunks(list(rep_cols), batch_size):
        exprs = []
        for r in batch:
            for k in use_lags:
                val_k = pl.col(r).shift(k).over([g_symbol, g_time])
                day_k = pl.col(g_date).shift(k).over([g_symbol, g_time])
                gap_k = (pl.col(g_date) - day_k).cast(pl.Int32)
                out = pl.when(gap_k.is_not_null() & (gap_k == k)).then(val_k).otherwise(None)
                if cast_f32:
                    out = out.cast(pl.Float32)
                exprs.append(out.alias(f"{r}_same_t_prev{k}"))
        lf_cur = lf_cur.with_columns(exprs)

    # 3) Mean/std over last |lags|
    L = len(use_lags)
    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in use_lags]
            vals = pl.concat_list([pl.col(c) for c in cols]).list.drop_nulls()
            m = vals.list.mean()
            s = vals.list.std(ddof=1)
            if cast_f32:
                m = m.cast(pl.Float32)
                s = s.cast(pl.Float32)
            exprs += [
                m.alias(f"{r}_same_t_last{L}_mean"),
                s.alias(f"{r}_same_t_last{L}_std"),
            ]
        lf_cur = lf_cur.with_columns(exprs)

    # 4) Normalized slope with recent weights (length == L)
    x = np.arange(L, 0, -1, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    x_lits = [pl.lit(float(v)) for v in x]

    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in use_lags]
            mean_ref = pl.col(f"{r}_same_t_last{L}_mean")
            std_ref = pl.col(f"{r}_same_t_last{L}_std")
            terms = [((pl.col(c) - mean_ref) / (std_ref + 1e-9)) * x_lits[i] for i, c in enumerate(cols)]
            terms = [
                pl.when(pl.col(c).is_not_null() & mean_ref.is_not_null() & std_ref.is_not_null())
                .then(t)
                .otherwise(pl.lit(0.0))
                for t, c in zip(terms, cols)
            ]
            n_eff = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in cols]).cast(pl.Float32)
            den = pl.when(n_eff > 0).then(n_eff).otherwise(pl.lit(1.0))
            slope = pl.sum_horizontal(terms) / den
            if cast_f32:
                slope = slope.cast(pl.Float32)
            exprs.append(slope.alias(f"{r}_same_t_last{L}_slope"))
        lf_cur = lf_cur.with_columns(exprs)

    # 5) Cross-responder prev1 stats (only if 1 in lags)
    if add_prev1_multirep and len(rep_cols) > 0 and (1 in use_lags):
        n_rep = len(rep_cols)
        prev1_cols = [f"{r}_same_t_prev1" for r in rep_cols]
        prev1_list = pl.concat_list([pl.col(c) for c in prev1_cols]).list.drop_nulls()
        m1 = prev1_list.list.mean()
        s1 = prev1_list.list.std(ddof=1)
        if cast_f32:
            m1 = m1.cast(pl.Float32)
            s1 = s1.cast(pl.Float32)
        lf_cur = lf_cur.with_columns(
            [
                m1.alias(f"prev1_same_t_mean_{n_rep}rep"),
                s1.alias(f"prev1_same_t_std_{n_rep}rep"),
            ]
        )

    # 6) Keep ascending (symbol, date, time)
    lf_cur = lf_cur.sort([g_symbol, g_date, g_time])
    return lf_cur


# =========================
# C) Tick-history features
# =========================
def fe_feat_history(
    *,
    lf: pl.LazyFrame,
    keys: Tuple[str, str, str] = ("symbol_id", "date_id", "time_id"),
    feature_cols: Sequence[str],
    is_sorted: bool = False,
    cast_f32: bool = True,
    batch_size: int = 30,
    lags: Iterable[int] = (1, 3),
    ret_periods: Iterable[int] = (1,),
    diff_periods: Iterable[int] = (1,),
    rz_windows: Iterable[int] = (5,),
    ewm_spans: Iterable[int] = (10,),
    keep_rmean_rstd: bool = True,
    cs_cols: Optional[Sequence[str]] = None,
) -> pl.LazyFrame:
    """
    Build tick-level history features on segmented continuous sequences:
    - Segment by (symbol, __streak_id) where gaps in dates start a new streak.
    - Within each streak, compute lag/diff/ret/rolling/ewm.
    - Optionally compute cross-sectional z-score/rank per (date_id, time_id).
    """
    g_sym, g_date, g_time = keys

    # Basic validation & sort
    need_cols = [*keys, *feature_cols]
    schema = lf.collect_schema().names()
    miss = [c for c in need_cols if c not in schema]
    if miss:
        raise KeyError(f"Columns not found: {miss}")

    lf_out = lf.select(need_cols)
    if not is_sorted:
        lf_out = lf_out.sort(list(keys))

    def _clean_pos_sorted_unique(x):
        if not x:
            return tuple()
        return tuple(sorted({int(v) for v in x if int(v) >= 1}))

    LAGS = _clean_pos_sorted_unique(lags)
    K_RET = _clean_pos_sorted_unique(ret_periods)
    K_DIFF = _clean_pos_sorted_unique(diff_periods)
    RZW = _clean_pos_sorted_unique(rz_windows)
    SPANS = _clean_pos_sorted_unique(ewm_spans)

    # 0) Gap/streak segmentation per symbol (do not cross gaps > 1 day)
    prev_row_date = pl.col(g_date).shift(1).over([g_sym])
    gap1 = (pl.col(g_date) - prev_row_date).cast(pl.Int32).alias("__gap1")  # same day=0, next day=1, gap>1
    lf_out = lf_out.with_columns(gap1)

    reset_seg = pl.when(pl.col("__gap1").is_null()).then(0).otherwise((pl.col("__gap1") > 1).cast(pl.Int32))
    streak_id = reset_seg.cum_sum().over([g_sym]).alias("__streak_id")
    lf_out = lf_out.with_columns(streak_id).drop("__gap1")

    by_sym_streak = [g_sym, "__streak_id"]

    def _chunks(lst, b):
        for i in range(0, len(lst), b):
            yield lst[i : i + b]

    # 1) Step-wise time-series features within streak
    # lag
    if LAGS:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                base = pl.col(c)
                for lag in LAGS:
                    v = base.shift(lag).over(by_sym_streak)
                    if cast_f32:
                        v = v.cast(pl.Float32)
                    exprs.append(v.alias(f"{c}__lag{lag}"))
            lf_out = lf_out.with_columns(exprs)

    # diff
    if K_DIFF:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                base = pl.col(c)
                for d in K_DIFF:
                    prev = base.shift(d).over(by_sym_streak)
                    diff = base - prev
                    if cast_f32:
                        diff = diff.cast(pl.Float32)
                    exprs.append(diff.alias(f"{c}__diff{d}"))
            lf_out = lf_out.with_columns(exprs)

    # ret (guard denominator + symmetric clipping)
    if K_RET:
        eps = pl.lit(1e-12)
        ret_cap = 10.0
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                base = pl.col(c)
                for r in K_RET:
                    prev = base.shift(r).over(by_sym_streak)
                    ret = pl.when(prev.is_not_null() & (prev.abs() > eps)).then((base / prev) - 1.0).otherwise(None)
                    if cast_f32:
                        ret = ret.cast(pl.Float32)
                    ret = ret.clip(-ret_cap, ret_cap)
                    exprs.append(ret.alias(f"{c}__ret{r}"))
            lf_out = lf_out.with_columns(exprs)

    # rolling r-mean / r-std / r-z
    if RZW:
        eps_std = pl.lit(1e-9)
        wins = sorted({int(w) for w in RZW if int(w) > 0})
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                base = pl.col(c)
                for w in wins:
                    rmean_expr = base.rolling_mean(window_size=w, min_periods=1).over(by_sym_streak)
                    rstd_expr = base.rolling_std(window_size=w, min_periods=2).over(by_sym_streak)
                    rz_expr = pl.when(rstd_expr.is_not_null() & (rstd_expr > eps_std)).then(
                        (base - rmean_expr) / (rstd_expr + eps_std)
                    ).otherwise(None)
                    if keep_rmean_rstd:
                        if cast_f32:
                            rmean_expr = rmean_expr.cast(pl.Float32)
                            rstd_expr = rstd_expr.cast(pl.Float32)
                        exprs.append(rmean_expr.alias(f"{c}__rmean{w}"))
                        exprs.append(rstd_expr.alias(f"{c}__rstd{w}"))
                    if cast_f32:
                        rz_expr = rz_expr.cast(pl.Float32)
                    exprs.append(rz_expr.alias(f"{c}__rz{w}"))
            lf_out = lf_out.with_columns(exprs)

    # ewm (per-streak)
    if SPANS:
        spans = sorted({int(s) for s in SPANS if int(s) > 0})
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                base = pl.col(c)
                for s in spans:
                    ema_expr = base.ewm_mean(span=s, adjust=False, min_periods=1, ignore_nulls=True).over(by_sym_streak)
                    if cast_f32:
                        ema_expr = ema_expr.cast(pl.Float32)
                    exprs.append(ema_expr.alias(f"{c}__ewm{s}"))
            lf_out = lf_out.with_columns(exprs)

    # 2) Cross-sectional features on (date_id, time_id)
    if cs_cols:
        by_cs = [g_date, g_time]
        cs_cols = [c for c in cs_cols if c in feature_cols]
        if cs_cols:
            eps_sig = pl.lit(1e-9)
            exprs = []
            for c in cs_cols:
                base = pl.col(c)
                n_valid = base.is_not_null().sum().over(by_cs).cast(pl.Int32)
                mu = base.mean().over(by_cs)
                sig = base.std(ddof=1).over(by_cs)
                z = pl.when(sig.is_not_null() & (sig > eps_sig)).then((base - mu) / (sig + eps_sig)).otherwise(None)
                rank_raw = base.rank(method="average").over(by_cs)
                csrank = pl.when(base.is_null())
                csrank = csrank.then(None).otherwise(
                    pl.when(n_valid > 1)
                    .then((rank_raw - 0.5) / n_valid.cast(pl.Float32))
                    .otherwise(pl.lit(0.5))
                ).cast(pl.Float32 if cast_f32 else pl.Float64)
                exprs += [z.alias(f"{c}__cs_z"), csrank.alias(f"{c}__csrank")]
            lf_out = lf_out.with_columns(exprs)

    # Drop temporary columns
    drops = ["__gap1", "__streak_id"]
    keep = [c for c in lf_out.collect_schema().names() if c not in drops]
    lf_out = lf_out.select(keep)
    return lf_out


# =========================
# Stage configurations
# =========================
@dataclass
class StageA:
    tail_lags: Sequence[int]
    tail_diffs: Sequence[int]
    rolling_windows: Optional[Sequence[int]]
    is_sorted: bool = False
    cast_f32: bool = True


@dataclass
class StageB:
    lags: Sequence[int]
    stats_rep_cols: Optional[Sequence[str]] = None
    add_prev1_multirep: bool = True
    batch_size: int = 5
    is_sorted: bool = False
    cast_f32: bool = True


@dataclass
class StageC:
    lags: Optional[Iterable[int]] = None
    ret_periods: Optional[Iterable[int]] = None
    diff_periods: Optional[Iterable[int]] = None
    rz_windows: Optional[Iterable[int]] = None
    ewm_spans: Optional[Iterable[int]] = None
    cs_cols: Optional[Sequence[str]] = None
    keep_rmean_rstd: bool = True
    batch_size: int = 10
    is_sorted: bool = False
    cast_f32: bool = True


# =========================
# Orchestrator
# =========================
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
    write_date_between: tuple[int, int] | None = None,
):
    g_symbol, g_date, g_time = keys

    def _save(lf_out: pl.LazyFrame, path: str):
        if write_date_between is None:
            raise ValueError("write_date_between must be specified to avoid date overlap")
        lo, hi = write_date_between

        sk = [g_date, g_time, g_symbol]
        df = (
            lf_out.filter(pl.col(g_date).is_between(lo, hi)).sort(sk).collect()
        )
        with fs.open(path, "wb") as f:
            df.write_parquet(f, compression="zstd")
        if cfg.get("debug", {}).get("check_time_monotone", True):
            assert_time_monotone(path, date_col=g_date, time_col=g_time)

    # Stage A
    if A is not None:
        lf_resp = lf_base.select([*keys, *rep_cols])
        lf_a_full = fe_resp_daily(
            lf_resp,
            keys=tuple(keys),
            rep_cols=rep_cols,
            is_sorted=A.is_sorted,
            cast_f32=A.cast_f32,
            tail_lags=A.tail_lags,
            tail_diffs=A.tail_diffs,
            rolling_windows=A.rolling_windows,
        )
        drop = set(keys) | set(rep_cols)
        a_cols = [c for c in lf_a_full.collect_schema().names() if c not in drop]
        _save(lf_a_full.select([*keys, *a_cols]), f"{out_dir}/stage_a.parquet")

    # Stage B
    if B is not None:
        lf_resp = lf_base.select([*keys, *rep_cols])
        lf_b_full = fe_resp_same_tick_xday(
            lf_resp,
            keys=tuple(keys),
            rep_cols=rep_cols,
            lags=B.lags,
            is_sorted=B.is_sorted,
            cast_f32=B.cast_f32,
            stats_rep_cols=B.stats_rep_cols,
            add_prev1_multirep=B.add_prev1_multirep,
            batch_size=B.batch_size,
        )
        drop = set(keys) | set(rep_cols)
        b_cols = [c for c in lf_b_full.collect_schema().names() if c not in drop]
        _save(lf_b_full.select([*keys, *b_cols]), f"{out_dir}/stage_b.parquet")

    # Stage C (each op outputs its own file)
    if C is not None:

        def _do_op(op_name: str, **op_flags):
            lf_src = lf_base.select([*keys, *feature_cols])
            lf_c = fe_feat_history(
                lf=lf_src,
                keys=tuple(keys),
                feature_cols=feature_cols,
                is_sorted=C.is_sorted,
                cast_f32=C.cast_f32,
                batch_size=C.batch_size,
                lags=op_flags.get("lags"),
                ret_periods=op_flags.get("ret_periods"),
                diff_periods=op_flags.get("diff_periods"),
                rz_windows=op_flags.get("rz_windows"),
                ewm_spans=op_flags.get("ewm_spans"),
                keep_rmean_rstd=C.keep_rmean_rstd,
                cs_cols=op_flags.get("cs_cols"),
            ).drop(feature_cols)  # only derived columns
            cols = [c for c in lf_c.collect_schema().names() if c not in keys]
            _save(lf_c.select([*keys, *cols]), f"{out_dir}/stage_c_{op_name}.parquet")

        if C.lags:
            _do_op("lags", lags=C.lags)
        if C.ret_periods:
            _do_op("ret", ret_periods=C.ret_periods)
        if C.diff_periods:
            _do_op("diff", diff_periods=C.diff_periods)
        if C.rz_windows:
            _do_op("rz", rz_windows=C.rz_windows)
        if C.ewm_spans:
            _do_op("ewm", ewm_spans=C.ewm_spans)
        if C.cs_cols:
            _do_op("csrank", cs_cols=C.cs_cols)
