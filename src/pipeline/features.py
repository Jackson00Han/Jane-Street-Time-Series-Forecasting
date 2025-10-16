# pipeline/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable, Optional, Tuple
import polars as pl
import numpy as np
from pipeline.io import cfg, fs, storage_options, P
from pipeline.validate import assert_time_monotone


# 特征工程

# A：响应列的“上一日尾部/日度摘要”
# fe_resp_daily 产出列命名与示例（对每个 rep 列分别生成同名模式）
#
# 设：
#   rep_cols = ["responder_0", "responder_6"]
#   tail_lags = [1, 2, 10]
#   tail_diffs = [1, 5]
#   rolling_windows = [3, 7]
#
# 聚合当日(日频)得到的“当日相对”列（先在 (symbol,date) 上聚合，再历史化）：
#   {r}_prev_tail_lag{L}              # 当日按 time 排序后倒数第 L 个值（L∈tail_lags ∪ {1, K+1}）
#   {r}_prevday_close                 # 当日收尾（= prev_tail_lag1）
#   {r}_prevday_mean                  # 当日均值
#   {r}_prevday_std                   # 当日标准差（ddof=1）
#   {r}_prev_tail_d{K}                # 当日尾部差分：lag1 - lag(K+1)，K∈tail_diffs
#   {r}_prev2day_close                # 前一日的收尾（= prevday_close.shift(1) over symbol）
#   {r}_overnight_gap                 # 当日收尾 - 前一日收尾
#   {r}_prevday_close_minus_mean      # 当日收尾 - 当日均值
#   {r}_close_roll{W}_mean            # 基于当日收尾的日度滚动均值，W∈rolling_windows
#   {r}_close_roll{W}_std             # 基于当日收尾的日度滚动标准差（ddof=1）
#
# 然后将上述“当日相对”的列统一转换为“对 d 生效的历史值”（TTL）：
#   - 若 prev_soft_days is None：总取“最近一次历史非空”的值
#   - 否则：仅当 (d - 最近一次非空日) ≤ prev_soft_days 才取，否则为 null
#   - 注意：转换后列名不变，但语义已是“历史值广播到 d 的所有 tick”
#
# 具体示例（r="responder_0"）在上述参数下，会出现：
#   responder_0_prev_tail_lag1
#   responder_0_prev_tail_lag2
#   responder_0_prev_tail_lag10
#   responder_0_prevday_close
#   responder_0_prevday_mean
#   responder_0_prevday_std
#   responder_0_prev_tail_d1
#   responder_0_prev_tail_d5
#   responder_0_prev2day_close
#   responder_0_overnight_gap
#   responder_0_prevday_close_minus_mean
#   responder_0_close_roll3_mean
#   responder_0_close_roll3_std
#   responder_0_close_roll7_mean
#   responder_0_close_roll7_std
#
# 同样的列会对 "responder_6" 生成一遍。

import polars as pl
from typing import Tuple, Sequence, Optional

def fe_resp_daily(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str, str, str] = ("symbol_id","date_id","time_id"),
    rep_cols: Sequence[str],
    is_sorted: bool = False,
    cast_f32: bool = True,
    tail_lags: Sequence[int] = (1,),
    tail_diffs: Sequence[int] = (1,),
    rolling_windows: Sequence[int] | None = (3,),
) -> pl.LazyFrame:
    """
    日级特征工程（最小必要版）：
      1) 对每个 (symbol, date) 统计：当日尾部倒序值、日收/均/方差，派生 dK、overnight、跨日 rolling；
      2) 将“当日量”转为“对 d 生效”（严格相邻自然日的 shift(1)）；
      3) 按 (symbol, date) 左连回 tick 级。
    """
    g_symbol, g_date, g_time = keys

    # 0) 输入排序（保障后续 over/rolling 等时序操作正确）
    if not is_sorted:
        lf = lf.sort([g_symbol, g_date, g_time])

    # 1) 当日聚合（尾部倒序 + 当日统计）
    need_L = sorted(set(tail_lags) | {k + 1 for k in tail_diffs} | {1})
    agg_exprs: list[pl.Expr] = []
    for r in rep_cols:
        # 当日内按 time 排序后的倒数第 L（长度不足 L → null）
        for L in need_L:
            agg_exprs.append(
                pl.when(pl.len() >= L)
                  .then(pl.col(r).sort_by(pl.col(g_time)).tail(L).first())
                  .otherwise(None)
                  .alias(f"{r}_prev_tail_lag{L}")
            )
        # 显式当日统计
        agg_exprs += [
            pl.col(r).sort_by(pl.col(g_time)).last().alias(f"{r}_prevday_close"),
            pl.col(r).mean().alias(f"{r}_prevday_mean"),
            pl.col(r).std(ddof=1).alias(f"{r}_prevday_std"),
        ]

    daily = (
        lf.group_by([g_symbol, g_date])
          .agg(agg_exprs)
          .sort([g_symbol, g_date])   # 保障 over(g_symbol) 的时序正确
    )

################************************************################
    # 添加两个关键列，把“是否相邻自然日”和“连续区间”显式化, 若想放宽条件，把gap1 调成为自己可接受的区间即可，如 gap1 <=5, 并修改下面的相应几个位置
    prev_day = pl.col(g_date).shift(1).over(g_symbol)
    gap1 = (pl.col(g_date) - prev_day).cast(pl.Int32)

    # 断档：gap != 1 的地方开新段；用累计和得到“连续段 id”
    streak_id = pl.when(gap1.fill_null(1) != 1).then(1).otherwise(0).cum_sum().over(g_symbol).alias("__streak_id")
    daily = daily.with_columns([gap1.alias("__gap1"), streak_id])
################************************************################   
    
    # 2) 当日派生：dK（last - 倒数第 K+1）
    daily = daily.with_columns([
        (pl.col(f"{r}_prev_tail_lag1") - pl.col(f"{r}_prev_tail_lag{K+1}")).alias(f"{r}_prev_tail_d{K}")
        for r in rep_cols for K in tail_diffs
    ])

    # 3) 跨日衍生：prev2day / overnight / close-mean

    daily = (
        daily.with_columns([
            # 只有 gap==1 才认“昨天”
            pl.when(pl.col("__gap1") == 1)
            .then(pl.col(f"{r}_prevday_close").shift(1).over(g_symbol))
            .otherwise(None)
            .alias(f"{r}_prev2day_close")
            for r in rep_cols
        ])
        .with_columns(
            [
                (pl.col(f"{r}_prevday_close") - pl.col(f"{r}_prevday_mean")).alias(f"{r}_prevday_close_minus_mean")
                for r in rep_cols
            ] + [
                # 如果不是严格“昨天”，overnight 置空
                (pl.col(f"{r}_prevday_close") - pl.col(f"{r}_prev2day_close")).alias(f"{r}_overnight_gap")
                for r in rep_cols
            ]
        )
    )


    # 4) 跨日 rolling（以当日收盘 close 为基）——用分段累加+差分，避免 rolling_*().over(...)
    if rolling_windows:
        wins = sorted({int(w) for w in rolling_windows if int(w) > 1})

        # 在每个 (symbol, __streak_id) 段内生成递增行号 __rn（1..段长）
        daily = daily.with_columns(
            pl.int_range(1, pl.len() + 1).over([g_symbol, "__streak_id"]).alias("__rn")
        )

        # 为每个 responder 预先构造分段累计和与平方累计和
        prep_exprs = []
        for r in rep_cols:
            base = pl.col(f"{r}_prevday_close")
            prep_exprs += [
                base.cum_sum().over([g_symbol, "__streak_id"]).alias(f"__cs_{r}"),
                (base * base).cum_sum().over([g_symbol, "__streak_id"]).alias(f"__cs2_{r}"),
            ]
        daily = daily.with_columns(prep_exprs)

        # 基于累计和做窗口差分，得到 mean/std（ddof=1；首 w-1 行使用更小样本数）
        roll_exprs = []
        rn = pl.col("__rn")
        for r in rep_cols:
            cs  = pl.col(f"__cs_{r}")
            cs2 = pl.col(f"__cs2_{r}")
            for w in wins:
                n = pl.when(rn >= w).then(pl.lit(w)).otherwise(rn).cast(pl.Float32)
                sum_w  = (cs  - cs.shift(w).over([g_symbol, "__streak_id"]))
                sum2_w = (cs2 - cs2.shift(w).over([g_symbol, "__streak_id"]))
                mean_w = (sum_w / n).alias(f"{r}_close_roll{w}_mean")
                var_w  = (sum2_w - (sum_w * sum_w) / n) / (n - 1.0)
                std_w  = pl.when(n >= 2.0).then(var_w.clip(0.0, None).sqrt()).otherwise(None) \
                        .alias(f"{r}_close_roll{w}_std")
                # 等价写法：var_w.clip(lower_bound=0.0)

                roll_exprs += [mean_w, std_w]

        daily = daily.with_columns(roll_exprs).drop(
            ["__rn", *[f"__cs_{r}" for r in rep_cols], *[f"__cs2_{r}" for r in rep_cols]]
        )


    # 5) 整体 shift(1)
    all_cols = set(daily.collect_schema().names())
    prev_cols = [
        c for c in all_cols
        if c not in (g_symbol, g_date, "__gap1", "__streak_id")
    ]

    hist_exprs = []
    for c in prev_cols:
        prev_val = pl.col(c).shift(1).over(g_symbol)
        resolved = pl.when(pl.col("__gap1") == 1).then(prev_val).otherwise(None)
        if cast_f32:
            resolved = resolved.cast(pl.Float32)
        hist_exprs.append(resolved.alias(c))
        
    daily_prev = daily.with_columns(hist_exprs).drop(["__gap1", "__streak_id"])
    
    
    # 6) 回拼 tick 级并固定顺序
    out = (
        lf.join(daily_prev, on=[g_symbol, g_date], how="left")
          .sort([g_symbol, g_date, g_time])
    )
    return out




# B：同 time_id 跨日的 prev{k} + 统计
def fe_resp_same_tick_xday(
    lf: pl.LazyFrame,
    *,
    keys: Tuple[str, str, str] = ("symbol_id", "date_id", "time_id"),
    rep_cols: Sequence[str],
    lags: Sequence[int],                      # 必填，例如 [1,2,3,4,5,6,7,14,21,30]
    is_sorted: bool = False,
    cast_f32: bool = True,
    stats_rep_cols: Optional[Sequence[str]] = None,
    add_prev1_multirep: bool = True,
    batch_size: int = 5,
) -> pl.LazyFrame:

    g_symbol, g_date, g_time = keys
    if not lags:
        raise ValueError("`lags` 不能为空")
    use_lags = sorted({int(x) for x in lags if int(x) > 0})
    if not use_lags:
        raise ValueError("`lags` 必须包含正整数")

    if stats_rep_cols is None:
        stats_rep_cols = list(rep_cols)

    # 1) 排序，确保因果正确：(symbol, time, date)
    if not is_sorted:
        lf = lf.sort([g_symbol, g_time, g_date])

    def _chunks(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i + k]

    lf_cur = lf

    # 2) 生成严格 prev{k}
    for batch in _chunks(list(rep_cols), batch_size):
        exprs = []
        for r in batch:
            for k in use_lags:
                val_k = pl.col(r).shift(k).over([g_symbol, g_time])
                day_k = pl.col(g_date).shift(k).over([g_symbol, g_time])
                gap_k = (pl.col(g_date) - day_k).cast(pl.Int32)  # 当前日 - 滞后日
                out = pl.when(gap_k.is_not_null() & (gap_k == k)).then(val_k).otherwise(None)
                if cast_f32:
                    out = out.cast(pl.Float32)
                exprs.append(out.alias(f"{r}_same_t_prev{k}"))
        lf_cur = lf_cur.with_columns(exprs)

    # 3) mean/std（基于 use_lags）
    L = len(use_lags)
    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in use_lags]
            vals = pl.concat_list([pl.col(c) for c in cols]).list.drop_nulls()
            m = vals.list.mean()
            s = vals.list.std(ddof=1)
            if cast_f32:
                m = m.cast(pl.Float32); s = s.cast(pl.Float32)
            exprs += [
                m.alias(f"{r}_same_t_last{L}_mean"),
                s.alias(f"{r}_same_t_last{L}_std"),
            ]
        lf_cur = lf_cur.with_columns(exprs)

    # 4) slope：最近权重最大（长度与 lags 一致）
    x = np.arange(L, 0, -1, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    x_lits = [pl.lit(float(v)) for v in x]

    for batch in _chunks([r for r in stats_rep_cols if r in rep_cols], batch_size):
        exprs = []
        for r in batch:
            cols = [f"{r}_same_t_prev{k}" for k in use_lags]
            mean_ref = pl.col(f"{r}_same_t_last{L}_mean")
            std_ref  = pl.col(f"{r}_same_t_last{L}_std")
            terms = [((pl.col(c) - mean_ref) / (std_ref + 1e-9)) * x_lits[i]
                     for i, c in enumerate(cols)]
            # null → 0，避免 sum_horizontal 传播 null
            terms = [pl.when(pl.col(c).is_not_null() & mean_ref.is_not_null() & std_ref.is_not_null())
                       .then(t).otherwise(pl.lit(0.0))
                     for t, c in zip(terms, cols)]
            n_eff = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in cols]).cast(pl.Float32)
            den   = pl.when(n_eff > 0).then(n_eff).otherwise(pl.lit(1.0))
            slope = pl.sum_horizontal(terms) / den
            if cast_f32:
                slope = slope.cast(pl.Float32)
            exprs.append(slope.alias(f"{r}_same_t_last{L}_slope"))
        lf_cur = lf_cur.with_columns(exprs)

    # 5) 跨 responder 的 prev1 统计（仅当 1 在 lags 中）
    if add_prev1_multirep and len(rep_cols) > 0 and (1 in use_lags):
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

    # 6) 输出仍保持 (symbol, date, time) 升序
    lf_cur = lf_cur.sort([g_symbol, g_date, g_time])
    return lf_cur



# C 系列：

# fe_feat_history（Stage C）生成列：命名、语义与示例
#
# 粒度与排序
#   - 以 by_grp = [symbol_id] 做分组，按 (symbol_id, date_id, time_id) 的时间顺序计算；
#   - 因此 C 系列的 lag/ret/diff 都是 **tick 级滞后（按行数）**，不是“天”。(L=967 大致≈ 1 天的 ticks)
#
# prev_soft_days 的作用（C 中）
#   - 对于 __lagL / __retK / __diffK / r-z / ewm 都会在“取到的上一条（或第 L 条）”上附加一个
#     “新鲜度上限”掩码：gap = (当前 date_id - 该历史值的 date_id)
#     * prev_soft_days is None  -> 不限制新鲜度（只要能取到第 L 条历史就保留）【常用于 tick 级 lag】
#     * prev_soft_days = D(int) -> 仅当 0 < gap ≤ D 才保留，否则置 null（用于希望只接受“最近 D 天内”的 tick 历史）
#
# 产出类别与命名
# 1) Tick 级滞后（可选）
#    {c}__lag{L}
#    - 第 L 条历史 tick 的取值；若设置 prev_soft_days=D，会额外要求该历史 tick 的日期新鲜度 gap ≤ D。
#
# 2) “收益率”式派生（可选）
#    {c}__ret{K}
#    - 定义为 cur/prev - 1，其中 prev 为第 K 条历史（若 K ∈ lags 列表则复用 {c}__lagK 的结果，已含掩码）。
#    - 若 prev 为 0 或 null，结果为 null。
#
# 3) 差分（可选）
#    {c}__diff{K} = cur - prevK
#    - prevK 取法与 ret 相同（若 K 在 lags 中则直接复用 {c}__lagK）。
#
# 4) Rolling r-z（可选）
#    {c}__rmean{W}, {c}__rstd{W}, {c}__rz{W}
#    - 以 t-1 的基准（等价于 lag1）为序列，做窗口为 W 的滚动 mean/std，并计算 r-z = (base - mean) / (std+eps)。
#    - keep_rmean_rstd=False 时仅输出 {c}__rz{W}。
#
# 5) EWM（可选）
#    {c}__ewm{S}
#    - 以 t-1 的基准序列做指数加权均值（span=S, adjust=False, ignore_nulls=True）。
#
# 6) 截面统计（可选，cs_cols 非空时）
#    {c}__cs_z, {c}__csrank
#    - 对同一 (date_id, time_id) 截面，基于该列的 t-1 值做截面 z-score 与百分位 rank∈[0,1]（n=1 -> 0.5）。
#
# 重要实现细节
#   - 若已在 C1 产出 {c}__lag1，则 r-z 与 ewm 会优先使用它作为 t-1 基准；否则内部自行构造 shift(1)。
#   - r-z / ewm 的“t-1 基准”也受 prev_soft_days 的新鲜度掩码影响（若设置了 D）。
#   - 所有 rolling/std 统一 ddof=1；数值稳定性使用 eps 防 0 除。
#
# 参数建议（常见）
#   lags:        [1,2,3,5,7,10,14,20,28,40,50,60,80,100]    # tick 级，覆盖短中期
#   ret_periods: [1,5,10,20,50]                             # 与 lags 对齐的若干档
#   diff_periods:[1,5,10,20,50]
#   rz_windows:  [5,10,20,60]
#   ewm_spans:   [5,10,20,60]
#   prev_soft_days: None  # tick 级 lag 通常不做“天级新鲜度”限制；若要限制，设一个合理天数如 3/5/10
#
# 示例（feature_cols 中含 ["feature_00","feature_07"]，选择部分档位）：
#   feature_00__lag1,  feature_00__lag7,  feature_00__lag50
#   feature_00__ret1,  feature_00__ret10, feature_00__diff5
#   feature_00__rmean10, feature_00__rstd10, feature_00__rz10
#   feature_00__ewm20
#   feature_07__lag1, feature_07__diff1, feature_07__rz60, feature_07__ewm5
#   （若 cs_cols 指定了 feature_00）→ feature_00__cs_z, feature_00__csrank
#
# 额外提示
#   - fe_pad_days 应至少覆盖你在 C 中“最长需要的天级新鲜度”与“用于 r-z/ewm 的有效历史”，
#     但由于 C 的 lag 是 tick 级，通常几十天的 pad 已很充裕。
#   - 大量滞后档位会迅速膨胀列数，建议先多给，再用特征筛选（FI、非空率、方差）做裁剪。

import polars as pl
from typing import Tuple, Sequence, Optional, Iterable

def fe_feat_history(
    *,
    lf: pl.LazyFrame,
    keys: Tuple[str,str,str] = ("symbol_id","date_id","time_id"),
    feature_cols: Sequence[str],
    is_sorted: bool = False,
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
    """
    协变量（feature0..N）的时序衍生（与 A/B 同风格，且 rolling/截面用 t-0）：
      - 严格自然日 k 日滞后（__lag{k}）：仅当 (date_now - date_shifted) == k 才保留
      - returns / diffs 基于“严格 lag”
      - rolling / EWM 使用 **t-0 当期值**，并在“连续段”内计算（断档处重置）
      - 截面 z / rank 使用 **t-0 当期值**，按 (date, time) 横截面计算
    注意：date_id 应为“自然日序号”（相邻自然日差为 1），不是 YYYYMMDD 原样数。
    """

    g_sym, g_date, g_time = keys
    by_grp = [g_sym]            # 组内：沿 date 的时序运算
    by_cs  = [g_date, g_time]   # 截面：同一时点 (date, time)

    # --- 校验 + 排序 ---
    need_cols = [*keys, *feature_cols]
    schema = lf.collect_schema().names()
    miss = [c for c in need_cols if c not in schema]
    if miss:
        raise KeyError(f"Columns not found: {miss}")

    lf_out = lf.select(need_cols)
    if not is_sorted:
        lf_out = lf_out.sort(list(keys))

    # 小工具
    def _chunks(lst, k):
        for i in range(0, len(lst), k):
            yield lst[i:i+k]

    def _clean_pos_sorted_unique(x):
        if not x:
            return tuple()
        return tuple(sorted({int(v) for v in x if int(v) >= 1}))

    LAGS   = _clean_pos_sorted_unique(lags)
    K_RET  = _clean_pos_sorted_unique(ret_periods)
    K_DIFF = _clean_pos_sorted_unique(diff_periods)
    RZW    = _clean_pos_sorted_unique(rz_windows)
    SPANS  = _clean_pos_sorted_unique(ewm_spans)

    # === 连续性分段：严格自然日（gap==1 连续；gap!=1 断档）===
    prev_day  = pl.col(g_date).shift(1).over(by_grp)
    gap1      = (pl.col(g_date) - prev_day).cast(pl.Int32).alias("__gap1")
    streak_id = (
        pl.when(pl.col("__gap1").fill_null(1) != 1).then(1).otherwise(0)
          .cum_sum().over(by_grp)
          .alias("__streak_id")
    )
    lf_out = lf_out.with_columns([gap1, streak_id])

    # C1 严格自然日 lag：只接受 gap_k == k
    if LAGS:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for L in LAGS:
                last_date_L = pl.col(g_date).shift(L).over(by_grp)
                gap_L = (pl.col(g_date) - last_date_L).cast(pl.Int32)
                keep_L = gap_L.is_not_null() & (gap_L == L)
                for c in batch:
                    e = pl.when(keep_L).then(pl.col(c).shift(L).over(by_grp)).otherwise(None)
                    if cast_f32:
                        e = e.cast(pl.Float32)
                    exprs.append(e.alias(f"{c}__lag{L}"))
            lf_out = lf_out.with_columns(exprs)

    # C2 returns：基于严格 lag（分母近 0 置空）
    if K_RET:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                cur = pl.col(c)  # t-0
                for k in K_RET:
                    if k in LAGS:
                        prev = pl.col(f"{c}__lag{k}")   # 已是严格 lag
                    else:
                        last_date_k = pl.col(g_date).shift(k).over(by_grp)
                        gap_k = (pl.col(g_date) - last_date_k).cast(pl.Int32)
                        keep_k = gap_k.is_not_null() & (gap_k == k)
                        prev = pl.when(keep_k).then(pl.col(c).shift(k).over(by_grp)).otherwise(None)
                    ret = pl.when(prev.is_not_null() & (prev.abs() > 1e-12)).then(cur / prev - 1.0).otherwise(None)
                    if cast_f32:
                        ret = ret.cast(pl.Float32)
                    exprs.append(ret.alias(f"{c}__ret{k}"))
            lf_out = lf_out.with_columns(exprs)

    # C3 diffs：基于严格 lag
    if K_DIFF:
        for batch in _chunks(feature_cols, batch_size):
            exprs = []
            for c in batch:
                cur = pl.col(c)  # t-0
                for k in K_DIFF:
                    if k in LAGS:
                        prevk = pl.col(f"{c}__lag{k}")  # 严格 lag
                    else:
                        last_date_k = pl.col(g_date).shift(k).over(by_grp)
                        gap_k = (pl.col(g_date) - last_date_k).cast(pl.Int32)
                        keep_k = gap_k.is_not_null() & (gap_k == k)
                        prevk = pl.when(keep_k).then(pl.col(c).shift(k).over(by_grp)).otherwise(None)
                    d = pl.when(prevk.is_not_null()).then(cur - prevk).otherwise(None)
                    if cast_f32:
                        d = d.cast(pl.Float32)
                    exprs.append(d.alias(f"{c}__diff{k}"))
            lf_out = lf_out.with_columns(exprs)

    # C4 rolling r-z：段内滚动，**t-0 当期值**
    if RZW:
        for batch in _chunks(feature_cols, batch_size):
            roll_exprs = []
            for c in batch:
                base = pl.col(c)  # t-0
                for w in RZW:
                    m  = base.rolling_mean(window_size=w, min_samples=1).over([*by_grp, "__streak_id"])
                    s  = base.rolling_std(window_size=w, ddof=1, min_samples=2).over([*by_grp, "__streak_id"])
                    rz = (base - m) / (s.fill_null(0.0) + 1e-9)
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

    # C5 EWM：段内计算，**t-0 当期值**
    if SPANS:
        for batch in _chunks(feature_cols, batch_size):
            ewm_exprs = []
            for c in batch:
                base = pl.col(c)  # t-0
                for s in SPANS:
                    ema = base.ewm_mean(span=int(s), adjust=False, ignore_nulls=True) \
                             .over([*by_grp, "__streak_id"])
                    if cast_f32:
                        ema = ema.cast(pl.Float32)
                    ewm_exprs.append(ema.alias(f"{c}__ewm{s}"))
            lf_out = lf_out.with_columns(ewm_exprs)

    # C6 截面 z/rank：**t-0 当期值**
    if cs_cols:
        cs_cols = [c for c in cs_cols if c in feature_cols]
        if cs_cols:
            cs_exprs = []
            for c in cs_cols:
                base = pl.col(c)  # t-0
                n_valid = base.is_not_null().cast(pl.Int32).sum().over(by_cs)
                mu      = base.mean().over(by_cs)
                sig     = base.std(ddof=1).over(by_cs)
                z = ((base - mu) / (sig.fill_null(0.0) + 1e-9)) \
                        .cast(pl.Float32 if cast_f32 else pl.Float64)
                rank_raw = base.rank(method="average").over(by_cs)
                csrank = pl.when(base.is_null()).then(None).otherwise(
                    pl.when(n_valid > 1)
                      .then((rank_raw - 0.5) / n_valid.cast(pl.Float32))
                      .otherwise(pl.lit(0.5))
                ).cast(pl.Float32 if cast_f32 else pl.Float64)
                cs_exprs += [z.alias(f"{c}__cs_z"), csrank.alias(f"{c}__csrank")]
            lf_out = lf_out.with_columns(cs_exprs)

    # 清理连续性临时列
    lf_out = lf_out.drop(["__gap1", "__streak_id"])
    return lf_out

from dataclasses import dataclass
from typing import Sequence, Optional, Iterable
import polars as pl

# -------- Stages configs --------

@dataclass
class StageA:
    tail_lags: Sequence[int]
    tail_diffs: Sequence[int]
    rolling_windows: Optional[Sequence[int]]
    is_sorted: bool = False
    cast_f32: bool = True
    # REMOVED: prev_soft_days  已在函数A中废弃

@dataclass
class StageB:
    lags: Sequence[int]
    stats_rep_cols: Optional[Sequence[str]] = None
    add_prev1_multirep: bool = True
    batch_size: int = 5
    is_sorted: bool = False
    cast_f32: bool = True
    # REMOVED: prev_soft_days  函数B本来也不需要

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
    batch_size: int = 10          # CHANGED: 不再 Optional，避免传 None
    is_sorted: bool = False
    cast_f32: bool = True
    # REMOVED: prev_soft_days  函数C已切换到严格自然日 + t-0

# -------- Orchestrator --------

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
        df = (
            lf_out
            .filter(pl.col(g_date).is_between(lo, hi))
            .sort(sk)
            .collect()
        )
        with fs.open(path, "wb") as f:   # 依赖外部 fs (fsspec)
            df.write_parquet(f, compression="zstd")
        if cfg.get("debug", {}).get("check_time_monotone", True):  # 依赖外部 cfg/断言函数
            assert_time_monotone(path, date_col=g_date, time_col=g_time)

    # ---------- A ----------
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
        )  # REMOVED: prev_soft_days
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
            lags=B.lags,                        # ← 只用离散 lags（严格 d-k 已在函数里处理）
            is_sorted=B.is_sorted,
            cast_f32=B.cast_f32,
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
                cast_f32=C.cast_f32,
                batch_size=C.batch_size,
                lags=op_flags.get("lags"),
                ret_periods=op_flags.get("ret_periods"),
                diff_periods=op_flags.get("diff_periods"),
                rz_windows=op_flags.get("rz_windows"),
                ewm_spans=op_flags.get("ewm_spans"),
                keep_rmean_rstd=C.keep_rmean_rstd,
                cs_cols=op_flags.get("cs_cols"),
            ).drop(feature_cols)  # 输出只包含派生列
            cols = [c for c in lf_c.collect_schema().names() if c not in keys]
            _save(lf_c.select([*keys, *cols]), f"{out_dir}/stage_c_{op_name}.parquet")

        if C.lags:         _do_op("lags",   lags=C.lags)
        if C.ret_periods:  _do_op("ret",    ret_periods=C.ret_periods)
        if C.diff_periods: _do_op("diff",   diff_periods=C.diff_periods)
        if C.rz_windows:   _do_op("rz",     rz_windows=C.rz_windows)
        if C.ewm_spans:    _do_op("ewm",    ewm_spans=C.ewm_spans)
        if C.cs_cols:      _do_op("csrank", cs_cols=C.cs_cols)
