from collections.abc import Callable
from typing import Any
from typing import cast

import numpy as np
import pandas as pd

StatFunc = Callable[[pd.Series], int | float]
WeightedStatFunc = Callable[[pd.Series, pd.Series], float]


def _hmean(x: pd.Series) -> float:
    """Unweighted harmonic mean: n / sum(1/x), excluding NaN and zero values.

    Zero and NaN values are excluded before computing because:
    - 1/0 is undefined (would produce inf or division error)
    - NaN != 0 evaluates to True in pandas, so a bare `!= 0` mask would
      silently include NaN as if it were a valid observation, causing
      1/NaN to be treated as 0 by .sum() and producing a spurious
      divide-by-zero warning.
    NaN is therefore dropped explicitly via .notna() before zero-filtering.
    """
    x = x[x.notna()]
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    return float(len(x) / (1.0 / x).sum())


_BASE_STATS: dict[str, StatFunc] = {
    # ── Count statistics ────────────────────────────────────────────────────
    # N, COUNT and SIZE may all be used WITHOUT a measure column (bare count).
    "N": lambda x: x.count(),  # backward-compatible alias for COUNT
    "COUNT": lambda x: x.count(),  # like pandas Series.count(): counts
    # only NON-MISSING values of the measure
    "SIZE": lambda x: x.size,  # like Python len() / numpy .size:
    # counts ALL rows, including those where
    # the measure is missing/NaN
    # NMISS requires a measure column (there must be something to be missing)
    "NMISS": lambda x: x.isna().sum(),
    # ── Descriptive statistics (all require a measure column) ────────────────
    "SUM": lambda x: x.sum(),
    "MEAN": lambda x: x.mean(),
    "MIN": lambda x: x.min(),
    "MAX": lambda x: x.max(),
    "STD": lambda x: x.std(),  # sample std dev (VARDEF=DF, n-1)
    "STDERR": lambda x: float(cast(Any, x.sem())),
    "VAR": lambda x: x.var(),  # sample variance (VARDEF=DF, n-1)
    "MEDIAN": lambda x: x.median(),
    "P1": lambda x: x.quantile(0.01),
    "P5": lambda x: x.quantile(0.05),
    "P10": lambda x: x.quantile(0.10),
    "P25": lambda x: x.quantile(0.25),
    "P75": lambda x: x.quantile(0.75),
    "P90": lambda x: x.quantile(0.90),
    "P95": lambda x: x.quantile(0.95),
    "P99": lambda x: x.quantile(0.99),
    "QRANGE": lambda x: x.quantile(0.75) - x.quantile(0.25),
    "GMEAN": lambda x: np.exp(np.log(x[x > 0]).mean()) if (x > 0).any() else np.nan,
    "HMEAN": lambda x: _hmean(x),
}

# Each function signature: (values: pd.Series, weights: pd.Series) -> float.
#
# Key design decisions:
#
#   N, COUNT, SIZE, NMISS — NEVER weighted, always plain counts.
#     Even when weight= is given, these statistics count observations, not
#     weight-sums. N/COUNT count non-missing values; SIZE counts all rows;
#     NMISS counts missing values of the measure column. This is a deliberate
#     departure from SAS's convention (where N=sum-of-weights) to match the
#     Python/pandas convention that count() and len() are plain row counts.
#
#   Missing measure values (NaN in the measure column) — each weighted
#     function calls _drop_nan_x(x, w) first to remove rows where the measure
#     is NaN. This is essential: pandas .sum() silently skips NaN in the
#     numerator (x*w), but the weight of that row would still be included in
#     w.sum() as a denominator if not explicitly removed, producing an
#     inflated denominator and a too-small weighted mean.
#
#   Missing weight values — rows with NaN weights are excluded entirely
#     from all statistics (including N/COUNT/SIZE) before any computation
#     by the caller (_compute_series / _compute_all_series).
#
#   Negative weights — treated as 0 via _clean_weights(); the row is still
#     counted in N/COUNT/SIZE but contributes nothing to weighted sums.
#
#   Weighted variance formula — reliability-weights unbiased estimator:
#     Var_w = [sum(w) / (sum(w)^2 - sum(w^2))] * sum(w*(x - x̄_w)^2)
#     This reduces to the standard sample variance (n-1 divisor) when all
#     weights equal 1, since then sum(w)=n and sum(w)^2-sum(w^2)=n(n-1).
#
# Per the SAS documentation: a weight of 0 counts the observation in N but
# contributes nothing to weighted sums; a negative weight is treated as 0
# (still counted in N); a missing weight excludes the observation entirely.
# This cleaning step is applied once in _clean_weights() before any of the
# functions below are called.


def _drop_nan_x(
    x: pd.Series,
    w: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Drop rows where the measure value x is NaN, keeping x and w aligned.

    This must happen before any weighted-stat formula runs: pandas .sum()
    silently skips NaN in the NUMERATOR (x*w), but the weight itself is a
    valid number even when x is missing, so w.sum() as a DENOMINATOR would
    incorrectly include that row's weight unless x's NaN rows are dropped
    from w too. Without this, MEAN/VAR/GMEAN/HMEAN/etc. all undercount the
    denominator whenever the measure column has any missing values.
    """
    mask = x.notna()
    return x[mask], w[mask]


def _wmean(
    x: pd.Series,
    w: pd.Series,
) -> float:
    # Weighted arithmetic mean: x_bar_w = (sum w*x) / (sum w)
    x, w = _drop_nan_x(x, w)
    wsum = w.sum()
    return (x * w).sum() / wsum if wsum else np.nan


def _wvar(
    x: pd.Series,
    w: pd.Series,
) -> float:
    # Weighted variance, UNBIASED (reliability-weights) estimator:
    #   Var_w = [ (sum w) / ((sum w)^2 - sum w^2) ] * sum( w*(x - xbar_w)^2 )
    # This reduces to the familiar sum((x-xbar)^2)/(n-1) when all w_i = 1,
    # since then sum(w)=n and (sum w)^2 - sum(w^2) = n^2 - n = n(n-1).
    x, w = _drop_nan_x(x, w)
    wsum = w.sum()
    if not wsum:
        return np.nan
    denom = wsum**2 - (w**2).sum()
    if denom <= 0:
        # Degenerate case (e.g. a single nonzero-weight observation):
        # not enough effective degrees of freedom to estimate variance.
        return np.nan
    xbar = _wmean(x, w)
    numerator = (w * (x - xbar) ** 2).sum()
    return float((wsum / denom) * numerator)


def _wstd(
    x: pd.Series,
    w: pd.Series,
) -> float:
    return float(np.sqrt(_wvar(x, w)))


def _wstderr(
    x: pd.Series,
    w: pd.Series,
) -> float:
    # Standard error of the weighted mean: SD_w / sqrt(sum w)
    x, w = _drop_nan_x(x, w)
    v = _wvar(x, w)
    wsum = float(w.sum())
    if pd.notna(v) and wsum:
        return float(np.sqrt(v / wsum))

    return np.nan


def _wpercentile(
    x: pd.Series,
    w: pd.Series,
    q: float,
) -> float:
    """Weighted percentile via linear interpolation on the weighted ECDF."""
    x, w = _drop_nan_x(x, w)
    if len(x) == 0 or w.sum() == 0:
        return np.nan
    x_arr = x.to_numpy(dtype=float)
    w_arr = w.to_numpy(dtype=float)

    order = np.argsort(x_arr)
    xs = x_arr[order]
    ws = w_arr[order]
    cw = np.cumsum(ws)
    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff)
    idx = min(idx, len(xs) - 1)
    return float(xs[idx])


def _wgmean(
    x: pd.Series,
    w: pd.Series,
) -> float:
    # Weighted geometric mean: exp( (sum w*ln x) / (sum w) )
    x, w = _drop_nan_x(x, w)
    mask = x > 0
    if not mask.any():
        return np.nan
    xw, ww = x[mask], w[mask]
    wsum = float(ww.sum())
    if not wsum:
        return np.nan
    return float(np.exp((ww * np.log(xw)).sum() / wsum))


def _whmean(
    x: pd.Series,
    w: pd.Series,
) -> float:
    # Weighted harmonic mean: (sum w) / (sum w/x)
    x, w = _drop_nan_x(x, w)
    mask = x != 0
    if not mask.any():
        return np.nan
    xw, ww = x[mask], w[mask]
    wsum = float(ww.sum())
    if not wsum:
        return np.nan
    return float(wsum / (ww / xw).sum())


def _clean_weights(weights: pd.Series) -> pd.Series:
    """Apply SAS PROC TABULATE's WEIGHT statement rules to a raw weight column.

    - Missing weight -> NaN (caller must drop these rows entirely).
    - Negative weight -> treated as 0 (observation still counted in N).
    - Zero or positive weight -> unchanged.
    """
    cleaned = weights.copy()
    cleaned[cleaned < 0] = 0
    return cleaned


_WEIGHTED_STATS: dict[str, WeightedStatFunc] = {
    "N": lambda x, w: x.count(),  # alias for COUNT, NEVER weighted
    "COUNT": lambda x, w: x.count(),  # NEVER weighted - plain count of non-missing values
    "SIZE": lambda x, w: x.size,  # NEVER weighted - plain count of ALL rows
    "NMISS": lambda x, w: x.isna().sum(),  # also never weighted - plain count
    "SUM": lambda x, w: (x * w).sum(),
    "MEAN": _wmean,
    "MIN": lambda x, w: x.min(),
    "MAX": lambda x, w: x.max(),
    "STD": _wstd,
    "STDERR": _wstderr,
    "VAR": _wvar,
    "MEDIAN": lambda x, w: _wpercentile(x, w, 0.50),
    "P1": lambda x, w: _wpercentile(x, w, 0.01),
    "P5": lambda x, w: _wpercentile(x, w, 0.05),
    "P10": lambda x, w: _wpercentile(x, w, 0.10),
    "P25": lambda x, w: _wpercentile(x, w, 0.25),
    "P75": lambda x, w: _wpercentile(x, w, 0.75),
    "P90": lambda x, w: _wpercentile(x, w, 0.90),
    "P95": lambda x, w: _wpercentile(x, w, 0.95),
    "P99": lambda x, w: _wpercentile(x, w, 0.99),
    "QRANGE": lambda x, w: _wpercentile(x, w, 0.75) - _wpercentile(x, w, 0.25),
    "GMEAN": _wgmean,
    "HMEAN": _whmean,
}

_PERCENT_STATS = {
    "PCTN",
    "ROWPCTN",
    "COLPCTN",
    "PCTSUM",
    "ROWPCTSUM",
    "COLPCTSUM",
}

ALL_STATS = set(_BASE_STATS) | _PERCENT_STATS


# ---------------------------------------------------------------------------
# Shared aggregation helpers (used by _compute_series and _compute_all_series)
# ---------------------------------------------------------------------------


def _agg_group_weighted(
    data: pd.DataFrame,
    groups: list[str],
    var: str,
    weight: str,
    wfunc: WeightedStatFunc,
    dropna: bool,
) -> pd.Series:
    cols = [var, weight]

    def _apply(g: pd.DataFrame) -> float:
        return wfunc(g[var], g[weight])

    if groups:
        return data.groupby(groups, dropna=dropna)[cols].apply(_apply)
    return pd.Series({"__total__": _apply(data[cols])})


def _agg_group(
    data: pd.DataFrame,
    groups: list[str],
    func: Callable[[pd.Series], float],
    wfunc: WeightedStatFunc | None,
    var: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    """Aggregate `func`/`wfunc` over `groups`, or return a bare row count.

    Shared by _compute_series and _compute_all_series - their per-function
    `_agg` closures were textually identical aside from how `func` was
    supplied (parameter vs. an enclosing-scope variable), so both now use
    this single explicit-parameter version.
    """
    if var is None:
        # No measure variable (bare N/NMISS): ALWAYS a plain row count,
        # never weighted - even when weight= is supplied. Rows with a
        # missing weight were already excluded from `data` upstream, and
        # zero/negative weights are still counted (only their VALUE was
        # cleaned to 0), so a plain count here is exactly right.
        if groups:
            return data.groupby(groups, dropna=dropna).size().rename(None)
        return pd.Series({"__total__": len(data)})

    if weight is not None and wfunc is not None:
        return _agg_group_weighted(data, groups, var, weight, wfunc, dropna)

    if groups:
        return data.groupby(groups, dropna=dropna)[var].agg(func)
    return pd.Series({"__total__": func(data[var])})


def _grand_value(
    data: pd.DataFrame,
    func: Callable[[pd.Series], float],
    wfunc: Callable[[pd.Series, pd.Series], float] | None,
    var: str | None,
    weight: str | None,
) -> float:
    if var is not None:
        if weight is not None and wfunc is not None:
            return float(wfunc(data[var], data[weight]))
        return float(func(data[var]))
    # No measure variable: ALWAYS a plain row count (N is never weighted)
    return float(len(data))


def _require_var_for_base_stat(var: str | None, stat: str) -> None:
    if var is None and stat not in ("N", "COUNT", "SIZE"):
        raise ValueError(
            f"Statistic '{stat}' requires a measure variable. "
            f"Omit measure= only when using N, COUNT, or SIZE (all plain "
            f"row counts with no measure). NMISS specifically counts "
            f"missing values OF a measure column, so it always needs "
            f"one, e.g. income*NMISS."
        )


def _require_var_for_percent_stat(
    var: str | None, stat: str, count_based: bool
) -> None:
    if not count_based and var is None:
        raise ValueError(
            f"Statistic '{stat}' requires a measure variable. "
            f"Omit measure= only when using count statistics: N, COUNT, SIZE, "
            f"NMISS, PCTN, ROWPCTN, COLPCTN."
        )


def _compute_base_stat_series(
    data: pd.DataFrame,
    groups: list[str],
    var: str | None,
    stat: str,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    _require_var_for_base_stat(var, stat)
    func = _BASE_STATS[stat] if var is not None else (lambda x: x.count())
    wfunc = _WEIGHTED_STATS.get(stat) if var is not None else None
    return _agg_group(data, groups, func, wfunc, var, weight, dropna)


def _resolve_percent_raw_func(
    count_based: bool, var: str | None
) -> tuple[Callable[[pd.Series], float], WeightedStatFunc | None]:
    if count_based:
        raw_func = _BASE_STATS["N"] if var is not None else (lambda x: x.count())
        raw_wfunc = _WEIGHTED_STATS["N"] if var is not None else None
    else:
        raw_func = _BASE_STATS["SUM"]
        raw_wfunc = _WEIGHTED_STATS["SUM"]
    return raw_func, raw_wfunc


# ---------------------------------------------------------------------------
# _compute_series and its helpers
# ---------------------------------------------------------------------------


def _row_pct_series(
    series: pd.Series,
    grand: float,
    r_groups: list[str],
    data: pd.DataFrame,
    raw_func: Callable[[pd.Series], float],
    raw_wfunc: WeightedStatFunc | None,
    var: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    if not r_groups:
        return 100.0 * series / grand

    denom = _agg_group(data, r_groups, raw_func, raw_wfunc, var, weight, dropna)
    n_r = len(r_groups)

    def row_pct(val: int | float, idx: object) -> float:
        key = idx[:n_r] if isinstance(idx, tuple) else (idx,)
        key = key[0] if len(key) == 1 else key
        d = denom.get(key, np.nan)
        return 100.0 * val / d if d else np.nan

    return pd.Series(
        {idx: row_pct(val, idx) for idx, val in series.items()},
        name=series.name,
    )


def _col_pct_series(
    series: pd.Series,
    grand: float,
    r_groups: list[str],
    c_groups: list[str],
    data: pd.DataFrame,
    raw_func: Callable[[pd.Series], float],
    raw_wfunc: WeightedStatFunc | None,
    var: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    if not c_groups:
        return 100.0 * series / grand

    denom = _agg_group(data, c_groups, raw_func, raw_wfunc, var, weight, dropna)
    n_r = len(r_groups)

    def col_pct(val: int | float, idx: object) -> float:
        if isinstance(idx, tuple):
            key = idx[n_r : n_r + len(c_groups)]
        else:
            key = (idx,)
        key = key[0] if len(key) == 1 else key
        d = denom.get(key, np.nan)
        return 100.0 * val / d if d else np.nan

    return pd.Series(
        {idx: col_pct(val, idx) for idx, val in series.items()},
        name=series.name,
    )


def _compute_percent_stat(
    data: pd.DataFrame,
    groups: list[str],
    var: str | None,
    stat: str,
    r_groups: list[str],
    c_groups: list[str],
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    count_based = "N" in stat
    _require_var_for_percent_stat(var, stat, count_based)

    raw_func, raw_wfunc = _resolve_percent_raw_func(count_based, var)

    series = _agg_group(data, groups, raw_func, raw_wfunc, var, weight, dropna)
    grand = _grand_value(
        data,
        raw_func if var is not None else (lambda x: len(x)),
        raw_wfunc,
        var,
        weight,
    )

    if stat in ("PCTN", "PCTSUM"):
        return 100.0 * series / grand

    if stat in ("ROWPCTN", "ROWPCTSUM"):
        return _row_pct_series(
            series, grand, r_groups, data, raw_func, raw_wfunc, var, weight, dropna
        )

    if stat in ("COLPCTN", "COLPCTSUM"):
        return _col_pct_series(
            series,
            grand,
            r_groups,
            c_groups,
            data,
            raw_func,
            raw_wfunc,
            var,
            weight,
            dropna,
        )

    raise ValueError(f"Unknown statistic: {stat}")


def _compute_series(
    data: pd.DataFrame,
    all_groups: list[str],
    var: str | None,
    stat: str,
    r_groups: list[str],
    c_groups: list[str],
    missing: bool,
    weight: str | None = None,
) -> pd.Series:
    """Compute an aggregated Series for one row/column specification pair.

    This function handles cases where neither specification contains an
    ALL/TOTAL token. Both dimensions therefore represent ordinary groupby
    breakdowns rather than marginal totals.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered input data used for the aggregation.
    all_groups : list[str]
        Combined row and column grouping columns.
    var : str | None
        Measure column name, or None for count-only statistics.
    stat : str
        Statistic keyword, for example ``"MEAN"``, ``"COLPCTN"``, or ``"N"``.
    r_groups : list[str]
        Grouping columns in the row dimension.
    c_groups : list[str]
        Grouping columns in the column dimension.
    missing : bool
        Whether missing group values should be included.
    weight : str | None
        Optional weight column name.

    Returns
    -------
    pd.Series
        Aggregated values indexed by the grouping columns in ``all_groups``.

    Notes
    -----
    Percentage statistics such as ``PCTN``, ``PCTSUM``, ``ROWPCTN``,
    ``COLPCTN``, ``ROWPCTSUM``, and ``COLPCTSUM`` are routed through this
    function. Their denominators depend on the statistic:

    - ``PCTN`` and ``PCTSUM`` use the grand total.
    - ``ROWPCTN`` and ``ROWPCTSUM`` use row subtotals.
    - ``COLPCTN`` and ``COLPCTSUM`` use column subtotals.

    Custom denominators such as ``PCTN<...>`` and ``PCTSUM<...>`` are handled
    separately by ``_compute_custom_pct``.

    ``N``, ``COUNT``, ``SIZE``, and ``NMISS`` are never weighted, even when
    ``weight`` is provided.
    """
    dropna = not missing

    # When a weight column is active, missing weights exclude the
    # observation entirely (per SAS WEIGHT statement rules). Negative
    # weights are cleaned to 0 (still counted in N) inside _clean_weights.
    if weight is not None:
        data = data[data[weight].notna()].copy()
        data[weight] = _clean_weights(data[weight])

    if stat in _BASE_STATS:
        return _compute_base_stat_series(data, all_groups, var, stat, weight, dropna)

    return _compute_percent_stat(
        data, all_groups, var, stat, r_groups, c_groups, weight, dropna
    )


# ---------------------------------------------------------------------------
# _compute_all_series and its helpers
# ---------------------------------------------------------------------------


def _all_col_pct_series(
    series: pd.Series,
    grand: float,
    groups_to_keep: list[str],
    c_groups: list[str],
    data: pd.DataFrame,
    raw_func: Callable[[pd.Series], float],
    raw_wfunc: WeightedStatFunc | None,
    var: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    # Denominator = total within each column group.
    # When ALL is on rows, c_groups are the column breakdown groups.
    # Divide each cell by the total for its column group.
    if not c_groups:
        # No column groups — divide by overall grand total
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * series / grand

    denom = _agg_group(data, c_groups, raw_func, raw_wfunc, var, weight, dropna)
    n_r_ctx = len(groups_to_keep) - len(c_groups)

    def _col_denom(idx: object) -> float:
        # idx is from groups_to_keep = r_context + c_groups
        # c_groups part starts after r_context groups
        if isinstance(idx, tuple):
            key = idx[n_r_ctx:]
        else:
            key = (idx,)
        key = key[0] if len(key) == 1 else key
        return float(denom.get(key, np.nan))

    def _safe_pct(val: int | float, denom_val: float) -> float:
        if np.isnan(denom_val) or denom_val == 0:
            return np.nan
        return 100.0 * float(val) / denom_val

    return pd.Series(
        {idx: _safe_pct(val, _col_denom(idx)) for idx, val in series.items()},
        name=series.name,
    )


def _all_row_pct_series(
    series: pd.Series,
    grand: float,
    r_groups: list[str],
    data: pd.DataFrame,
    raw_func: Callable[[pd.Series], float],
    raw_wfunc: WeightedStatFunc | None,
    var: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    # Denominator = total within each row group.
    # When ALL is on cols, r_groups are the row breakdown groups.
    if not r_groups:
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * series / grand

    denom = _agg_group(data, r_groups, raw_func, raw_wfunc, var, weight, dropna)

    def _row_denom(idx: object) -> float:
        if isinstance(idx, tuple):
            key = idx[: len(r_groups)]
        else:
            key = (idx,)
        key = key[0] if len(key) == 1 else key
        return float(denom.get(key, np.nan))

    def _safe_row_pct(val: float, denom_val: int | float) -> float:
        if np.isnan(denom_val) or denom_val == 0:
            return np.nan
        return 100.0 * val / denom_val

    return pd.Series(
        {idx: _safe_row_pct(val, _row_denom(idx)) for idx, val in series.items()},
        name=series.name,
    )


def _compute_all_base_stat(
    data: pd.DataFrame,
    groups_to_keep: list[str],
    var: str | None,
    stat: str,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    return _compute_base_stat_series(data, groups_to_keep, var, stat, weight, dropna)


def _all_series_grand(
    data: pd.DataFrame,
    var: str | None,
    weight: str | None,
    count_based: bool,
    raw_wfunc: WeightedStatFunc | None,
) -> float:
    if count_based:
        # N/NMISS-based grand total is ALWAYS a plain row count, never
        # weighted - even when weight= is supplied and var is set.
        return float(data[var].count() if var is not None else len(data))
    if weight is not None and raw_wfunc is not None:
        return float(raw_wfunc(data[var], data[weight]))
    return float(data[var].sum())


def _compute_all_percent_stat(
    data: pd.DataFrame,
    groups_to_keep: list[str],
    var: str | None,
    stat: str,
    r_groups: list[str],
    c_groups: list[str],
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    count_based = "N" in stat
    _require_var_for_percent_stat(var, stat, count_based)

    raw_func, raw_wfunc = _resolve_percent_raw_func(count_based, var)
    grand = _all_series_grand(data, var, weight, count_based, raw_wfunc)

    series = _agg_group(data, groups_to_keep, raw_func, raw_wfunc, var, weight, dropna)

    if stat in ("COLPCTN", "COLPCTSUM"):
        return _all_col_pct_series(
            series,
            grand,
            groups_to_keep,
            c_groups,
            data,
            raw_func,
            raw_wfunc,
            var,
            weight,
            dropna,
        )

    if stat in ("ROWPCTN", "ROWPCTSUM"):
        return _all_row_pct_series(
            series, grand, r_groups, data, raw_func, raw_wfunc, var, weight, dropna
        )

    # PCTN / PCTSUM: always use overall grand total
    return 100.0 * series / grand


def _compute_all_series(
    data: pd.DataFrame,
    groups_to_keep: list[str],
    var: str | None,
    stat: str,
    missing: bool,
    r_groups: list[str] | None = None,
    c_groups: list[str] | None = None,
    weight: str | None = None,
) -> pd.Series:
    """Compute an aggregated Series involving an ALL/TOTAL margin.

    This handles the three ALL cases:
      - ALL on rows only  (has_all_r=True,  has_all_c=False)
      - ALL on cols only  (has_all_r=False, has_all_c=True)
      - ALL on both dims  (has_all_r=True,  has_all_c=True)

    In each case, groups_to_keep is the subset of groupby columns that
    still vary (i.e. the groups from the OTHER dimension that provide the
    cross-breakdown), and the ALL dimension is collapsed.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered input data.

    groups_to_keep : list[str]
        Groupby columns to aggregate over on the non-ALL side.

    var : str | None
        Measure column name, or None for count-only statistics.

    stat : str
        Statistic keyword.

    missing : bool
        Whether missing values should be retained in grouping operations.

    r_groups : list[str] | None
        Row-dimension groupby columns, used for the COLPCTN denominator.

    c_groups : list[str] | None
        Column-dimension groupby columns, used for the ROWPCTN denominator.

    weight : str | None
        Optional weight column name.

    Returns
    -------
    pd.Series
        Aggregated values for the requested statistic and ALL/TOTAL margin,
        indexed by the grouping columns that remain after collapsing the ALL
        dimension.

    Notes
    -----
    For COLPCTN/COLPCTSUM: denominator = column total (grouped by c_groups).
    For ROWPCTN/ROWPCTSUM: denominator = row total   (grouped by r_groups).
    For PCTN/PCTSUM:       denominator = overall grand total.

    Grand-total ALL cells (denominator = grand total) always yield 100%
    for PCTN/PCTSUM, and the correct fraction for subtotal ALL rows.

    N, COUNT, SIZE and NMISS are never weighted.
    """
    r_groups = r_groups or []
    c_groups = c_groups or []
    dropna = not missing

    if weight is not None:
        data = data[data[weight].notna()].copy()
        data[weight] = _clean_weights(data[weight])

    if stat in _BASE_STATS:
        return _compute_all_base_stat(data, groups_to_keep, var, stat, weight, dropna)

    return _compute_all_percent_stat(
        data, groups_to_keep, var, stat, r_groups, c_groups, weight, dropna
    )


def _parse_denom_def(denom_str: str) -> list[str]:
    """Parse a denominator definition into ordered uppercase tokens.

    Each token is one of:

    - An uppercase measure variable name, for example ``"INCOME"``.
    - An uppercase class variable name, for example ``"GENDER"``.
    - ``"ALL"`` or ``"TOTAL"`` as a grand-total fallback.

    Multiple tokens are tried from left to right by ``_compute_custom_pct``.
    The first token that matches the current subtable is used as the
    denominator. Placing ``ALL`` or ``TOTAL`` last provides a fallback when
    none of the named variables participate.

    Parameters
    ----------
    denom_str : str
        Content of the ``<...>`` denominator definition.

    Returns
    -------
    list[str]
        Ordered uppercase denominator tokens.

    Examples
    --------
    ``"income"`` becomes ``["INCOME"]``.

    ``"gender all"`` becomes ``["GENDER", "ALL"]``.

    ``"origin total"`` becomes ``["ORIGIN", "TOTAL"]``.
    """
    return [t.upper() for t in denom_str.strip().split()]


# ---------------------------------------------------------------------------
# _compute_custom_pct and its helpers
# ---------------------------------------------------------------------------


def _agg_sum_series(
    data: pd.DataFrame,
    groups: list[str],
    var_col: str | None,
    weight: str | None,
    dropna: bool,
) -> pd.Series:
    """Sum-based aggregation used for custom-denominator numerator/denominator.

    Distinct from _agg_group: always sums (never a stat-function dispatch),
    and its own weighted-sum implementation that also drops rows with a
    missing measure value (in addition to missing weight).
    """
    if var_col is not None:
        if weight is not None:
            # Weighted sum: drop rows with missing weight or missing measure,
            # clean negative weights to 0, then compute sum(w * x)
            d = data.dropna(subset=[weight]).copy()
            d[weight] = _clean_weights(d[weight])
            d = d.dropna(subset=[var_col])

            if groups:
                grouped = d.groupby(groups, dropna=dropna)[[var_col, weight]]
                result = grouped.apply(lambda g: float((g[var_col] * g[weight]).sum()))
                return cast(pd.Series, result.rename(None))

            return pd.Series({"__total__": float((d[var_col] * d[weight]).sum())})

        if groups:
            return data.groupby(groups, dropna=dropna)[var_col].sum()

        return pd.Series({"__total__": data[var_col].sum()})

    if groups:
        return data.groupby(groups, dropna=dropna).size().rename(None)

    return pd.Series({"__total__": len(data)})


def _norm_for_lookup(v: object) -> object:
    if v is None:
        return "__nan__"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "__nan__"
    except (TypeError, ValueError):
        pass
    if v is pd.NA or v is pd.NaT:
        return "__nan__"
    return v


def _norm_lookup_key(k: object) -> object:
    if isinstance(k, tuple):
        return tuple(_norm_for_lookup(v) for v in k)
    return _norm_for_lookup(k)


def _find_innermost_breakdown(
    r_path_order: list[tuple[Any, ...]] | None,
    c_path_order: list[tuple[Any, ...]] | None,
) -> tuple[str | None, str | None]:
    """Find the innermost breakdown variable for a spec.

    Looks at r_path_order and c_path_order combined, taking the last entry
    that is either 'group' (a real groupby column) or 'all' (a Total row).

    Returns (kind, orig) where kind is 'group'/'all'/None, and orig is the
    original column name for 'group', else None.
    """
    combined_po = list(r_path_order or []) + list(c_path_order or [])
    for entry in reversed(combined_po):
        kind = entry[0]
        if kind == "group":
            return "group", (entry[2] if len(entry) > 2 else None)
        if kind == "all":
            return "all", None
    return None, None


def _resolve_measure_denom(
    denom_tokens: list[str],
    measure_upper: list[str],
    measure_map: dict[str, str],
    all_groups: list[str],
) -> tuple[str, list[str]] | None:
    """Resolve a denom token that names a measure (always takes priority).

    Returns a ratio between two measures, using the same grouping as the
    numerator.
    """
    measure_tok = next((t for t in denom_tokens if t in measure_upper), None)
    if measure_tok is None:
        return None
    return measure_map[measure_tok], all_groups


def _resolve_denom_for_group_innermost(
    denom_tokens: list[str],
    innermost_orig: str,
    groupby_upper: list[str],
    groupby_map: dict[str, str],
    all_groups: list[str],
    var: str | None,
) -> tuple[str | None, list[str]] | None:
    """Resolve the denominator when this spec is broken down by innermost_orig.

    Find the token in denom_def naming this variable; collapse it to get
    the parent subtotal. If not found, fall back left-to-right.
    """
    inner_upper = innermost_orig.upper()
    if inner_upper in denom_tokens and inner_upper in groupby_upper:
        return var, [g for g in all_groups if g != innermost_orig]

    for tok in denom_tokens:
        if tok in groupby_upper:
            col = groupby_map[tok]
            return var, [g for g in all_groups if g != col]
        if tok in ("ALL", "TOTAL"):
            return var, []
    return None


def _resolve_denom_for_all_innermost(
    denom_tokens: list[str],
    groupby_upper: list[str],
    groupby_map: dict[str, str],
    all_groups: list[str],
    var: str | None,
) -> tuple[str | None, list[str]] | None:
    """Resolve the denominator when this spec produces an ALL/Total row.

    Use the first TOTAL/ALL token (grand total), or fall back left-to-right
    over groupby tokens only.
    """
    for tok in denom_tokens:
        if tok in ("ALL", "TOTAL"):
            return var, []

    for tok in denom_tokens:
        if tok in groupby_upper:
            col = groupby_map[tok]
            return var, [g for g in all_groups if g != col]
    return None


def _resolve_denom_for_unknown_innermost(
    denom_tokens: list[str],
    measure_upper: list[str],
    measure_map: dict[str, str],
    groupby_upper: list[str],
    groupby_map: dict[str, str],
    all_groups: list[str],
    var: str | None,
) -> tuple[str | None, list[str]] | None:
    """No innermost known (e.g. pure stat spec with no groupby): left-to-right."""
    for tok in denom_tokens:
        if tok in measure_upper:
            return measure_map[tok], all_groups
        if tok in groupby_upper:
            col = groupby_map[tok]
            return var, [g for g in all_groups if g != col]
        if tok in ("ALL", "TOTAL"):
            return var, []
    return None


def _resolve_denominator(
    denom_tokens: list[str],
    innermost_kind: str | None,
    innermost_orig: str | None,
    measure_upper: list[str],
    measure_map: dict[str, str],
    groupby_upper: list[str],
    groupby_map: dict[str, str],
    all_groups: list[str],
    var: str | None,
) -> tuple[str | None, list[str]]:
    """Resolve which denominator to use for this spec's rows.

    - measure token: ratio between two measures (denom_groups = all_groups)
    - 'all'/'total' innermost: use the first TOTAL/ALL token in denom_def,
      or fall back to grand total
    - 'group' innermost with orig_col X: find the token matching X and
      collapse X from all_groups to get the parent subtotal

    Default (no token resolves anything): denom_var_col = var (same measure
    as numerator), denom_groups = [] (grand total).
    """
    measure_result = _resolve_measure_denom(
        denom_tokens, measure_upper, measure_map, all_groups
    )
    if measure_result is not None:
        return measure_result

    if innermost_kind == "group" and innermost_orig is not None:
        result = _resolve_denom_for_group_innermost(
            denom_tokens, innermost_orig, groupby_upper, groupby_map, all_groups, var
        )
    elif innermost_kind == "all":
        result = _resolve_denom_for_all_innermost(
            denom_tokens, groupby_upper, groupby_map, all_groups, var
        )
    else:
        result = _resolve_denom_for_unknown_innermost(
            denom_tokens,
            measure_upper,
            measure_map,
            groupby_upper,
            groupby_map,
            all_groups,
            var,
        )

    if result is None:
        return var, []
    return result


def _custom_pct_denom_key(
    idx_t: tuple[Any, ...],
    denom_groups: list[str],
    r_groups: list[str],
    c_groups: list[str],
) -> object:
    denom_key_vals = []
    for g in denom_groups:
        if g in r_groups:
            pos = r_groups.index(g)
            denom_key_vals.append(idx_t[pos] if pos < len(idx_t) else None)
        elif g in c_groups:
            pos = len(r_groups) + c_groups.index(g)
            denom_key_vals.append(idx_t[pos] if pos < len(idx_t) else None)

    if len(denom_key_vals) == 0:
        return "__total__"
    if len(denom_key_vals) == 1:
        return denom_key_vals[0]
    return tuple(denom_key_vals)


def _safe_custom_pct(num: float, denom_val: Any) -> float:
    try:
        if denom_val and not np.isnan(float(denom_val)) and denom_val != 0:
            return float(100.0 * num / denom_val)
        return np.nan
    except (TypeError, ValueError):
        return np.nan


def _build_custom_pct_results(
    num_series: pd.Series,
    denom_lookup: dict[object, Any],
    denom_groups: list[str],
    r_groups: list[str],
    c_groups: list[str],
) -> dict[Any, float]:
    results = {}
    for idx, num in num_series.items():
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        denom_key = _custom_pct_denom_key(idx_t, denom_groups, r_groups, c_groups)
        denom_val = denom_lookup.get(_norm_lookup_key(denom_key), np.nan)
        results[idx] = _safe_custom_pct(num, denom_val)
    return results


def _compute_custom_pct(
    data: pd.DataFrame,
    r_groups: list[str],
    c_groups: list[str],
    var: str | None,
    stat: str,
    denom_def: str,
    groupby: list[str],
    measure: list[str],
    missing: bool,
    weight: str | None = None,
    r_path_order: list[tuple[Any, ...]] | None = None,
    c_path_order: list[tuple[Any, ...]] | None = None,
) -> pd.Series:
    """Compute a percentage statistic with a user-defined denominator.

    This implements the ``PCTN<...>`` and ``PCTSUM<...>`` syntax, where the
    content of ``<...>`` specifies how the denominator should be resolved.

    Parameters
    ----------
    data : pd.DataFrame
        Input data used to compute the numerator and denominator.

    r_groups : list[str]
        Grouping columns in the row dimension.

    c_groups : list[str]
        Grouping columns in the column dimension.

    var : str | None
        Measure column used for the numerator, or None for a bare count-based
        PCTN with no measure variable.

    stat : str
        Percentage statistic to compute.

    denom_def : str
        Denominator definition from the ``<...>`` expression.

    groupby : list[str]
        Available grouping variables.

    measure : list[str]
        Available measure variables.

    missing : bool
        Whether missing grouping values should be included.

    weight : str | None
        Optional name of the weight column.

    r_path_order : list[tuple[Any, ...]] | None
        Parsed row path used to determine the innermost breakdown variable.

    c_path_order : list[tuple[Any, ...]] | None
        Parsed column path used to determine the innermost breakdown variable.

    Returns
    -------
    pd.Series
        Percentage values indexed by the relevant grouping keys.

    Notes
    -----
    Denominator resolution is performed separately for each table
    specification.

    If the innermost entry is ``"all"``, the first ``"ALL"`` or ``"TOTAL"``
    token is preferred. A grouping variable that is not part of the current
    grouping may instead define a parent subtotal.

    If the innermost entry is a grouping variable with original column
    ``X``, the denominator token matching ``X`` is preferred. The denominator
    is then obtained by collapsing ``X`` from the current grouping.

    A denominator token that refers to a measure variable uses the same
    grouping as the numerator.

    If no preferred token matches, denominator tokens are tried from left to
    right, with the grand total used as the final fallback.

    When ``weight`` is supplied, weighted sums are used for measure-based
    numerator and denominator calculations. Rows with missing weights are
    excluded, and negative weights are treated according to the package's
    weight-cleaning rules.

    Examples
    --------
    ``pctn<total gender age_group>`` with the row specification
    ``origin*(Total gender age_group)`` produces denominator behaviour such
    that the Total row uses the grand total, while the Gender and Age-group
    rows use the corresponding parent subtotal.

    ``pctsum<income>`` uses ``income`` as the denominator measure with the
    same grouping as the numerator.

    ``pctn<gender all>`` prefers the gender subtotal and falls back to the
    grand total.
    """
    measure_upper = [m.upper() for m in (measure or [])]
    groupby_upper = [g.upper() for g in (groupby or [])]
    groupby_map = {g.upper(): g for g in (groupby or [])}
    measure_map = {m.upper(): m for m in (measure or [])}

    denom_tokens = _parse_denom_def(denom_def)
    all_groups = list(dict.fromkeys(r_groups + c_groups))
    dropna = not missing

    innermost_kind, innermost_orig = _find_innermost_breakdown(
        r_path_order, c_path_order
    )

    denom_var_col, denom_groups = _resolve_denominator(
        denom_tokens,
        innermost_kind,
        innermost_orig,
        measure_upper,
        measure_map,
        groupby_upper,
        groupby_map,
        all_groups,
        var,
    )

    num_series = _agg_sum_series(data, all_groups, var, weight, dropna)
    denom_series = _agg_sum_series(data, denom_groups, denom_var_col, weight, dropna)

    # Build normalised-key lookup so NaN groupby values match correctly
    denom_lookup = {_norm_lookup_key(k): v for k, v in denom_series.items()}

    results = _build_custom_pct_results(
        num_series, denom_lookup, denom_groups, r_groups, c_groups
    )
    return pd.Series(results)
