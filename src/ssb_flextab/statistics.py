from collections.abc import Callable

import numpy as np
import pandas as pd


def _hmean(x):
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
    return len(x) / (1.0 / x).sum()

_BASE_STATS: dict[str, Callable] = {
    # ── Count statistics ────────────────────────────────────────────────────
    # N, COUNT and SIZE may all be used WITHOUT a measure column (bare count).
    "N":      lambda x: x.count(),   # backward-compatible alias for COUNT
    "COUNT":  lambda x: x.count(),   # like pandas Series.count(): counts
                                      # only NON-MISSING values of the measure
    "SIZE":   lambda x: x.size,      # like Python len() / numpy .size:
                                      # counts ALL rows, including those where
                                      # the measure is missing/NaN
    # NMISS requires a measure column (there must be something to be missing)
    "NMISS":  lambda x: x.isna().sum(),
    # ── Descriptive statistics (all require a measure column) ────────────────
    "SUM":    lambda x: x.sum(),
    "MEAN":   lambda x: x.mean(),
    "MIN":    lambda x: x.min(),
    "MAX":    lambda x: x.max(),
    "STD":    lambda x: x.std(),      # sample std dev (VARDEF=DF, n-1)
    "STDERR": lambda x: x.sem(),      # standard error of the mean
    "VAR":    lambda x: x.var(),      # sample variance (VARDEF=DF, n-1)
    "MEDIAN": lambda x: x.median(),
    "P1":     lambda x: x.quantile(0.01),
    "P5":     lambda x: x.quantile(0.05),
    "P10":    lambda x: x.quantile(0.10),
    "P25":    lambda x: x.quantile(0.25),
    "P75":    lambda x: x.quantile(0.75),
    "P90":    lambda x: x.quantile(0.90),
    "P95":    lambda x: x.quantile(0.95),
    "P99":    lambda x: x.quantile(0.99),
    "QRANGE": lambda x: x.quantile(0.75) - x.quantile(0.25),
    "GMEAN":  lambda x: np.exp(np.log(x[x > 0]).mean()) if (x > 0).any() else np.nan,
    "HMEAN":  lambda x: _hmean(x),
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

def _drop_nan_x(x, w):
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

def _wmean(x, w):
    # Weighted arithmetic mean: x_bar_w = (sum w*x) / (sum w)
    x, w = _drop_nan_x(x, w)
    wsum = w.sum()
    return (x * w).sum() / wsum if wsum else np.nan

def _wvar(x, w):
    # Weighted variance, UNBIASED (reliability-weights) estimator:
    #   Var_w = [ (sum w) / ((sum w)^2 - sum w^2) ] * sum( w*(x - xbar_w)^2 )
    # This reduces to the familiar sum((x-xbar)^2)/(n-1) when all w_i = 1,
    # since then sum(w)=n and (sum w)^2 - sum(w^2) = n^2 - n = n(n-1).
    x, w = _drop_nan_x(x, w)
    wsum = w.sum()
    if not wsum:
        return np.nan
    denom = wsum ** 2 - (w ** 2).sum()
    if denom <= 0:
        # Degenerate case (e.g. a single nonzero-weight observation):
        # not enough effective degrees of freedom to estimate variance.
        return np.nan
    xbar = _wmean(x, w)
    numerator = (w * (x - xbar) ** 2).sum()
    return (wsum / denom) * numerator

def _wstd(x, w):
    v = _wvar(x, w)
    return np.sqrt(v) if pd.notna(v) else np.nan

def _wstderr(x, w):
    # Standard error of the weighted mean: SD_w / sqrt(sum w)
    x, w = _drop_nan_x(x, w)
    v = _wvar(x, w)
    wsum = w.sum()
    return np.sqrt(v / wsum) if pd.notna(v) and wsum else np.nan

def _wpercentile(x, w, q):
    """Weighted percentile via linear interpolation on the weighted ECDF."""
    x, w = _drop_nan_x(x, w)
    if len(x) == 0 or w.sum() == 0:
        return np.nan
    order = np.argsort(x.values)
    xs = x.values[order]
    ws = w.values[order]
    cw = np.cumsum(ws)
    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff)
    idx = min(idx, len(xs) - 1)
    return xs[idx]

def _wgmean(x, w):
    # Weighted geometric mean: exp( (sum w*ln x) / (sum w) )
    x, w = _drop_nan_x(x, w)
    mask = x > 0
    if not mask.any():
        return np.nan
    xw, ww = x[mask], w[mask]
    wsum = ww.sum()
    if not wsum:
        return np.nan
    return np.exp((ww * np.log(xw)).sum() / wsum)

def _whmean(x, w):
    # Weighted harmonic mean: (sum w) / (sum w/x)
    x, w = _drop_nan_x(x, w)
    mask = x != 0
    if not mask.any():
        return np.nan
    xw, ww = x[mask], w[mask]
    wsum = ww.sum()
    if not wsum:
        return np.nan
    return wsum / (ww / xw).sum()

def _clean_weights(weights: pd.Series) -> pd.Series:
    """Apply SAS PROC TABULATE's WEIGHT statement rules to a raw weight column:
    - missing weight  -> NaN (caller must drop these rows entirely)
    - negative weight -> treated as 0 (observation still counted in N)
    - zero / positive -> unchanged
    """
    cleaned = weights.copy()
    cleaned[cleaned < 0] = 0
    return cleaned

_WEIGHTED_STATS: dict[str, Callable] = {
    "N":      lambda x, w: x.count(),   # alias for COUNT, NEVER weighted
    "COUNT":  lambda x, w: x.count(),   # NEVER weighted - plain count of non-missing values
    "SIZE":   lambda x, w: x.size,      # NEVER weighted - plain count of ALL rows
    "NMISS":  lambda x, w: x.isna().sum(),  # also never weighted - plain count
    "SUM":    lambda x, w: (x * w).sum(),
    "MEAN":   _wmean,
    "MIN":    lambda x, w: x.min(),
    "MAX":    lambda x, w: x.max(),
    "STD":    _wstd,
    "STDERR": _wstderr,
    "VAR":    _wvar,
    "MEDIAN": lambda x, w: _wpercentile(x, w, 0.50),
    "P1":     lambda x, w: _wpercentile(x, w, 0.01),
    "P5":     lambda x, w: _wpercentile(x, w, 0.05),
    "P10":    lambda x, w: _wpercentile(x, w, 0.10),
    "P25":    lambda x, w: _wpercentile(x, w, 0.25),
    "P75":    lambda x, w: _wpercentile(x, w, 0.75),
    "P90":    lambda x, w: _wpercentile(x, w, 0.90),
    "P95":    lambda x, w: _wpercentile(x, w, 0.95),
    "P99":    lambda x, w: _wpercentile(x, w, 0.99),
    "QRANGE": lambda x, w: _wpercentile(x, w, 0.75) - _wpercentile(x, w, 0.25),
    "GMEAN":  _wgmean,
    "HMEAN":  _whmean,
}

_PERCENT_STATS = {
    "PCTN", "ROWPCTN", "COLPCTN",
    "PCTSUM", "ROWPCTSUM", "COLPCTSUM",
}

ALL_STATS = set(_BASE_STATS) | _PERCENT_STATS

def _compute_series(data, all_groups, var, stat, r_groups, c_groups, missing, weight=None):
    """Compute an aggregated Series for a single (row_spec, col_spec) pair
    when neither spec carries an ALL/TOTAL token — i.e. both dimensions
    are pure groupby-value breakdowns with no marginal totals.

    Parameters
    ----------
    data       : filtered/cleaned DataFrame (weight rows already dropped)
    all_groups : combined list of r_groups + c_groups (the groupby keys)
    var        : measure column name, or None for count-only stats
    stat       : statistic keyword (e.g. 'MEAN', 'COLPCTN', 'N')
    r_groups   : groupby columns in the row dimension
    c_groups   : groupby columns in the column dimension
    missing    : passed as dropna=not-missing to groupby
    weight     : optional weight column name

    Returns:
    -------
    pd.Series  indexed by the all_groups groupby key(s)

    Notes:
    -----
    Percentage stats (PCTN, PCTSUM, ROWPCTN, COLPCTN, ROWPCTSUM,
    COLPCTSUM) route through this function. Their denominator is
    determined by the stat name:
      PCTN/PCTSUM       -> grand total (all rows)
      ROWPCTN/ROWPCTSUM -> row subtotal (grouped by r_groups)
      COLPCTN/COLPCTSUM -> column subtotal (grouped by c_groups)

    Custom denominators (PCTN<...>, PCTSUM<...>) are NOT routed here —
    they go directly to _compute_custom_pct() from the main dispatch loop.

    N, COUNT, SIZE and NMISS are never weighted even when weight= is given.
    """
    dropna = not missing

    # When a weight column is active, missing weights exclude the
    # observation entirely (per SAS WEIGHT statement rules). Negative
    # weights are cleaned to 0 (still counted in N) inside _clean_weights.
    if weight is not None:
        data = data[data[weight].notna()].copy()
        data[weight] = _clean_weights(data[weight])

    def _agg(groups, func, wfunc=None):
        if var is not None:
            if weight is not None and wfunc is not None:
                cols = [var, weight]
                def _apply(g):
                    return wfunc(g[var], g[weight])
                if groups:
                    return data.groupby(groups, dropna=dropna)[cols].apply(_apply)
                return pd.Series({"__total__": _apply(data[cols])})
            if groups:
                return data.groupby(groups, dropna=dropna)[var].agg(func)
            return pd.Series({"__total__": func(data[var])})
        else:
            # No measure variable (bare N/NMISS): ALWAYS a plain row count,
            # never weighted - even when weight= is supplied. Rows with a
            # missing weight were already excluded from `data` above, and
            # zero/negative weights are still counted (only their VALUE was
            # cleaned to 0), so a plain count here is exactly right.
            if groups:
                return data.groupby(groups, dropna=dropna).size().rename(None)
            return pd.Series({"__total__": len(data)})

    def _grand(func, wfunc=None):
        if var is not None:
            if weight is not None and wfunc is not None:
                return wfunc(data[var], data[weight])
            return func(data[var])
        # No measure variable: ALWAYS a plain row count (N is never weighted)
        return len(data)

    if stat in _BASE_STATS:
        if var is None and stat not in ("N", "COUNT", "SIZE"):
            raise ValueError(
                f"Statistic '{stat}' requires a measure variable. "
                f"Omit measure= only when using N, COUNT, or SIZE (all plain "
                f"row counts with no measure). NMISS specifically counts "
                f"missing values OF a measure column, so it always needs "
                f"one, e.g. income*NMISS."
            )
        func  = _BASE_STATS[stat] if var is not None else (lambda x: x.count())
        wfunc = _WEIGHTED_STATS.get(stat) if var is not None else None
        return _agg(all_groups, func, wfunc)

    count_based = "N" in stat
    if not count_based and var is None:
        raise ValueError(
            f"Statistic '{stat}' requires a measure variable. "
            f"Omit measure= only when using count statistics: N, COUNT, SIZE, "
            f"NMISS, PCTN, ROWPCTN, COLPCTN."
        )

    if count_based:
        raw_func  = _BASE_STATS["N"] if var is not None else (lambda x: x.count())
        raw_wfunc = _WEIGHTED_STATS["N"] if var is not None else None
    else:
        raw_func  = _BASE_STATS["SUM"]
        raw_wfunc = _WEIGHTED_STATS["SUM"]

    series = _agg(all_groups, raw_func, raw_wfunc)
    grand  = _grand(raw_func if var is not None else (lambda x: len(x)), raw_wfunc)

    if stat in ("PCTN", "PCTSUM"):
        return 100.0 * series / grand

    if stat in ("ROWPCTN", "ROWPCTSUM"):
        if r_groups:
            denom = _agg(r_groups, raw_func, raw_wfunc)
            n_r = len(r_groups)
            def row_pct(val, idx):
                key = idx[:n_r] if isinstance(idx, tuple) else (idx,)
                key = key[0] if len(key) == 1 else key
                d = denom.get(key, np.nan)
                return 100.0 * val / d if d else np.nan
            return pd.Series(
                {idx: row_pct(val, idx) for idx, val in series.items()},
                name=series.name,
            )
        return 100.0 * series / grand

    if stat in ("COLPCTN", "COLPCTSUM"):
        if c_groups:
            denom = _agg(c_groups, raw_func, raw_wfunc)
            n_r = len(r_groups)
            def col_pct(val, idx):
                if isinstance(idx, tuple):
                    key = idx[n_r:n_r + len(c_groups)]
                else:
                    key = (idx,)
                key = key[0] if len(key) == 1 else key
                d = denom.get(key, np.nan)
                return 100.0 * val / d if d else np.nan
            return pd.Series(
                {idx: col_pct(val, idx) for idx, val in series.items()},
                name=series.name,
            )
        return 100.0 * series / grand

    raise ValueError(f"Unknown statistic: {stat}")

def _compute_all_series(data, groups_to_keep, var, stat, missing,
                        r_groups=None, c_groups=None, weight=None):
    """Compute an aggregated Series for a spec that involves an ALL/TOTAL
    marginal total in at least one dimension.

    This handles the three ALL cases:
      - ALL on rows only  (has_all_r=True,  has_all_c=False)
      - ALL on cols only  (has_all_r=False, has_all_c=True)
      - ALL on both dims  (has_all_r=True,  has_all_c=True)

    In each case, groups_to_keep is the subset of groupby columns that
    still vary (i.e. the groups from the OTHER dimension that provide the
    cross-breakdown), and the ALL dimension is collapsed.

    Parameters
    ----------
    data           : filtered DataFrame
    groups_to_keep : groupby columns to aggregate over (the non-ALL side)
    var            : measure column name, or None for count-only stats
    stat           : statistic keyword
    missing        : controls dropna= in groupby
    r_groups       : row-dimension groupby columns (for COLPCTN denominator)
    c_groups       : col-dimension groupby columns (for ROWPCTN denominator)
    weight         : optional weight column name

    Notes:
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

    def _agg(groups, wfunc=None):
        if var is not None:
            if weight is not None and wfunc is not None:
                cols = [var, weight]
                def _apply(g):
                    return wfunc(g[var], g[weight])
                if groups:
                    return data.groupby(groups, dropna=dropna)[cols].apply(_apply)
                return pd.Series({"__total__": _apply(data[cols])})
            if groups:
                return data.groupby(groups, dropna=dropna)[var].agg(func)
            return pd.Series({"__total__": func(data[var])})
        else:
            # No measure variable (bare N/NMISS): ALWAYS a plain row count,
            # never weighted - even when weight= is supplied.
            if groups:
                return data.groupby(groups, dropna=dropna).size().rename(None)
            return pd.Series({"__total__": len(data)})

    if stat in _BASE_STATS:
        if var is None and stat not in ("N", "COUNT", "SIZE"):
            raise ValueError(
                f"Statistic '{stat}' requires a measure variable. "
                f"Omit measure= only when using N, COUNT, or SIZE (all plain "
                f"row counts with no measure). NMISS specifically counts "
                f"missing values OF a measure column, so it always needs "
                f"one, e.g. income*NMISS."
            )
        func  = _BASE_STATS[stat] if var is not None else (lambda x: x.count())
        wfunc = _WEIGHTED_STATS.get(stat) if var is not None else None
        return _agg(groups_to_keep, wfunc)

    count_based = "N" in stat
    if not count_based and var is None:
        raise ValueError(
            f"Statistic '{stat}' requires a measure variable. "
            f"Omit measure= only when using count statistics: N, COUNT, SIZE, "
            f"NMISS, PCTN, ROWPCTN, COLPCTN."
        )

    if count_based:
        raw_func  = _BASE_STATS["N"] if var is not None else (lambda x: x.count())
        raw_wfunc = _WEIGHTED_STATS["N"] if var is not None else None
        # N/NMISS-based grand total is ALWAYS a plain row count, never
        # weighted - even when weight= is supplied and var is set.
        grand = data[var].count() if var is not None else len(data)
    else:
        raw_func  = _BASE_STATS["SUM"]
        raw_wfunc = _WEIGHTED_STATS["SUM"]
        grand = raw_wfunc(data[var], data[weight]) if weight is not None else data[var].sum()

    func = raw_func  # used inside _agg's non-weighted branch

    series = _agg(groups_to_keep, raw_wfunc)

    if stat in ("COLPCTN", "COLPCTSUM"):
        # Denominator = total within each column group.
        # When ALL is on rows, c_groups are the column breakdown groups.
        # Divide each cell by the total for its column group.
        if c_groups:
            denom = _agg(c_groups, raw_wfunc)  # total per column group
            def _col_denom(idx):
                # idx is from groups_to_keep = r_context + c_groups
                # c_groups part starts after r_context groups
                n_r_ctx = len(groups_to_keep) - len(c_groups)
                if isinstance(idx, tuple):
                    key = idx[n_r_ctx:]
                else:
                    key = (idx,)
                key = key[0] if len(key) == 1 else key
                return denom.get(key, np.nan)
            def _safe_pct(val, denom_val):
                if denom_val is np.nan or denom_val == 0:
                    return np.nan
                return 100.0 * val / denom_val
            return pd.Series(
                {idx: _safe_pct(val, _col_denom(idx))
                 for idx, val in series.items()},
                name=series.name,
            )
        # No column groups — divide by overall grand total
        with np.errstate(invalid='ignore', divide='ignore'):
            return 100.0 * series / grand

    if stat in ("ROWPCTN", "ROWPCTSUM"):
        # Denominator = total within each row group.
        # When ALL is on cols, r_groups are the row breakdown groups.
        if r_groups:
            denom = _agg(r_groups, raw_wfunc)  # total per row group
            def _row_denom(idx):
                if isinstance(idx, tuple):
                    key = idx[:len(r_groups)]
                else:
                    key = (idx,)
                key = key[0] if len(key) == 1 else key
                return denom.get(key, np.nan)
            def _safe_row_pct(val, denom_val):
                if denom_val is np.nan or denom_val == 0:
                    return np.nan
                return 100.0 * val / denom_val
            return pd.Series(
                {idx: _safe_row_pct(val, _row_denom(idx))
                 for idx, val in series.items()},
                name=series.name,
            )
        with np.errstate(invalid='ignore', divide='ignore'):
            return 100.0 * series / grand

    # PCTN / PCTSUM: always use overall grand total
    return 100.0 * series / grand

def _parse_denom_def(denom_str: str) -> list[str]:
    """Parse the content of a <...> denominator definition into an ordered
    list of uppercase token strings.

    Each token is one of:
      - An uppercase measure variable name (e.g. 'INCOME')
      - An uppercase class variable name   (e.g. 'GENDER')
      - 'ALL' or 'TOTAL'                   (grand-total fallback)

    Multiple tokens are tried in left-to-right order by _compute_custom_pct;
    the first token that "matches" the current subtable is used as the
    denominator.  Placing ALL/TOTAL last provides a fallback for cells where
    none of the named variables participate.

    Examples:
    --------
    'income'       -> ['INCOME']
    'gender all'   -> ['GENDER', 'ALL']
    'origin total' -> ['ORIGIN', 'TOTAL']
    """
    return [t.upper() for t in denom_str.strip().split()]



def _compute_custom_pct(
    data: pd.DataFrame,
    r_groups: list,
    c_groups: list,
    var: str,
    stat: str,
    denom_def: str,
    groupby: list,
    measure: list,
    missing: bool,
    weight: str | None = None,
    r_path_order: list | None = None,
    c_path_order: list | None = None,
) -> pd.Series:
    """Compute a percentage statistic with a user-defined denominator.

    This implements the PCTN<...> and PCTSUM<...> syntax, where the content
    of <...> specifies what the denominator should be.

    Denominator token resolution — PER ROW
    ---------------------------------------
    SAS picks the denominator token that matches the **innermost breakdown
    variable** of the current spec, not just the first globally-applicable
    token.  This is derived from the innermost 'group' or 'all' entry in
    r_path_order / c_path_order:

      - If the innermost entry is 'all' (a Total/ALL row):
            use the first token in denom_def that is 'ALL'/'TOTAL', or the
            first class variable that is NOT in the current all_groups (so
            it acts as the parent subtotal level).

      - If the innermost entry is 'group' with orig_col X:
            find the token in denom_def that matches X (case-insensitive).
            The denominator is then the subtotal obtained by collapsing X
            from all_groups — i.e. groupby(all_groups minus X).

      - Measure variable names in denom_def always use the same groups as
        the numerator (ratio of two measures within the same breakdown).

      - If no matching token is found, fall back left-to-right through the
        token list as before, or to grand total as a final fallback.

    Examples:
    --------
    pctn<total gender age_group>  with row spec origin*(Total gender age_group):
      Total row  -> innermost=all  -> token 'total' -> grand total denom
      gender row -> innermost=gender -> token 'gender' -> origin subtotal denom
      age_group row -> innermost=age_group -> token 'age_group' -> origin subtotal

    pctsum<income>  (ratio of two measures, same groups always)
    pctn<gender all>  (gender subtotal, fallback to grand total)
    """
    measure_upper  = [m.upper() for m in (measure or [])]
    groupby_upper  = [g.upper() for g in (groupby  or [])]
    groupby_map    = {g.upper(): g for g in (groupby or [])}
    measure_map    = {m.upper(): m for m in (measure or [])}

    denom_tokens = _parse_denom_def(denom_def)
    all_groups   = list(dict.fromkeys(r_groups + c_groups))
    dropna       = not missing

    def _agg_series(groups, var_col):
        if var_col is not None:
            if weight is not None:
                # Weighted sum: drop rows with missing weight or missing measure,
                # clean negative weights to 0, then compute sum(w * x)
                d = data.dropna(subset=[weight]).copy()
                d[weight] = _clean_weights(d[weight])
                d = d.dropna(subset=[var_col])
                if groups:
                    return (d.groupby(groups, dropna=dropna)
                              .apply(lambda g: (g[var_col] * g[weight]).sum(),
                                     include_groups=False)
                              .rename(None))
                return pd.Series({"__total__": (d[var_col] * d[weight]).sum()})
            else:
                if groups:
                    return data.groupby(groups, dropna=dropna)[var_col].sum()
                return pd.Series({"__total__": data[var_col].sum()})
        else:
            if groups:
                return data.groupby(groups, dropna=dropna).size().rename(None)
            return pd.Series({"__total__": len(data)})

    def _norm_for_lookup(v):
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

    def _norm_lookup_key(k):
        if isinstance(k, tuple):
            return tuple(_norm_for_lookup(v) for v in k)
        return _norm_for_lookup(k)

    # Determine the innermost breakdown variable for this spec.
    # We look at r_path_order and c_path_order combined, taking the last entry
    # that is either 'group' (a real groupby column) or 'all' (a Total row).
    combined_po = list(r_path_order or []) + list(c_path_order or [])
    innermost_kind    = None   # 'group' or 'all'
    innermost_orig    = None   # original column name for 'group', None for 'all'

    for entry in reversed(combined_po):
        kind = entry[0]
        if kind == "group":
            innermost_kind = "group"
            innermost_orig = entry[2] if len(entry) > 2 else None
            break
        elif kind == "all":
            innermost_kind = "all"
            innermost_orig = None
            break

    # Resolve which denominator to use for THIS spec's rows:
    #   - measure token: ratio between two measures (denom_groups = all_groups)
    #   - 'all'/'total' innermost: use the first TOTAL/ALL token in denom_def,
    #     or fall back to grand total
    #   - 'group' innermost with orig_col X: find the token matching X and
    #     collapse X from all_groups to get the parent subtotal
    denom_var_col = var   # default: same measure as numerator
    denom_groups  = []    # default: grand total

    # First check if any token is a measure name (always takes priority)
    measure_tok = next((t for t in denom_tokens if t in measure_upper), None)
    if measure_tok is not None:
        denom_var_col = measure_map[measure_tok]
        denom_groups  = all_groups

    elif innermost_kind == "group" and innermost_orig is not None:
        # This spec produces rows broken down by innermost_orig.
        # Find the token in denom_def that names this variable.
        inner_upper = innermost_orig.upper()
        if inner_upper in denom_tokens and inner_upper in groupby_upper:
            # Collapse innermost_orig: denominator = subtotal excluding this var
            denom_groups  = [g for g in all_groups if g != innermost_orig]
            denom_var_col = var
        else:
            # Token not found for this variable; try left-to-right as fallback
            for tok in denom_tokens:
                if tok in groupby_upper:
                    col = groupby_map[tok]
                    denom_groups  = [g for g in all_groups if g != col]
                    denom_var_col = var
                    break
                elif tok in ("ALL", "TOTAL"):
                    denom_groups  = []
                    denom_var_col = var
                    break

    elif innermost_kind == "all":
        # This spec produces an ALL/Total row.
        # Use the first TOTAL/ALL token in denom_def (i.e. the grand total),
        # OR if no such token exists fall back left-to-right.
        found = False
        for tok in denom_tokens:
            if tok in ("ALL", "TOTAL"):
                denom_groups  = []
                denom_var_col = var
                found = True
                break
        if not found:
            # No TOTAL token; use left-to-right resolution
            for tok in denom_tokens:
                if tok in groupby_upper:
                    col = groupby_map[tok]
                    denom_groups  = [g for g in all_groups if g != col]
                    denom_var_col = var
                    break

    else:
        # No innermost known (e.g. pure stat spec with no groupby); left-to-right
        for tok in denom_tokens:
            if tok in measure_upper:
                denom_var_col = measure_map[tok]
                denom_groups  = all_groups
                break
            elif tok in groupby_upper:
                col = groupby_map[tok]
                denom_groups  = [g for g in all_groups if g != col]
                denom_var_col = var
                break
            elif tok in ("ALL", "TOTAL"):
                denom_groups  = []
                denom_var_col = var
                break

    # Compute numerator and denominator series
    num_series   = _agg_series(all_groups,   var)
    denom_series = _agg_series(denom_groups, denom_var_col)

    # Build normalised-key lookup so NaN groupby values match correctly
    denom_lookup = {_norm_lookup_key(k): v for k, v in denom_series.items()}

    # Build result
    results = {}
    for idx, num in num_series.items():
        idx_t = idx if isinstance(idx, tuple) else (idx,)
        denom_key_vals = []
        for g in denom_groups:
            if g in r_groups:
                pos = r_groups.index(g)
                denom_key_vals.append(idx_t[pos] if pos < len(idx_t) else None)
            elif g in c_groups:
                pos = len(r_groups) + c_groups.index(g)
                denom_key_vals.append(idx_t[pos] if pos < len(idx_t) else None)
        if len(denom_key_vals) == 0:
            denom_key = "__total__"
        elif len(denom_key_vals) == 1:
            denom_key = denom_key_vals[0]
        else:
            denom_key = tuple(denom_key_vals)
        denom_val = denom_lookup.get(_norm_lookup_key(denom_key), np.nan)
        try:
            if denom_val and not np.isnan(float(denom_val)) and denom_val != 0:
                results[idx] = 100.0 * num / denom_val
            else:
                results[idx] = np.nan
        except (TypeError, ValueError):
            results[idx] = np.nan
    return pd.Series(results)


