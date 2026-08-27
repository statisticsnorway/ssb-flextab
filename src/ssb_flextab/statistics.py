from collections.abc import Callable

import numpy as np
import pandas as pd

def _hmean(x):
    """
    Unweighted harmonic mean: n / sum(1/x), excluding NaN and zero values.

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

# ---------------------------------------------------------------------------
# 2. WEIGHTED STATISTICS  (used when weight= is supplied to flextab())
# ---------------------------------------------------------------------------
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
    """
    Drop rows where the measure value x is NaN, keeping x and w aligned.

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
    """
    Apply SAS PROC TABULATE's WEIGHT statement rules to a raw weight column:
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
