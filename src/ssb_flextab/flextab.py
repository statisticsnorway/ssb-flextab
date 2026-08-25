"""
flextab.py
===========
A Python implementation of SAS PROC TABULATE.

Public API
----------
  flextab(...)           -> FlextabResult  (a pd.DataFrame subclass)
  flextab_to_string(result, fmt, na_rep) -> str

Naming conventions
------------------
  measure  -> SAS VAR   (numeric analysis variables)
  groupby  -> SAS CLASS (categorical grouping variables)

TABLE EXPRESSION SYNTAX
=======================

Dimension separators
--------------------
  row_expr , col_expr
  The comma separates the row and column dimensions (max 2 dimensions).

Operators inside a dimension
-----------------------------
  A * B          cross / nest:  each A value crossed with each B value
  A B            concatenation: A then B side-by-side (space-separated)
  ( A B )        grouping:      treat A B as one unit for * and format=

Tokens
------
  groupby_col    a CLASS variable; contributes 2 header levels (label + value)
  stat           a statistic keyword (see STATISTICS below)
  measure_col    a VAR variable; must be crossed with a statistic
  ALL or TOTAL   marginal total across the current groupby variable
                 (these two keywords are synonymous)

Labels
------
  name='Label'   inline label for any token:
                   origin='Region'  -> display "Region" instead of "origin"
                   msrp=''          -> suppress the label entirely
  Both single and double quotes are accepted.

Format specifications
---------------------
  *format=W.D        W total width (ignored), D decimal places, decimal point
  *format=W,D        decimal comma (European style)
  *format=W.D_        decimal point + thousands separator  (comma separator)
  *format=W,D_        decimal comma + thousands dot         (1.234,56)
  *format=W.Ds        decimal point + thousands separator  (space separator)
  *format=W,Ds        decimal comma + space thousands       (1 234,56)

  format= attaches to the immediately preceding token, or to every leaf
  inside the preceding group:
    mean*format=7,1*income          -> format only the mean
    (mean gmean)*format=7,1*income  -> format both mean and gmean
    mean*format=9,3 gmean)*format=7,1*income  -> mean keeps 9,3; gmean gets 7,1

Custom denominator definitions
-------------------------------
  PCTN<denom>   or  PCTSUM<denom>

  Place a space-separated list of tokens inside < > immediately after
  PCTN or PCTSUM to define a custom denominator:

    token = a measure variable name   -> denominator = sum of that measure
                                         in the same groups (ratio between
                                         two measure columns)
    token = a class variable name     -> denominator = sum/count summed over
                                         all values of that class variable
                                         (e.g. the column or row subtotal)
    token = ALL or TOTAL              -> grand total across everything

  Multiple tokens are tried left-to-right; the first one that "participates
  in the current subtable" is used.  Use ALL/TOTAL as a fallback.

  Examples:
    tax*pctsum<income>               -> tax as % of income (ratio)
    income*pctsum<gender all>        -> income as % of gender subtotal;
                                        falls back to grand total when
                                        gender is not in the subtable
    pctn<origin all>                 -> row count as % of origin subtotal

STATISTICS
==========

Count statistics (work with or without a measure column)
---------------------------------------------------------
  N        alias for COUNT (backward-compatible)
  COUNT    like pandas Series.count() — counts NON-MISSING values of the
           measure; when no measure is given, counts all rows
  SIZE     like Python len() / numpy .size — counts ALL rows regardless
           of missingness in the measure

Measure statistics (require a measure column)
---------------------------------------------
  NMISS    count of missing values in the measure column
  SUM      sum of values
  MEAN     arithmetic mean
  MIN      minimum
  MAX      maximum
  STD      standard deviation (VARDEF=DF, i.e. divided by n-1)
  STDERR   standard error of the mean
  VAR      variance (VARDEF=DF)
  MEDIAN   50th percentile
  P1 P5 P10 P25 P75 P90 P95 P99   percentiles
  QRANGE   interquartile range (P75 - P25)
  GMEAN    geometric mean (positive values only)
  HMEAN    harmonic mean (non-zero values only)

Percentage statistics (work with or without a measure column)
-------------------------------------------------------------
  PCTN        N as % of the grand-total N
  ROWPCTN     N as % of the row-total N
  COLPCTN     N as % of the column-total N
  PCTSUM      SUM as % of the grand-total SUM  (requires measure)
  ROWPCTSUM   SUM as % of the row-total SUM    (requires measure)
  COLPCTSUM   SUM as % of the column-total SUM (requires measure)

  All six accept a custom denominator: PCTN<...>, PCTSUM<gender all>, etc.

WEIGHTED STATISTICS
===================
  When weight= is supplied, weighted versions of the following statistics
  are used:
    SUM    -> sum(w * x)
    MEAN   -> sum(w*x) / sum(w)          (weighted arithmetic mean)
    VAR    -> [sum(w) / (sum(w)^2 - sum(w^2))] * sum(w*(x-xbar)^2)
              (reliability-weights unbiased estimator; reduces to the
              ordinary n-1 divisor when all weights are 1)
    STD    -> sqrt(VAR)
    STDERR -> sqrt(VAR / sum(w))
    MEDIAN / percentiles -> weighted empirical CDF
    GMEAN  -> exp(sum(w*log(x)) / sum(w))  (positive values only)
    HMEAN  -> sum(w) / sum(w/x)            (non-zero values only)

  N, COUNT, SIZE, NMISS are NEVER weighted — they always return plain
  (unweighted) row counts or counts of missing values.

  Weight rules (matching SAS PROC TABULATE WEIGHT statement):
    weight = 0       -> counted in N; contributes 0 to weighted sums
    weight < 0       -> treated as 0 (still counted in N)
    weight = missing -> row excluded entirely from all calculations

HEADER LEVEL RULES
==================
  Headers are built bottom-up:
  - The innermost (rightmost in a * chain) token occupies the lowest level
  - Each outer nesting adds one level up
  - Groupby variables contribute 2 levels (label row + value row)
  - stat/var/ALL tokens contribute 1 level each
  - When a group label is suppressed with ='', its label level is dropped
  - Entirely-blank levels are removed from the final MultiIndex
  - Shorter specs align at the bottom (front-padded), not the top

EXAMPLES
========
  # Basic grouped table
  flextab(data=df, groupby=["origin"], measure=["msrp"],
           table="origin, msrp*(N MEAN)")

  # Totals with custom labels, European decimal format
  flextab(data=df, groupby=["origin","type"],
           table="origin*(type total='Sub'), msrp=''*(N mean*format=7,1)")

  # Multiple stats, custom denominator
  flextab(data=df, groupby=["gender","age"], measure=["income","tax"],
           table="(all gender)*(all age), income tax tax*pctsum<income>")

  # Weighted statistics
  flextab(data=df, groupby=["region"], measure=["income"],
           table="region, income*(N MEAN VAR)", weight="survey_weight")

  # Grouped format, na_rep, label remapping, sort by dict order
  flextab(data=df, groupby=["gender"], measure=["income"],
           table="gender, (N MEAN GMEAN)*format=7,1*income",
           labels={"gender": {1: "Male", 2: "Female"}},
           sort_by="index", na_rep="-")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. STATISTICS REGISTRY
# ---------------------------------------------------------------------------

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

_PERCENT_STATS = {
    "PCTN", "ROWPCTN", "COLPCTN",
    "PCTSUM", "ROWPCTSUM", "COLPCTSUM",
}

ALL_STATS = set(_BASE_STATS) | _PERCENT_STATS


# ---------------------------------------------------------------------------
# 3. TOKENIZER
# ---------------------------------------------------------------------------

def _tokenize(expr: str) -> list[tuple]:
    """
    Tokenize a single TABLE dimension expression into a flat list of tokens.

    Token types emitted:
      ('NAME', name, label)     a variable or keyword name; label is the
                                 string from name='...' or None if absent
      ('FMT',  fmt_spec)        a format specification from format=W.D[_s]
      ('DENOM', inner)          denominator definition from <tok1 tok2 ...>
      ('OP',   char)            operator character: '*' or '(' or ')'
      ('SP',   ' ')             whitespace (used as a concatenation signal)

    The pattern tries alternatives left-to-right so that format=7,2 is
    captured as FMT before the comma could be misread as a dimension
    separator, and <...> is captured as DENOM before its content could
    be misread as NAME tokens.

    Label quoting:
      name="Label"  or  name='Label'   (either quote style accepted)
      name=""        or  name=''        suppresses the label (empty string)
    """
    pattern = re.compile(
        # format=7.2[_|s]  -> decimal point, 2 decimals
        # format=7,2[_|s]  -> decimal comma,  2 decimals
        # trailing _ or s  -> thousands separator (comma/dot or space)
        r'(?P<fmt>format)\s*=\s*(?P<fmt_spec>[0-9]+[.,][0-9]+[_s]*)'
        # denominator definition: <token1 token2 ...>
        r'|(?P<denom><[^>]*>)'
        r'|(?P<labeled>[A-Za-z_][A-Za-z0-9_%]*)\s*=\s*'
        r'(?:"(?P<dq_label>[^"]*)"|\'(?P<sq_label>[^\']*)\')'
        r'|(?P<name>[A-Za-z_][A-Za-z0-9_%]*)'
        r'|(?P<op>[*()])'
        r'|(?P<space>\s+)',
    )
    tokens = []
    for m in pattern.finditer(expr):
        if m.group("fmt"):
            tokens.append(("FMT", m.group("fmt_spec")))
        elif m.group("denom"):
            # Strip the angle brackets and whitespace, split into tokens
            inner = m.group("denom")[1:-1].strip()
            tokens.append(("DENOM", inner))
        elif m.group("labeled"):
            label = m.group("dq_label") if m.group("dq_label") is not None else m.group("sq_label")
            tokens.append(("NAME", m.group("labeled"), label))
        elif m.group("name"):
            tokens.append(("NAME", m.group("name"), None))
        elif m.group("op"):
            tokens.append(("OP", m.group("op")))
        elif m.group("space"):
            tokens.append(("SP", " "))
    cleaned = []
    for t in tokens:
        if t[0] == "SP" and cleaned and cleaned[-1][0] == "SP":
            continue
        cleaned.append(t)
    while cleaned and cleaned[0][0] == "SP":
        cleaned.pop(0)
    while cleaned and cleaned[-1][0] == "SP":
        cleaned.pop()
    return cleaned


# ---------------------------------------------------------------------------
# 4. AST NODE
# ---------------------------------------------------------------------------

@dataclass
class DimNode:
    """
    A node in the parsed TABLE expression tree.

    Attributes
    ----------
    kind : str
        One of:
          'var'    — a measure or class variable name, or a statistic keyword
          'all'    — the ALL/TOTAL marginal-total keyword
          'cross'  — a * b  (children = [a, b])
          'concat' — a b    (children = [a, b, ...])
          'group'  — (...)  (children = [inner_node])

    name : str or None
        The original token text (uppercased for stat/all keywords).

    label : str or None
        The display label from name='Label' syntax.
        None  → use the default (name or 'TOTAL')
        ''    → suppress the label level in the header entirely

    fmt : str or None
        Raw format specification string from *format=W.D[_s] syntax,
        e.g. '7,1' or '12.0s'.  Set by _apply_fmt() and read by
        _classify_path() to populate spec['fmt'].

    denom : str or None
        Denominator definition from PCTN<...> / PCTSUM<...> syntax,
        e.g. 'income' or 'gender all'.  Read by _classify_path() to
        populate spec['denom_def'], which routes the cell to
        _compute_custom_pct().

    children : list of DimNode
        Sub-nodes for cross/concat/group kinds.
    """
    kind: str
    name: Optional[str] = None
    label: Optional[str] = None
    fmt:   Optional[str] = None
    denom: Optional[str] = None
    children: list["DimNode"] = field(default_factory=list)

    def display_label(self) -> str:
        if self.label is None:
            return self.name if self.name else ""
        return self.label

    def __repr__(self):
        suffix = f"='{self.label}'" if self.label is not None else ""
        if self.kind == "var":
            return f"{self.name}{suffix}"
        if self.kind == "all":
            return f"ALL{suffix}"
        sep = " * " if self.kind == "cross" else " "
        inner = sep.join(repr(c) for c in self.children)
        if self.kind == "group":
            return f"({inner})"
        return inner


# ---------------------------------------------------------------------------
# 5. RECURSIVE-DESCENT PARSER
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[tuple]):
        self.tokens = tokens
        self.pos = 0

    def _skip_sp(self):
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
            self.pos += 1

    def peek(self) -> Optional[tuple]:
        p = self.pos
        while p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        return self.tokens[p] if p < len(self.tokens) else None

    def consume_op(self, val: str):
        self._skip_sp()
        t = self.tokens[self.pos]
        if t != ("OP", val):
            raise SyntaxError(f"Expected operator '{val}', got {t}")
        self.pos += 1

    def parse(self) -> DimNode:
        return self._parse_concat()

    def _parse_concat(self) -> DimNode:
        nodes = [self._parse_cross()]
        while True:
            saved = self.pos
            while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
                self.pos += 1
            if self.pos >= len(self.tokens):
                break
            nxt = self.tokens[self.pos]
            if nxt[0] == "OP" and nxt[1] in (")", "*"):
                self.pos = saved
                break
            nodes.append(self._parse_cross())
        return nodes[0] if len(nodes) == 1 else DimNode(kind="concat", children=nodes)

    def _parse_cross(self) -> DimNode:
        nodes = [self._parse_atom()]
        while True:
            p = self.peek()
            if p and p[0] == "OP" and p[1] == "*":
                # Look ahead past the '*' (and any space) to see if a FMT
                # token follows. If so, this '*format=...' applies to the
                # PRECEDING node (e.g. mean*format=7.1), not a new atom.
                saved = self.pos
                self._skip_sp()
                self.pos += 1  # consume '*'
                self._skip_sp()
                if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "FMT":
                    fmt_spec = self.tokens[self.pos][1]
                    self.pos += 1
                    self._apply_fmt(nodes[-1], fmt_spec)
                    continue
                # Not a format suffix — restore and parse as a normal cross atom
                self.pos = saved
                self._skip_sp()
                self.pos += 1
                nodes.append(self._parse_atom())
            else:
                break
        return nodes[0] if len(nodes) == 1 else DimNode(kind="cross", children=nodes)

    def _apply_fmt(self, node: DimNode, fmt_spec: str):
        """
        Apply a format= spec to a node.

        For a leaf node (var/all), the format is set directly on it - this
        is the simple "mean*format=7,1" case.

        For a group node (i.e. node came from "(...)" - e.g. the result of
        "(mean gmean)*format=7,1"), the format must be propagated to every
        LEAF inside the group's subtree instead of being set on the group
        node itself. This is necessary because _expand_node flattens away
        group/cross/concat wrapper nodes when building leaf paths, so any
        .fmt set only on the group node would never be seen by
        _classify_path - it has to live on the var/all leaves directly.

        A leaf's own format (set via "var*format=..." written explicitly
        inside the group) takes precedence and is NOT overwritten - this
        lets "(mean*format=7,1 gmean)*format=8,2" give gmean the outer 8,2
        while mean keeps its own explicit 7,1.
        """
        if node.kind in ("var", "all"):
            if node.fmt is None:
                node.fmt = fmt_spec
            return
        # group / cross / concat: recurse into every child
        for child in node.children:
            self._apply_fmt(child, fmt_spec)

    def _parse_atom(self) -> DimNode:
        self._skip_sp()
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of expression")
        t = self.tokens[self.pos]
        if t[0] == "OP" and t[1] == "(":
            self.pos += 1
            inner = self._parse_concat()
            self.consume_op(")")
            return DimNode(kind="group", children=[inner])
        if t[0] == "NAME":
            self.pos += 1
            name_upper = t[1].upper()
            label = t[2]
            # Consume an immediately following FMT token if present
            fmt   = self._consume_fmt()
            denom = self._consume_denom()
            if name_upper in ("ALL", "TOTAL"):
                return DimNode(kind="all", name="ALL", label=label, fmt=fmt, denom=denom)
            return DimNode(kind="var", name=t[1], label=label, fmt=fmt, denom=denom)
        raise SyntaxError(f"Unexpected token: {t}")

    def _consume_fmt(self) -> Optional[str]:
        """Consume and return a FMT token immediately following the current
        position (skipping a single space), or return None."""
        p = self.pos
        # Allow one optional space between token and format=
        if p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        if p < len(self.tokens) and self.tokens[p][0] == "FMT":
            self.pos = p + 1
            return self.tokens[p][1]
        return None

    def _consume_denom(self) -> Optional[str]:
        """Consume and return a DENOM token (e.g. the 'income' from
        pctsum<income>) immediately after the current position, or None."""
        p = self.pos
        if p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        if p < len(self.tokens) and self.tokens[p][0] == "DENOM":
            self.pos = p + 1
            return self.tokens[p][1]
        return None


def _split_dimensions(expr: str) -> list[str]:
    """
    Split on top-level commas (not inside parentheses or quoted strings).

    A comma that is the DECIMAL SEPARATOR in a format=W,D[_|s] spec
    (immediately following the width digits of "format=", e.g.
    "format=7,2" or "format=7,2_") is NOT treated as a dimension separator.
    """
    parts, current, depth = [], [], 0
    in_str = False
    str_char = None
    i = 0
    while i < len(expr):
        ch = expr[i]
        if not in_str and ch in ("'", '"'):
            in_str = True
            str_char = ch
            current.append(ch)
        elif in_str and ch == str_char:
            in_str = False
            str_char = None
            current.append(ch)
        elif in_str:
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            # Check if this comma is the decimal separator in "format=<digits>,"
            so_far = "".join(current)
            if re.search(r'format\s*=\s*[0-9]+$', so_far):
                current.append(ch)  # decimal comma in format=W,D — not a separator
            else:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def parse_table(table_str: str) -> tuple:
    dims = _split_dimensions(table_str)
    if len(dims) > 2:
        raise ValueError("TABLE supports at most 2 dimensions (row, col). The page dimension is not supported.")
    result = []
    for dim in dims:
        tokens = _tokenize(dim.strip())
        result.append(_Parser(tokens).parse())
    return tuple(result)


# ---------------------------------------------------------------------------
# 6. PATH EXPANSION
# ---------------------------------------------------------------------------

def _expand_node(node: DimNode) -> list[list[DimNode]]:
    from itertools import product as iproduct
    if node.kind in ("var", "all"):
        return [[node]]
    if node.kind == "group":
        return _expand_node(node.children[0])
    if node.kind == "concat":
        result = []
        for child in node.children:
            result.extend(_expand_node(child))
        return result
    if node.kind == "cross":
        child_paths = [_expand_node(c) for c in node.children]
        return [
            [leaf for path in combo for leaf in path]
            for combo in iproduct(*child_paths)
        ]
    raise ValueError(f"Unknown node kind: {node.kind}")


def _expand_node_with_branch(node: DimNode):
    """
    Like _expand_node, but additionally returns a top-level branch index for
    each path, used to order specs that come from a TOP-LEVEL concatenation
    (space-separated dimension root) in written left-to-right order.

    Only the OUTERMOST node matters: if the dimension root is itself a
    'concat' (e.g. "origin ALL='Total'" or "origin * type ALL='Subtotal'"
    which parses as concat(cross(origin,type), all)), each top-level child
    gets its own branch index (0, 1, 2, ...) in written order. All paths
    expanded from that child share that branch index.

    If the root is NOT a concat (e.g. pure cross/group like
    "(n rowpctn)*(all age)*(all inc)"), every path gets branch 0 — meaning
    the positional lexicographic sort (via path_order) is solely
    responsible for ordering.

    Returns: list of (branch_index, path) tuples.
    """
    if node.kind == "concat":
        result = []
        for branch_idx, child in enumerate(node.children):
            for path in _expand_node(child):
                result.append((branch_idx, path))
        return result
    return [(0, path) for path in _expand_node(node)]


# ---------------------------------------------------------------------------
# 7. PATH CLASSIFICATION
# ---------------------------------------------------------------------------

def _classify_path(path, measure_list, groupby_list):
    measure_map = {m.upper(): m for m in measure_list}
    groupby_map = {g.upper(): g for g in groupby_list}

    group_keys = []
    var        = None
    var_label  = None
    stat       = None
    stat_label = None
    has_all    = False
    all_label  = None

    for node in path:
        upper = node.name.upper() if node.name else ""
        if node.kind == "all":
            has_all   = True
            all_label = node.label
        elif upper in groupby_map:
            orig = groupby_map[upper]
            lbl  = node.label if node.label is not None else orig
            group_keys.append((orig, lbl))
        elif upper in measure_map:
            orig      = measure_map[upper]
            var       = orig
            var_label = node.label if node.label is not None else orig
        elif upper in ALL_STATS:
            stat       = upper
            stat_label = node.label if node.label is not None else upper
        else:
            raise ValueError(
                f"Token {node.name!r} not found in measure=, groupby=, or known statistics.\n"
                f"Tip: if your label uses the same quote character as the surrounding "
                f"Python string, Python will terminate the string early.\n"
                f"Known statistics: {sorted(ALL_STATS)}"
            )

    path_order = []
    for node in path:
        upper = node.name.upper() if node.name else ""
        if node.kind == "all":
            path_order.append(("all", node.label if node.label is not None else "TOTAL", None))
        elif upper in groupby_map:
            orig_name = groupby_map[upper]
            lbl = node.label if node.label is not None else orig_name
            # Store original column name as third element so _key_to_label
            # can detect whether the label was explicitly renamed by the user.
            path_order.append(("group", lbl, orig_name))
        elif upper in measure_map:
            lbl = node.label if node.label is not None else measure_map[upper]
            path_order.append(("var", lbl, None))
        elif upper in ALL_STATS:
            lbl = node.label if node.label is not None else upper
            path_order.append(("stat", lbl, None))

    # Collect any format specs from the path nodes.
    # The innermost (last) non-None fmt wins so e.g. stat*format=7.1
    # overrides a measure-level format.
    fmt = None
    for node in path:
        if node.fmt is not None:
            fmt = node.fmt

    # Collect denom_def from any node that carries one (stat node with <...>)
    denom_def = None
    for node in path:
        if node.denom is not None:
            denom_def = node.denom

    return {
        "group_keys": group_keys,
        "var":        var,
        "var_label":  var_label,
        "stat":       stat,
        "stat_label": stat_label,
        "has_all":    has_all,
        "all_label":  all_label,
        "path_order": path_order,
        "fmt":        fmt,
        "denom_def":  denom_def,
    }


# ---------------------------------------------------------------------------
# 8. AGGREGATION
# ---------------------------------------------------------------------------

def _compute_series(data, all_groups, var, stat, r_groups, c_groups, missing, weight=None):
    """
    Compute an aggregated Series for a single (row_spec, col_spec) pair
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

    Returns
    -------
    pd.Series  indexed by the all_groups groupby key(s)

    Notes
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


# ---------------------------------------------------------------------------
# 9. ALL (MARGINAL TOTAL) COMPUTATION
# ---------------------------------------------------------------------------

def _compute_all_series(data, groups_to_keep, var, stat, missing,
                        r_groups=None, c_groups=None, weight=None):
    """
    Compute an aggregated Series for a spec that involves an ALL/TOTAL
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
                n_c_ctx = len(groups_to_keep) - len(r_groups)
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


# ---------------------------------------------------------------------------
# 10. CUSTOM DENOMINATOR PERCENTAGE COMPUTATION
# ---------------------------------------------------------------------------

def _parse_denom_def(denom_str: str) -> list[str]:
    """
    Parse the content of a <...> denominator definition into an ordered
    list of uppercase token strings.

    Each token is one of:
      - An uppercase measure variable name (e.g. 'INCOME')
      - An uppercase class variable name   (e.g. 'GENDER')
      - 'ALL' or 'TOTAL'                   (grand-total fallback)

    Multiple tokens are tried in left-to-right order by _compute_custom_pct;
    the first token that "matches" the current subtable is used as the
    denominator.  Placing ALL/TOTAL last provides a fallback for cells where
    none of the named variables participate.

    Examples
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
    weight: str = None,
    r_path_order: list = None,
    c_path_order: list = None,
) -> pd.Series:
    """
    Compute a percentage statistic with a user-defined denominator.

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

    Examples
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
        if v is None: return "__nan__"
        try:
            if isinstance(v, float) and np.isnan(v): return "__nan__"
        except (TypeError, ValueError): pass
        if v is pd.NA or v is pd.NaT: return "__nan__"
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


# ---------------------------------------------------------------------------
# 11. MAIN PUBLIC FUNCTION
# ---------------------------------------------------------------------------

_SENTINEL = "__total__"


def flextab(
    data: pd.DataFrame,
    measure: "str | list" = None,
    groupby: "str | list" = None,
    table: str = None,
    include_missing_in_groupby: bool = True,
    fmt: str = "{:.1f}",
    na_rep: str = None,
    labels: dict = None,
    sort_by: str = 'code',
    weight: str = None,
    row_header: str = None,
    style: dict = None,
) -> pd.DataFrame:
    """
    Build a cross-tabulation table, equivalent to SAS PROC TABULATE.

    Returns a FlextabResult — a pd.DataFrame subclass that keeps the
    underlying values numeric (for further computation or export) but
    applies format= specs from the TABLE expression when displayed.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    measure : str or list of str, optional
        Numeric analysis variable name(s) (SAS: VAR). A single column can be
        passed as a plain string, e.g. measure="income" instead of
        measure=["income"]. Omit for count-only tables that use N, COUNT,
        SIZE or percent statistics.

    groupby : str or list of str, optional
        Categorical grouping variable name(s) (SAS: CLASS). A single column
        can be passed as a plain string, e.g. groupby="origin" instead of
        groupby=["origin"].

    table : str, optional
        TABLE expression.  Syntax summary:

          row_expr , col_expr      comma separates row and column dimensions
          A * B                   cross/nest: A values crossed with B values
          A B                     concatenation: A then B (space-separated)
          ( A B )                 group: treat as one unit for * and format=
          ALL  or  TOTAL          marginal total (synonymous keywords)
          name='Label'            rename any token; name='' suppresses it

        Format specifications (attach to any preceding token or group):
          *format=W.D             D decimals, decimal point
          *format=W,D             D decimals, decimal comma (European)
          *format=W.D_            decimal point + comma thousands separator
          *format=W,D_            decimal comma + dot thousands separator
          *format=W.Ds            decimal point + space thousands separator
          *format=W,Ds            decimal comma + space thousands separator

          Apply one format to several stats at once:
            (mean gmean)*format=7,1*income

        Custom denominator definitions (after PCTN or PCTSUM):
          PCTN<denom>  or  PCTSUM<denom>
          where <denom> is a space-separated list of:
            measure_col  -> denominator = sum of that measure in same group
            groupby_col  -> denominator = subtotal within all values of
                            that class variable
            ALL or TOTAL -> grand total
          Multiple tokens: first one that applies to the subtable is used.

          Examples:
            tax*pctsum<income>           tax as % of income
            income*pctsum<gender all>    % of gender subtotal; ALL fallback
            pctn<origin all>             count % of origin subtotal

    include_missing_in_groupby : bool, default True
        When True, NaN values in groupby columns appear as their own group
        level. When False, rows with missing groupby values are excluded.

    fmt : str, default "{:.1f}"
        Default Python format string for numeric cells that have no
        per-column format= spec in the TABLE expression.  Used by print(),
        repr(), and flextab_to_string() unless overridden there.

    na_rep : str, optional
        Text shown in place of NaN / missing cells (e.g. na_rep='-' or
        na_rep='.'). When None (default), NaN cells remain as NaN floats,
        which is best for further numeric operations.

    labels : dict of dict, optional
        Remap groupby values to display labels. Outer key = column name,
        inner dict maps original values to display labels.  Groupby
        aggregation always uses the original codes; remapping is applied
        only for display.
        Example: {'gender': {1: 'Male', 2: 'Female'},
                  'region': {'N': 'North', 'S': 'South'}}

    sort_by : {'code', 'index', 'label'}, default 'code'
        Controls the order of groupby levels:
          'code'   sort by original data values (before label remapping)
          'index'  sort by position in the labels dict (dict insertion
                   order) — use when the dict defines the desired order
          'label'  sort alphabetically by the display label text
        'index' and 'label' only affect columns that appear in labels=;
        columns without a label dict always sort by 'code'.

    weight : str, optional
        Name of a numeric column to use as a frequency weight
        (SAS: WEIGHT statement).  Rules:
          weight = 0       counted in N; contributes 0 to weighted sums
          weight < 0       treated as 0 (still counted in N)
          weight = missing row excluded entirely
        Weighted formulas used:
          SUM    sum(w * x)
          MEAN   sum(w*x) / sum(w)
          VAR    reliability-weights unbiased: sum(w)/(sum(w)^2-sum(w^2))
                 * sum(w*(x-xbar)^2)  [reduces to n-1 when all w_i = 1]
          STD    sqrt(VAR)
          STDERR sqrt(VAR / sum(w))
          GMEAN  exp(sum(w*log(x)) / sum(w))
          HMEAN  sum(w) / sum(w/x)
          Percentiles: weighted empirical CDF
        N, COUNT, SIZE and NMISS are NEVER weighted.

    row_header : str, optional
        Name to assign to the row index.  Equivalent to setting
        result.index.name (flat index) or result.index.names[0]
        (MultiIndex) after the call.
        Example: row_header='Region' labels the leftmost index column.

    style : dict, optional
        Colour styling for display and Excel export.  Omit any key to leave
        that area unstyled.  Colours may be specified as:
          - Named string:   'blue', 'red', 'lightgrey', 'navy', …
          - Hex string:     '#4472C4'  or  '4472C4'
          - RGB tuple:      (70, 114, 196)

        Every key below uses the SAME technique: the value is either a
        single colour (applied to every row) or a 2-tuple (colour0,
        colour1) that CYCLES through all rows: row 0 → colour0, row 1 →
        colour1, row 2 → colour0, and so on.

        Keys:
          'header_bg'      background colour for the column header cells
          'header_fg'      foreground (text) colour for column header cells
          'row_bg'         background colour for the row index cells
                           (the groupby label/value cells on the left)
          'row_fg'         foreground colour for the row index cells
          'row_header_bg'  background colour for the row header cell — the
                           corner cell showing the text passed via the
                           row_header= argument to flextab()
          'row_header_fg'  foreground colour for the row header cell
          'cell_bg'        background colour for the table's data cells
          'cell_fg'        foreground colour for the table's data cells

        The styling is applied in three places:
          1. Jupyter HTML display (_repr_html_): inline CSS on <tr>/<th>
          2. print() / repr(): terminal output is uncoloured (plain text)
          3. result.to_excel('file.xlsx'): openpyxl PatternFill + Font

        Example::

            style={
                'header_bg': '#4472C4', 'header_fg': 'white',
                'row_bg':    'lightgrey', 'row_fg': 'black',
                'row_header_bg': '#4472C4', 'row_header_fg': 'white',
                'cell_bg': ('white', '#EBF3FB'),  # row0=white, row1=pale blue
            }

    Returns
    -------
    FlextabResult
        A pd.DataFrame subclass.  Numeric values are preserved for
        computation (.sum(), arithmetic, etc.).  format= specs from the
        TABLE expression are applied automatically by print(), repr(),
        and Jupyter cell display.

        Call result.to_excel(path) to export with full number formatting
        and colour styling preserved via openpyxl post-processing.
        Use flextab_to_string(result, fmt, na_rep) for explicit string
        rendering control.
    """
    # Allow passing a single column name as a plain string instead of a
    # one-element list, e.g. measure="income" instead of measure=["income"].
    if isinstance(measure, str):
        measure = [measure]
    if isinstance(groupby, str):
        groupby = [groupby]

    measure = measure or []
    groupby = groupby or []
    missing = include_missing_in_groupby
    labels  = labels  or {}
    # Flat lookup: original_value -> display_label for any groupby column.
    # Used by _fmt_val inside _key_to_label to remap codes to labels.
    _label_map = labels  # kept separate so groupby always uses original codes

    if table is None:
        if measure:
            if groupby:
                row_expr = " ".join(groupby)
                col_expr = " ".join(f"{v} * (N MEAN)" for v in measure)
                table = f"{row_expr}, {col_expr}"
            else:
                table = " ".join(f"{v} * (N MEAN)" for v in measure)
        else:
            if groupby:
                row_expr = " ".join(groupby)
                table = f"{row_expr}, N"
            else:
                table = "N"

    dims = parse_table(table)
    if len(dims) == 1:
        row_dim, col_dim = None, dims[0]
    else:
        row_dim, col_dim = dims[0], dims[1]

    def expand_dim(dim_node):
        if dim_node is None:
            return [{"group_keys": [], "var": None, "var_label": None,
                     "stat": None, "stat_label": None,
                     "has_all": False, "all_label": None,
                     "path_order": [], "branch": 0}]
        specs = []
        for branch_idx, path in _expand_node_with_branch(dim_node):
            spec = _classify_path(path, measure, groupby)
            spec["branch"] = branch_idx
            specs.append(spec)
        return specs

    col_specs = expand_dim(col_dim)
    row_specs = expand_dim(row_dim)

    def spec_header(spec):
        # Build a header tuple that uniquely identifies this spec.
        # For group entries, use the label when non-blank.
        # When the label is blank (suppressed with =''), fall back to the
        # original column name so that e.g. origin='' and type='' produce
        # distinct headers ('origin',) and ('type',) rather than both
        # collapsing to ('',), which would cause sorting to treat all their
        # values as belonging to the same spec and sort them together.
        # The orig_name is used ONLY as an internal discriminator here —
        # it does not affect what gets displayed in the table header.
        parts = []
        for entry in spec["path_order"]:
            label    = entry[1]
            orig     = entry[2] if len(entry) > 2 else None
            is_group = entry[0] == "group"
            if is_group and not label and orig:
                # Blank label on a group token → use orig_name internally
                parts.append(f"\x00{orig}")  # prefix ensures no collision with real labels
            else:
                parts.append(label)
        return tuple(parts) if parts else ("",)

    def orig_groups(spec):
        return [col for col, _ in spec["group_keys"]]

    cells: dict = {}
    row_hdr_path: dict = {}
    col_hdr_path: dict = {}

    for r_spec in row_specs:
        r_hdr    = spec_header(r_spec)
        r_groups = orig_groups(r_spec)
        row_hdr_path.setdefault(r_hdr, (r_spec["path_order"], r_spec["branch"]))

        for c_spec in col_specs:
            c_hdr    = spec_header(c_spec)
            c_groups = orig_groups(c_spec)
            col_hdr_path.setdefault(c_hdr, (c_spec["path_order"], c_spec["branch"]))

            var  = r_spec["var"]  or c_spec["var"]
            stat = r_spec["stat"] or c_spec["stat"]

            if stat is None:
                stat = "N"

            # Custom denominator definition from <...> syntax
            denom_def = r_spec.get("denom_def") or c_spec.get("denom_def")

            all_groups = list(dict.fromkeys(r_groups + c_groups))

            has_all_r = r_spec["has_all"]
            has_all_c = c_spec["has_all"]

            if denom_def is not None and stat.upper() in ("PCTSUM", "PCTN"):
                # Custom-denominator percentage: route to dedicated function
                series = _compute_custom_pct(
                    data=data,
                    r_groups=r_groups,
                    c_groups=c_groups,
                    var=var,
                    stat=stat,
                    denom_def=denom_def,
                    groupby=groupby,
                    measure=measure,
                    missing=missing,
                    weight=weight,
                    r_path_order=r_spec["path_order"],
                    c_path_order=c_spec["path_order"],
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            elif not has_all_r and not has_all_c:
                series = _compute_series(
                    data, all_groups, var, stat, r_groups, c_groups, missing,
                    weight=weight
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            elif has_all_r and not has_all_c:
                # ALL on rows: c_groups drive the column denominator for COLPCTN
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(data, keep, var, stat, missing,
                                             r_groups=r_groups, c_groups=c_groups,
                                             weight=weight)
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            elif not has_all_r and has_all_c:
                # ALL on cols: r_groups drive the row denominator for ROWPCTN
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(data, keep, var, stat, missing,
                                             r_groups=r_groups, c_groups=c_groups,
                                             weight=weight)
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            else:
                # ALL on both: pass both for correct denominator selection
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(data, keep, var, stat, missing,
                                             r_groups=r_groups, c_groups=c_groups,
                                             weight=weight)
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

    def _index_value_key(orig_col, v):
        """
        Sort key for sort_by='index': order by each value's POSITION in the
        labels dict as written by the caller (dict insertion order).

        Looks up v ONLY in labels[orig_col] (the dict belonging to this
        specific groupby column), never in other columns' label dicts, so
        a raw value like 1 used in two different columns can never borrow
        the wrong column's label/order.

        Values not present in the label dict (or when the column has no
        label dict at all) fall back to their normalised string form,
        sorted after all explicitly labelled values.
        """
        col_labels = _label_map.get(orig_col) if orig_col else None
        if col_labels and v in col_labels:
            keys_in_order = list(col_labels.keys())
            return (0, keys_in_order.index(v))
        return (1, _na_safe_str(v))

    def _label_text_value_key(orig_col, v):
        """
        Sort key for sort_by='label': order alphabetically by the DISPLAY
        LABEL TEXT (the dict's value), not by dict-write order and not by
        the raw code.

        Looks up v ONLY in labels[orig_col], so values are never resolved
        against the wrong column's dict. Values without a label fall back
        to their normalised string form, sorted after all explicitly
        labelled values.
        """
        col_labels = _label_map.get(orig_col) if orig_col else None
        if col_labels and v in col_labels:
            return (0, str(col_labels[v]))
        return (1, _na_safe_str(v))

    def _na_safe_str(v):
        if v is None or v == _NAN_SENTINEL:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _sort_keys(keys, hdr_path):
        """Sort row/col keys, respecting sort_by='code', 'index', or 'label'."""
        if sort_by == 'index' and _label_map:
            value_key_fn = _index_value_key
        elif sort_by == 'label' and _label_map:
            value_key_fn = _label_text_value_key
        else:
            value_key_fn = None
        return _sort_row_keys(keys, hdr_path, value_key_fn=value_key_fn)

    all_row_keys = _sort_keys(list(dict.fromkeys(rk for rk in cells)), row_hdr_path)
    all_col_keys = _sort_keys(list(dict.fromkeys(ck for rk in cells for ck in cells[rk])), col_hdr_path)

    matrix = np.full((len(all_row_keys), len(all_col_keys)), np.nan)
    rk_pos = {rk: i for i, rk in enumerate(all_row_keys)}
    ck_pos = {ck: j for j, ck in enumerate(all_col_keys)}

    for rk, col_dict in cells.items():
        for ck, val in col_dict.items():
            matrix[rk_pos[rk], ck_pos[ck]] = val

    def _fmt_val(v, col_name=None):
        """
        Format a group key value for display, applying label remapping.

        col_name : the original groupby column this value belongs to. Only
                   that column's label dict (labels[col_name]) is consulted,
                   so the same raw value (e.g. 1) used in two different
                   groupby columns never gets the wrong column's label.
                   If col_name is None or not in _label_map, no remapping
                   is applied beyond the nan/sentinel handling.
        """
        if v is None or v == _NAN_SENTINEL:
            return "nan"
        if isinstance(v, float):
            try:
                if np.isnan(v):
                    return "nan"
            except (TypeError, ValueError):
                pass
        if col_name is not None:
            col_labels = _label_map.get(col_name)
            if col_labels and v in col_labels:
                return str(col_labels[v])
        return str(v)

    def _compute_slot_layout(all_path_orders):
        """
        Compute display slots per position across all specs.

        Each cross-position gets:
          2 slots  if ANY spec has kind='group' there WITH a non-blank label
                   (groupby variables need a label row + a value row)
          1 slot   otherwise (stat/var/all, or group with label='' suppressed)

        Returns a list of ints, outermost first.
        """
        if not all_path_orders:
            return []
        max_len = max((len(po) for po in all_path_orders), default=0)
        slots = []
        for i in range(max_len):
            has_labeled_group = any(
                i < len(po)
                and po[i][0] == "group"
                and po[i][1]          # label is non-blank
                for po in all_path_orders
            )
            slots.append(2 if has_labeled_group else 1)
        return slots

    def _key_to_label_slotted(hdr, data_key, path_order, slots):
        """
        Build a fixed-length index tuple using a pre-computed slot layout.

        D = sum(slots) levels total. Slots assigned bottom-up: the innermost
        (rightmost) path_order position occupies the lowest (rightmost) slots.

        For a spec whose path_order is SHORTER than the full slot list (i.e.
        it has fewer cross-positions than the deepest spec), its tokens are
        placed starting at the BOTTOM of the available slots — front-padding
        with blanks — so that shallower specs always align at the bottom
        level alongside deeper specs' innermost values.

        Within a position:
          - stat/var/ALL (1 slot): value -> low slot; high slot blank
          - group with non-blank label (2 slots): label -> high slot, value -> low slot
          - group with blank label (1 slot): value -> low slot only (label suppressed)
        """
        if not path_order:
            return hdr

        D = sum(slots)
        is_total = (not data_key) or data_key == (_SENTINEL,)
        dvals = [] if is_total else list(
            data_key if isinstance(data_key, tuple) else (data_key,)
        )
        data_iter = iter(dvals)

        row = [""] * D
        group_slots_pos = [i for i, e in enumerate(path_order) if e[0] == "group"]

        # Slots used by THIS spec's path_order positions
        my_slots = slots[-len(path_order):]   # align from bottom
        my_D = sum(my_slots)
        # Offset: how many bottom-slots this spec doesn't use (front-pad)
        offset = D - my_D

        def slot_range(local_pos):
            # local_pos is the index within path_order (0 = outermost of THIS spec)
            # map to the global slots list (bottom-aligned)
            global_pos = len(slots) - len(path_order) + local_pos
            low  = sum(slots[global_pos + 1:])
            high = low + slots[global_pos] - 1
            return D - 1 - high, D - 1 - low  # (hi_idx, lo_idx)

        for pos, entry in enumerate(path_order):
            kind      = entry[0]
            label     = entry[1]
            orig_name = entry[2] if len(entry) > 2 else None
            hi_idx, lo_idx = slot_range(pos)

            if kind != "group":
                row[lo_idx] = label
            else:
                if is_total:
                    pass
                else:
                    val     = next(data_iter, None)
                    val_str = _fmt_val(val, col_name=orig_name)
                    is_last = (pos == group_slots_pos[-1])
                    global_pos = len(slots) - len(path_order) + pos
                    has_label_slot = (slots[global_pos] == 2)
                    # In the 2-slot system, every group position has a dedicated
                    # label slot (hi_idx) and value slot (lo_idx). Whether a
                    # group is "preceding" or "last" no longer matters for slot
                    # allocation — both always emit label at hi_idx (if non-blank)
                    # and value at lo_idx. The old "preceding group → value only"
                    # rule was a 1-slot workaround; it dropped labels that now
                    # have their own dedicated row.
                    if label and has_label_slot:
                        row[hi_idx] = label
                    row[lo_idx] = val_str

        return tuple(row) if any(row) else ("",)

    def make_index(keys, hdr_path):
        """
        Convert (hdr, data_key) pairs to a MultiIndex using slot-based layout.

        Every spec in the dimension produces a fixed-length tuple of the same
        depth D = sum(slots), where the slot layout is computed globally so
        that all specs align correctly:
          - groupby variables occupy 2 slots (label row + value row)
          - stat / var / ALL tokens occupy 1 slot each
          - shorter specs are bottom-aligned (front-padded with blanks) so
            their innermost token always lands at the same absolute level as
            the innermost token of deeper specs

        After building the tuples, any level that is blank across ALL columns
        is dropped — these are structural artefacts (e.g. the label slot of a
        group whose label was suppressed with ='') that carry no information.

        Examples
        --------
        Expression ``n colpctn*(all age_group)`` produces specs:
          [stat:N]                         → ('N',)        1 position
          [stat:COLPCTN, all:TOTAL]        → ('COLPCTN','TOTAL')
          [stat:COLPCTN, group:age_group]  → ('COLPCTN','age_group','10-19')

        Slot layout: pos0=stat(1 slot), pos1=group/all(2 slots because age_group
        has a non-blank label) → D=3.  After bottom-alignment and blank-dropping:
          N          → ('',       'N')      ← blank level 0 dropped, level 2 dropped
          COLPCTN/TOTAL → ('COLPCTN','TOTAL','')
          COLPCTN/age=10-19 → ('COLPCTN','age_group','10-19')

        Expression ``origin * (type total='Subtotal')`` produces specs:
          [group:origin, group:type]   → ('origin','Asia','type','SUV')
          [group:origin, all:Subtotal] → ('origin','Asia','Subtotal','')
        Both have D=4; the Subtotal row's trailing '' is kept so 'Asia' aligns
        vertically with 'Asia' in the detail rows (2-slot group positions).
        """
        all_po = [hdr_path.get(hdr, ([], 0))[0] for hdr, _ in keys]
        slots  = _compute_slot_layout(all_po)
        D      = sum(slots)

        if D == 0:
            return pd.Index([""] * len(keys))

        labels = [
            _key_to_label_slotted(
                hdr, dk,
                hdr_path.get(hdr, ([], 0))[0],
                slots,
            )
            for hdr, dk in keys
        ]

        # Drop levels that are blank in every column
        if labels and len(labels[0]) > 1:
            keep = [
                i for i in range(len(labels[0]))
                if any(t[i] for t in labels)
            ]
            if len(keep) < len(labels[0]):
                labels = [tuple(t[i] for i in keep) for t in labels]

        D_final = len(labels[0]) if labels else 0
        if D_final == 0:
            return pd.Index([""] * len(keys))
        if D_final == 1:
            return pd.Index([t[0] for t in labels])
        return pd.MultiIndex.from_tuples(labels)

    row_idx = make_index(all_row_keys, row_hdr_path)
    col_idx = make_index(all_col_keys, col_hdr_path)

    base = pd.DataFrame(matrix, index=row_idx, columns=col_idx)

    # Build a column-format map: col_index_position -> formatter callable.
    # Each column spec may carry a fmt spec from the TABLE expression.
    col_fmt_map = {}
    for j, ck in enumerate(all_col_keys):
        c_hdr, c_data = ck
        for c_spec in col_specs:
            if spec_header(c_spec) == c_hdr and c_spec.get("fmt"):
                col_fmt_map[j] = _parse_fmt_spec(c_spec["fmt"])
                break

    # Build a row-format map: row_index_position -> formatter callable.
    # Each row spec may carry a fmt spec from the TABLE expression (e.g.
    # when the statistic with format= lives in the row dimension instead
    # of the column dimension, as in "n*format=6,0 rowpctn*format=7,1, ...").
    row_fmt_map = {}
    for i, rk in enumerate(all_row_keys):
        r_hdr, r_data = rk
        for r_spec in row_specs:
            if spec_header(r_spec) == r_hdr and r_spec.get("fmt"):
                row_fmt_map[i] = _parse_fmt_spec(r_spec["fmt"])
                break

    # Replace NaN cells with na_rep text when requested.
    if na_rep is not None:
        base = base.astype(object)
        base = base.where(base.notna(), other=na_rep)
        # na_rep forces object dtype -> col_fmt_map can't be applied to numerics
        # reliably anymore for the repr path, but flextab_to_string still works
        # via pd.isna() check before formatting.

    result = FlextabResult(base)
    result.attrs["col_fmt_map"] = col_fmt_map
    result.attrs["row_fmt_map"] = row_fmt_map
    result.attrs["default_fmt"] = fmt
    result.attrs["style"]       = style or {}

    # Apply row_header: name the row index so it prints as a column label
    if row_header is not None:
        if isinstance(result.index, pd.MultiIndex):
            result.index.names = [row_header] + list(result.index.names[1:])
        else:
            result.index.name = row_header

    return result


# ---------------------------------------------------------------------------
# 12. ROW-KEY SORT
# ---------------------------------------------------------------------------

def _sort_row_keys(row_keys: list, hdr_path: dict = None, value_key_fn=None) -> list:
    """
    Re-order row/column keys to follow the TABLE expression's written order.

    Two ordering rules combine, applied in this priority:

    1. BRANCH (primary) - if the dimension's root node is a top-level
       concatenation (space-separated, e.g. "origin ALL='Total'" or
       "origin * type ALL='Subtotal'"), each top-level concat child keeps
       its own written left-to-right position as the dominant sort key.
       This ensures e.g. "origin ALL='Total'" puts Total AFTER all origin
       rows (concat = independent blocks in written order), rather than
       interleaving ALL by value.
       When the dimension has no top-level concat (pure '*' nesting, e.g.
       "(n rowpctn)*(all age)*(all inc)"), every spec shares branch 0 and
       this rule has no effect - ordering is fully determined by rule 2.

    2. POSITIONAL LEXICOGRAPHIC (secondary, within the same branch) -
       every spec's path_order has one entry per '*'-crossed position:
         ('group', label, orig_col) - a real groupby variable (e.g. 'age')
         ('all',   label, None)     - the ALL keyword in that position
         ('stat',  label, None)     - a statistic name (e.g. 'N', 'ROWPCTN')
         ('var',   label, orig_col) - a measure name
       Across specs, position i always refers to the same semantic role
       (whichever spec has kind='group' there defines it as a groupby
       position; otherwise it's a stat/var position).
       For each (hdr, data_key), one sort element is built per position:
         - groupby position: (0, '') for ALL (sorts first), or
           (1, sort_value) for a real value, where sort_value comes from
           value_key_fn(orig_col, value) if provided, else the normalised
           string representation of value.
         - stat/var position: ordinal by first-seen label order (e.g.
           'N' before 'ROWPCTN').
       Comparing these tuples left-to-right reproduces the nested-loop
       order of the TABLE expression's '*' structure, with the
       LAST-written dimension varying fastest.

    Grand-total (sentinel) data keys are treated as ALL for every groupby
    position in that spec.

    hdr_path     : dict mapping hdr tuple -> (path_order, branch_index).
    value_key_fn : optional callable (orig_col, value) -> sortable key,
                   used instead of the default string normalisation for
                   real groupby values. Lets callers sort by something
                   other than the raw code's string form - e.g. by each
                   value's position in a user-supplied label dict
                   (sort_by='index'), or by the label text itself
                   (sort_by='label').
    """
    hdr_path = hdr_path or {}

    def _normalise(v):
        if v is None or v == _NAN_SENTINEL:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _value_key(orig_col, v):
        if value_key_fn is not None:
            return value_key_fn(orig_col, v)
        return _normalise(v)

    def _po(hdr):
        return hdr_path.get(hdr, ([], 0))[0]

    def _branch(hdr):
        return hdr_path.get(hdr, ([], 0))[1]

    # Determine the maximum path length and, for each position, whether ANY
    # spec has a real groupby variable there (kind='group').
    max_len = 0
    is_group_position: dict[int, bool] = {}
    for hdr, _ in row_keys:
        po = _po(hdr)
        max_len = max(max_len, len(po))
        for i, entry in enumerate(po):
            if entry[0] == "group":
                is_group_position[i] = True
            else:
                is_group_position.setdefault(i, False)

    # For STAT/VAR (non-group) positions, assign an ordinal to each distinct
    # label in first-seen order, so e.g. 'N' < 'ROWPCTN' is preserved.
    #
    # IMPORTANT: this must be scoped PER BRANCH, not globally across the
    # whole dimension. Each top-level concatenated piece of the TABLE
    # expression (e.g. "sum*w n*format=... colpctn*format=... (SUM
    # COLPCTSUM n nmiss ...)*format=...*income") is its own branch, and the
    # branch index already determines their relative order (branch is the
    # primary sort key, compared before any of this). If label_order were
    # shared globally, a stat name seen early in one branch (e.g. a
    # standalone "n*format=6,0") would claim a low ordinal that then leaks
    # into an unrelated later branch containing the SAME stat name again
    # (e.g. "n" inside "(SUM COLPCTSUM n nmiss ...)"), silently reordering
    # that branch's internal stats to match the EARLIER branch's first
    # appearance instead of THIS branch's own written order.
    label_order: dict[tuple, dict[int, dict]] = {}
    for hdr, _ in row_keys:
        po = _po(hdr)
        branch = _branch(hdr)
        branch_orders = label_order.setdefault(branch, {})
        for i, entry in enumerate(po):
            if not is_group_position.get(i, False):
                d = branch_orders.setdefault(i, {})
                if entry[1] not in d:
                    d[entry[1]] = len(d)

    # For each group-position, compute which distinct original column names
    # appear there across all specs, in first-seen order. Specs sharing the
    # same orig_col at a given position should have their values sorted
    # together (e.g. all origin values); specs with different orig_cols at
    # that position should be fully separated (e.g. gender block then
    # age_group block, even when they share value strings like '1' and '2').
    pos_col_ordinal: dict[int, dict] = {}
    for hdr, _ in row_keys:
        po = _po(hdr)
        for i, entry in enumerate(po):
            if is_group_position.get(i, False) and entry[0] == "group":
                orig_col = entry[2] if len(entry) > 2 else entry[1]
                d = pos_col_ordinal.setdefault(i, {})
                if orig_col not in d:
                    d[orig_col] = len(d)

    def sort_key(rk):
        hdr, dk = rk
        po = _po(hdr)
        branch = _branch(hdr)

        if dk == (_SENTINEL,):
            dk = ()

        dk_iter = iter(dk if isinstance(dk, tuple) else (dk,))
        parts = []
        for i in range(max_len):
            if i >= len(po):
                parts.append((0, 0, ""))
                continue
            entry = po[i]
            if is_group_position.get(i, False):
                if entry[0] == "group":
                    val = next(dk_iter, "")
                    orig_col = entry[2] if len(entry) > 2 else None
                    # col_ord separates specs whose position-i variables differ
                    # (e.g. gender vs age_group both at position 1).
                    # Specs sharing the same orig_col at this position compare
                    # by value alone (col_ord ties → compare val).
                    col_ord = pos_col_ordinal.get(i, {}).get(orig_col, 0)
                    parts.append((1, col_ord, _value_key(orig_col, val)))
                else:
                    # ALL/TOTAL at a group position: sort first within its col_ord=0
                    parts.append((0, 0, ""))
            else:
                ordinal = label_order.get(branch, {}).get(i, {}).get(entry[1], 0)
                parts.append((ordinal,))
        return (branch,) + tuple(parts)

    return sorted(row_keys, key=sort_key)


# ---------------------------------------------------------------------------
# 13. INTERNAL CELL-FILL HELPER
# ---------------------------------------------------------------------------

# Canonical sentinel for all missing-value types in group keys.
# Using a distinct string ensures consistent hashing regardless of whether
# pandas returns float nan, pd.NA, pd.NaT, or None for missing group levels.
_NAN_SENTINEL = "__nan__"


def _normalise_key(val):
    """
    Normalise a group key value so that all missing-value representations
    (float nan, pd.NA, pd.NaT, None) map to a single canonical object.
    This prevents duplicate row/column keys when different aggregation calls
    (e.g. groupby on a column vs .size()) return different NA types.
    """
    if val is None:
        return _NAN_SENTINEL
    if val is pd.NA or val is pd.NaT:
        return _NAN_SENTINEL
    try:
        if isinstance(val, float) and np.isnan(val):
            return _NAN_SENTINEL
    except (TypeError, ValueError):
        pass
    return val


def _normalise_idx(idx_tuple):
    """Apply _normalise_key to every element of an index tuple."""
    return tuple(_normalise_key(v) for v in idx_tuple)


def _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups):
    """Distribute a grouped Series into the cells dict."""
    n_r = len(r_groups)
    n_c = len(c_groups)

    for idx_val, value in series.items():
        if isinstance(series.index, pd.MultiIndex):
            idx_tuple = idx_val
        else:
            idx_tuple = (idx_val,)

        if idx_tuple == (_SENTINEL,):
            r_data = (_SENTINEL,)
            c_data = (_SENTINEL,)
        else:
            # Normalise before slicing so all NA types hash consistently
            idx_tuple = _normalise_idx(idx_tuple)
            r_data = idx_tuple[:n_r] if n_r else (_SENTINEL,)
            c_data = idx_tuple[n_r:n_r + n_c] if n_c else (_SENTINEL,)

        rk = (r_hdr, r_data)
        ck = (c_hdr, c_data)
        if rk not in cells:
            cells[rk] = {}
        cells[rk][ck] = value



# ---------------------------------------------------------------------------
# 14. FORMAT SPEC PARSER
# ---------------------------------------------------------------------------

def _parse_fmt_spec(spec: str):
    """
    Parse a format specification string and return a callable formatter.

    Called internally when a *format=... suffix is encountered in the TABLE
    expression.  The resulting formatter is stored in result.attrs['col_fmt_map']
    or result.attrs['row_fmt_map'] and applied per-cell by _format_dataframe().

    Syntax:  W<sep>D[modifier]
      W         total width (accepted for SAS compatibility, ignored in output
                since Python handles column width automatically)
      <sep>     the decimal separator character:
                  '.'  decimal point  (standard)
                  ','  decimal comma  (European style)
      D         number of decimal places
      modifier  optional trailing character(s):
                  '_'  add a thousands separator using the OTHER separator
                       character: if sep='.', thousands uses ','; if sep=',',
                       thousands uses '.'
                  's'  add a thousands separator using a SPACE

    The W and sep together determine the output style — W itself is not
    used for padding since flextab() returns string-valued cells.

    Examples
    --------
      "7.1"   -> 1 decimal, point:              1 234.6
      "7,2"   -> 2 decimals, comma:             1 234,56
      "12.0_" -> 0 decimals, comma thousands:   1,235
      "7,2_"  -> 2 decimals, European:          1.234,56
      "9.0s"  -> 0 decimals, space thousands:   1 235
      "7,2s"  -> 2 decimals, comma + space:     1 234,56

    Returns
    -------
    Callable (value: Any) -> str
        A formatting function.  Non-numeric values are returned as str(value).
    """
    m = re.fullmatch(r'[0-9]+([.,])([0-9]+)([_s]*)', spec.strip())
    if not m:
        raise ValueError(f"Invalid format spec {spec!r}. Expected W.D[_|s] or W,D[_|s].")
    dec_sep   = m.group(1)          # '.' or ',' — the decimal separator
    decimals  = int(m.group(2))
    modifiers = m.group(3)
    use_comma       = (dec_sep == ',')  # decimal comma instead of decimal point
    use_thousands   = '_' in modifiers  # thousands grouping (opposite char of dec_sep)
    use_space_thous = 's' in modifiers  # thousands grouping with space

    def _fmt(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)

        if use_thousands or use_space_thous:
            # Python's "," format spec inserts a comma thousands separator
            s = f"{v:,.{decimals}f}"
        else:
            s = f"{v:.{decimals}f}"

        if use_space_thous:
            if use_comma:
                # s looks like "1,234.56" (thousands=',' decimal='.').
                # Replace thousands ',' with space, then decimal '.' with ','.
                s = s.replace(',', ' ').replace('.', ',')
            else:
                s = s.replace(',', ' ')
        elif use_comma:
            # Swap . and , for European style: 1,234.56 -> 1.234,56
            s = s.replace(',', '\u00b6').replace('.', ',').replace('\u00b6', '.')

        return s

    return _fmt


# ---------------------------------------------------------------------------
# 15. PRETTY-PRINT HELPER
# ---------------------------------------------------------------------------

def _format_dataframe(result, fmt="{:.3f}", na_rep="."):
    """
    Build a string-valued copy of result with per-column AND per-row formats
    applied.

    A format= spec can appear on either the row or the column dimension of
    the TABLE expression (e.g. "n*format=6,0 rowpctn*format=7,1, ..." puts
    the format on the row stat instead of a column stat). For each cell,
    the column's format (result.attrs["col_fmt_map"]) takes precedence if
    present; otherwise the row's format (result.attrs["row_fmt_map"]) is
    used; otherwise the default fmt string applies.
    """
    col_fmt_map = result.attrs.get("col_fmt_map", {})
    row_fmt_map = result.attrs.get("row_fmt_map", {})

    def fmt_val(v, row_pos, col_pos):
        if pd.isna(v):
            return na_rep
        formatter = col_fmt_map.get(col_pos) or row_fmt_map.get(row_pos)
        if formatter is not None:
            return formatter(v)
        try:
            return fmt.format(float(v))
        except Exception:
            return str(v)

    n_rows, n_cols = result.shape
    data = {
        j: [fmt_val(result.iat[i, j], i, j) for i in range(n_rows)]
        for j in range(n_cols)
    }
    formatted = pd.DataFrame(data, index=result.index)
    formatted.columns = result.columns
    return formatted


def flextab_to_string(result, fmt="{:.3f}", na_rep="."):
    """
    Render a flextab() result as a formatted string.

    Parameters
    ----------
    result : FlextabResult
        The DataFrame returned by flextab().

    fmt : str, default "{:.3f}"
        Python format string applied to numeric cells that have no
        per-cell format= spec from the TABLE expression.
        Examples: "{:.1f}", "{:,.0f}", "{:.2%}"

    na_rep : str, default "."
        Text shown in place of NaN / missing cells.

    Returns
    -------
    str
        A fixed-width string suitable for printing.

    Notes
    -----
    Per-cell format= specs from the TABLE expression (e.g. *format=7,1
    or *format=12.0s) take precedence over the fmt parameter for their
    specific cells.  The fmt parameter acts as the default for any cell
    without an explicit format= spec.

    Examples
    --------
    print(flextab_to_string(r))                   # default fmt
    print(flextab_to_string(r, fmt="{:.0f}"))     # 0 decimals everywhere
    print(flextab_to_string(r, na_rep="-"))       # dash for missing
    """
    return _format_dataframe(result, fmt=fmt, na_rep=na_rep).to_string()


# ---------------------------------------------------------------------------
# 16. FLEXTAB RESULT CLASS (DISPLAY & EXCEL EXPORT)
# ---------------------------------------------------------------------------

class FlextabResult(pd.DataFrame):
    """
    DataFrame subclass returned by flextab().

    Numeric values are fully preserved — arithmetic, .sum(), slicing, and
    all other DataFrame operations work exactly as on a plain DataFrame.
    The only difference is in how the result is *displayed* and *exported*:

      * print(result)              applies format= specs via __str__
      * repr(result)               same
      * bare expression in Jupyter applies format= specs and CSS colours
                                   via _repr_html_
      * result.to_excel(path)      writes an Excel file with number
                                   formatting and colour styling preserved

    Use flextab_to_string(result, fmt, na_rep) for full string control.

    Attrs (result.attrs)
    --------------------
    col_fmt_map  : dict {col_position -> callable}  column formatters
    row_fmt_map  : dict {row_position -> callable}  row formatters
    default_fmt  : str   fallback Python format string
    style        : dict  colour styling (see flextab() style= parameter)
    """

    _metadata = []

    @property
    def _constructor(self):
        return FlextabResult

    # ── Internal colour helpers ────────────────────────────────────────────

    @staticmethod
    def _to_hex(color) -> str:
        """
        Normalise a colour specification to a 6-character uppercase hex string
        (no '#' prefix) suitable for openpyxl and CSS.

        Accepts:
          Named string   'blue', 'red', 'lightgrey', …  (CSS colour names)
          Hex string     '#4472C4'  or  '4472C4'
          RGB tuple      (70, 114, 196)
        """
        if color is None:
            return None
        # RGB tuple
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return "{:02X}{:02X}{:02X}".format(*[int(c) for c in color])
        s = str(color).strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
            return s.upper()
        # Named colour: convert via a small lookup of common names
        _names = {
            "black": "000000", "white": "FFFFFF",
            "red": "FF0000", "green": "008000", "blue": "0000FF",
            "yellow": "FFFF00", "orange": "FFA500", "purple": "800080",
            "grey": "808080", "gray": "808080",
            "lightgrey": "D3D3D3", "lightgray": "D3D3D3",
            "darkgrey": "A9A9A9", "darkgray": "A9A9A9",
            "navy": "000080", "teal": "008080", "maroon": "800000",
            "silver": "C0C0C0", "lime": "00FF00", "cyan": "00FFFF",
            "magenta": "FF00FF", "pink": "FFC0CB", "beige": "F5F5DC",
        }
        key = s.lower()
        if key in _names:
            return _names[key]
        raise ValueError(
            f"Unrecognised colour {color!r}. "
            "Use a hex string ('#4472C4'), an RGB tuple (70,114,196), "
            "or a basic CSS colour name ('blue', 'red', …)."
        )

    @staticmethod
    def _resolve_color(spec, idx: int):
        """
        Resolve a style-dict colour spec for row/band index `idx`.

        Every style key (header_bg/fg, row_bg/fg, row_header_bg/fg,
        cell_bg/fg) uses this same technique:
          None            -> no colour
          single colour   -> used for every row (named str / hex str / RGB tuple)
          2-tuple         -> (colour0, colour1) cycling: idx % 2 selects
                             which of the two colours this row gets

        An RGB tuple like (70, 114, 196) is a single colour, not a cycling
        pair — it's distinguished from a genuine 2-colour tuple by length
        (3 vs 2) and by its first element being an int in 0-255 range.
        """
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)) and len(spec) == 2 \
                and not (isinstance(spec[0], int) and len(spec) == 3):
            return spec[idx % 2]
        return spec

    @staticmethod
    def _fmt_to_excel_numfmt(formatter) -> str:
        """
        Convert a _parse_fmt_spec formatter to an Excel number format string.

        Inspects the closure variables of the formatter to reliably determine
        decimals, decimal-comma, and thousands-separator settings — rather than
        trying to parse the formatted output, which breaks for space-thousands
        and edge cases.

        Excel number format syntax:
          #,##0    -> comma thousands separator (Excel always uses , for thousands)
          0.00     -> decimal point with 2 places
          For decimal-comma locales, Excel uses its own locale settings;
          we write the standard #,##0.00 format and rely on the user's
          regional Excel settings, since Excel number formats are locale-
          independent internally.
        """
        try:
            # Access the closure to get the exact formatting parameters
            fvars = formatter.__code__.co_freevars
            fvals = {k: v.cell_contents for k, v in
                     zip(fvars, formatter.__closure__)}
            decimals        = fvals.get("decimals", 0)
            use_comma       = fvals.get("use_comma", False)       # decimal comma
            use_thousands   = fvals.get("use_thousands", False)   # _ separator
            use_space_thous = fvals.get("use_space_thous", False) # s separator

            # Build the Excel number format. Excel always uses ',' for thousands
            # grouping internally regardless of locale display.
            thous = use_thousands or use_space_thous
            if decimals > 0:
                dec_part = "." + "0" * decimals
            else:
                dec_part = ""
            if thous:
                return f'#,##0{dec_part}'
            else:
                return f'0{dec_part}'
        except Exception:
            # Fall back to parsing the formatted output if closure inspection fails
            try:
                sample = formatter(1234.5)
                # Count decimal places
                for sep in (',', '.'):
                    if sep in sample:
                        return f'0.{"0" * len(sample.split(sep)[-1])}'
                return "0"
            except Exception:
                return "General"

    # ── Display ───────────────────────────────────────────────────────────

    def __repr__(self):
        try:
            fmt = self.attrs.get("default_fmt", "{:.1f}")
            return _format_dataframe(self, fmt=fmt, na_rep=".").to_string()
        except Exception:
            return super().__repr__()

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        """
        Jupyter/IPython HTML display with format= specs AND inline CSS
        colours from the style= parameter applied.

        Style keys and what they colour (see flextab()'s style= docstring
        for the full picture):
          header_bg/fg      — column header cells, EXCLUDING the row_header
                              cell (cycled by header row)
          row_bg/fg         — row index cells, i.e. the groupby label/value
                              cells on the left (cycled by data row)
          row_header_bg/fg  — the corner cell holding the row_header= text
          cell_bg/fg        — the table's data cells (cycled by data row)

        Each key is applied directly to its own <th>/<td> cells rather than
        via a <tr style=...> — CSS background/colour is inherited by child
        cells, so styling at the row level would leak an unset area's
        colour in from its neighbour (e.g. cell_bg bleeding into the row
        index cells when row_bg isn't set). Every key colours only the
        cells it names; an unset key leaves its cells uncoloured, even
        when a neighbouring key IS set.

        Every key accepts a single colour or a 2-tuple that cycles through
        rows — resolved via FlextabResult._resolve_color().
        """
        try:
            fmt = self.attrs.get("default_fmt", "{:.1f}")
            style = self.attrs.get("style", {})
            formatted = _format_dataframe(self, fmt=fmt, na_rep=".")

            def _css(bg=None, fg=None):
                parts = []
                if bg:
                    hex_bg = FlextabResult._to_hex(bg)
                    parts.append(f"background-color:#{hex_bg}")
                if fg:
                    hex_fg = FlextabResult._to_hex(fg)
                    parts.append(f"color:#{hex_fg}")
                return ";".join(parts)

            _resolve = FlextabResult._resolve_color

            header_bg_spec = style.get("header_bg")
            header_fg_spec = style.get("header_fg")
            row_bg_spec    = style.get("row_bg")
            row_fg_spec    = style.get("row_fg")
            rh_bg_spec     = style.get("row_header_bg")
            rh_fg_spec     = style.get("row_header_fg")
            cell_bg_spec   = style.get("cell_bg")
            cell_fg_spec   = style.get("cell_fg")

            # Text of the row_header cell(s), so we can single it out among
            # the <th> cells in the header area.
            if isinstance(self.index, pd.MultiIndex):
                row_header_names = {n for n in self.index.names if n}
            else:
                row_header_names = {self.index.name} if self.index.name else set()

            html = formatted.to_html(border=0)

            any_style = any([header_bg_spec, header_fg_spec, row_bg_spec, row_fg_spec,
                              rh_bg_spec, rh_fg_spec, cell_bg_spec, cell_fg_spec])
            if any_style:
                import re as _re
                import html as _html
                lines = html.splitlines()
                out = []
                data_row_idx = 0
                header_row_idx = 0
                in_thead = False
                in_thead_row = False

                def _style_one_th(m, css_str):
                    if not css_str:
                        return m.group(0)
                    return _re.sub(
                        r'<th\b([^>]*)>',
                        lambda mm: f'<th{mm.group(1)} style="{css_str}">',
                        m.group(0),
                    )

                for line in lines:
                    stripped = line.strip()
                    if "<thead>" in stripped:
                        in_thead = True
                    if "</thead>" in stripped:
                        in_thead = False

                    if in_thead and stripped.startswith("<tr"):
                        in_thead_row = True

                    elif in_thead and in_thead_row and stripped.startswith("<th"):
                        # Is this the row_header cell (holds the row_header=
                        # text), or a plain column-header cell?
                        text = _html.unescape(_re.sub(r'<[^>]+>', '', stripped)).strip()
                        if row_header_names and text in row_header_names:
                            rh_bg = _resolve(rh_bg_spec, header_row_idx)
                            rh_fg = _resolve(rh_fg_spec, header_row_idx)
                            css_str = _css(rh_bg, rh_fg)
                        else:
                            hdr_bg = _resolve(header_bg_spec, header_row_idx)
                            hdr_fg = _resolve(header_fg_spec, header_row_idx)
                            css_str = _css(hdr_bg, hdr_fg)
                        if css_str:
                            line = _re.sub(
                                r'<th\b[^>]*>.*?</th>',
                                lambda m, _c=css_str: _style_one_th(m, _c),
                                line,
                            )

                    elif not in_thead and stripped.startswith("<tr>"):
                        pass  # no row-level styling — each cell below styles itself

                    elif not in_thead and stripped.startswith("<th"):
                        # Row index cell (groupby LABEL or VALUE cell on the left).
                        rb = _resolve(row_bg_spec, data_row_idx)
                        rf = _resolve(row_fg_spec, data_row_idx)
                        row_css = _css(rb, rf)
                        if row_css:
                            line = _re.sub(
                                r'<th\b([^>]*?)>',
                                lambda m: f'<th{m.group(1)} style="{row_css}">',
                                line,
                            )

                    elif not in_thead and stripped.startswith("<td"):
                        # Data cell.
                        cb = _resolve(cell_bg_spec, data_row_idx)
                        cf = _resolve(cell_fg_spec, data_row_idx)
                        cell_css = _css(cb, cf)
                        if cell_css:
                            line = _re.sub(
                                r'<td\b([^>]*?)>',
                                lambda m: f'<td{m.group(1)} style="{cell_css}">',
                                line,
                            )

                    if "</tr>" in stripped:
                        if in_thead:
                            header_row_idx += 1
                            in_thead_row = False
                        else:
                            data_row_idx += 1

                    out.append(line)
                html = "\n".join(out)

            return html
        except Exception:
            return super()._repr_html_()

    # ── Excel export ──────────────────────────────────────────────────────

    def to_excel(self, excel_writer, sheet_name="Sheet1", **kwargs):
        """
        Write to an Excel file with number formatting and colour styling
        preserved via openpyxl post-processing.

        Parameters
        ----------
        excel_writer : str, Path, or ExcelWriter
            File path (str/Path) or an open pd.ExcelWriter.
        sheet_name : str, default 'Sheet1'
        **kwargs    : passed through to pd.DataFrame.to_excel()

        Number formatting
        -----------------
        Cells that have a format= spec in the TABLE expression receive the
        equivalent Excel number format (e.g. format=7,1 → '#,##0.0' with
        European decimal-comma adjustment).  This means the values remain
        numeric in Excel and sort/sum correctly, while displaying with the
        requested decimal places and separators.

        Colour styling
        --------------
        All keys from the style= dict passed to flextab() are applied:
          header_bg     / header_fg      — column header cells
          row_bg        / row_fg         — row index cells (the groupby
                                            label/value cells on the left)
          row_header_bg / row_header_fg  — the corner cell holding the
                                            row_header= text
          cell_bg       / cell_fg        — the table's data cells

        Every key accepts a single colour (applied to every row) or a
        (colour0, colour1) 2-tuple that cycles through rows.

        Notes
        -----
        If excel_writer is a file path (str or Path), the file is written
        and post-processed in one step. If it is an open ExcelWriter, the
        sheet is formatted immediately after writing but the caller must
        call ExcelWriter.close() / use it as a context manager to save.
        """
        import io
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import PatternFill, Font
            HAS_OPENPYXL = True
        except ImportError:
            HAS_OPENPYXL = False

        col_fmt_map = self.attrs.get("col_fmt_map", {})
        row_fmt_map = self.attrs.get("row_fmt_map", {})
        style       = self.attrs.get("style", {})

        is_path = isinstance(excel_writer, (str, __import__("pathlib").Path))

        # ── Write the plain DataFrame first ──────────────────────────────
        buf = io.BytesIO() if (HAS_OPENPYXL and is_path) else None
        target = buf if buf is not None else excel_writer
        super().to_excel(target, sheet_name=sheet_name, **kwargs)

        if not HAS_OPENPYXL:
            return  # openpyxl not available: plain write is all we can do

        # ── Post-process with openpyxl ───────────────────────────────────
        if buf is not None:
            buf.seek(0)
            wb = load_workbook(buf)
        else:
            # ExcelWriter with openpyxl engine: access the workbook directly
            try:
                wb = excel_writer.book
            except AttributeError:
                return  # can't access workbook; skip formatting

        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active

        # pandas writes n_col_header_rows for the column MultiIndex levels,
        # PLUS an extra row for the index name(s) when any index level has a
        # name set (e.g. row_header="Origin and type").  Detect this extra row
        # by comparing the actual sheet row count to what we'd expect without it.
        n_col_header_rows = self.columns.nlevels
        n_row_index_cols  = self.index.nlevels
        n_data_rows       = len(self)
        n_data_cols       = len(self.columns)

        expected_rows_no_name = n_col_header_rows + n_data_rows
        actual_rows = ws.max_row
        # If pandas wrote an extra row for the index name, adjust the offset
        index_name_rows = max(0, actual_rows - expected_rows_no_name)
        first_data_row = n_col_header_rows + 1 + index_name_rows

        def _fill(hex_color):
            if not hex_color:
                return None
            return PatternFill(start_color=hex_color, end_color=hex_color,
                               fill_type="solid")

        def _font(hex_color):
            if not hex_color:
                return None
            return Font(color=hex_color)

        def _apply(cell, bg_hex=None, fg_hex=None):
            if bg_hex:
                cell.fill = _fill(bg_hex)
            if fg_hex:
                cell.font = _font(fg_hex)

        header_bg_spec = style.get("header_bg")
        header_fg_spec = style.get("header_fg")
        row_bg_spec    = style.get("row_bg")
        row_fg_spec    = style.get("row_fg")
        rh_bg_spec     = style.get("row_header_bg")
        rh_fg_spec     = style.get("row_header_fg")
        cell_bg_spec   = style.get("cell_bg")
        cell_fg_spec   = style.get("cell_fg")

        def _resolve_hex(spec, idx):
            """Resolve a style spec (single colour or cycling 2-tuple) to hex for row idx."""
            val = FlextabResult._resolve_color(spec, idx)
            return FlextabResult._to_hex(val) if val is not None else None

        # The row_header= text lives in the LAST header row's index columns —
        # either its own dedicated row (when pandas writes one) or merged
        # into the same row as the column headers (when it doesn't). Those
        # cells are excluded from the header_bg/fg loop below and coloured
        # separately by row_header_bg/fg, so header_bg never bleeds into
        # them when row_header_bg is left unset.
        has_row_header = bool(self.index.names[0]) if isinstance(self.index, pd.MultiIndex) \
            else bool(self.index.name)
        row_header_row = (first_data_row - 1) if has_row_header else None

        # ── Header rows (column headers, excluding the row_header cells) ───
        for r in range(1, first_data_row):
            header_row_idx = r - 1
            hdr_bg = _resolve_hex(header_bg_spec, header_row_idx)
            hdr_fg = _resolve_hex(header_fg_spec, header_row_idx)
            skip_cols = n_row_index_cols if r == row_header_row else 0
            for c in range(skip_cols + 1, n_row_index_cols + n_data_cols + 1):
                _apply(ws.cell(r, c), hdr_bg, hdr_fg)

        # ── Row header cell(s): styled only by row_header_bg/fg ────────────
        if row_header_row is not None:
            rh_header_row_idx = row_header_row - 1
            rh_bg = _resolve_hex(rh_bg_spec, rh_header_row_idx)
            rh_fg = _resolve_hex(rh_fg_spec, rh_header_row_idx)
            for c in range(1, n_row_index_cols + 1):
                _apply(ws.cell(row_header_row, c), rh_bg, rh_fg)

        # ── Data rows: row index cells get row_bg/fg ONLY, data cells get
        #    cell_bg/fg ONLY — neither falls back to the other. ────────────
        for data_row_idx in range(n_data_rows):
            xl_row = first_data_row + data_row_idx
            cur_cell_bg = _resolve_hex(cell_bg_spec, data_row_idx)
            cur_cell_fg = _resolve_hex(cell_fg_spec, data_row_idx)
            cur_row_bg  = _resolve_hex(row_bg_spec, data_row_idx)
            cur_row_fg  = _resolve_hex(row_fg_spec, data_row_idx)

            for c in range(1, n_row_index_cols + n_data_cols + 1):
                cell = ws.cell(xl_row, c)
                if c <= n_row_index_cols:
                    _apply(cell, cur_row_bg, cur_row_fg)
                else:
                    _apply(cell, cur_cell_bg, cur_cell_fg)

        # ── Number formats ────────────────────────────────────────────────
        for data_col_idx in range(n_data_cols):
            formatter = col_fmt_map.get(data_col_idx)
            if formatter:
                num_fmt = self._fmt_to_excel_numfmt(formatter)
                xl_col = n_row_index_cols + 1 + data_col_idx
                for data_row_idx in range(n_data_rows):
                    xl_row = first_data_row + data_row_idx
                    cell = ws.cell(xl_row, xl_col)
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = num_fmt

        for data_row_idx in range(n_data_rows):
            formatter = row_fmt_map.get(data_row_idx)
            if formatter:
                num_fmt = self._fmt_to_excel_numfmt(formatter)
                xl_row = first_data_row + data_row_idx
                for data_col_idx in range(n_data_cols):
                    xl_col = n_row_index_cols + 1 + data_col_idx
                    cell = ws.cell(xl_row, xl_col)
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = num_fmt

        # ── Save ─────────────────────────────────────────────────────────
        if buf is not None:
            wb.save(excel_writer)

