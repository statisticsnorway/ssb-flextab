from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd

from .parser import (
    parse_table,
    _expand_node_with_branch,
    _classify_path,
)

from .statistics import (
    _BASE_STATS,
    _WEIGHTED_STATS,
    _clean_weights,
    _compute_series,
    _compute_all_series,
    _compute_custom_pct
)

from .result import FlextabResult, _format_dataframe

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

# ---------------------------------------------------------------------------
# 1. STATISTICS REGISTRY
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. AGGREGATION
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. CUSTOM DENOMINATOR PERCENTAGE COMPUTATION
# ---------------------------------------------------------------------------





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


def flextab_to_markdown(result, fmt="{:.3f}", na_rep=".", sep=" / ") -> str:
    """
    Render a flextab() result as a plain Markdown table, with flattened,
    human-readable column headers instead of the raw index tuples that
    pandas' inherited DataFrame.to_markdown() shows for a MultiIndex, and
    with format= specs from the TABLE expression applied to the numbers.

    Standard Markdown tables can't merge cells (no colspan/rowspan) and
    can't stack more than one header row, so the nested, visually "merged"
    header layout from the notebook display can't be reproduced in a
    plain Markdown table - not a flextab limitation, a Markdown one. This
    function's compromise is to flatten every column's levels into ONE
    readable label per column: a ('SUM', 'Income') heading becomes
    "SUM / Income" (see `sep`), and blank levels (e.g. a label suppressed
    with name='') are dropped rather than left as an empty segment.

    If the exact merged-header look is what you need in a document,
    embed the HTML rendering instead of a Markdown table - most Markdown
    processors (GitHub, MkDocs, Jupyter Book, Pandoc) pass raw HTML
    through untouched, colspan and all:

        with open("table.md", "w") as f:
            f.write(result._repr_html_())

    (Some renderers, GitHub included, strip inline `style` attributes
    from embedded HTML for security, so `style=` colouring may not
    survive - the table structure and merged headers still will.)

    Parameters
    ----------
    result : FlextabResult
        The DataFrame returned by flextab().

    fmt : str, default "{:.3f}"
        Python format string applied to numeric cells that have no
        per-cell format= spec from the TABLE expression - same behaviour
        as flextab_to_string().

    na_rep : str, default "."
        Text shown in place of NaN / missing cells.

    sep : str, default " / "
        Separator used to join a MultiIndex column's levels into one
        header label.

    Returns
    -------
    str
        A GitHub-flavoured Markdown table.

    Examples
    --------
    print(flextab_to_markdown(r))
    with open("table.md", "w") as f:
        f.write(flextab_to_markdown(r, na_rep="-"))
    """
    formatted = _format_dataframe(result, fmt=fmt, na_rep=na_rep)

    def _escape(text: str) -> str:
        # A literal "|" would otherwise break the Markdown table's
        # column structure.
        return text.replace("|", "\\|")

    def _flatten(key) -> str:
        if isinstance(key, tuple):
            parts = [str(p) for p in key if str(p) != ""]
            label = sep.join(parts) if parts else ""
        else:
            label = str(key)
        return _escape(label)

    col_labels = [_flatten(c) for c in formatted.columns]

    if isinstance(formatted.index, pd.MultiIndex):
        index_names = [_escape(n) if n else "" for n in formatted.index.names]
        index_rows = [
            [_escape(str(v)) for v in (idx if isinstance(idx, tuple) else (idx,))]
            for idx in formatted.index
        ]
    else:
        index_names = [_escape(formatted.index.name) if formatted.index.name else ""]
        index_rows = [[_escape(str(idx))] for idx in formatted.index]

    header = index_names + col_labels
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for idx_vals, (_, row) in zip(index_rows, formatted.iterrows()):
        cells = idx_vals + [_escape(str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 17. SMOKE TESTS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    demo = pd.DataFrame({
        "origin":     np.random.choice(["Asia", "Europe", "USA"], n),
        "type":       np.random.choice(["Sedan", "SUV", "Truck"], n),
        "msrp":       np.random.normal(35000, 12000, n).clip(10000),
        "horsepower": np.random.normal(220, 60, n).clip(80),
    })

    sep = "=" * 70

    print(sep)
    print("Example 1 — 1D, no groupby, stat labels")
    print(sep)
    r1 = flextab(
        data=demo, measure=["msrp", "horsepower"],
        table="msrp='' * (N MEAN='Average' STD='Std Dev') horsepower='' * (N MEAN='Average')",
    )
    print(flextab_to_string(r1))

    print(f"\n{sep}")
    print("Example 2 — 2D: origin renamed, msrp suppressed")
    print(sep)
    r2 = flextab(
        data=demo, measure=["msrp"], groupby=["origin"],
        table="origin='Region', msrp='' * (N MEAN='Mean' ROWPCTN='Row %')",
    )
    print(flextab_to_string(r2))

    print(f"\n{sep}")
    print("Example 3 — ALL='Total' in row dimension")
    print(sep)
    r3 = flextab(
        data=demo, measure=["msrp"], groupby=["origin"],
        table="origin ALL='Total', msrp='' * (N MEAN='Mean' MIN MAX)",
    )
    print(flextab_to_string(r3))

    print(f"\n{sep}")
    print("Example 4 — ALL in both dimensions (grand-total row + col)")
    print(sep)
    r4 = flextab(
        data=demo, measure=["msrp"], groupby=["origin", "type"],
        table="origin * type ALL='Subtotal', msrp='' * (N MEAN='Mean') ALL='Grand Total'",
    )
    print(flextab_to_string(r4))

    print(f"\n{sep}")
    print("Example 5 — Percent stats with labels")
    print(sep)
    r5 = flextab(
        data=demo, measure=["msrp"], groupby=["origin"],
        table="origin='Region' ALL='Total', msrp='' * (PCTN='% of Total' COLPCTN='Col %')",
    )
    print(flextab_to_string(r5))

    print(f"\n{sep}")
    print("Example 6 — Nested groupby + ALL subtotal")
    print(sep)
    r6 = flextab(
        data=demo, measure=["msrp", "horsepower"],
        groupby=["origin", "type"],
        table="origin * (type ALL='Subtotal'), msrp='' * (N MEAN='Mean') horsepower='' * MEAN='Mean'",
    )
    print(flextab_to_string(r6))

    print(f"\n{sep}")
    print("Example 7 — nan rows: single nan per groupby value")
    print(sep)
    demo2 = pd.DataFrame({"origin": ["Asia", None, "USA"], "msrp": [30000, 40000, None]})
    r7 = flextab(
        data=demo2, groupby=["origin"], measure=["msrp"],
        table="origin, msrp*(sum n) n",
        include_missing_in_groupby=True)
    print(flextab_to_string(r7))
    assert len(r7) == 3, f"Expected 3 rows, got {len(r7)}"
    # Index is now (label, value) tuples e.g. ('origin','nan') — check value level
    idx_values = [t[-1] if isinstance(t, tuple) else t for t in r7.index]
    assert idx_values.count("nan") == 1, f"nan should appear exactly once, got: {idx_values}"
    print("OK: 3 rows, nan appears once")
