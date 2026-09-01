from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .formatting import _parse_fmt_spec
from .formatting import flextab_to_string
from .parser import DimNode
from .parser import _classify_path
from .parser import _expand_node_with_branch
from .parser import parse_table
from .result import FlextabResult
from .statistics import _compute_all_series
from .statistics import _compute_custom_pct
from .statistics import _compute_series

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

_SENTINEL = "__total__"


def flextab(
    data: pd.DataFrame,
    measure: str | list | None = None,
    groupby: str | list | None = None,
    table: str | None = None,
    include_missing_in_groupby: bool = True,
    fmt: str = "{:.1f}",
    na_rep: str | None = None,
    labels: dict | None = None,
    sort_by: str = "code",
    weight: str | None = None,
    row_header: str | None = None,
    style: dict | None = None,
) -> FlextabResult:
    """Build a cross-tabulation table similar to SAS PROC TABULATE.

    Returns a ``FlextabResult``, a ``pd.DataFrame`` subclass that keeps the
    underlying values numeric for further computation or export, while
    applying ``format=`` specifications from the TABLE expression when
    displayed.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    measure : str | list | None
        Numeric analysis variable name or names (SAS: VAR). A single column
        can be passed as a plain string, for example ``measure="income"``
        instead of ``measure=["income"]``.

        Omit this argument for count-only tables that use ``N``, ``COUNT``,
        ``SIZE``, or percentage statistics that do not require a measure.

    groupby : str | list | None
        Categorical grouping variable name or names (SAS: CLASS). A single
        column can be passed as a plain string, for example
        ``groupby="origin"`` instead of ``groupby=["origin"]``.

    table : str | None
        TABLE expression describing the row and column dimensions.

        Basic syntax:

        - ``row_expr , col_expr`` separates row and column dimensions.
        - ``A * B`` crosses or nests A with B.
        - ``A B`` concatenates A and B in written order.
        - ``(A B)`` groups an expression so it is treated as one unit.
        - ``ALL`` or ``TOTAL`` adds a marginal total.
        - ``name='Label'`` renames a token.
        - ``name=''`` suppresses the corresponding label level.

        Format specifications may be attached to a preceding token or group:

        - ``*format=W.D``: D decimals, decimal point.
        - ``*format=W,D``: D decimals, decimal comma.
        - ``*format=W.D_``: decimal point and comma thousands separator.
        - ``*format=W,D_``: decimal comma and dot thousands separator.
        - ``*format=W.Ds``: decimal point and space thousands separator.
        - ``*format=W,Ds``: decimal comma and space thousands separator.

        A format can be applied to several statistics at once, for example::

            (mean gmean)*format=7,1*income

        Custom denominator definitions are supported for ``PCTN`` and
        ``PCTSUM`` using::

            PCTN<denom>
            PCTSUM<denom>

        The denominator definition is a space-separated list containing one
        or more of:

        - A measure column, meaning that measure is used as the denominator
          within the same grouping.
        - A groupby column, meaning the denominator is the subtotal obtained
          by collapsing that class variable.
        - ``ALL`` or ``TOTAL``, meaning the grand total.

        When several denominator tokens are supplied, the token matching the
        current subtable is preferred, with later tokens available as
        fallbacks.

        Examples include::

            tax*pctsum<income>

        which expresses tax as a percentage of income, and::

            income*pctsum<gender all>

        which uses a gender subtotal where applicable and ``ALL`` as a
        fallback.

    include_missing_in_groupby : bool
        Whether missing values in groupby columns should appear as their own
        group level.

        If True, missing values are retained as grouping levels. If False,
        rows with missing values in grouping columns are excluded from the
        corresponding grouping operation.

    fmt : str
        Default Python format string for numeric cells that do not have an
        explicit ``format=`` specification in the TABLE expression.

        The default is ``"{:.1f}"``. This format is used by ``print()``,
        ``repr()``, and ``flextab_to_string()`` unless overridden there.

    na_rep : str | None
        Text shown in place of NaN or missing cells, for example ``"-"`` or
        ``"."``.

        If None, missing cells remain as numeric NaN values. This is normally
        preferable when the returned table will be used for further numeric
        operations.

    labels : dict | None
        Mapping from original groupby values to display labels.

        The outer dictionary key is the groupby column name. The inner
        dictionary maps original values to display labels. Aggregation and
        sorting by code use the original values; remapping is applied only
        for display.

        Example::

            {
                "gender": {1: "Male", 2: "Female"},
                "region": {"N": "North", "S": "South"},
            }

    sort_by : str
        Controls the order of groupby levels.

        The default is ``"code"``. Supported values are:

        - ``"code"``: sort by original data values before label remapping.
        - ``"index"``: sort according to insertion order in the corresponding
          ``labels`` dictionary.
        - ``"label"``: sort alphabetically by the display-label text.

        ``"index"`` and ``"label"`` affect only columns that have a mapping in
        ``labels``. Columns without a label dictionary are sorted by their
        original values.

    weight : str | None
        Name of a numeric column used as a weight (SAS: WEIGHT statement).

        Weight handling follows these rules:

        - A weight of zero remains part of unweighted counts but contributes
          zero to weighted statistics.
        - A negative weight is treated as zero and remains part of unweighted
          counts.
        - A missing weight excludes the observation entirely.

        Weighted statistics use the following definitions:

        - ``SUM``: ``sum(w * x)``.
        - ``MEAN``: ``sum(w * x) / sum(w)``.
        - ``VAR``: unbiased reliability-weight variance,
          ``sum(w) / (sum(w)^2 - sum(w^2)) * sum(w * (x - xbar)^2)``.
        - ``STD``: square root of weighted variance.
        - ``STDERR``: ``sqrt(VAR / sum(w))``.
        - ``GMEAN``: ``exp(sum(w * log(x)) / sum(w))``.
        - ``HMEAN``: ``sum(w) / sum(w / x)``.
        - Percentiles: weighted empirical cumulative distribution.

        ``N``, ``COUNT``, ``SIZE``, and ``NMISS`` are never weighted.

    row_header : str | None
        Name assigned to the row index.

        For a flat index, this corresponds to ``result.index.name``. For a
        MultiIndex, it is assigned to the first index level.

        Example::

            row_header="Region"

        labels the leftmost index column as ``Region``.

    style : dict | None
        Colour styling applied to notebook display and Excel export.

        Colours may be specified as:

        - Named strings such as ``"blue"``, ``"red"``, ``"lightgrey"``, or
          ``"navy"``.
        - Hexadecimal strings such as ``"#4472C4"`` or ``"4472C4"``.
        - RGB tuples such as ``(70, 114, 196)``.

        Each style key may contain either one colour, which is applied
        uniformly, or a two-element sequence of colours that alternates
        between rows.

        Supported keys are:

        - ``header_bg``: column-header background colour.
        - ``header_fg``: column-header foreground colour.
        - ``row_bg``: row-index background colour.
        - ``row_fg``: row-index foreground colour.
        - ``row_header_bg``: background colour of the row-header corner cell.
        - ``row_header_fg``: foreground colour of the row-header corner cell.
        - ``cell_bg``: data-cell background colour.
        - ``cell_fg``: data-cell foreground colour.

        Styling is applied in three contexts:

        1. Jupyter HTML display via ``_repr_html_()``.
        2. Plain ``print()`` and ``repr()`` output, which remains uncoloured.
        3. Excel export through ``result.to_excel()``, using openpyxl styling.

        Example::

            style={
                "header_bg": "#4472C4",
                "header_fg": "white",
                "row_bg": "lightgrey",
                "row_fg": "black",
                "row_header_bg": "#4472C4",
                "row_header_fg": "white",
                "cell_bg": ("white", "#EBF3FB"),
            }ning the denominator is the subtotal obtained
          by collapsing that class variable.
        - ``ALL`` or ``TOTAL``, meaning the grand total.

        When several denominator tokens are supplied, the token matching the
        current subtable is preferred, with later tokens available as
        fallbacks.

        Examples include::

            tax*pctsum<income>

        which expresses tax as a percentage of income, and::

            income*pctsum<gender all>

        which uses a gender subtotal where applicable and ``ALL`` as a
        fallback.

    Returns
    -------
    FlextabResult
        A ``pd.DataFrame`` subclass containing the numeric table values.

        Numeric values are preserved for operations such as ``sum`` and
        arithmetic. ``format=`` specifications from the TABLE expression are
        applied automatically by ``print()``, ``repr()``, and Jupyter display.

        Use ``result.to_excel(path)`` to export the table with number
        formatting and colour styling. Use ``flextab_to_string()`` or
        ``flextab_to_markdown()`` for explicit textual rendering.
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
    labels = labels or {}
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

    def expand_dim(dim_node: DimNode | None):
        if dim_node is None:
            return [
                {
                    "group_keys": [],
                    "var": None,
                    "var_label": None,
                    "stat": None,
                    "stat_label": None,
                    "has_all": False,
                    "all_label": None,
                    "path_order": [],
                    "branch": 0,
                }
            ]
        specs = []
        for branch_idx, path in _expand_node_with_branch(dim_node):
            spec = _classify_path(path, measure, groupby)
            spec["branch"] = branch_idx
            specs.append(spec)
        return specs

    col_specs = expand_dim(col_dim)
    row_specs = expand_dim(row_dim)

    def spec_header(spec: dict[str, Any]) -> tuple[str, ...]:
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
            label = entry[1]
            orig = entry[2] if len(entry) > 2 else None
            is_group = entry[0] == "group"
            if is_group and not label and orig:
                # Blank label on a group token → use orig_name internally
                parts.append(
                    f"\x00{orig}"
                )  # prefix ensures no collision with real labels
            else:
                parts.append(label)
        return tuple(parts) if parts else ("",)

    def orig_groups(spec: dict[str, Any]) -> list[str]:
        return [col for col, _ in spec["group_keys"]]

    cells: dict = {}
    row_hdr_path: dict = {}
    col_hdr_path: dict = {}

    for r_spec in row_specs:
        r_hdr = spec_header(r_spec)
        r_groups = orig_groups(r_spec)
        row_hdr_path.setdefault(r_hdr, (r_spec["path_order"], r_spec["branch"]))

        for c_spec in col_specs:
            c_hdr = spec_header(c_spec)
            c_groups = orig_groups(c_spec)
            col_hdr_path.setdefault(c_hdr, (c_spec["path_order"], c_spec["branch"]))

            var = r_spec["var"] or c_spec["var"]
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
                    data,
                    all_groups,
                    var,
                    stat,
                    r_groups,
                    c_groups,
                    missing,
                    weight=weight,
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            elif has_all_r and not has_all_c:
                # ALL on rows: c_groups drive the column denominator for COLPCTN
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(
                    data,
                    keep,
                    var,
                    stat,
                    missing,
                    r_groups=r_groups,
                    c_groups=c_groups,
                    weight=weight,
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            elif not has_all_r and has_all_c:
                # ALL on cols: r_groups drive the row denominator for ROWPCTN
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(
                    data,
                    keep,
                    var,
                    stat,
                    missing,
                    r_groups=r_groups,
                    c_groups=c_groups,
                    weight=weight,
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

            else:
                # ALL on both: pass both for correct denominator selection
                keep = list(dict.fromkeys(r_groups + c_groups))
                series = _compute_all_series(
                    data,
                    keep,
                    var,
                    stat,
                    missing,
                    r_groups=r_groups,
                    c_groups=c_groups,
                    weight=weight,
                )
                _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)

    def _index_value_key(
        orig_col: str | None,
        v: Any,
    ) -> tuple[int, int | str]:
        """Return the sort key for ``sort_by='index'``.

        Order values by their position in the labels dictionary, using the
        insertion order provided by the caller.

        Looks up ``v`` only in ``labels[orig_col]`` (the dictionary belonging to
        this specific groupby column), never in other columns' label dictionaries,
        so a raw value like 1 used in two different columns can never borrow the
        wrong column's label or order.

        Values not present in the label dictionary, or when the column has no
        label dictionary at all, fall back to their normalised string form and
        are sorted after all explicitly labelled values.
        """
        col_labels = _label_map.get(orig_col) if orig_col else None
        if col_labels and v in col_labels:
            keys_in_order = list(col_labels.keys())
            return (0, keys_in_order.index(v))
        return (1, _na_safe_str(v))

    def _label_text_value_key(
        orig_col: str | None,
        v: Any,
    ) -> tuple[int, str]:
        """Sort key for sort_by='label': order alphabetically by the DISPLAY.

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

    def _na_safe_str(v: Any) -> str:
        if v is None or v == _NAN_SENTINEL:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _sort_keys(
        keys: list[tuple[Any, Any]],
        hdr_path: dict[Any, tuple[list[Any], int]],
    ) -> list[tuple[Any, Any]]:
        """Sort row/col keys, respecting sort_by='code', 'index', or 'label'."""
        if sort_by == "index" and _label_map:
            value_key_fn = _index_value_key
        elif sort_by == "label" and _label_map:
            value_key_fn = _label_text_value_key
        else:
            value_key_fn = None
        return _sort_row_keys(keys, hdr_path, value_key_fn=value_key_fn)

    all_row_keys = _sort_keys(list(dict.fromkeys(rk for rk in cells)), row_hdr_path)
    all_col_keys = _sort_keys(
        list(dict.fromkeys(ck for rk in cells for ck in cells[rk])), col_hdr_path
    )

    matrix = np.full((len(all_row_keys), len(all_col_keys)), np.nan)
    rk_pos = {rk: i for i, rk in enumerate(all_row_keys)}
    ck_pos = {ck: j for j, ck in enumerate(all_col_keys)}

    for rk, col_dict in cells.items():
        for ck, val in col_dict.items():
            matrix[rk_pos[rk], ck_pos[ck]] = val

    def _fmt_val(
        v: Any,
        col_name: str | None = None,
    ) -> str:
        """Format a group key value for display, applying label remapping.

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

    def _compute_slot_layout(
        all_path_orders: list[list[tuple[Any, ...]]],
    ) -> list[int]:
        """Compute display slots per position across all specs.

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
                i < len(po) and po[i][0] == "group" and po[i][1]  # label is non-blank
                for po in all_path_orders
            )
            slots.append(2 if has_labeled_group else 1)
        return slots

    def _key_to_label_slotted(
        hdr: Any,
        data_key: Any,
        path_order: list[tuple[Any, ...]],
        slots: list[int],
    ) -> Any:
        """Build a fixed-length index tuple using a pre-computed slot layout.

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
        dvals = (
            []
            if is_total
            else list(data_key if isinstance(data_key, tuple) else (data_key,))
        )
        data_iter = iter(dvals)

        row = [""] * D

        def slot_range(local_pos: int) -> tuple[int, int]:
            # local_pos is the index within path_order (0 = outermost of THIS spec)
            # map to the global slots list (bottom-aligned)
            global_pos = len(slots) - len(path_order) + local_pos
            low = sum(slots[global_pos + 1 :])
            high = low + slots[global_pos] - 1
            return D - 1 - high, D - 1 - low  # (hi_idx, lo_idx)

        for pos, entry in enumerate(path_order):
            kind = entry[0]
            label = entry[1]
            orig_name = entry[2] if len(entry) > 2 else None
            hi_idx, lo_idx = slot_range(pos)

            if kind != "group":
                row[lo_idx] = label
            else:
                if is_total:
                    pass
                else:
                    val = next(data_iter, None)
                    val_str = _fmt_val(val, col_name=orig_name)
                    global_pos = len(slots) - len(path_order) + pos
                    has_label_slot = slots[global_pos] == 2
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

    def make_index(
        keys: list[tuple[Any, Any]],
        hdr_path: dict[Any, tuple[list[Any], int]],
    ) -> pd.Index:
        """Convert header/data-key pairs to an index using slot-based layout.

        Every specification in the dimension produces a fixed-length tuple of the
        same depth, ``D = sum(slots)``, where the slot layout is computed globally
        so that all specifications align correctly:

        - Groupby variables occupy two slots: one label row and one value row.
        - Statistic, measure, and ALL/TOTAL tokens occupy one slot each.
        - Shorter specifications are bottom-aligned and front-padded with blanks
        so their innermost token lands at the same absolute level as the
        innermost token of deeper specifications.

        After building the tuples, any level that is blank across all columns is
        dropped. These blank levels are structural artefacts, for example the
        label slot of a group whose label was suppressed with ``=''``.

        Parameters
        ----------
        keys : list[tuple[Any, Any]]
            Sequence of ``(header, data_key)`` pairs to convert to index labels.
        hdr_path : dict[Any, tuple[list[Any], int]]
            Mapping from each header to its path-order metadata and top-level
            branch index.

        Returns
        -------
        pd.Index
            An Index or MultiIndex containing the aligned display labels.

        Examples
        --------
        The expression ``n colpctn*(all age_group)`` produces specifications such
        as::

            [stat:N]
            [stat:COLPCTN, all:TOTAL]
            [stat:COLPCTN, group:age_group]

        Their logical labels are::

            ('N',)
            ('COLPCTN', 'TOTAL')
            ('COLPCTN', 'age_group', '10-19')

        The slot layout has one slot for the statistic position and two slots for
        the group/ALL position because ``age_group`` has a non-blank label. This
        gives ``D = 3``.

        After bottom alignment and removal of levels that are blank everywhere,
        the entries align so that totals and group values occupy the same logical
        positions.

        The expression ``origin * (type total='Subtotal')`` similarly produces::

            [group:origin, group:type]
            [group:origin, all:Subtotal]

        which can be represented as::

            ('origin', 'Asia', 'type', 'SUV')
            ('origin', 'Asia', 'Subtotal', '')

        Both have the same slot depth, so ``'Asia'`` remains vertically aligned
        between detail and subtotal rows.
        """
        all_po = [hdr_path.get(hdr, ([], 0))[0] for hdr, _ in keys]
        slots = _compute_slot_layout(all_po)
        D = sum(slots)

        if D == 0:
            return pd.Index([""] * len(keys))

        labels = [
            _key_to_label_slotted(
                hdr,
                dk,
                hdr_path.get(hdr, ([], 0))[0],
                slots,
            )
            for hdr, dk in keys
        ]

        # Drop levels that are blank in every column
        if labels and len(labels[0]) > 1:
            keep = [i for i in range(len(labels[0])) if any(t[i] for t in labels)]
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
        c_hdr, _ = ck
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
        r_hdr, _ = rk
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
    result.attrs["style"] = style or {}

    # Apply row_header: name the row index so it prints as a column label
    if row_header is not None:
        if isinstance(result.index, pd.MultiIndex):
            result.index.names = [row_header, *result.index.names[1:]]
        else:
            result.index.name = row_header

    return result


def _sort_row_keys(
    row_keys: list, hdr_path: dict | None = None, value_key_fn: str | None = None
) -> list:
    """Re-order row/column keys to follow the TABLE expression's written order.

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

    def _normalise(v: Any) -> str:
        if v is None or v == _NAN_SENTINEL:
            return ""
        try:
            if isinstance(v, float) and np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _value_key(orig_col: str | None, v: Any) -> Any:
        if value_key_fn is not None:
            return value_key_fn(orig_col, v)
        return _normalise(v)

    def _po(hdr: Any) -> list[Any]:
        return hdr_path.get(hdr, ([], 0))[0]

    def _branch(hdr: Any) -> int:
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

    def sort_key(rk: tuple[Any, Any]) -> tuple[Any, ...]:
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
        return (branch, *parts)

    return sorted(row_keys, key=sort_key)


# Canonical sentinel for all missing-value types in group keys.
# Using a distinct string ensures consistent hashing regardless of whether
# pandas returns float nan, pd.NA, pd.NaT, or None for missing group levels.
_NAN_SENTINEL = "__nan__"


def _normalise_key(val: Any) -> Any:
    """Normalise a group key value so that all missing-value representations.

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


def _normalise_idx(idx_tuple: tuple):
    """Apply _normalise_key to every element of an index tuple."""
    return tuple(_normalise_key(v) for v in idx_tuple)


def _fill_cells(
    cells: dict[Any, dict[Any, Any]],
    series: pd.Series,
    r_hdr: Any,
    c_hdr: Any,
    r_groups: list[str],
    c_groups: list[str],
) -> None:
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
            c_data = idx_tuple[n_r : n_r + n_c] if n_c else (_SENTINEL,)

        rk = (r_hdr, r_data)
        ck = (c_hdr, c_data)
        if rk not in cells:
            cells[rk] = {}
        cells[rk][ck] = value


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    demo = pd.DataFrame(
        {
            "origin": np.random.choice(["Asia", "Europe", "USA"], n),
            "type": np.random.choice(["Sedan", "SUV", "Truck"], n),
            "msrp": np.random.normal(35000, 12000, n).clip(10000),
            "horsepower": np.random.normal(220, 60, n).clip(80),
        }
    )

    sep = "=" * 70

    print(sep)
    print("Example 1 — 1D, no groupby, stat labels")
    print(sep)
    r1 = flextab(
        data=demo,
        measure=["msrp", "horsepower"],
        table="msrp='' * (N MEAN='Average' STD='Std Dev') horsepower='' * (N MEAN='Average')",
    )
    print(flextab_to_string(r1))

    print(f"\n{sep}")
    print("Example 2 — 2D: origin renamed, msrp suppressed")
    print(sep)
    r2 = flextab(
        data=demo,
        measure=["msrp"],
        groupby=["origin"],
        table="origin='Region', msrp='' * (N MEAN='Mean' ROWPCTN='Row %')",
    )
    print(flextab_to_string(r2))

    print(f"\n{sep}")
    print("Example 3 — ALL='Total' in row dimension")
    print(sep)
    r3 = flextab(
        data=demo,
        measure=["msrp"],
        groupby=["origin"],
        table="origin ALL='Total', msrp='' * (N MEAN='Mean' MIN MAX)",
    )
    print(flextab_to_string(r3))

    print(f"\n{sep}")
    print("Example 4 — ALL in both dimensions (grand-total row + col)")
    print(sep)
    r4 = flextab(
        data=demo,
        measure=["msrp"],
        groupby=["origin", "type"],
        table="origin * type ALL='Subtotal', msrp='' * (N MEAN='Mean') ALL='Grand Total'",
    )
    print(flextab_to_string(r4))

    print(f"\n{sep}")
    print("Example 5 — Percent stats with labels")
    print(sep)
    r5 = flextab(
        data=demo,
        measure=["msrp"],
        groupby=["origin"],
        table="origin='Region' ALL='Total', msrp='' * (PCTN='% of Total' COLPCTN='Col %')",
    )
    print(flextab_to_string(r5))

    print(f"\n{sep}")
    print("Example 6 — Nested groupby + ALL subtotal")
    print(sep)
    r6 = flextab(
        data=demo,
        measure=["msrp", "horsepower"],
        groupby=["origin", "type"],
        table="origin * (type ALL='Subtotal'), msrp='' * (N MEAN='Mean') horsepower='' * MEAN='Mean'",
    )
    print(flextab_to_string(r6))

    print(f"\n{sep}")
    print("Example 7 — nan rows: single nan per groupby value")
    print(sep)
    demo2 = pd.DataFrame(
        {"origin": ["Asia", None, "USA"], "msrp": [30000, 40000, None]}
    )
    r7 = flextab(
        data=demo2,
        groupby=["origin"],
        measure=["msrp"],
        table="origin, msrp*(sum n) n",
        include_missing_in_groupby=True,
    )
    print(flextab_to_string(r7))
    assert len(r7) == 3, f"Expected 3 rows, got {len(r7)}"
    # Index is now (label, value) tuples e.g. ('origin','nan') — check value level
    idx_values = [t[-1] if isinstance(t, tuple) else t for t in r7.index]
    assert (
        idx_values.count("nan") == 1
    ), f"nan should appear exactly once, got: {idx_values}"
    print("OK: 3 rows, nan appears once")
