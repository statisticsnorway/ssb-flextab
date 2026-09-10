from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any
from typing import cast

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

# Canonical sentinel for all missing-value types in group keys.
# Using a distinct string ensures consistent hashing regardless of whether
# pandas returns float nan, pd.NA, pd.NaT, or None for missing group levels.
_NAN_SENTINEL = "__nan__"


# ---------------------------------------------------------------------------
# Argument normalisation / default TABLE expression
# ---------------------------------------------------------------------------


def _normalize_measure_groupby(
    measure: str | list[str] | None,
    groupby: str | list[str] | None,
) -> tuple[list[str], list[str]]:
    """Allow a single column name to be passed as a plain string."""
    if isinstance(measure, str):
        measure = [measure]
    if isinstance(groupby, str):
        groupby = [groupby]
    return measure or [], groupby or []


def _default_table_expr(measure: list[str], groupby: list[str]) -> str:
    """Build the default TABLE expression when none is supplied."""
    if measure:
        col_expr = " ".join(f"{v} * (N MEAN)" for v in measure)
        if groupby:
            row_expr = " ".join(groupby)
            return f"{row_expr}, {col_expr}"
        return col_expr
    if groupby:
        row_expr = " ".join(groupby)
        return f"{row_expr}, N"
    return "N"


def _split_dims(
    dims: list[DimNode],
) -> tuple[DimNode | None, DimNode]:
    """Split the parsed TABLE expression into (row_dim, col_dim)."""
    if len(dims) == 1:
        return None, dims[0]
    return dims[0], dims[1]


# ---------------------------------------------------------------------------
# Dimension spec expansion
# ---------------------------------------------------------------------------


def _expand_dim(
    dim_node: DimNode | None,
    measure: list[str],
    groupby: list[str],
) -> list[dict[str, Any]]:
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


def _spec_header(spec: dict[str, Any]) -> tuple[Any, ...]:
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
            parts.append(f"\x00{orig}")  # prefix ensures no collision with real labels
        else:
            parts.append(label)
    header = tuple(parts) if parts else ("",)
    # Two DISTINCT top-level (space-separated) entries can still produce
    # identical label text — e.g. "total sex total education total
    # region" repeats the default 'TOTAL' label three times, and a user
    # may also deliberately reuse the same explicit label twice. Without
    # a further discriminator these specs would collapse onto the same
    # dict key in `cells`/`row_hdr_path`/`col_hdr_path` and silently
    # overwrite one another, dropping all but the last occurrence.
    # `spec["branch"]` is the top-level concat branch index (unique per
    # space-separated entry), so folding it in here keeps same-text
    # repeats as distinct rows/columns. It is never shown to the user —
    # display text is built from `path_order` alone in
    # `_key_to_label_slotted`.
    return (*header, ("\x00branch", spec["branch"]))


def _orig_groups(spec: dict[str, Any]) -> list[str]:
    return [col for col, _ in spec["group_keys"]]


# ---------------------------------------------------------------------------
# Cell computation
# ---------------------------------------------------------------------------


def _resolve_stat_var(
    r_spec: dict[str, Any],
    c_spec: dict[str, Any],
) -> tuple[str | None, str, Any]:
    var = r_spec["var"] or c_spec["var"]
    stat = r_spec["stat"] or c_spec["stat"]
    if stat is None:
        stat = "N"
    # Custom denominator definition from <...> syntax
    denom_def = r_spec.get("denom_def") or c_spec.get("denom_def")
    return var, stat, denom_def


def _select_series(
    data: pd.DataFrame,
    r_spec: dict[str, Any],
    c_spec: dict[str, Any],
    r_groups: list[str],
    c_groups: list[str],
    var: str | None,
    stat: str,
    denom_def: Any,
    groupby: list[str],
    measure: list[str],
    missing: bool,
    weight: str | None,
) -> pd.Series:
    if denom_def is not None and stat.upper() in ("PCTSUM", "PCTN"):
        # Custom-denominator percentage: route to dedicated function
        return _compute_custom_pct(
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

    groups_union = list(dict.fromkeys(r_groups + c_groups))

    if not r_spec["has_all"] and not c_spec["has_all"]:
        return _compute_series(
            data, groups_union, var, stat, r_groups, c_groups, missing, weight=weight
        )

    # ALL on rows, columns, or both: r_groups/c_groups drive the correct
    # denominator selection for ROWPCTN/COLPCTN inside _compute_all_series.
    return _compute_all_series(
        data,
        groups_union,
        var,
        stat,
        missing,
        r_groups=r_groups,
        c_groups=c_groups,
        weight=weight,
    )


def _fill_cell(
    data: pd.DataFrame,
    cells: dict[Any, dict[Any, Any]],
    r_spec: dict[str, Any],
    c_spec: dict[str, Any],
    r_hdr: Any,
    c_hdr: Any,
    r_groups: list[str],
    c_groups: list[str],
    groupby: list[str],
    measure: list[str],
    missing: bool,
    weight: str | None,
) -> None:
    var, stat, denom_def = _resolve_stat_var(r_spec, c_spec)
    series = _select_series(
        data,
        r_spec,
        c_spec,
        r_groups,
        c_groups,
        var,
        stat,
        denom_def,
        groupby,
        measure,
        missing,
        weight,
    )
    _fill_cells(cells, series, r_hdr, c_hdr, r_groups, c_groups)


def _build_cells(
    data: pd.DataFrame,
    row_specs: list[dict[str, Any]],
    col_specs: list[dict[str, Any]],
    groupby: list[str],
    measure: list[str],
    missing: bool,
    weight: str | None,
) -> tuple[
    dict[Any, dict[Any, Any]],
    dict[Any, tuple[list[Any], int]],
    dict[Any, tuple[list[Any], int]],
]:
    cells: dict[Any, dict[Any, Any]] = {}
    row_hdr_path: dict[Any, tuple[list[Any], int]] = {}
    col_hdr_path: dict[Any, tuple[list[Any], int]] = {}

    for r_spec in row_specs:
        r_hdr = _spec_header(r_spec)
        r_groups = _orig_groups(r_spec)
        row_hdr_path.setdefault(r_hdr, (r_spec["path_order"], r_spec["branch"]))

        for c_spec in col_specs:
            c_hdr = _spec_header(c_spec)
            c_groups = _orig_groups(c_spec)
            col_hdr_path.setdefault(c_hdr, (c_spec["path_order"], c_spec["branch"]))

            _fill_cell(
                data,
                cells,
                r_spec,
                c_spec,
                r_hdr,
                c_hdr,
                r_groups,
                c_groups,
                groupby,
                measure,
                missing,
                weight,
            )

    return cells, row_hdr_path, col_hdr_path


# ---------------------------------------------------------------------------
# Sort-by value-key helpers
# ---------------------------------------------------------------------------


def _na_safe_str(v: Any) -> str:
    if v is None or v == _NAN_SENTINEL:
        return ""
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def _index_value_key(
    orig_col: str | None,
    v: Any,
    label_map: dict[str, dict[Any, str]],
) -> tuple[int, int | str]:
    """Return the sort key for ``sort_by='index'``.

    Order values by their position in the labels dictionary, using the
    insertion order provided by the caller.

    Looks up ``v`` only in ``label_map[orig_col]`` (the dictionary belonging
    to this specific groupby column), never in other columns' label
    dictionaries, so a raw value like 1 used in two different columns can
    never borrow the wrong column's label or order.

    Values not present in the label dictionary, or when the column has no
    label dictionary at all, fall back to their normalised string form and
    are sorted after all explicitly labelled values.
    """
    col_labels = label_map.get(orig_col) if orig_col else None
    if col_labels and v in col_labels:
        keys_in_order = list(col_labels.keys())
        return (0, keys_in_order.index(v))
    return (1, _na_safe_str(v))


def _label_text_value_key(
    orig_col: str | None,
    v: Any,
    label_map: dict[str, dict[Any, str]],
) -> tuple[int, str]:
    """Sort key for sort_by='label': order alphabetically by the DISPLAY.

    LABEL TEXT (the dict's value), not by dict-write order and not by
    the raw code.

    Looks up v ONLY in label_map[orig_col], so values are never resolved
    against the wrong column's dict. Values without a label fall back
    to their normalised string form, sorted after all explicitly
    labelled values.
    """
    col_labels = label_map.get(orig_col) if orig_col else None
    if col_labels and v in col_labels:
        return (0, str(col_labels[v]))
    return (1, _na_safe_str(v))


def _sort_keys(
    keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]],
    sort_by: str,
    label_map: dict[str, dict[Any, str]],
) -> list[tuple[Any, Any]]:
    """Sort row/col keys, respecting sort_by='code', 'index', or 'label'."""
    if sort_by == "index" and label_map:
        value_key_fn = partial(_index_value_key, label_map=label_map)
    elif sort_by == "label" and label_map:
        value_key_fn = partial(_label_text_value_key, label_map=label_map)
    else:
        value_key_fn = None
    return _sort_row_keys(keys, hdr_path, value_key_fn=value_key_fn)


# ---------------------------------------------------------------------------
# Display-label formatting helpers
# ---------------------------------------------------------------------------


def _fmt_val(
    v: Any,
    label_map: dict[str, dict[Any, str]],
    col_name: str | None = None,
) -> str:
    """Format a group key value for display, applying label remapping.

    col_name : the original groupby column this value belongs to. Only
               that column's label dict (label_map[col_name]) is consulted,
               so the same raw value (e.g. 1) used in two different
               groupby columns never gets the wrong column's label.
               If col_name is None or not in label_map, no remapping
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
        col_labels = label_map.get(col_name)
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


def _slot_range(
    local_pos: int,
    num_positions: int,
    slots: list[int],
    D: int,
) -> tuple[int, int]:
    # local_pos is the index within path_order (0 = outermost of THIS spec)
    # map to the global slots list (bottom-aligned)
    global_pos = len(slots) - num_positions + local_pos
    low = sum(slots[global_pos + 1 :])
    high = low + slots[global_pos] - 1
    return D - 1 - high, D - 1 - low  # (hi_idx, lo_idx)


def _place_group_entry(
    row: list[str],
    hi_idx: int,
    lo_idx: int,
    label: str,
    orig_name: str | None,
    is_total: bool,
    data_iter: Any,
    slots: list[int],
    global_pos: int,
    label_map: dict[str, dict[Any, str]],
) -> None:
    if is_total:
        return
    val = next(data_iter, None)
    val_str = _fmt_val(val, label_map, col_name=orig_name)
    # In the 2-slot system, every group position has a dedicated
    # label slot (hi_idx) and value slot (lo_idx). Whether a
    # group is "preceding" or "last" no longer matters for slot
    # allocation — both always emit label at hi_idx (if non-blank)
    # and value at lo_idx. The old "preceding group → value only"
    # rule was a 1-slot workaround; it dropped labels that now
    # have their own dedicated row.
    has_label_slot = slots[global_pos] == 2
    if label and has_label_slot:
        row[hi_idx] = label
    row[lo_idx] = val_str


def _key_to_label_slotted(
    hdr: Any,
    data_key: Any,
    path_order: list[tuple[Any, ...]],
    slots: list[int],
    label_map: dict[str, dict[Any, str]],
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
    num_positions = len(path_order)

    for pos, entry in enumerate(path_order):
        kind = entry[0]
        label = entry[1]
        orig_name = entry[2] if len(entry) > 2 else None
        hi_idx, lo_idx = _slot_range(pos, num_positions, slots, D)

        if kind != "group":
            row[lo_idx] = label
        else:
            global_pos = len(slots) - num_positions + pos
            _place_group_entry(
                row,
                hi_idx,
                lo_idx,
                label,
                orig_name,
                is_total,
                data_iter,
                slots,
                global_pos,
                label_map,
            )

    return tuple(row) if any(row) else ("",)


def _drop_blank_levels(labels: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Drop index levels that are blank in every row."""
    if labels and len(labels[0]) > 1:
        keep = [i for i in range(len(labels[0])) if any(t[i] for t in labels)]
        if len(keep) < len(labels[0]):
            labels = [tuple(t[i] for i in keep) for t in labels]
    return labels


def _make_index(
    keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]],
    label_map: dict[str, dict[Any, str]],
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
    label_map : dict[str, dict[Any, str]]
        Mapping from groupby column name to a display-label dictionary.

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
            label_map,
        )
        for hdr, dk in keys
    ]

    labels = _drop_blank_levels(labels)

    n_levels = len(labels[0]) if labels else 0
    if n_levels == 0:
        return pd.Index([""] * len(keys))
    if n_levels == 1:
        return cast(pd.Index, pd.Index([t[0] for t in labels]))
    return pd.MultiIndex.from_tuples(labels)


# ---------------------------------------------------------------------------
# Format-map / matrix / result assembly
# ---------------------------------------------------------------------------


def _build_fmt_map(
    keys: list[tuple[Any, Any]],
    specs: list[dict[str, Any]],
) -> dict[int, Callable[[Any], str]]:
    """Map key position -> formatter callable for a dimension's format= specs."""
    fmt_map: dict[int, Callable[[Any], str]] = {}
    for i, key in enumerate(keys):
        hdr, _ = key
        for spec in specs:
            if _spec_header(spec) == hdr and spec.get("fmt"):
                fmt_map[i] = _parse_fmt_spec(spec["fmt"])
                break
    return fmt_map


def _build_matrix(
    cells: dict[Any, dict[Any, Any]],
    all_row_keys: list[tuple[Any, Any]],
    all_col_keys: list[tuple[Any, Any]],
) -> np.ndarray:
    matrix = np.full((len(all_row_keys), len(all_col_keys)), np.nan)
    rk_pos = {rk: i for i, rk in enumerate(all_row_keys)}
    ck_pos = {ck: j for j, ck in enumerate(all_col_keys)}

    for rk, col_dict in cells.items():
        for ck, val in col_dict.items():
            matrix[rk_pos[rk], ck_pos[ck]] = val

    return matrix


def _apply_na_rep(base: pd.DataFrame, na_rep: str | None) -> pd.DataFrame:
    # Replace NaN cells with na_rep text when requested.
    if na_rep is None:
        return base
    base = base.astype(object)
    # na_rep forces object dtype -> col_fmt_map can't be applied to numerics
    # reliably anymore for the repr path, but flextab_to_string still works
    # via pd.isna() check before formatting.
    return base.where(base.notna(), other=na_rep)


def _apply_row_header(result: FlextabResult, row_header: str | None) -> None:
    # Apply row_header: name the row index so it prints as a column label
    if row_header is None:
        return
    if isinstance(result.index, pd.MultiIndex):
        result.index.names = [row_header, *result.index.names[1:]]
    else:
        result.index.name = row_header


def _build_result(
    base: pd.DataFrame,
    col_fmt_map: dict[int, Callable[[Any], str]],
    row_fmt_map: dict[int, Callable[[Any], str]],
    fmt: str,
    style: dict[str, Any] | None,
    na_rep: str | None,
    row_header: str | None,
) -> FlextabResult:
    base = _apply_na_rep(base, na_rep)

    result = FlextabResult(base)
    result.attrs["col_fmt_map"] = col_fmt_map
    result.attrs["row_fmt_map"] = row_fmt_map
    result.attrs["default_fmt"] = fmt
    result.attrs["style"] = style or {}

    _apply_row_header(result, row_header)

    return result


def flextab(
    data: pd.DataFrame,
    measure: str | list[str] | None = None,
    groupby: str | list[str] | None = None,
    table: str | None = None,
    include_missing_in_groupby: bool = True,
    fmt: str = "{:.1f}",
    na_rep: str | None = None,
    labels: dict[str, dict[Any, str]] | None = None,
    sort_by: str = "code",
    weight: str | None = None,
    row_header: str | None = None,
    style: dict[str, Any] | None = None,
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

    measure : str | list[str] | None
        Numeric analysis variable name or names (SAS: VAR). A single column
        can be passed as a plain string, for example ``measure="income"``
        instead of ``measure=["income"]``.

        Omit this argument for count-only tables that use ``N``, ``COUNT``,
        ``SIZE``, or percentage statistics that do not require a measure.

    groupby : str | list[str] | None
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

    labels : dict[str, dict[Any, str]] | None
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

    style : dict[str, Any] | None
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
            }

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
    measure, groupby = _normalize_measure_groupby(measure, groupby)
    missing = include_missing_in_groupby
    # Flat lookup: original_value -> display_label for any groupby column.
    # Used by _fmt_val (via _make_index) to remap codes to labels, and by
    # _sort_keys for sort_by='index'/'label'.
    label_map = labels or {}

    if table is None:
        table = _default_table_expr(measure, groupby)

    dims = parse_table(table)
    row_dim, col_dim = _split_dims(dims)

    col_specs = _expand_dim(col_dim, measure, groupby)
    row_specs = _expand_dim(row_dim, measure, groupby)

    cells, row_hdr_path, col_hdr_path = _build_cells(
        data, row_specs, col_specs, groupby, measure, missing, weight
    )

    all_row_keys = _sort_keys(
        list(dict.fromkeys(rk for rk in cells)), row_hdr_path, sort_by, label_map
    )
    all_col_keys = _sort_keys(
        list(dict.fromkeys(ck for rk in cells for ck in cells[rk])),
        col_hdr_path,
        sort_by,
        label_map,
    )

    matrix = _build_matrix(cells, all_row_keys, all_col_keys)

    # Build column-format and row-format maps: index position -> formatter
    # callable, sourced from each spec's format= specification (if any).
    col_fmt_map = _build_fmt_map(all_col_keys, col_specs)
    row_fmt_map = _build_fmt_map(all_row_keys, row_specs)

    row_idx = _make_index(all_row_keys, row_hdr_path, label_map)
    col_idx = _make_index(all_col_keys, col_hdr_path, label_map)

    base = pd.DataFrame(matrix, index=row_idx, columns=col_idx)

    return _build_result(base, col_fmt_map, row_fmt_map, fmt, style, na_rep, row_header)


# ---------------------------------------------------------------------------
# Row/column key ordering
# ---------------------------------------------------------------------------


def _get_path_order(hdr: Any, hdr_path: dict[Any, tuple[list[Any], int]]) -> list[Any]:
    entry: tuple[list[Any], int] = hdr_path.get(hdr, ([], 0))
    return entry[0]


def _get_branch(hdr: Any, hdr_path: dict[Any, tuple[list[Any], int]]) -> int:
    entry: tuple[list[Any], int] = hdr_path.get(hdr, ([], 0))
    return entry[1]


def _compute_position_group_flags(
    row_keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]],
) -> tuple[int, dict[int, bool]]:
    """Determine the max path length and which positions hold a real groupby var."""
    max_len = 0
    is_group_position: dict[int, bool] = {}
    for hdr, _ in row_keys:
        po = _get_path_order(hdr, hdr_path)
        max_len = max(max_len, len(po))
        for i, entry in enumerate(po):
            if entry[0] == "group":
                is_group_position[i] = True
            else:
                is_group_position.setdefault(i, False)
    return max_len, is_group_position


def _compute_label_order(
    row_keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]],
    is_group_position: dict[int, bool],
) -> dict[int, dict[int, dict[Any, int]]]:
    """Assign a first-seen ordinal to each STAT/VAR label, scoped per branch.

    IMPORTANT: this must be scoped PER BRANCH, not globally across the
    whole dimension. Each top-level concatenated piece of the TABLE
    expression (e.g. "sum*w n*format=... colpctn*format=... (SUM
    COLPCTSUM n nmiss ...)*format=...*income") is its own branch, and the
    branch index already determines their relative order (branch is the
    primary sort key, compared before any of this). If label_order were
    shared globally, a stat name seen early in one branch (e.g. a
    standalone "n*format=6,0") would claim a low ordinal that then leaks
    into an unrelated later branch containing the SAME stat name again
    (e.g. "n" inside "(SUM COLPCTSUM n nmiss ...)"), silently reordering
    that branch's internal stats to match the EARLIER branch's first
    appearance instead of THIS branch's own written order.
    """
    label_order: dict[int, dict[int, dict[Any, int]]] = {}
    for hdr, _ in row_keys:
        po = _get_path_order(hdr, hdr_path)
        branch = _get_branch(hdr, hdr_path)
        branch_orders = label_order.setdefault(branch, {})
        for i, entry in enumerate(po):
            if not is_group_position.get(i, False):
                d = branch_orders.setdefault(i, {})
                if entry[1] not in d:
                    d[entry[1]] = len(d)
    return label_order


def _compute_col_ordinal(
    row_keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]],
    is_group_position: dict[int, bool],
) -> dict[int, dict[Any, int]]:
    """For each group-position, order the distinct original column names.

    Computed in first-seen order across all specs. Specs sharing the
    same orig_col at a given position should have their values sorted
    together (e.g. all origin values); specs with different orig_cols at
    that position should be fully separated (e.g. gender block then
    age_group block, even when they share value strings like '1' and '2').
    """
    pos_col_ordinal: dict[int, dict[Any, int]] = {}
    for hdr, _ in row_keys:
        po = _get_path_order(hdr, hdr_path)
        for i, entry in enumerate(po):
            if is_group_position.get(i, False) and entry[0] == "group":
                orig_col = entry[2] if len(entry) > 2 else entry[1]
                d = pos_col_ordinal.setdefault(i, {})
                if orig_col not in d:
                    d[orig_col] = len(d)
    return pos_col_ordinal


def _value_sort_key(
    orig_col: str | None,
    v: Any,
    value_key_fn: Callable[[str | None, Any], tuple[int, int | str]] | None,
) -> Any:
    if value_key_fn is not None:
        return value_key_fn(orig_col, v)
    return _na_safe_str(v)


def _row_sort_part(
    i: int,
    po: list[Any],
    is_group_position: dict[int, bool],
    pos_col_ordinal: dict[int, dict[Any, int]],
    label_order: dict[int, dict[int, dict[Any, int]]],
    branch: int,
    dk_iter: Any,
    value_key_fn: Callable[[str | None, Any], tuple[int, int | str]] | None,
) -> tuple[Any, ...]:
    """Build one element of a row/col sort key, for cross-position ``i``."""
    if i >= len(po):
        return (0, 0, "")

    entry = po[i]

    if not is_group_position.get(i, False):
        ordinal = label_order.get(branch, {}).get(i, {}).get(entry[1], 0)
        return (ordinal,)

    if entry[0] != "group":
        # ALL/TOTAL at a group position: sort first within its col_ord=0
        return (0, 0, "")

    val = next(dk_iter, "")
    orig_col = entry[2] if len(entry) > 2 else None
    # col_ord separates specs whose position-i variables differ
    # (e.g. gender vs age_group both at position 1).
    # Specs sharing the same orig_col at this position compare
    # by value alone (col_ord ties → compare val).
    col_ord = pos_col_ordinal.get(i, {}).get(orig_col, 0)
    return (1, col_ord, _value_sort_key(orig_col, val, value_key_fn))


def _row_sort_key(
    rk: tuple[Any, Any],
    hdr_path: dict[Any, tuple[list[Any], int]],
    max_len: int,
    is_group_position: dict[int, bool],
    pos_col_ordinal: dict[int, dict[Any, int]],
    label_order: dict[int, dict[int, dict[Any, int]]],
    value_key_fn: Callable[[str | None, Any], tuple[int, int | str]] | None,
) -> tuple[Any, ...]:
    hdr, dk = rk
    po = _get_path_order(hdr, hdr_path)
    branch = _get_branch(hdr, hdr_path)

    if dk == (_SENTINEL,):
        dk = ()

    dk_iter = iter(dk if isinstance(dk, tuple) else (dk,))
    parts = [
        _row_sort_part(
            i,
            po,
            is_group_position,
            pos_col_ordinal,
            label_order,
            branch,
            dk_iter,
            value_key_fn,
        )
        for i in range(max_len)
    ]
    return (branch, *parts)


def _sort_row_keys(
    row_keys: list[tuple[Any, Any]],
    hdr_path: dict[Any, tuple[list[Any], int]] | None = None,
    value_key_fn: Callable[[str | None, Any], tuple[int, int | str]] | None = None,
) -> list[tuple[Any, Any]]:
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

    max_len, is_group_position = _compute_position_group_flags(row_keys, hdr_path)
    label_order = _compute_label_order(row_keys, hdr_path, is_group_position)
    pos_col_ordinal = _compute_col_ordinal(row_keys, hdr_path, is_group_position)

    key_fn = partial(
        _row_sort_key,
        hdr_path=hdr_path,
        max_len=max_len,
        is_group_position=is_group_position,
        pos_col_ordinal=pos_col_ordinal,
        label_order=label_order,
        value_key_fn=value_key_fn,
    )

    return sorted(row_keys, key=key_fn)


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


def _normalise_idx(idx_tuple: tuple[Any, ...]) -> tuple[Any, ...]:
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
        idx_tuple: tuple[Any, ...]
        if isinstance(series.index, pd.MultiIndex):
            idx_tuple = cast("tuple[Any, ...]", idx_val)
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
    generator = np.random.default_rng(42)
    n = 200
    demo = pd.DataFrame(
        {
            "origin": generator.choice(["Asia", "Europe", "USA"], n),
            "type": generator.choice(["Sedan", "SUV", "Truck"], n),
            "msrp": generator.normal(35000, 12000, n).clip(10000),
            "horsepower": generator.normal(220, 60, n).clip(80),
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
