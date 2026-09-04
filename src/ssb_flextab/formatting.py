from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .result import FlextabResult

import re

import pandas as pd


def _parse_fmt_spec(spec: str) -> Callable[[Any], str]:
    """Parse a TABLE format specification into a formatter callable.

    The formatter is used internally for ``format=`` specifications in the
    TABLE expression and is stored in ``result.attrs`` for later display and
    export formatting.

    Supported syntax is ``W<sep>D[modifier]``, where:

    - ``W`` is the total width. It is accepted for SAS compatibility but is
      not used for output padding.
    - ``<sep>`` is either ``"."`` for decimal point or ``","`` for decimal
      comma.
    - ``D`` is the number of decimal places.
    - ``modifier`` may be ``"_"`` for thousands grouping using the opposite
      separator, or ``"s"`` for space-separated thousands.

    Parameters
    ----------
    spec : str
        Format specification, for example ``"7.1"``, ``"7,2_"``, or
        ``"9.0s"``.

    Returns
    -------
    Callable
        Formatter callable accepting a value and returning its formatted
        string representation.

    Raises
    ------
    ValueError
        If ``spec`` does not match the supported ``W.D`` or ``W,D`` syntax.

    Examples
    --------
    ``"7.1"``
        One decimal place with decimal point.

    ``"7,2"``
        Two decimal places with decimal comma.

    ``"12.0_"``
        No decimal places and thousands grouping.

    ``"7,2_"``
        Two decimal places with European-style separators.

    ``"9.0s"``
        No decimal places and space-separated thousands.
    """
    m = re.fullmatch(r"[0-9]+([.,])([0-9]+)([_s]*)", spec.strip())
    if not m:
        raise ValueError(
            f"Invalid format spec {spec!r}. Expected W.D[_|s] or W,D[_|s]."
        )
    dec_sep = m.group(1)  # '.' or ',' — the decimal separator
    decimals = int(m.group(2))
    modifiers = m.group(3)
    use_comma = dec_sep == ","  # decimal comma instead of decimal point
    use_thousands = "_" in modifiers  # thousands grouping (opposite char of dec_sep)
    use_space_thous = "s" in modifiers  # thousands grouping with space

    def _fmt(value: Any) -> str:
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
                s = s.replace(",", " ").replace(".", ",")
            else:
                s = s.replace(",", " ")
        elif use_comma:
            # Swap . and , for European style: 1,234.56 -> 1.234,56
            s = s.replace(",", "\u00b6").replace(".", ",").replace("\u00b6", ".")

        return s

    return _fmt


def _format_dataframe(
    result: pd.DataFrame, fmt: str = "{:.3f}", na_rep: str = "."
) -> pd.DataFrame:
    """Build a string-valued copy of result with per-column AND per-row formats applied.

    A format= spec can appear on either the row or the column dimension of
    the TABLE expression (e.g. "n*format=6,0 rowpctn*format=7,1, ..." puts
    the format on the row stat instead of a column stat). For each cell,
    the column's format (result.attrs["col_fmt_map"]) takes precedence if
    present; otherwise the row's format (result.attrs["row_fmt_map"]) is
    used; otherwise the default fmt string applies.
    """
    col_fmt_map = result.attrs.get("col_fmt_map", {})
    row_fmt_map = result.attrs.get("row_fmt_map", {})

    def fmt_val(v: Any, row_pos: int, col_pos: int) -> str:
        if pd.isna(v):
            return na_rep
        formatter = col_fmt_map.get(col_pos) or row_fmt_map.get(row_pos)
        if formatter is not None:
            return str(formatter(v))
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


def flextab_to_string(
    result: "FlextabResult", fmt: str = "{:.3f}", na_rep: str = "."
) -> str:
    """Render a ``flextab()`` result as a formatted string.

    Parameters
    ----------
    result : FlextabResult
        Result returned by ``flextab()``.

    fmt : str
        Python format string applied to numeric cells without an explicit
        ``format=`` specification in the TABLE expression.

    na_rep : str
        Text shown in place of missing values.

    Returns
    -------
    str
        Fixed-width string representation of the table.

    Notes
    -----
    Per-cell ``format=`` specifications from the TABLE expression take
    precedence over ``fmt``. The ``fmt`` argument therefore acts as the
    fallback format for cells without an explicit specification.

    Examples
    --------
    Use the default formatting::

        print(flextab_to_string(r))

    Use zero decimal places::

        print(flextab_to_string(r, fmt="{:.0f}"))

    Display missing values as a dash::

        print(flextab_to_string(r, na_rep="-"))
    """
    return _format_dataframe(result, fmt=fmt, na_rep=na_rep).to_string()


def flextab_to_markdown(
    result: "FlextabResult",
    fmt: str = "{:.3f}",
    na_rep: str = ".",
    sep: str = " / ",
) -> str:
    """Render a ``flextab()`` result as a plain Markdown table.

    MultiIndex column headers are flattened into readable labels instead of
    being represented as raw tuples. Explicit ``format=`` specifications from
    the TABLE expression are applied to numeric values before rendering.

    Markdown does not support merged cells or multiple header rows. MultiIndex
    column levels are therefore joined into a single label using ``sep``.
    Empty levels are omitted.

    Parameters
    ----------
    result : FlextabResult
        Result returned by ``flextab()``.

    fmt : str
        Python format string applied to numeric cells without an explicit
        ``format=`` specification in the TABLE expression.

    na_rep : str
        Text shown in place of missing values.

    sep : str
        Separator used to join MultiIndex column levels into a single header
        label.

    Returns
    -------
    str
        GitHub-flavoured Markdown representation of the table.

    Notes
    -----
    Markdown tables cannot reproduce the merged-header structure used in the
    HTML representation. For documents that support raw HTML, use
    ``result._repr_html_()`` when the original multi-level header layout is
    required.

    Some Markdown renderers may remove inline CSS styling from embedded HTML.

    Examples
    --------
    Render with default settings::

        print(flextab_to_markdown(r))

    Use a dash for missing values::

        print(flextab_to_markdown(r, na_rep="-"))
    """
    formatted = _format_dataframe(result, fmt=fmt, na_rep=na_rep)

    def _escape(text: str) -> str:
        # A literal "|" would otherwise break the Markdown table's
        # column structure.
        return text.replace("|", "\\|")

    def _flatten(key: object) -> str:
        if isinstance(key, tuple):
            parts = [str(p) for p in key if str(p) != ""]
            label = sep.join(parts) if parts else ""
        else:
            label = str(key)
        return _escape(label)

    col_labels = [_flatten(c) for c in formatted.columns]

    if isinstance(formatted.index, pd.MultiIndex):
        index_names = [_escape(str(n)) if n else "" for n in formatted.index.names]
        index_rows = [
            [_escape(str(v)) for v in (idx if isinstance(idx, tuple) else (idx,))]
            for idx in formatted.index
        ]
    else:
        index_names = [
            _escape(str(formatted.index.name)) if formatted.index.name else ""
        ]
        index_rows = [[_escape(str(idx))] for idx in formatted.index]

    header = index_names + col_labels
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for idx_vals, (_, row) in zip(
        index_rows,
        formatted.iterrows(),
        strict=True,
    ):
        cells = idx_vals + [_escape(str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
