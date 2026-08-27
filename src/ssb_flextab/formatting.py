import re

import pandas as pd


def _parse_fmt_spec(spec: str):
    """Parse a format specification string and return a callable formatter.

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

    Examples:
    --------
      "7.1"   -> 1 decimal, point:              1 234.6
      "7,2"   -> 2 decimals, comma:             1 234,56
      "12.0_" -> 0 decimals, comma thousands:   1,235
      "7,2_"  -> 2 decimals, European:          1.234,56
      "9.0s"  -> 0 decimals, space thousands:   1 235
      "7,2s"  -> 2 decimals, comma + space:     1 234,56

    Returns:
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

def _format_dataframe(result, fmt="{:.3f}", na_rep="."):
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
    """Render a flextab() result as a formatted string.

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

    Returns:
    -------
    str
        A fixed-width string suitable for printing.

    Notes:
    -----
    Per-cell format= specs from the TABLE expression (e.g. *format=7,1
    or *format=12.0s) take precedence over the fmt parameter for their
    specific cells.  The fmt parameter acts as the default for any cell
    without an explicit format= spec.

    Examples:
    --------
    print(flextab_to_string(r))                   # default fmt
    print(flextab_to_string(r, fmt="{:.0f}"))     # 0 decimals everywhere
    print(flextab_to_string(r, na_rep="-"))       # dash for missing
    """
    return _format_dataframe(result, fmt=fmt, na_rep=na_rep).to_string()


def flextab_to_markdown(result, fmt="{:.3f}", na_rep=".", sep=" / ") -> str:
    """Render a flextab() result as a plain Markdown table.
    
    The table contains flattened, human-readable column headers instead of 
    the raw index tuples that pandas' inherited DataFrame.to_markdown()
    shows for a MultiIndex, and with format= specs from the TABLE expression
    applied to the numbers.

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

    Returns:
    -------
    str
        A GitHub-flavoured Markdown table.

    Examples:
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
    for idx_vals, (_, row) in zip(
        index_rows,
        formatted.iterrows(),
        strict=True,
    ):
        cells = idx_vals + [_escape(str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)