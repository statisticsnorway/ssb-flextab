from __future__ import annotations

import html as html_stdlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from typing import ClassVar

import pandas as pd
from openpyxl.cell.cell import Cell
from openpyxl.cell.cell import MergedCell
from openpyxl.styles.fills import PatternFill

from .formatting import _format_dataframe

_NAMED_COLORS: dict[str, str] = {
    "black": "000000",
    "white": "FFFFFF",
    "red": "FF0000",
    "green": "008000",
    "blue": "0000FF",
    "yellow": "FFFF00",
    "orange": "FFA500",
    "purple": "800080",
    "grey": "808080",
    "gray": "808080",
    "lightgrey": "D3D3D3",
    "lightgray": "D3D3D3",
    "darkgrey": "A9A9A9",
    "darkgray": "A9A9A9",
    "navy": "000080",
    "teal": "008080",
    "maroon": "800000",
    "silver": "C0C0C0",
    "lime": "00FF00",
    "cyan": "00FFFF",
    "magenta": "FF00FF",
    "pink": "FFC0CB",
    "beige": "F5F5DC",
}


# ---------------------------------------------------------------------------
# Colour helpers (module-level; FlextabResult._to_hex / _resolve_color
# delegate to these so the public staticmethod API is unchanged)
# ---------------------------------------------------------------------------


def _rgb_tuple_to_hex(color: Any) -> str:
    return "{:02X}{:02X}{:02X}".format(*[int(c) for c in color])


def _is_hex_digits(s: str) -> bool:
    return len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s)


def _normalise_color(color: Any) -> str | None:
    """Normalise a colour specification to a 6-character uppercase hex string.

    (no '#' prefix) suitable for openpyxl and CSS.

    Accepts:
      Named string   'blue', 'red', 'lightgrey', …  (CSS colour names)
      Hex string     '#4472C4'  or  '4472C4'
      RGB tuple      (70, 114, 196)
    """
    if color is None:
        return None
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return _rgb_tuple_to_hex(color)
    s = str(color).strip()
    if s.startswith("#"):
        s = s[1:]
    if _is_hex_digits(s):
        return s.upper()
    key = s.lower()
    if key in _NAMED_COLORS:
        return _NAMED_COLORS[key]
    raise ValueError(
        f"Unrecognised colour {color!r}. "
        "Use a hex string ('#4472C4'), an RGB tuple (70,114,196), "
        "or a basic CSS colour name ('blue', 'red', …)."
    )


def _is_cycling_pair(spec: Any) -> bool:
    return (
        isinstance(spec, (list, tuple))
        and len(spec) == 2
        and not (isinstance(spec[0], int) and len(spec) == 3)
    )


def _resolve_color(spec: Any, idx: int) -> Any:
    """Resolve a style-dict colour spec for row/band index `idx`.

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
    if _is_cycling_pair(spec):
        return spec[idx % 2]
    return spec


def _resolve_hex(spec: Any, idx: int) -> str | None:
    """Resolve a style spec straight to a hex colour string for row `idx`."""
    val = _resolve_color(spec, idx)
    return _normalise_color(val) if val is not None else None


# ---------------------------------------------------------------------------
# Excel number-format helpers
# ---------------------------------------------------------------------------


def _numfmt_from_closure(formatter: Callable[..., Any]) -> str:
    """Build an Excel number format by inspecting a formatter's closure vars.

    Raises if the closure can't be inspected this way.
    """
    closure = formatter.__closure__
    if closure is None:
        raise ValueError("Formatter has no closure")

    fvars = formatter.__code__.co_freevars
    fvals = {k: v.cell_contents for k, v in zip(fvars, closure, strict=True)}
    decimals = fvals.get("decimals", 0)
    use_thousands = fvals.get("use_thousands", False)  # _ separator
    use_space_thous = fvals.get("use_space_thous", False)  # s separator

    # Build the Excel number format. Excel always uses ',' for thousands
    # grouping internally regardless of locale display.
    thous = use_thousands or use_space_thous
    dec_part = "." + "0" * decimals if decimals > 0 else ""
    if thous:
        return f"#,##0{dec_part}"
    return f"0{dec_part}"


def _numfmt_from_sample(formatter: Callable[..., Any]) -> str:
    """Fallback: infer decimal places by parsing the formatted output."""
    try:
        sample = formatter(1234.5)
        for sep in (",", "."):
            if sep in sample:
                return f'0.{"0" * len(sample.split(sep)[-1])}'
        return "0"
    except Exception:
        return "General"


def _fmt_to_excel_numfmt(formatter: Callable[..., Any]) -> str:
    """Convert a _parse_fmt_spec formatter to an Excel number format string.

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
        return _numfmt_from_closure(formatter)
    except Exception:
        return _numfmt_from_sample(formatter)


# ---------------------------------------------------------------------------
# Shared style-spec container (used by both HTML display and Excel export)
# ---------------------------------------------------------------------------


@dataclass
class _StyleSpecs:
    header_bg: Any = None
    header_fg: Any = None
    row_bg: Any = None
    row_fg: Any = None
    row_header_bg: Any = None
    row_header_fg: Any = None
    cell_bg: Any = None
    cell_fg: Any = None

    def any_set(self) -> bool:
        return any(
            [
                self.header_bg,
                self.header_fg,
                self.row_bg,
                self.row_fg,
                self.row_header_bg,
                self.row_header_fg,
                self.cell_bg,
                self.cell_fg,
            ]
        )


def _style_specs_from_dict(style: dict[str, Any]) -> _StyleSpecs:
    return _StyleSpecs(
        header_bg=style.get("header_bg"),
        header_fg=style.get("header_fg"),
        row_bg=style.get("row_bg"),
        row_fg=style.get("row_fg"),
        row_header_bg=style.get("row_header_bg"),
        row_header_fg=style.get("row_header_fg"),
        cell_bg=style.get("cell_bg"),
        cell_fg=style.get("cell_fg"),
    )


def _row_header_name_set(index: pd.Index) -> set[Any]:
    if isinstance(index, pd.MultiIndex):
        return {n for n in index.names if n}
    return {index.name} if index.name else set()


# ---------------------------------------------------------------------------
# HTML styling helpers (used by FlextabResult._repr_html_)
# ---------------------------------------------------------------------------


def _css_string(bg: Any = None, fg: Any = None) -> str:
    parts = []
    if bg:
        hex_bg = _normalise_color(bg)
        parts.append(f"background-color:#{hex_bg}")
    if fg:
        hex_fg = _normalise_color(fg)
        parts.append(f"color:#{hex_fg}")
    return ";".join(parts)


def _style_one_th(m: re.Match[str], css_str: str | None) -> str:
    if not css_str:
        return m.group(0)
    return re.sub(
        r"<th\b([^>]*)>",
        lambda mm: f'<th{mm.group(1)} style="{css_str}">',
        m.group(0),
    )


def _replace_header(m: re.Match[str], css_str: str) -> str:
    return _style_one_th(m, css_str)


def _replace_row_header(m: re.Match[str], css_str: str) -> str:
    return f'<th{m.group(1)} style="{css_str}">'


def _replace_cell(m: re.Match[str], css_str: str) -> str:
    return f'<td{m.group(1)} style="{css_str}">'


@dataclass
class _HtmlRowState:
    in_thead: bool = False
    in_thead_row: bool = False
    header_row_idx: int = 0
    data_row_idx: int = 0


def _style_thead_cell_line(
    line: str,
    stripped: str,
    state: _HtmlRowState,
    specs: _StyleSpecs,
    row_header_names: set[Any],
) -> str:
    # Is this the row_header cell (holds the row_header= text), or a plain
    # column-header cell?
    text = html_stdlib.unescape(re.sub(r"<[^>]+>", "", stripped)).strip()
    if row_header_names and text in row_header_names:
        css_str = _css_string(
            _resolve_color(specs.row_header_bg, state.header_row_idx),
            _resolve_color(specs.row_header_fg, state.header_row_idx),
        )
    else:
        css_str = _css_string(
            _resolve_color(specs.header_bg, state.header_row_idx),
            _resolve_color(specs.header_fg, state.header_row_idx),
        )
    if not css_str:
        return line
    return re.sub(
        r"<th\b[^>]*>.*?</th>",
        partial(_replace_header, css_str=css_str),
        line,
    )


def _style_row_index_line(line: str, state: _HtmlRowState, specs: _StyleSpecs) -> str:
    # Row index cell (groupby LABEL or VALUE cell on the left).
    css_str = _css_string(
        _resolve_color(specs.row_bg, state.data_row_idx),
        _resolve_color(specs.row_fg, state.data_row_idx),
    )
    if not css_str:
        return line
    return re.sub(
        r"<th\b([^>]*?)>",
        partial(_replace_row_header, css_str=css_str),
        line,
    )


def _style_data_cell_line(line: str, state: _HtmlRowState, specs: _StyleSpecs) -> str:
    css_str = _css_string(
        _resolve_color(specs.cell_bg, state.data_row_idx),
        _resolve_color(specs.cell_fg, state.data_row_idx),
    )
    if not css_str:
        return line
    return re.sub(
        r"<td\b([^>]*?)>",
        partial(_replace_cell, css_str=css_str),
        line,
    )


def _update_thead_open_close(stripped: str, state: _HtmlRowState) -> None:
    if "<thead>" in stripped:
        state.in_thead = True
    if "</thead>" in stripped:
        state.in_thead = False


def _advance_row_counters(stripped: str, state: _HtmlRowState) -> None:
    if "</tr>" in stripped:
        if state.in_thead:
            state.header_row_idx += 1
            state.in_thead_row = False
        else:
            state.data_row_idx += 1


def _style_html_line(
    line: str,
    state: _HtmlRowState,
    specs: _StyleSpecs,
    row_header_names: set[Any],
) -> str:
    stripped = line.strip()
    _update_thead_open_close(stripped, state)

    if state.in_thead and stripped.startswith("<tr"):
        state.in_thead_row = True
    elif state.in_thead and state.in_thead_row and stripped.startswith("<th"):
        line = _style_thead_cell_line(line, stripped, state, specs, row_header_names)
    elif not state.in_thead and stripped.startswith("<tr>"):
        pass  # no row-level styling — each cell below styles itself
    elif not state.in_thead and stripped.startswith("<th"):
        line = _style_row_index_line(line, state, specs)
    elif not state.in_thead and stripped.startswith("<td"):
        line = _style_data_cell_line(line, state, specs)

    _advance_row_counters(stripped, state)
    return line


def _apply_html_styles(
    html: str, specs: _StyleSpecs, row_header_names: set[Any]
) -> str:
    state = _HtmlRowState()
    lines = html.splitlines()
    out = [_style_html_line(line, state, specs, row_header_names) for line in lines]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Excel export helpers (used by FlextabResult.to_excel)
# ---------------------------------------------------------------------------


@dataclass
class _ExcelLayout:
    n_col_header_rows: int
    n_row_index_cols: int
    n_data_rows: int
    n_data_cols: int
    first_data_row: int
    has_row_header: bool
    row_header_row: int | None


def _compute_excel_layout(result: pd.DataFrame, ws: Any) -> _ExcelLayout:
    # pandas writes n_col_header_rows for the column MultiIndex levels,
    # PLUS an extra row for the index name(s) when any index level has a
    # name set (e.g. row_header="Origin and type"). Detect this extra row
    # by comparing the actual sheet row count to what we'd expect without it.
    n_col_header_rows = result.columns.nlevels
    n_row_index_cols = result.index.nlevels
    n_data_rows = len(result)
    n_data_cols = len(result.columns)

    expected_rows_no_name = n_col_header_rows + n_data_rows
    actual_rows = ws.max_row
    index_name_rows = max(0, actual_rows - expected_rows_no_name)
    first_data_row = n_col_header_rows + 1 + index_name_rows

    has_row_header = (
        bool(result.index.names[0])
        if isinstance(result.index, pd.MultiIndex)
        else bool(result.index.name)
    )
    row_header_row = (first_data_row - 1) if has_row_header else None

    return _ExcelLayout(
        n_col_header_rows=n_col_header_rows,
        n_row_index_cols=n_row_index_cols,
        n_data_rows=n_data_rows,
        n_data_cols=n_data_cols,
        first_data_row=first_data_row,
        has_row_header=has_row_header,
        row_header_row=row_header_row,
    )


def _try_import_openpyxl() -> tuple[Any, Any, Any] | None:
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font
        from openpyxl.styles import PatternFill as PatternFillCls

        return load_workbook, Font, PatternFillCls
    except ImportError:
        return None


def _resolve_workbook(
    excel_writer: str | Path | pd.ExcelWriter,
    is_path: bool,
    buf: Any,
    load_workbook: Callable[..., Any],
) -> Any:
    if buf is not None:
        buf.seek(0)
        return load_workbook(buf)
    if is_path:
        return None  # unreachable in practice, but keeps mypy honest
    try:
        return excel_writer.book  # type: ignore[union-attr]
    except AttributeError:
        return None


def _make_fill(pattern_fill_cls: type[PatternFill], hex_color: str) -> PatternFill:
    return pattern_fill_cls(
        start_color=hex_color, end_color=hex_color, fill_type="solid"
    )


def _make_font(font_cls: Any, hex_color: str) -> Any:
    return font_cls(color=hex_color)


def _apply_cell_colors(
    cell: Cell | MergedCell,
    pattern_fill_cls: Any,
    font_cls: Any,
    bg_hex: str | None = None,
    fg_hex: str | None = None,
) -> None:
    if bg_hex:
        cell.fill = _make_fill(pattern_fill_cls, bg_hex)
    if fg_hex:
        cell.font = _make_font(font_cls, fg_hex)


def _style_header_rows(
    ws: Any, specs: _StyleSpecs, layout: _ExcelLayout, pattern_fill_cls: Any, font_cls: Any
) -> None:
    for r in range(1, layout.first_data_row):
        header_row_idx = r - 1
        hdr_bg = _resolve_hex(specs.header_bg, header_row_idx)
        hdr_fg = _resolve_hex(specs.header_fg, header_row_idx)
        skip_cols = layout.n_row_index_cols if r == layout.row_header_row else 0
        for c in range(skip_cols + 1, layout.n_row_index_cols + layout.n_data_cols + 1):
            _apply_cell_colors(ws.cell(r, c), pattern_fill_cls, font_cls, hdr_bg, hdr_fg)


def _style_row_header_cells(
    ws: Any, specs: _StyleSpecs, layout: _ExcelLayout, pattern_fill_cls: Any, font_cls: Any
) -> None:
    # The row_header= text lives in the LAST header row's index columns —
    # either its own dedicated row (when pandas writes one) or merged into
    # the same row as the column headers (when it doesn't). These cells are
    # styled separately by row_header_bg/fg so header_bg never bleeds into
    # them when row_header_bg is left unset.
    if layout.row_header_row is None:
        return
    rh_header_row_idx = layout.row_header_row - 1
    rh_bg = _resolve_hex(specs.row_header_bg, rh_header_row_idx)
    rh_fg = _resolve_hex(specs.row_header_fg, rh_header_row_idx)
    for c in range(1, layout.n_row_index_cols + 1):
        _apply_cell_colors(
            ws.cell(layout.row_header_row, c), pattern_fill_cls, font_cls, rh_bg, rh_fg
        )


def _style_data_rows(
    ws: Any, specs: _StyleSpecs, layout: _ExcelLayout, pattern_fill_cls: Any, font_cls: Any
) -> None:
    # Row index cells get row_bg/fg ONLY, data cells get cell_bg/fg ONLY —
    # neither falls back to the other.
    for data_row_idx in range(layout.n_data_rows):
        xl_row = layout.first_data_row + data_row_idx
        cur_cell_bg = _resolve_hex(specs.cell_bg, data_row_idx)
        cur_cell_fg = _resolve_hex(specs.cell_fg, data_row_idx)
        cur_row_bg = _resolve_hex(specs.row_bg, data_row_idx)
        cur_row_fg = _resolve_hex(specs.row_fg, data_row_idx)

        for c in range(1, layout.n_row_index_cols + layout.n_data_cols + 1):
            cell = ws.cell(xl_row, c)
            if c <= layout.n_row_index_cols:
                _apply_cell_colors(cell, pattern_fill_cls, font_cls, cur_row_bg, cur_row_fg)
            else:
                _apply_cell_colors(cell, pattern_fill_cls, font_cls, cur_cell_bg, cur_cell_fg)


def _apply_col_number_formats(
    ws: Any, col_fmt_map: dict[int, Callable[[Any], str]], layout: _ExcelLayout
) -> None:
    for data_col_idx in range(layout.n_data_cols):
        formatter = col_fmt_map.get(data_col_idx)
        if not formatter:
            continue
        num_fmt = _fmt_to_excel_numfmt(formatter)
        xl_col = layout.n_row_index_cols + 1 + data_col_idx
        for data_row_idx in range(layout.n_data_rows):
            xl_row = layout.first_data_row + data_row_idx
            cell = ws.cell(xl_row, xl_col)
            if isinstance(cell.value, (int, float)):
                cell.number_format = num_fmt


def _apply_row_number_formats(
    ws: Any, row_fmt_map: dict[int, Callable[[Any], str]], layout: _ExcelLayout
) -> None:
    for data_row_idx in range(layout.n_data_rows):
        formatter = row_fmt_map.get(data_row_idx)
        if not formatter:
            continue
        num_fmt = _fmt_to_excel_numfmt(formatter)
        xl_row = layout.first_data_row + data_row_idx
        for data_col_idx in range(layout.n_data_cols):
            xl_col = layout.n_row_index_cols + 1 + data_col_idx
            cell = ws.cell(xl_row, xl_col)
            if isinstance(cell.value, (int, float)):
                cell.number_format = num_fmt


class FlextabResult(pd.DataFrame):
    """DataFrame subclass returned by ``flextab()``.

    Numeric values are fully preserved. Arithmetic, ``.sum()``, slicing, and
    other DataFrame operations work as on a regular ``pd.DataFrame``.

    The main differences are in display and export behaviour:

    - ``print(result)`` applies ``format=`` specifications via ``__str__``.
    - ``repr(result)`` applies the same formatting.
    - Jupyter display applies ``format=`` specifications and CSS colours via
    ``_repr_html_``.
    - ``result.to_excel(path)`` writes an Excel file with number formatting and
    colour styling preserved.

    Use ``flextab_to_string(result, fmt, na_rep)`` for explicit control over
    plain-text rendering.

    Notes
    -----
    Display and export metadata are stored in ``result.attrs``. Relevant keys
    include:

    ``col_fmt_map``
        Mapping from column position to formatter callable.

    ``row_fmt_map``
        Mapping from row position to formatter callable.

    ``default_fmt``
        Fallback Python format string.

    ``style``
        Colour styling configuration. See the ``style`` parameter in
        ``flextab()``.
    """

    _metadata: ClassVar[list[str]] = []

    @property
    def _constructor(self) -> type[FlextabResult]:
        return FlextabResult

    # ── Internal colour helpers ────────────────────────────────────────────
    # (kept as staticmethods for API compatibility; logic lives in the
    # module-level functions above so it can be unit tested and reused by
    # both _repr_html_ and to_excel without going through the class.)

    @staticmethod
    def _to_hex(color: Any) -> str | None:
        """Normalise a colour specification to a 6-character uppercase hex string.

        (no '#' prefix) suitable for openpyxl and CSS.

        Accepts:
          Named string   'blue', 'red', 'lightgrey', …  (CSS colour names)
          Hex string     '#4472C4'  or  '4472C4'
          RGB tuple      (70, 114, 196)
        """
        return _normalise_color(color)

    @staticmethod
    def _resolve_color(spec: Any, idx: int) -> Any:
        """Resolve a style-dict colour spec for row/band index `idx`.

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
        return _resolve_color(spec, idx)

    @staticmethod
    def _fmt_to_excel_numfmt(
        formatter: Callable[..., Any],
    ) -> str:
        """Convert a _parse_fmt_spec formatter to an Excel number format string.

        Inspects the closure variables of the formatter to reliably determine
        decimals, decimal-comma, and thousands-separator settings — rather than
        trying to parse the formatted output, which breaks for space-thousands
        and edge cases.
        """
        return _fmt_to_excel_numfmt(formatter)

    # ── Display ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return the formatted string representation of the result."""
        try:
            fmt = self.attrs.get("default_fmt", "{:.1f}")
            return str(_format_dataframe(self, fmt=fmt, na_rep=".").to_string())
        except Exception:
            return str(super().__repr__())

    def __str__(self) -> str:
        """Return the result as a formatted string."""
        return self.__repr__()

    def _repr_html_(self) -> str:
        """Jupyter/IPython HTML display.

        With format= specs AND inline CSS colours from the style= parameter
        applied.

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

            specs = _style_specs_from_dict(style)
            row_header_names = _row_header_name_set(self.index)

            html = formatted.to_html(border=0)

            if specs.any_set():
                html = _apply_html_styles(html, specs, row_header_names)

            return str(html)
        except Exception:
            # pandas-stubs doesn't declare this IPython display hook on
            # DataFrame, even though it exists at runtime.
            return str(super()._repr_html_())  # type: ignore[misc]

    # ── Excel export ──────────────────────────────────────────────────────

    def to_excel(  # type: ignore[misc, override]
        self,
        excel_writer: str | Path | pd.ExcelWriter,
        sheet_name: str = "Sheet1",
        **kwargs: Any,
    ) -> None:
        """Write the result to Excel with formatting and colour styling.

        Parameters
        ----------
        excel_writer : str | Path | pd.ExcelWriter
            File path or open pandas Excel writer.

        sheet_name : str
            Name of the worksheet to write to.

        **kwargs : Any
            Additional keyword arguments passed to ``pd.DataFrame.to_excel()``.

        Returns
        -------
        None
            The result is written to the supplied file path or Excel writer.

        Notes
        -----
        Cells with a ``format=`` specification in the TABLE expression receive a
        corresponding Excel number format. For example, ``format=7,1`` is converted
        to an Excel number format with one decimal place and thousands grouping.
        Values remain numeric in Excel and can therefore be sorted, summed, and used
        in formulas.

        Excel number formats are stored using Excel's locale-independent format
        syntax. Decimal and thousands separators are displayed according to the
        user's regional Excel settings.

        Colour styling from the ``style`` dictionary supplied to ``flextab()`` is
        also applied during export. Supported style keys are:

        - ``header_bg`` and ``header_fg`` for column header cells.
        - ``row_bg`` and ``row_fg`` for row index cells.
        - ``row_header_bg`` and ``row_header_fg`` for the row-header cell.
        - ``cell_bg`` and ``cell_fg`` for data cells.

        Each style key may contain either a single colour, applied uniformly, or a
        two-element sequence of colours that alternates between rows.

        If ``excel_writer`` is a file path, the file is written and post-processed
        in one operation. If it is an open ``pd.ExcelWriter``, the worksheet is
        formatted immediately after writing, but the caller remains responsible for
        closing the writer or using it as a context manager.
        """
        import io

        openpyxl_imports = _try_import_openpyxl()

        col_fmt_map = self.attrs.get("col_fmt_map", {})
        row_fmt_map = self.attrs.get("row_fmt_map", {})
        style = self.attrs.get("style", {})

        is_path = isinstance(excel_writer, (str, Path))

        # ── Write the plain DataFrame first ──────────────────────────────
        buf = io.BytesIO() if (openpyxl_imports is not None and is_path) else None
        target = buf if buf is not None else excel_writer
        super().to_excel(target, sheet_name=sheet_name, **kwargs)

        if openpyxl_imports is None:
            return  # openpyxl not available: plain write is all we can do

        load_workbook, font_cls, pattern_fill_cls = openpyxl_imports

        # ── Post-process with openpyxl ───────────────────────────────────
        wb = _resolve_workbook(excel_writer, is_path, buf, load_workbook)
        if wb is None:
            return  # can't access/load the workbook; skip formatting

        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active
        if ws is None:
            return  # no worksheet available; skip formatting

        layout = _compute_excel_layout(self, ws)
        specs = _style_specs_from_dict(style)

        # ── Header rows (column headers, excluding the row_header cells) ───
        _style_header_rows(ws, specs, layout, pattern_fill_cls, font_cls)

        # ── Row header cell(s): styled only by row_header_bg/fg ────────────
        _style_row_header_cells(ws, specs, layout, pattern_fill_cls, font_cls)

        # ── Data rows ────────────────────────────────────────────────────
        _style_data_rows(ws, specs, layout, pattern_fill_cls, font_cls)

        # ── Number formats ────────────────────────────────────────────────
        _apply_col_number_formats(ws, col_fmt_map, layout)
        _apply_row_number_formats(ws, row_fmt_map, layout)

        # ── Save ─────────────────────────────────────────────────────────
        if buf is not None:
            wb.save(excel_writer)
