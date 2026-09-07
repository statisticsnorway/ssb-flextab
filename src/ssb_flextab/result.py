from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from typing import ClassVar

import pandas as pd
from openpyxl.cell.cell import Cell
from openpyxl.cell.cell import MergedCell
from openpyxl.styles.fills import PatternFill

from .formatting import _format_dataframe


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

    @staticmethod
    def _to_hex(color: Any) -> str | None:
        """Normalise a colour specification to a 6-character uppercase hex string.

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
        key = s.lower()
        if key in _names:
            return _names[key]
        raise ValueError(
            f"Unrecognised colour {color!r}. "
            "Use a hex string ('#4472C4'), an RGB tuple (70,114,196), "
            "or a basic CSS colour name ('blue', 'red', …)."
        )

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
        if spec is None:
            return None
        if (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and not (isinstance(spec[0], int) and len(spec) == 3)
        ):
            return spec[idx % 2]
        return spec

    @staticmethod
    def _fmt_to_excel_numfmt(
        formatter: Callable[..., Any],
    ) -> str:
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
            # Access the closure to get the exact formatting parameters
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
            if decimals > 0:
                dec_part = "." + "0" * decimals
            else:
                dec_part = ""
            if thous:
                return f"#,##0{dec_part}"
            else:
                return f"0{dec_part}"
        except Exception:
            # Fall back to parsing the formatted output if closure inspection fails
            try:
                sample = formatter(1234.5)
                # Count decimal places
                for sep in (",", "."):
                    if sep in sample:
                        return f'0.{"0" * len(sample.split(sep)[-1])}'
                return "0"
            except Exception:
                return "General"

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

            def _css(
                bg: Any = None,
                fg: Any = None,
            ) -> str:
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
            row_bg_spec = style.get("row_bg")
            row_fg_spec = style.get("row_fg")
            rh_bg_spec = style.get("row_header_bg")
            rh_fg_spec = style.get("row_header_fg")
            cell_bg_spec = style.get("cell_bg")
            cell_fg_spec = style.get("cell_fg")

            # Text of the row_header cell(s), so we can single it out among
            # the <th> cells in the header area.
            if isinstance(self.index, pd.MultiIndex):
                row_header_names = {n for n in self.index.names if n}
            else:
                row_header_names = {self.index.name} if self.index.name else set()

            html = formatted.to_html(border=0)

            any_style = any(
                [
                    header_bg_spec,
                    header_fg_spec,
                    row_bg_spec,
                    row_fg_spec,
                    rh_bg_spec,
                    rh_fg_spec,
                    cell_bg_spec,
                    cell_fg_spec,
                ]
            )
            if any_style:
                import html as _html
                import re as _re

                lines = html.splitlines()
                out = []
                data_row_idx = 0
                header_row_idx = 0
                in_thead = False
                in_thead_row = False

                def _style_one_th(
                    m: re.Match[str],
                    css_str: str | None,
                ) -> str:
                    if not css_str:
                        return m.group(0)

                    return _re.sub(
                        r"<th\b([^>]*)>",
                        lambda mm: f'<th{mm.group(1)} style="{css_str}">',
                        m.group(0),
                    )

                def _replace_header(
                    m: re.Match[str],
                    css_str: str,
                ) -> str:
                    return _style_one_th(m, css_str)

                def _replace_row_header(
                    m: re.Match[str],
                    css_str: str,
                ) -> str:
                    return f'<th{m.group(1)} style="{css_str}">'

                def _replace_cell(
                    m: re.Match[str],
                    css_str: str,
                ) -> str:
                    return f'<td{m.group(1)} style="{css_str}">'

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
                        text = _html.unescape(_re.sub(r"<[^>]+>", "", stripped)).strip()
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
                                r"<th\b[^>]*>.*?</th>",
                                partial(_replace_header, css_str=css_str),
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
                                r"<th\b([^>]*?)>",
                                partial(_replace_row_header, css_str=row_css),
                                line,
                            )

                    elif not in_thead and stripped.startswith("<td"):
                        # Data cell.
                        cb = _resolve(cell_bg_spec, data_row_idx)
                        cf = _resolve(cell_fg_spec, data_row_idx)
                        cell_css = _css(cb, cf)
                        if cell_css:
                            line = _re.sub(
                                r"<td\b([^>]*?)>",
                                partial(_replace_cell, css_str=cell_css),
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

        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Font
            from openpyxl.styles import PatternFill

            HAS_OPENPYXL = True
        except ImportError:
            HAS_OPENPYXL = False

        col_fmt_map = self.attrs.get("col_fmt_map", {})
        row_fmt_map = self.attrs.get("row_fmt_map", {})
        style = self.attrs.get("style", {})

        is_path = isinstance(excel_writer, (str, Path))

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
            # ExcelWriter with openpyxl engine: access the workbook directly.
            # `is_path` is False here, so `excel_writer` is not a str/Path;
            # narrow it explicitly for mypy rather than relying on that
            # control-flow fact alone.
            if isinstance(excel_writer, (str, Path)):
                return  # unreachable in practice, but keeps mypy honest
            try:
                wb = excel_writer.book
            except AttributeError:
                return  # can't access workbook; skip formatting

        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active
        if ws is None:
            return  # no worksheet available; skip formatting

        # pandas writes n_col_header_rows for the column MultiIndex levels,
        # PLUS an extra row for the index name(s) when any index level has a
        # name set (e.g. row_header="Origin and type").  Detect this extra row
        # by comparing the actual sheet row count to what we'd expect without it.
        n_col_header_rows = self.columns.nlevels
        n_row_index_cols = self.index.nlevels
        n_data_rows = len(self)
        n_data_cols = len(self.columns)

        expected_rows_no_name = n_col_header_rows + n_data_rows
        actual_rows = ws.max_row
        # If pandas wrote an extra row for the index name, adjust the offset
        index_name_rows = max(0, actual_rows - expected_rows_no_name)
        first_data_row = n_col_header_rows + 1 + index_name_rows

        def _fill(hex_color: str) -> PatternFill:
            return PatternFill(
                start_color=hex_color,
                end_color=hex_color,
                fill_type="solid",
            )

        def _font(hex_color: str) -> Font:
            return Font(color=hex_color)

        def _apply(
            cell: Cell | MergedCell,
            bg_hex: str | None = None,
            fg_hex: str | None = None,
        ) -> None:
            if bg_hex:
                cell.fill = _fill(bg_hex)
            if fg_hex:
                cell.font = _font(fg_hex)

        header_bg_spec = style.get("header_bg")
        header_fg_spec = style.get("header_fg")
        row_bg_spec = style.get("row_bg")
        row_fg_spec = style.get("row_fg")
        rh_bg_spec = style.get("row_header_bg")
        rh_fg_spec = style.get("row_header_fg")
        cell_bg_spec = style.get("cell_bg")
        cell_fg_spec = style.get("cell_fg")

        def _resolve_hex(spec: Any, idx: int) -> str | None:
            """Resolve a style spec to a hexadecimal colour for row ``idx``."""
            val = FlextabResult._resolve_color(spec, idx)
            return FlextabResult._to_hex(val) if val is not None else None

        # The row_header= text lives in the LAST header row's index columns —
        # either its own dedicated row (when pandas writes one) or merged
        # into the same row as the column headers (when it doesn't). Those
        # cells are excluded from the header_bg/fg loop below and coloured
        # separately by row_header_bg/fg, so header_bg never bleeds into
        # them when row_header_bg is left unset.
        has_row_header = (
            bool(self.index.names[0])
            if isinstance(self.index, pd.MultiIndex)
            else bool(self.index.name)
        )
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
            cur_row_bg = _resolve_hex(row_bg_spec, data_row_idx)
            cur_row_fg = _resolve_hex(row_fg_spec, data_row_idx)

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
