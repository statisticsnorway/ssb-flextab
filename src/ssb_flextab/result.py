from __future__ import annotations

import numpy as np
import pandas as pd

from .formatting import _format_dataframe

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

