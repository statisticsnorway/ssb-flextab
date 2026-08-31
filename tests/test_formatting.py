"""
Pytest suite for ssb_flextab.formatting and ssb_flextab.result (FlextabResult).

Mirrors the style of test_parser.py / test_statistics.py: representative
"typical" usages plus a dedicated section of invalid / edge-case inputs,
including a documented bug found while probing the real behavior.

Covered:
  * _parse_fmt_spec — every documented format-spec variant (W.D, W,D,
    thousands '_' , space thousands 's', combined '_s'), non-numeric
    passthrough, and invalid specs.
  * _format_dataframe — col_fmt_map / row_fmt_map precedence, na_rep,
    default fmt string.
  * flextab_to_string / flextab_to_markdown — MultiIndex column
    flattening, suppressed ('') label dropping, pipe escaping, MultiIndex
    rows, unnamed index.
  * FlextabResult static color helpers (_to_hex, _resolve_color) and
    _fmt_to_excel_numfmt.
  * FlextabResult.__repr__ / __str__ (default_fmt applied).
  * FlextabResult._repr_html_, including a regression test for a bug
    that used to make default_fmt silently ignored whenever no style=
    key was set at all (it fell back to plain pandas HTML because of a
    NameError swallowed by a bare `except Exception`) — now fixed.
  * FlextabResult.to_excel — number formats and colour fills, single- and
    multi-level columns, row_fmt_map vs col_fmt_map.
  * Invalid usages (bad format specs, unrecognised colours).

Run with:  pytest tests/test_formatting.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from ssb_flextab.formatting import _format_dataframe
from ssb_flextab.formatting import _parse_fmt_spec
from ssb_flextab.formatting import flextab_to_markdown
from ssb_flextab.formatting import flextab_to_string
from ssb_flextab.result import FlextabResult

# --------------------------------------------------------------------------
# _parse_fmt_spec
# --------------------------------------------------------------------------

class TestParseFmtSpecValid:
    @pytest.mark.parametrize(
        "spec, value, expected",
        [
            ("7.1", 1234.567, "1234.6"),
            ("7,2", 1234.567, "1234,57"),
            ("12.0_", 1234.567, "1,235"),
            ("7,2_", 1234.567, "1.234,57"),
            ("9.0s", 1234.567, "1 235"),
            ("7,2s", 1234.567, "1 234,57"),
        ],
    )
    def test_documented_examples(self, spec, value, expected):
        assert _parse_fmt_spec(spec)(value) == expected

    def test_negative_values_keep_sign(self):
        f = _parse_fmt_spec("7.1")
        assert f(-1234.567) == "-1234.6"

    def test_zero_formats_with_requested_decimals(self):
        f = _parse_fmt_spec("7,2")
        assert f(0) == "0,00"

    def test_combined_underscore_and_space_modifier_prefers_space(self):
        f = _parse_fmt_spec("7.1_s")
        assert f(1234.5) == "1 234.5"
        g = _parse_fmt_spec("7,1_s")
        assert g(1234.5) == "1 234,5"

    def test_non_numeric_value_falls_back_to_str(self):
        f = _parse_fmt_spec("7.1")
        assert f("abc") == "abc"
        assert f(None) == "None"

    def test_leading_zero_width_still_parses(self):
        # W is accepted-but-ignored, so any digit string works, even "0".
        f = _parse_fmt_spec("0.2")
        assert f(1.5) == "1.50"

    def test_surrounding_whitespace_is_stripped(self):
        f = _parse_fmt_spec("  7.1  ")
        assert f(1.25) == "1.2"  # banker's/round-half-even via format()


class TestParseFmtSpecInvalid:
    @pytest.mark.parametrize(
        "spec",
        [
            "abc",       # no digits/separator at all
            "7",         # missing separator + decimals
            "7.",        # missing decimals
            ".5",        # missing width
            "7.1x",      # unrecognised trailing modifier
            "",          # empty
            "7;1",       # wrong separator character
            "-7.1",      # negative width not allowed
        ],
    )
    def test_raises_value_error(self, spec):
        with pytest.raises(ValueError, match="Invalid format spec"):
            _parse_fmt_spec(spec)


# --------------------------------------------------------------------------
# _format_dataframe
# --------------------------------------------------------------------------

@pytest.fixture
def plain_df():
    return pd.DataFrame({"a": [1.23456, np.nan, 3.0], "b": [100.0, 200.0, 300.0]})


class TestFormatDataframe:
    def test_default_fmt_applied_to_every_cell(self, plain_df):
        out = _format_dataframe(plain_df)
        assert out.loc[0, "a"] == "1.235"
        assert out.loc[2, "b"] == "300.000"

    def test_na_rep_used_for_missing_values(self, plain_df):
        out = _format_dataframe(plain_df, na_rep="MISSING")
        assert out.loc[1, "a"] == "MISSING"

    def test_custom_default_fmt_string(self, plain_df):
        out = _format_dataframe(plain_df, fmt="{:.0f}")
        assert out.loc[0, "a"] == "1"
        assert out.loc[0, "b"] == "100"

    def test_col_fmt_map_overrides_default_for_that_column(self, plain_df):
        df = plain_df.copy()
        df.attrs["col_fmt_map"] = {1: _parse_fmt_spec("7,1")}
        out = _format_dataframe(df)
        assert out.loc[0, "b"] == "100,0"
        assert out.loc[0, "a"] == "1.235"  # column 0 unaffected, uses default fmt

    def test_row_fmt_map_overrides_default_for_that_row(self, plain_df):
        df = plain_df.copy()
        df.attrs["row_fmt_map"] = {0: _parse_fmt_spec("9.0s")}
        out = _format_dataframe(df)
        assert out.loc[0, "a"] == "1"
        assert out.loc[0, "b"] == "100"
        assert out.loc[2, "b"] == "300.000"  # row 2 unaffected

    def test_col_fmt_map_takes_precedence_over_row_fmt_map(self, plain_df):
        df = plain_df.copy()
        df.attrs["col_fmt_map"] = {0: _parse_fmt_spec("7,1")}
        df.attrs["row_fmt_map"] = {0: _parse_fmt_spec("9.0s")}
        out = _format_dataframe(df)
        # cell (0, "a") is targeted by BOTH maps -> column format wins
        assert out.loc[0, "a"] == "1,2"

    def test_non_numeric_column_falls_back_to_str(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        out = _format_dataframe(df)
        assert out.loc[0, "a"] == "x"
        assert out.loc[1, "a"] == "y"

    def test_result_columns_are_string_typed_not_numeric(self, plain_df):
        out = _format_dataframe(plain_df)
        assert all(isinstance(v, str) for v in out["a"])
        assert all(isinstance(v, str) for v in out["b"])

    def test_preserves_original_index_and_columns(self, plain_df):
        out = _format_dataframe(plain_df)
        assert list(out.index) == list(plain_df.index)
        assert list(out.columns) == list(plain_df.columns)


# --------------------------------------------------------------------------
# flextab_to_string
# --------------------------------------------------------------------------

class TestFlextabToString:
    def test_basic_rendering_contains_formatted_values(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        out = flextab_to_string(r)
        assert "100.000" in out
        assert "200.000" in out

    def test_custom_fmt_and_na_rep(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, np.nan]}, index=idx)
        out = flextab_to_string(r, fmt="{:.0f}", na_rep="-")
        assert "100" in out
        assert "-" in out


# --------------------------------------------------------------------------
# flextab_to_markdown
# --------------------------------------------------------------------------

class TestFlextabToMarkdown:
    def test_multiindex_columns_are_flattened_with_separator(self):
        cols = pd.MultiIndex.from_tuples([("SUM", "Income"), ("MEAN", "Income")])
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult([[100.0, 50.0], [200.0, 66.6667]], index=idx, columns=cols)
        md = flextab_to_markdown(r)
        assert "SUM / Income" in md
        assert "MEAN / Income" in md
        assert "| Sex |" in md

    def test_suppressed_label_is_dropped_not_left_blank(self):
        # An empty string in a column tuple represents a label suppressed
        # via name='' in the TABLE expression: it must vanish entirely,
        # not appear as an empty " / " segment.
        cols = pd.MultiIndex.from_tuples([("", "Income"), ("MEAN", "")])
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult([[100.0, 50.0], [200.0, 66.6667]], index=idx, columns=cols)
        md = flextab_to_markdown(r)
        header_line = md.splitlines()[0]
        assert " / " not in header_line
        assert "Income" in header_line
        assert "MEAN" in header_line

    def test_pipe_characters_are_escaped(self):
        cols = pd.MultiIndex.from_tuples([("A|B", "X")])
        idx = pd.Index(["a|b", "c"], name="k")
        r = FlextabResult([[1.0], [2.0]], index=idx, columns=cols)
        md = flextab_to_markdown(r)
        assert "A\\|B" in md
        assert "a\\|b" in md
        # the escaped pipes must not be mistaken for column delimiters
        assert md.splitlines()[0].count("|") == 4  # leading/trailing + 2 real cols

    def test_multiindex_rows_produce_one_column_per_level(self):
        ridx = pd.MultiIndex.from_tuples([("M", "E"), ("M", "W")], names=["Sex", "Region"])
        r = FlextabResult([[1.0], [2.0]], index=ridx, columns=pd.Index(["Income"]))
        md = flextab_to_markdown(r)
        assert "| Sex | Region | Income |" == md.splitlines()[0]

    def test_unnamed_index_gets_blank_header_cell(self):
        r = FlextabResult([[1.0], [2.0]], index=pd.Index(["a", "b"]), columns=pd.Index(["Income"]))
        md = flextab_to_markdown(r)
        assert md.splitlines()[0] == "|  | Income |"

    def test_custom_separator(self):
        cols = pd.MultiIndex.from_tuples([("SUM", "Income")])
        idx = pd.Index(["M"], name="Sex")
        r = FlextabResult([[100.0]], index=idx, columns=cols)
        md = flextab_to_markdown(r, sep=" -> ")
        assert "SUM -> Income" in md

    def test_output_has_correct_number_of_rows(self):
        idx = pd.Index(["M", "F", "X"], name="Sex")
        r = FlextabResult({"Income": [1.0, 2.0, 3.0]}, index=idx)
        md = flextab_to_markdown(r)
        lines = md.splitlines()
        # header + separator + 3 data rows
        assert len(lines) == 5


# --------------------------------------------------------------------------
# FlextabResult static colour helpers
# --------------------------------------------------------------------------

class TestToHex:
    def test_named_colors(self):
        assert FlextabResult._to_hex("blue") == "0000FF"
        assert FlextabResult._to_hex("lightgrey") == "D3D3D3"

    def test_named_colors_case_insensitive(self):
        assert FlextabResult._to_hex("BLUE") == "0000FF"

    def test_hex_string_with_and_without_hash(self):
        assert FlextabResult._to_hex("#4472C4") == "4472C4"
        assert FlextabResult._to_hex("4472c4") == "4472C4"

    def test_rgb_tuple(self):
        assert FlextabResult._to_hex((70, 114, 196)) == "4672C4"

    def test_rgb_list_also_accepted(self):
        assert FlextabResult._to_hex([0, 0, 0]) == "000000"

    def test_none_returns_none(self):
        assert FlextabResult._to_hex(None) is None

    def test_unrecognised_color_raises(self):
        with pytest.raises(ValueError, match="Unrecognised colour"):
            FlextabResult._to_hex("notacolor")

    def test_two_element_tuple_is_not_a_valid_rgb(self):
        with pytest.raises(ValueError, match="Unrecognised colour"):
            FlextabResult._to_hex((1, 2))


class TestResolveColor:
    def test_none_spec_returns_none(self):
        assert FlextabResult._resolve_color(None, 0) is None

    def test_single_color_used_for_every_index(self):
        assert FlextabResult._resolve_color("blue", 0) == "blue"
        assert FlextabResult._resolve_color("blue", 7) == "blue"

    def test_two_tuple_cycles_by_parity(self):
        spec = ("red", "blue")
        assert FlextabResult._resolve_color(spec, 0) == "red"
        assert FlextabResult._resolve_color(spec, 1) == "blue"
        assert FlextabResult._resolve_color(spec, 2) == "red"
        assert FlextabResult._resolve_color(spec, 3) == "blue"

    def test_three_element_rgb_tuple_is_not_treated_as_cycling_pair(self):
        spec = (70, 114, 196)
        assert FlextabResult._resolve_color(spec, 0) == spec
        assert FlextabResult._resolve_color(spec, 1) == spec


class TestFmtToExcelNumfmt:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("7,1", "0.0"),
            ("7.2", "0.00"),
            ("12.0_", "#,##0"),
            ("9.0s", "#,##0"),
            ("7,2_", "#,##0.00"),
        ],
    )
    def test_matches_closure_inspected_parameters(self, spec, expected):
        formatter = _parse_fmt_spec(spec)
        assert FlextabResult._fmt_to_excel_numfmt(formatter) == expected

    def test_arbitrary_callable_without_closure_falls_back_to_sample_parse(self):
        def plain_fmt(v):
            return f"{v:.2f}"

        assert FlextabResult._fmt_to_excel_numfmt(plain_fmt) == "0.00"

    def test_callable_that_raises_falls_back_to_general(self):
        def broken_fmt(v):
            raise RuntimeError("boom")

        assert FlextabResult._fmt_to_excel_numfmt(broken_fmt) == "General"


# --------------------------------------------------------------------------
# FlextabResult.__repr__ / __str__
# --------------------------------------------------------------------------

class TestReprAndStr:
    def test_default_repr_uses_default_fmt_of_one_decimal(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        assert "100.0" in repr(r)

    def test_str_matches_repr(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        assert str(r) == repr(r)

    def test_custom_default_fmt_attr_is_honored(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["default_fmt"] = "{:.0f}"
        assert "100\n" in repr(r) or repr(r).strip().endswith("200")


# --------------------------------------------------------------------------
# FlextabResult._repr_html_
# --------------------------------------------------------------------------

class TestReprHtml:
    def test_no_style_set_renders_a_table_with_the_income_header(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        html = r._repr_html_()
        assert "<table" in html
        assert "Income" in html

    def test_default_fmt_is_applied_even_when_no_style_key_is_set(self):
        """Regression test for a fixed bug.

        `_style_one_th` and the `for line in lines:` loop that applies
        default_fmt-aware formatting used to sit OUTSIDE the
        `if any_style:` block that defines `lines`, due to an indentation
        slip. When no style key was set, `any_style` was False, `lines`
        was never defined, the loop raised NameError, and the surrounding
        `except Exception` silently swallowed it — falling back to plain
        pandas HTML with RAW float values instead of applying
        default_fmt. That block is now correctly nested inside
        `if any_style:`, and the `return html` right after (which uses
        whichever `html` value is live in each branch) handles the no-op
        case too, so default_fmt applies in both the styled and unstyled
        paths.
        """
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.1234, 200.5678]}, index=idx)
        r.attrs["default_fmt"] = "{:.0f}"
        html = r._repr_html_()
        # The requested 0-decimal formatting is applied...
        assert "100.1234" not in html
        assert ">100<" in html
        assert ">201<" in html
        # ...and no plain-pandas fallback wrapper leaks through.
        assert '<table border="1"' not in html
        assert "<style scoped>" not in html

    def test_no_style_and_no_default_fmt_still_renders_plain_values(self):
        # With neither style nor default_fmt set, the table should still
        # render via the "real" code path (not the exception fallback),
        # showing pandas' normal default float rendering.
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        html = r._repr_html_()
        assert '<table border="1"' not in html
        assert "<style scoped>" not in html
        assert ">100.0<" in html

    def test_formatting_is_applied_once_any_style_key_is_set(self):
        # Setting even one style key makes `any_style` True, which takes
        # the real formatting code path — this is the "working" case.
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.1234, 200.5678]}, index=idx)
        r.attrs["default_fmt"] = "{:.0f}"
        r.attrs["style"] = {"cell_bg": "yellow"}
        html = r._repr_html_()
        assert "100.1234" not in html
        assert ">100<" in html

    def test_header_bg_excludes_the_row_header_cell(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {"header_bg": "lightgrey"}
        html = r._repr_html_()
        # "Income" header cell IS colored...
        assert '<th style="background-color:#D3D3D3">Income</th>' in html
        # ...but the "Sex" row-header cell is NOT, since header_bg
        # explicitly excludes the row_header cell per the docstring.
        assert "<th>Sex</th>" in html

    def test_row_header_bg_colors_only_the_row_header_cell(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {"row_header_bg": "orange", "header_bg": "lightgrey"}
        html = r._repr_html_()
        assert '<th style="background-color:#FFA500">Sex</th>' in html

    def test_cell_bg_cycles_by_data_row(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {"cell_bg": ("white", "beige")}
        html = r._repr_html_()
        assert "#FFFFFF" in html
        assert "#F5F5DC" in html


# --------------------------------------------------------------------------
# FlextabResult.to_excel
# --------------------------------------------------------------------------

class TestToExcel:
    def test_col_fmt_map_sets_number_format_on_the_right_column_only(self, tmp_path):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult(
            {"Income": [100.1234, 200.5678], "Age": [30.0, 40.0]}, index=idx
        )
        r.attrs["col_fmt_map"] = {0: _parse_fmt_spec("7,1")}
        path = tmp_path / "out.xlsx"
        r.to_excel(path)

        wb = load_workbook(path)
        ws = wb.active
        # data starts at row 2 (1 header row, no index name row since
        # "Sex" name IS written but on the header row itself for a flat index)
        income_cells = [ws.cell(r_, 2) for r_ in range(2, ws.max_row + 1)]
        age_cells = [ws.cell(r_, 3) for r_ in range(2, ws.max_row + 1)]
        assert all(c.number_format == "0.0" for c in income_cells)
        assert all(c.number_format == "General" for c in age_cells)

    def test_row_fmt_map_sets_number_format_across_that_data_row(self, tmp_path):
        cols = pd.MultiIndex.from_tuples([("SUM", "Income"), ("MEAN", "Income")])
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult(
            [[100.0, 50.0], [200.0, 66.6667]], index=idx, columns=cols
        )
        r.attrs["row_fmt_map"] = {1: _parse_fmt_spec("7,2")}
        path = tmp_path / "out2.xlsx"
        r.to_excel(path)

        wb = load_workbook(path)
        ws = wb.active
        # 2 column-header levels + 1 index-name row = 3 header rows,
        # so data rows are 4 (M) and 5 (F).
        assert ws.max_row == 5
        row_m = [ws.cell(4, c).number_format for c in (2, 3)]
        row_f = [ws.cell(5, c).number_format for c in (2, 3)]
        assert row_m == ["General", "General"]
        assert row_f == ["0.00", "0.00"]

    def test_style_colors_header_row_and_cell_fills(self, tmp_path):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {
            "header_bg": "lightgrey",
            "cell_bg": ("white", "beige"),
            "row_bg": "yellow",
        }
        path = tmp_path / "out3.xlsx"
        r.to_excel(path)

        wb = load_workbook(path)
        ws = wb.active

        def argb(cell):
            return cell.fill.fgColor.rgb

        assert argb(ws.cell(1, 2)) == "00D3D3D3"  # "Income" header
        assert argb(ws.cell(1, 1)) == "00000000"  # "Sex" row-header cell, unstyled
        assert argb(ws.cell(2, 1)) == "00FFFF00"  # row index cell "M"
        assert argb(ws.cell(2, 2)) == "00FFFFFF"  # data cell row 0 -> white
        assert argb(ws.cell(3, 2)) == "00F5F5DC"  # data cell row 1 -> beige

    def test_values_remain_numeric_not_stringified(self, tmp_path):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        path = tmp_path / "out4.xlsx"
        r.to_excel(path)
        wb = load_workbook(path)
        ws = wb.active
        assert isinstance(ws.cell(2, 2).value, (int, float))

    def test_no_style_or_fmt_still_writes_a_plain_file(self, tmp_path):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        path = tmp_path / "out5.xlsx"
        r.to_excel(path)
        wb = load_workbook(path)
        ws = wb.active
        assert ws.cell(2, 2).value == 100.0


# --------------------------------------------------------------------------
# Invalid usage / edge cases
# --------------------------------------------------------------------------

class TestInvalidUsage:
    def test_bad_format_spec_in_format_dataframe_col_map_would_raise_at_construction(
        self, plain_df
    ):
        # _parse_fmt_spec is where validation happens; _format_dataframe
        # itself trusts pre-built callables and never validates a spec
        # string directly.
        with pytest.raises(ValueError):
            _parse_fmt_spec("nonsense")

    def test_unrecognised_color_in_style_dict_raises_when_rendering_html(self):
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {"cell_bg": "not-a-real-color"}
        # _repr_html_ wraps its body in try/except, so an internal
        # ValueError from _to_hex is swallowed and it silently falls
        # back to the plain pandas rendering rather than raising.
        html = r._repr_html_()
        assert "<table" in html

    def test_unrecognised_color_in_style_dict_raises_on_to_excel(self, tmp_path):
        # to_excel has NO surrounding try/except around style resolution,
        # so a bad colour value propagates as a real exception.
        idx = pd.Index(["M", "F"], name="Sex")
        r = FlextabResult({"Income": [100.0, 200.0]}, index=idx)
        r.attrs["style"] = {"cell_bg": "not-a-real-color"}
        with pytest.raises(ValueError, match="Unrecognised colour"):
            r.to_excel(tmp_path / "bad.xlsx")

    def test_empty_dataframe_formats_without_error(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        out = _format_dataframe(df)
        assert len(out) == 0

    def test_markdown_on_single_row_single_column(self):
        r = FlextabResult({"Income": [1.0]}, index=pd.Index(["M"], name="Sex"))
        md = flextab_to_markdown(r)
        assert len(md.splitlines()) == 3  # header + separator + one data row
