from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ssb_flextab import FlextabResult
from ssb_flextab import flextab


class TestResultObject:

    def test_returns_flextab_result_subclass(self, df: pd.DataFrame) -> None:
        r = flextab(data=df, groupby="sex", table="sex, N")
        assert isinstance(r, FlextabResult)
        assert isinstance(r, pd.DataFrame)

    def test_values_stay_numeric_for_further_computation(
        self, df: pd.DataFrame
    ) -> None:
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*SUM"
        )
        assert r["SUM"].sum() == pytest.approx(df["income"].sum())

    def test_to_excel_writes_a_file(self, df: pd.DataFrame, tmp_path: Path) -> None:
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(N MEAN)",
            style={"header_bg": "#4472C4", "header_fg": "white"},
        )
        out_path = tmp_path / "out.xlsx"
        r.to_excel(out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_repr_html_runs_without_error(self, df: pd.DataFrame) -> None:
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(N MEAN)",
            style={"cell_bg": ("white", "#EBF3FB")},
        )
        html = r._repr_html_()
        assert "<table" in html

    def test_repr_html_applies_the_requested_colours(self, df: pd.DataFrame) -> None:
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(N MEAN)",
            style={"header_bg": "#4472C4", "header_fg": "white"},
        )
        html = r._repr_html_()
        assert "background-color:#4472C4" in html
        assert "color:#FFFFFF" in html

    def test_to_excel_applies_header_fill(
        self, df: pd.DataFrame, tmp_path: Path
    ) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(N MEAN)",
            style={"header_bg": "#4472C4"},
        )
        out_path = tmp_path / "out.xlsx"
        r.to_excel(out_path)
        wb = openpyxl.load_workbook(out_path)
        ws = wb.active
        fills = {ws.cell(1, c).fill.fgColor.rgb for c in range(1, ws.max_column + 1)}
        assert "004472C4" in fills or "FF4472C4" in fills


class TestColorHelpers:
    """Direct checks of FlextabResult's colour-normalisation helpers.

    These back both _repr_html_ (CSS colours) and to_excel (openpyxl fills
    and fonts), so covering them in isolation is cheap and catches
    colour-parsing regressions closer to the source than the end-to-end
    style= tests above.
    """

    @pytest.mark.parametrize(
        ("color", "expected"),
        [
            ("#4472C4", "4472C4"),
            ("4472c4", "4472C4"),  # no '#', lowercase -> normalised & uppercased
            ("blue", "0000FF"),
            ("white", "FFFFFF"),
            ((70, 114, 196), "4672C4"),
        ],
    )
    def test_to_hex_normalises_supported_formats(
        self, color: str | tuple[int, int, int], expected: str
    ) -> None:
        assert FlextabResult._to_hex(color) == expected

    def test_to_hex_none_stays_none(self) -> None:
        assert FlextabResult._to_hex(None) is None

    def test_to_hex_rejects_unrecognised_color(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised colour"):
            FlextabResult._to_hex("not-a-color")

    def test_resolve_color_single_value_used_for_every_row(self) -> None:
        for idx in range(4):
            assert FlextabResult._resolve_color("blue", idx) == "blue"

    def test_resolve_color_two_tuple_cycles_by_parity(self) -> None:
        spec = ("white", "#EBF3FB")
        assert FlextabResult._resolve_color(spec, 0) == "white"
        assert FlextabResult._resolve_color(spec, 1) == "#EBF3FB"
        assert FlextabResult._resolve_color(spec, 2) == "white"
        assert FlextabResult._resolve_color(spec, 3) == "#EBF3FB"

    def test_resolve_color_rgb_triple_is_not_treated_as_a_cycling_pair(self) -> None:
        # A 3-element RGB tuple is a single colour, not (colorA, colorB).
        rgb = (70, 114, 196)
        assert FlextabResult._resolve_color(rgb, 0) == rgb
        assert FlextabResult._resolve_color(rgb, 1) == rgb

    def test_resolve_color_none_stays_none(self) -> None:
        assert FlextabResult._resolve_color(None, 0) is None
