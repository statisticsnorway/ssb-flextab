import pandas as pd
import pytest

from ssb_flextab import FlextabResult
from ssb_flextab import flextab

from tests.helpers import cell

class TestResultObject:

    def test_returns_flextab_result_subclass(self, df):
        r = flextab(data=df, groupby="sex", table="sex, N")
        assert isinstance(r, FlextabResult)
        assert isinstance(r, pd.DataFrame)

    def test_values_stay_numeric_for_further_computation(self, df):
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*SUM"
        )
        assert r["SUM"].sum() == pytest.approx(df["income"].sum())

    def test_to_excel_writes_a_file(self, df, tmp_path):
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

    def test_repr_html_runs_without_error(self, df):
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(N MEAN)",
            style={"cell_bg": ("white", "#EBF3FB")},
        )
        html = r._repr_html_()
        assert "<table" in html
