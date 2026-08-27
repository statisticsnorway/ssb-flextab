"""pytest test suite for flextab.py

Run with:
    pytest test_flextab.py -v

Every expected value below is computed independently with plain pandas/
numpy (not by calling flextab's own internals), so a test failure means
flextab's output has actually diverged from the documented behaviour —
not that the test is just re-deriving the same code path.
"""
import numpy as np
import pandas as pd
import pytest

from ssb_flextab import FlextabResult
from ssb_flextab import flextab
from ssb_flextab import flextab_to_string

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    return pd.DataFrame({
        "sex":        ["1", "1", "2", "1", None, "2", "2", "1", "2", "2"],
        "age_group":  ["2", "1", "2", "3", "1", "2", "2", "2", "3", None],
        "region":     [None, "2", "1", "2", "2", "3", "3", "2", "2", "1"],
        "education":  ["3", "2", "3", "1", "3", "3", "3", "1", "3", "2"],
        "income":     [300, 100, 450, 200, 650, 750, None, 850, 400, 350],
        "tax":        [100, 10, 200, 90, 340, 370, 30, None, 150, 150],
        "weight":     [1.5, 3.2, 1.7, 2.2, 6.1, 4.2, 1.9, 4.8, None, 8.2],
    })


@pytest.fixture
def labels():
    return {
        "sex":       {"1": "Males", "2": "Females"},
        "age_group": {"1": "0-19", "2": "20-66", "3": "67+"},
        "region":    {"1": "West", "2": "East", "3": "Central"},
        "education": {"3": "Higher education", "2": "Secondary school",
                       "1": "Elementary school"},
    }


# ---------------------------------------------------------------------------
# Small helper: pull a scalar cell out of a result by (row_key, col_key)
# ---------------------------------------------------------------------------

def cell(result, row, col):
    return result.loc[row, col]


# ---------------------------------------------------------------------------
# 1. Counts & missing-value handling
# ---------------------------------------------------------------------------

class TestCounts:

    def test_bare_n_with_no_groupby_or_measure(self, df):
        r = flextab(data=df, table="N")
        assert r.shape == (1, 1)
        assert cell(r, "", "N") == len(df)

    def test_groupby_counts_match_value_counts(self, df):
        r = flextab(data=df, groupby="sex", table="sex, N")
        expected = df["sex"].value_counts(dropna=False)
        assert cell(r, ("sex", "1"), "N") == expected["1"]
        assert cell(r, ("sex", "2"), "N") == expected["2"]
        # Missing values show up in value_counts()'s index as either None
        # or float nan depending on the pandas version, so don't index by
        # np.nan directly (KeyError on some versions) - just count misses.
        assert cell(r, ("sex", "nan"), "N") == df["sex"].isna().sum()
        assert r["N"].sum() == len(df)

    def test_include_missing_false_drops_nan_group(self, df):
        r = flextab(data=df, groupby="sex", table="sex, N",
                    include_missing_in_groupby=False)
        row_values = [idx[-1] if isinstance(idx, tuple) else idx for idx in r.index]
        assert "nan" not in row_values
        assert r["N"].sum() == df["sex"].notna().sum()

    def test_n_counts_nonmissing_measure_values(self, df):
        # sex == "2" has 5 rows but one has income missing -> N should be 4
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*N")
        assert cell(r, ("sex", "2"), "N") == 4
        assert cell(r, ("sex", "1"), "N") == 4

    def test_size_counts_all_rows_including_missing_measure(self, df):
        # SIZE counts every row in the group, unlike N/COUNT
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*SIZE")
        assert cell(r, ("sex", "2"), "SIZE") == 5   # includes the missing income row
        assert cell(r, ("sex", "1"), "SIZE") == 4


# ---------------------------------------------------------------------------
# 2. Descriptive statistics on a measure
# ---------------------------------------------------------------------------

class TestDescriptiveStats:

    def test_sum_and_mean_by_group(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(SUM MEAN)")
        expected_sum = df.groupby("sex", dropna=False)["income"].sum()
        expected_mean = df.groupby("sex", dropna=False)["income"].mean()
        for key in ["1", "2"]:
            assert cell(r, ("sex", key), "SUM") == pytest.approx(expected_sum[key])
            assert cell(r, ("sex", key), "MEAN") == pytest.approx(expected_mean[key])

    def test_std_and_var_use_sample_ddof(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(STD VAR)")
        expected_std = df.groupby("sex", dropna=False)["income"].std()  # ddof=1 default
        expected_var = df.groupby("sex", dropna=False)["income"].var()
        for key in ["1", "2"]:
            assert cell(r, ("sex", key), "STD") == pytest.approx(expected_std[key])
            assert cell(r, ("sex", key), "VAR") == pytest.approx(expected_var[key])

    def test_median_and_percentiles(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(MEDIAN P25 P75)")
        sub = df.loc[df["sex"] == "2", "income"]
        assert cell(r, ("sex", "2"), "MEDIAN") == pytest.approx(sub.median())
        assert cell(r, ("sex", "2"), "P25") == pytest.approx(sub.quantile(0.25))
        assert cell(r, ("sex", "2"), "P75") == pytest.approx(sub.quantile(0.75))

    def test_min_max(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(MIN MAX)")
        sub = df.loc[df["sex"] == "1", "income"]
        assert cell(r, ("sex", "1"), "MIN") == sub.min()
        assert cell(r, ("sex", "1"), "MAX") == sub.max()

    def test_nmiss_counts_missing_measure_values(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*NMISS")
        # income is missing exactly once, for a sex == "2" row (index 6)
        assert cell(r, ("sex", "2"), "NMISS") == 1
        assert cell(r, ("sex", "1"), "NMISS") == 0

    def test_gmean_uses_positive_values_only(self, df):
        r = flextab(data=df, measure="income", table="income=''*GMEAN")
        positive = df["income"].dropna()
        positive = positive[positive > 0]
        expected = np.exp(np.log(positive).mean())
        assert cell(r, "", "GMEAN") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. Percentage statistics
# ---------------------------------------------------------------------------

class TestPercentages:

    def test_pctn_rowpctn_colpctn_two_way(self, df):
        r = flextab(data=df, groupby=["sex", "region"],
                    table="sex, region*(N PCTN ROWPCTN COLPCTN)")

        # sex=="1" & region=="2" -> 3 of 4 "1" rows, 3 of 10 overall,
        # 3 of the 4 rows where region=="2"
        n_cell = cell(r, ("sex", "1"), ("region", "2", "N"))
        assert n_cell == 3

        pctn = cell(r, ("sex", "1"), ("region", "2", "PCTN"))
        assert pctn == pytest.approx(100 * 3 / len(df))

        rowpctn = cell(r, ("sex", "1"), ("region", "2", "ROWPCTN"))
        sex1_total = (df["sex"] == "1").sum()
        assert rowpctn == pytest.approx(100 * 3 / sex1_total)

        colpctn = cell(r, ("sex", "1"), ("region", "2", "COLPCTN"))
        region2_total = (df["region"] == "2").sum()
        assert colpctn == pytest.approx(100 * 3 / region2_total)

    def test_pctsum_and_rowpctsum_single_dimension(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(PCTSUM ROWPCTSUM)")
        grand_total = df["income"].sum()
        sex1_total = df.loc[df["sex"] == "1", "income"].sum()
        pctsum = cell(r, ("sex", "1"), "PCTSUM")
        assert pctsum == pytest.approx(100 * sex1_total / grand_total)
        # with only a row dimension (no columns to break out), the "row
        # subtotal" IS the cell itself, so ROWPCTSUM is always 100%
        rowpctsum = cell(r, ("sex", "1"), "ROWPCTSUM")
        assert rowpctsum == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 4. Custom denominator (PCTSUM<...> / PCTN<...>)
# ---------------------------------------------------------------------------

class TestCustomDenominator:

    def test_tax_as_pct_of_income_same_group(self, df):
        r = flextab(data=df, groupby=["sex", "region"], measure=["income", "tax"],
                    table="sex*region, tax*pctsum<income>")
        # sex=="2", region=="1" -> two rows (index 2, 9)
        sub = df[(df["sex"] == "2") & (df["region"] == "1")]
        expected = 100 * sub["tax"].sum() / sub["income"].sum()
        got = cell(r, ("sex", "2", "region", "1"), ("tax", "PCTSUM"))
        assert got == pytest.approx(expected)

    def test_pctn_with_all_fallback(self, df):
        r = flextab(data=df, groupby="sex", table="sex, pctn<region all>")
        # 'region' isn't grouped in this table at all (only sex is), so the
        # 'region' token in the denom list can't match anything and the
        # ALL/TOTAL fallback kicks in -> same as a plain grand-total PCTN
        n_sex1 = (df["sex"] == "1").sum()
        expected = 100 * n_sex1 / len(df)
        got = cell(r, ("sex", "1"), "PCTN")
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. ALL / TOTAL marginal
# ---------------------------------------------------------------------------

class TestAllTotal:

    def test_all_row_is_grand_total(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex ALL='Total', income=''*(N MEAN)")
        # the ALL/TOTAL row's outer label level is blank, not 'sex' -
        # only the detail rows repeat the groupby column's own label
        total_row = cell(r, ("", "Total"), "N")
        assert total_row == df["income"].count()
        assert cell(r, ("", "Total"), "MEAN") == pytest.approx(df["income"].mean())

    def test_all_in_both_dimensions(self, df):
        r = flextab(
            data=df, groupby=["sex", "region"], measure="income",
            table="sex ALL='Row total', income=''*N ALL='Col total'",
        )
        # bare ALL (no measure attached) counts every row in the data,
        # regardless of missing values in any particular measure
        grand_n = cell(r, ("", "Row total"), "Col total")
        assert grand_n == len(df)


# ---------------------------------------------------------------------------
# 6. Weighted statistics
# ---------------------------------------------------------------------------

class TestWeighted:

    def _weighted_ref(self, group, x="income", w="weight"):
        d = group.dropna(subset=[w])
        d = d[d[x].notna()]
        wsum = d[w].sum()
        wmean = (d[x] * d[w]).sum() / wsum if wsum else np.nan
        return d[w].sum(), (d[x] * d[w]).sum(), wmean

    def test_weighted_sum_and_mean(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(SUM MEAN)", weight="weight")
        for key in ["1", "2"]:
            group = df[df["sex"] == key]
            _, expected_sum, expected_mean = self._weighted_ref(group)
            assert cell(r, ("sex", key), "SUM") == pytest.approx(expected_sum)
            assert cell(r, ("sex", key), "MEAN") == pytest.approx(expected_mean)

    def test_n_is_never_weighted(self, df):
        # weight has one missing value (index 8, sex == "2"); that row is
        # excluded entirely, but N still counts plain rows, not weight-sums
        r = flextab(data=df, groupby="sex", table="sex, N", weight="weight")
        sex2_rows_with_weight = ((df["sex"] == "2") & df["weight"].notna()).sum()
        assert cell(r, ("sex", "2"), "N") == sex2_rows_with_weight


# ---------------------------------------------------------------------------
# 7. Labels & sort_by
# ---------------------------------------------------------------------------

class TestLabelsAndSort:

    def test_labels_relabel_index_values(self, df, labels):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*N", labels=labels)
        row_values = {idx[-1] for idx in r.index}
        assert row_values == {"Males", "Females", "nan"}

    def test_sort_by_label_is_alphabetical(self, df, labels):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*N", labels=labels, sort_by="label")
        row_values = [idx[-1] for idx in r.index]
        # "Females" < "Males" < "nan" alphabetically
        assert row_values == ["Females", "Males", "nan"]

    def test_sort_by_index_uses_dict_order(self, df, labels):
        r = flextab(data=df, groupby="age_group", measure="income",
                    table="age_group, income=''*N", labels=labels, sort_by="index")
        row_values = [idx[-1] for idx in r.index]
        # labels['age_group'] dict order is '1','2','3' -> 0-19, 20-66, 67+
        assert row_values == ["0-19", "20-66", "67+", "nan"]

    def test_default_sort_by_code(self, df):
        # sort_by='code' (default): sorts on the raw string values "1"<"2"
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*N")
        row_values = [idx[-1] for idx in r.index]
        assert row_values.index("1") < row_values.index("2")


# ---------------------------------------------------------------------------
# 8. Format specs & na_rep
# ---------------------------------------------------------------------------

class TestFormatting:

    def test_format_spec_controls_decimal_places(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*MEAN*format=8,0")
        text = flextab_to_string(r)
        # 0 decimal places -> no "." in the formatted mean values
        mean_lines = [
            line
            for line in text.splitlines()
            if "1" in line or "2" in line
        ]
        assert any("." not in line.split()[-1] for line in mean_lines)

    def test_na_rep_used_for_missing_cells(self, df):
        r = flextab(data=df, groupby=["sex", "region"],
                    table="sex, region*N")
        text = flextab_to_string(r, na_rep="MISSING")
        assert "MISSING" in text

    def test_default_fmt_applies_when_no_format_spec(self, df):
        r = flextab(data=df, measure="income", table="income=''*MEAN",
                    fmt="{:.3f}")
        text = flextab_to_string(r, fmt="{:.3f}")
        assert "." in text  # 3 decimal places rendered


# ---------------------------------------------------------------------------
# 9. Two-dimensional nested tables
# ---------------------------------------------------------------------------

class TestNestedTables:

    def test_nested_groupby_row_dimension_shape(self, df):
        r = flextab(
            data=df, groupby=["sex", "region"], measure="income",
            table="sex*region, income=''*N",
        )
        # every distinct (sex, region) combo present in the data should be a row
        combos = df.dropna(subset=[]).groupby(["sex", "region"], dropna=False).size()
        assert len(r) == len(combos)

    def test_two_measures_side_by_side(self, df):
        r = flextab(
            data=df, groupby="sex", measure=["income", "tax"],
            table="sex, income*MEAN tax*MEAN",
        )
        assert cell(r, ("sex", "1"), ("income", "MEAN")) is not None
        assert r.shape[1] == 2  # one MEAN column per measure
        assert list(r.columns.get_level_values(0)) == ["income", "tax"]


# ---------------------------------------------------------------------------
# 10. row_header
# ---------------------------------------------------------------------------

class TestRowHeader:

    def test_row_header_sets_index_name(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*N", row_header="Sex")
        if isinstance(r.index, pd.MultiIndex):
            assert r.index.names[0] == "Sex"
        else:
            assert r.index.name == "Sex"


# ---------------------------------------------------------------------------
# 11. Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_stat_requiring_measure_without_measure_raises(self, df):
        with pytest.raises(ValueError, match="requires a measure variable"):
            flextab(data=df, groupby="sex", table="sex, MEAN")

    def test_too_many_dimensions_raises(self, df):
        with pytest.raises(ValueError, match="at most 2 dimensions"):
            flextab(data=df, groupby="sex", table="sex, sex, sex")

    def test_unknown_token_raises(self, df):
        with pytest.raises(ValueError, match="not found in measure"):
            flextab(data=df, groupby="sex", table="sex, not_a_real_column")


# ---------------------------------------------------------------------------
# 12. Result type & export smoke tests
# ---------------------------------------------------------------------------

class TestResultObject:

    def test_returns_flextab_result_subclass(self, df):
        r = flextab(data=df, groupby="sex", table="sex, N")
        assert isinstance(r, FlextabResult)
        assert isinstance(r, pd.DataFrame)

    def test_values_stay_numeric_for_further_computation(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*SUM")
        assert r["SUM"].sum() == pytest.approx(df["income"].sum())

    def test_to_excel_writes_a_file(self, df, tmp_path):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(N MEAN)",
                    style={"header_bg": "#4472C4", "header_fg": "white"})
        out_path = tmp_path / "out.xlsx"
        r.to_excel(out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_repr_html_runs_without_error(self, df):
        r = flextab(data=df, groupby="sex", measure="income",
                    table="sex, income=''*(N MEAN)",
                    style={"cell_bg": ("white", "#EBF3FB")})
        html = r._repr_html_()
        assert "<table" in html
