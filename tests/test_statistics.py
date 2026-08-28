import numpy as np
import pytest

from ssb_flextab import flextab

from tests.helpers import cell

class TestDescriptiveStats:

    def test_sum_and_mean_by_group(self, df):
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*(SUM MEAN)"
        )
        expected_sum = df.groupby("sex", dropna=False)["income"].sum()
        expected_mean = df.groupby("sex", dropna=False)["income"].mean()
        for key in ["1", "2"]:
            assert cell(r, ("sex", key), "SUM") == pytest.approx(expected_sum[key])
            assert cell(r, ("sex", key), "MEAN") == pytest.approx(expected_mean[key])

    def test_std_and_var_use_sample_ddof(self, df):
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*(STD VAR)"
        )
        expected_std = df.groupby("sex", dropna=False)["income"].std()  # ddof=1 default
        expected_var = df.groupby("sex", dropna=False)["income"].var()
        for key in ["1", "2"]:
            assert cell(r, ("sex", key), "STD") == pytest.approx(expected_std[key])
            assert cell(r, ("sex", key), "VAR") == pytest.approx(expected_var[key])

    def test_median_and_percentiles(self, df):
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(MEDIAN P25 P75)",
        )
        sub = df.loc[df["sex"] == "2", "income"]
        assert cell(r, ("sex", "2"), "MEDIAN") == pytest.approx(sub.median())
        assert cell(r, ("sex", "2"), "P25") == pytest.approx(sub.quantile(0.25))
        assert cell(r, ("sex", "2"), "P75") == pytest.approx(sub.quantile(0.75))

    def test_min_max(self, df):
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*(MIN MAX)"
        )
        sub = df.loc[df["sex"] == "1", "income"]
        assert cell(r, ("sex", "1"), "MIN") == sub.min()
        assert cell(r, ("sex", "1"), "MAX") == sub.max()

    def test_nmiss_counts_missing_measure_values(self, df):
        r = flextab(
            data=df, groupby="sex", measure="income", table="sex, income=''*NMISS"
        )
        # income is missing exactly once, for a sex == "2" row (index 6)
        assert cell(r, ("sex", "2"), "NMISS") == 1
        assert cell(r, ("sex", "1"), "NMISS") == 0

    def test_gmean_uses_positive_values_only(self, df):
        r = flextab(data=df, measure="income", table="income=''*GMEAN")
        positive = df["income"].dropna()
        positive = positive[positive > 0]
        expected = np.exp(np.log(positive).mean())
        assert cell(r, "", "GMEAN") == pytest.approx(expected)

class TestWeighted:

    def _weighted_ref(self, group, x="income", w="weight"):
        d = group.dropna(subset=[w])
        d = d[d[x].notna()]
        wsum = d[w].sum()
        wmean = (d[x] * d[w]).sum() / wsum if wsum else np.nan
        return d[w].sum(), (d[x] * d[w]).sum(), wmean

    def test_weighted_sum_and_mean(self, df):
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*(SUM MEAN)",
            weight="weight",
        )
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

