"""
Pytest suite for tabulate.statistics (weighted/unweighted stat functions and
the row/column/percent aggregation engine).

Mirrors the style of test_parser.py: representative "typical" usages plus a
dedicated section of invalid / edge-case inputs.

Covered:
  * low-level scalar helpers: _hmean, _clean_weights, _drop_nan_x, _wmean,
    _wvar, _wstd, _wstderr, _wpercentile, _wgmean, _whmean
  * _parse_denom_def
  * _compute_series (plain groupby stats: MEAN/N/SUM/PCTN/ROWPCTN/COLPCTN, ...)
  * _compute_all_series (ALL/TOTAL margins on rows, columns, or both)
  * _compute_custom_pct (PCTN<...>/PCTSUM<...> user-defined denominators)
  * invalid usages (missing measure for stats that require one, unknown
    statistic keyword, degenerate/empty inputs)

Run with:  pytest tests/test_statistics.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ssb_flextab.statistics import (
    ALL_STATS,
    _BASE_STATS,
    _PERCENT_STATS,
    _WEIGHTED_STATS,
    _clean_weights,
    _compute_all_series,
    _compute_custom_pct,
    _compute_series,
    _drop_nan_x,
    _hmean,
    _parse_denom_def,
    _wgmean,
    _whmean,
    _wmean,
    _wpercentile,
    _wstd,
    _wstderr,
    _wvar,
)


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def df():
    """5-row toy dataset: 2 sexes crossed unevenly with 2 regions."""
    return pd.DataFrame(
        {
            "sex": ["M", "M", "F", "F", "F"],
            "region": ["E", "W", "E", "E", "W"],
            "income": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def approx(v):
    return pytest.approx(v, rel=1e-9)


# --------------------------------------------------------------------------
# _hmean — unweighted harmonic mean
# --------------------------------------------------------------------------

class TestHmean:
    def test_matches_hand_computed_value(self):
        x = pd.Series([1.0, 2.0, 4.0])
        assert _hmean(x) == approx(3 / (1 + 0.5 + 0.25))

    def test_excludes_nan_and_zero(self):
        x = pd.Series([1.0, 0.0, np.nan, 4.0])
        # only 1 and 4 survive: 2 / (1/1 + 1/4) = 2 / 1.25 = 1.6
        assert _hmean(x) == approx(1.6)

    def test_all_invalid_values_returns_nan(self):
        x = pd.Series([0.0, 0.0, np.nan])
        assert math.isnan(_hmean(x))

    def test_empty_series_returns_nan(self):
        assert math.isnan(_hmean(pd.Series([], dtype=float)))


# --------------------------------------------------------------------------
# _clean_weights / _drop_nan_x
# --------------------------------------------------------------------------

class TestWeightCleaning:
    def test_negative_weights_zeroed_positive_untouched(self):
        w = pd.Series([1.0, -2.0, 0.0, 3.0])
        assert _clean_weights(w).tolist() == [1.0, 0.0, 0.0, 3.0]

    def test_does_not_mutate_input(self):
        w = pd.Series([-1.0, 2.0])
        _clean_weights(w)
        assert w.tolist() == [-1.0, 2.0]

    def test_drop_nan_x_keeps_x_and_w_aligned(self):
        x = pd.Series([1.0, np.nan, 3.0])
        w = pd.Series([10.0, 20.0, 30.0])
        xx, ww = _drop_nan_x(x, w)
        assert xx.tolist() == [1.0, 3.0]
        assert ww.tolist() == [10.0, 30.0]

    def test_drop_nan_x_on_all_valid_is_a_no_op(self):
        x = pd.Series([1.0, 2.0])
        w = pd.Series([5.0, 6.0])
        xx, ww = _drop_nan_x(x, w)
        assert xx.tolist() == [1.0, 2.0]
        assert ww.tolist() == [5.0, 6.0]


# --------------------------------------------------------------------------
# Weighted statistic functions
# --------------------------------------------------------------------------

class TestWeightedMean:
    def test_equal_weights_matches_plain_mean(self):
        x = pd.Series([1.0, 2.0, 3.0])
        w = pd.Series([1.0, 1.0, 1.0])
        assert _wmean(x, w) == approx(x.mean())

    def test_unequal_weights_matches_hand_computation(self):
        x = pd.Series([1.0, 2.0, 3.0])
        w = pd.Series([1.0, 2.0, 3.0])
        assert _wmean(x, w) == approx((1 * 1 + 2 * 2 + 3 * 3) / (1 + 2 + 3))

    def test_zero_total_weight_returns_nan(self):
        x = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        assert math.isnan(_wmean(x, w))

    def test_nan_measure_row_excluded_from_both_numerator_and_denominator(self):
        x = pd.Series([1.0, np.nan, 3.0])
        w = pd.Series([10.0, 20.0, 30.0])
        # the NaN row's weight (20) must NOT leak into the denominator
        assert _wmean(x, w) == approx((1 * 10 + 3 * 30) / (10 + 30))


class TestWeightedVariance:
    def test_all_weights_equal_one_matches_sample_variance(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert _wvar(x, w) == approx(x.var())  # pandas default ddof=1

    def test_single_observation_is_degenerate_nan(self):
        x = pd.Series([5.0])
        w = pd.Series([2.0])
        assert math.isnan(_wvar(x, w))

    def test_zero_total_weight_returns_nan(self):
        x = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        assert math.isnan(_wvar(x, w))

    def test_std_is_sqrt_of_var(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert _wstd(x, w) == approx(math.sqrt(_wvar(x, w)))

    def test_std_propagates_nan_from_degenerate_variance(self):
        x = pd.Series([5.0])
        w = pd.Series([2.0])
        assert math.isnan(_wstd(x, w))

    def test_stderr_is_std_over_sqrt_wsum(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])
        expected = _wstd(x, w) / math.sqrt(w.sum())
        assert _wstderr(x, w) == approx(expected)

    def test_stderr_nan_when_wsum_zero(self):
        x = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        assert math.isnan(_wstderr(x, w))


class TestWeightedPercentile:
    def test_equal_weights_median(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        w = pd.Series([1.0] * 5)
        assert _wpercentile(x, w, 0.5) == 3.0

    def test_q_zero_returns_minimum(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        w = pd.Series([1.0] * 5)
        assert _wpercentile(x, w, 0.0) == 1.0

    def test_q_one_returns_maximum(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        w = pd.Series([1.0] * 5)
        assert _wpercentile(x, w, 1.0) == 5.0

    def test_heavier_weight_pulls_percentile_toward_that_value(self):
        # A single heavily-weighted low value should dominate the low end
        # of the weighted ECDF.
        x = pd.Series([1.0, 100.0])
        w = pd.Series([99.0, 1.0])
        assert _wpercentile(x, w, 0.5) == 1.0

    def test_all_nan_measure_returns_nan(self):
        x = pd.Series([np.nan, np.nan])
        w = pd.Series([1.0, 2.0])
        assert math.isnan(_wpercentile(x, w, 0.5))

    def test_zero_total_weight_returns_nan(self):
        x = pd.Series([1.0, 2.0])
        w = pd.Series([0.0, 0.0])
        assert math.isnan(_wpercentile(x, w, 0.5))


class TestWeightedGeometricAndHarmonicMean:
    def test_wgmean_equal_weights_matches_unweighted_gmean(self):
        x = pd.Series([1.0, 2.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0])
        expected = math.exp((math.log(1) + math.log(2) + math.log(4)) / 3)
        assert _wgmean(x, w) == approx(expected)

    def test_wgmean_excludes_non_positive_values(self):
        x = pd.Series([-1.0, 0.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0])
        assert _wgmean(x, w) == approx(4.0)

    def test_wgmean_all_non_positive_returns_nan(self):
        x = pd.Series([-1.0, 0.0])
        w = pd.Series([1.0, 1.0])
        assert math.isnan(_wgmean(x, w))

    def test_whmean_equal_weights_matches_unweighted_hmean(self):
        x = pd.Series([1.0, 2.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0])
        assert _whmean(x, w) == approx(_hmean(x))

    def test_whmean_excludes_zero_values(self):
        x = pd.Series([0.0, 2.0, 4.0])
        w = pd.Series([1.0, 1.0, 1.0])
        expected = (1.0 + 1.0) / (1.0 / 2.0 + 1.0 / 4.0)
        assert _whmean(x, w) == approx(expected)

    def test_whmean_all_zero_returns_nan(self):
        x = pd.Series([0.0, 0.0])
        w = pd.Series([1.0, 1.0])
        assert math.isnan(_whmean(x, w))


# --------------------------------------------------------------------------
# _parse_denom_def
# --------------------------------------------------------------------------

class TestParseDenomDef:
    def test_single_token_uppercased(self):
        assert _parse_denom_def("income") == ["INCOME"]

    def test_multiple_tokens_preserve_order(self):
        assert _parse_denom_def("gender all") == ["GENDER", "ALL"]

    def test_extra_whitespace_is_collapsed(self):
        assert _parse_denom_def("  origin   total ") == ["ORIGIN", "TOTAL"]

    def test_empty_string_yields_empty_list(self):
        assert _parse_denom_def("") == []


# --------------------------------------------------------------------------
# ALL_STATS / stat-table consistency
# --------------------------------------------------------------------------

class TestStatTables:
    def test_all_stats_is_union_of_base_and_percent(self):
        assert ALL_STATS == set(_BASE_STATS) | _PERCENT_STATS

    def test_every_base_stat_has_a_weighted_counterpart(self):
        # Every key in _BASE_STATS must have a matching entry in
        # _WEIGHTED_STATS so weight= works uniformly across all stats.
        missing = set(_BASE_STATS) - set(_WEIGHTED_STATS)
        assert missing == set()

    def test_count_stats_are_never_in_weighted_formula_dict_by_accident(self):
        # N/COUNT/SIZE/NMISS are present but their weighted lambdas ignore
        # the weight argument entirely — sanity check the ignore actually
        # holds for a couple of them.
        x = pd.Series([1.0, 2.0, np.nan])
        assert _WEIGHTED_STATS["N"](x, pd.Series([1.0, 2.0, 3.0])) == x.count()
        assert _WEIGHTED_STATS["N"](x, pd.Series([999.0, 999.0, 999.0])) == x.count()


# --------------------------------------------------------------------------
# _compute_series — plain (non-ALL) groupby aggregation
# --------------------------------------------------------------------------

class TestComputeSeriesBasic:
    def test_mean_by_single_group(self, df):
        s = _compute_series(df, ["sex"], "income", "MEAN", ["sex"], [], missing=False)
        assert s["M"] == approx(15.0)  # (10+20)/2
        assert s["F"] == approx(40.0)  # (30+40+50)/3

    def test_bare_n_needs_no_measure(self, df):
        s = _compute_series(df, ["sex"], None, "N", ["sex"], [], missing=False)
        assert s["M"] == 2
        assert s["F"] == 3

    def test_sum_by_cross_of_two_groups(self, df):
        s = _compute_series(
            df, ["sex", "region"], "income", "SUM", ["sex"], ["region"], missing=False
        )
        assert s[("M", "E")] == 10.0
        assert s[("M", "W")] == 20.0
        assert s[("F", "E")] == 70.0  # 30 + 40
        assert s[("F", "W")] == 50.0

    def test_grand_total_when_no_groups(self, df):
        s = _compute_series(df, [], "income", "SUM", [], [], missing=False)
        assert s["__total__"] == 150.0

    def test_missing_true_keeps_nan_group_missing_false_drops_it(self):
        d = pd.DataFrame({"sex": ["M", "M", "F", None], "income": [10.0, 20.0, 30.0, 40.0]})
        dropped = _compute_series(d, ["sex"], "income", "SUM", ["sex"], [], missing=False)
        kept = _compute_series(d, ["sex"], "income", "SUM", ["sex"], [], missing=True)
        assert set(dropped.index) == {"M", "F"}
        assert len(kept) == 3


class TestComputeSeriesPercent:
    def test_pctn_sums_to_100_across_full_cross(self, df):
        s = _compute_series(
            df, ["sex", "region"], None, "PCTN", ["sex"], ["region"], missing=False
        )
        assert s.sum() == approx(100.0)

    def test_pctn_cell_values(self, df):
        # 5 rows total; (M,E)=1 row -> 20%, (M,W)=1 -> 20%,
        # (F,E)=2 -> 40%, (F,W)=1 -> 20%
        s = _compute_series(
            df, ["sex", "region"], None, "PCTN", ["sex"], ["region"], missing=False
        )
        assert s[("M", "E")] == approx(20.0)
        assert s[("F", "E")] == approx(40.0)

    def test_rowpctn_sums_to_100_within_each_row(self, df):
        s = _compute_series(
            df, ["sex", "region"], None, "ROWPCTN", ["sex"], ["region"], missing=False
        )
        assert s["M"].sum() == approx(100.0)
        assert s["F"].sum() == approx(100.0)

    def test_colpctn_sums_to_100_within_each_column(self, df):
        s = _compute_series(
            df, ["sex", "region"], None, "COLPCTN", ["sex"], ["region"], missing=False
        )
        # index levels are unnamed here, so group by position (level 1 = region)
        by_region = s.groupby(level=1).sum()
        assert by_region["E"] == approx(100.0)
        assert by_region["W"] == approx(100.0)

    def test_rowpctn_without_row_groups_falls_back_to_grand_total(self, df):
        # r_groups=[] means "row pct" degenerates to "pct of grand total"
        s = _compute_series(
            df, ["region"], None, "ROWPCTN", [], ["region"], missing=False
        )
        assert s.sum() == approx(100.0)


class TestComputeSeriesWeighted:
    def test_weighted_mean_matches_hand_computation(self):
        d = pd.DataFrame(
            {
                "sex": ["M", "M", "F", "F"],
                "income": [10.0, 20.0, 30.0, 40.0],
                "wt": [1.0, 3.0, 1.0, 2.0],
            }
        )
        s = _compute_series(d, ["sex"], "income", "MEAN", ["sex"], [], missing=False, weight="wt")
        assert s["M"] == approx((10 * 1 + 20 * 3) / (1 + 3))
        assert s["F"] == approx((30 * 1 + 40 * 2) / (1 + 2))

    def test_negative_weight_zeroed_but_row_still_counted_in_n(self):
        d = pd.DataFrame(
            {
                "sex": ["F", "F"],
                "income": [30.0, 40.0],
                "wt": [-1.0, 2.0],
            }
        )
        mean = _compute_series(d, ["sex"], "income", "MEAN", ["sex"], [], missing=False, weight="wt")
        n = _compute_series(d, ["sex"], None, "N", ["sex"], [], missing=False, weight="wt")
        # negative weight -> 0, so only the 40-row contributes to the mean
        assert mean["F"] == approx(40.0)
        # but N is a plain count, unaffected by the weight value
        assert n["F"] == 2

    def test_missing_weight_row_excluded_even_from_n(self):
        d = pd.DataFrame(
            {
                "sex": ["M", "M"],
                "income": [10.0, 20.0],
                "wt": [1.0, np.nan],
            }
        )
        n = _compute_series(d, ["sex"], None, "N", ["sex"], [], missing=False, weight="wt")
        assert n["M"] == 1


# --------------------------------------------------------------------------
# _compute_all_series — ALL/TOTAL margins
# --------------------------------------------------------------------------

class TestComputeAllSeries:
    def test_all_on_rows_sums_by_remaining_column_group(self, df):
        s = _compute_all_series(
            df, ["region"], "income", "SUM", missing=False, r_groups=[], c_groups=["region"]
        )
        assert s["E"] == 80.0  # 10+30+40
        assert s["W"] == 70.0  # 20+50

    def test_all_on_rows_colpctn_is_always_100(self, df):
        # The ALL row IS the column total, so COLPCTN against it is 100%.
        s = _compute_all_series(
            df, ["region"], None, "COLPCTN", missing=False, r_groups=[], c_groups=["region"]
        )
        assert s["E"] == approx(100.0)
        assert s["W"] == approx(100.0)

    def test_all_on_cols_rowpctn_is_always_100(self, df):
        s = _compute_all_series(
            df, ["sex"], None, "ROWPCTN", missing=False, r_groups=["sex"], c_groups=[]
        )
        assert s["M"] == approx(100.0)
        assert s["F"] == approx(100.0)

    def test_all_on_both_dims_collapses_to_grand_total(self, df):
        s = _compute_all_series(
            df, [], "income", "SUM", missing=False, r_groups=[], c_groups=[]
        )
        assert s["__total__"] == 150.0

        n = _compute_all_series(df, [], None, "N", missing=False, r_groups=[], c_groups=[])
        assert n["__total__"] == 5


# --------------------------------------------------------------------------
# _compute_custom_pct — PCTN<...> / PCTSUM<...>
# --------------------------------------------------------------------------

class TestComputeCustomPct:
    def test_denom_token_matching_innermost_groupby_collapses_it(self, df):
        # pctsum<region> with rows broken down by sex*region: since "region"
        # is the innermost row variable, the denominator collapses region
        # and sums income by sex only.
        r_path_order = [("group", "sex", "sex"), ("group", "region", "region")]
        s = _compute_custom_pct(
            df,
            r_groups=["sex", "region"],
            c_groups=[],
            var="income",
            stat="PCTSUM",
            denom_def="region",
            groupby=["sex", "region"],
            measure=["income"],
            missing=False,
            r_path_order=r_path_order,
            c_path_order=[],
        )
        # sex totals: M=30 (10+20), F=120 (30+40+50)
        assert s[("M", "E")] == approx(10 / 30 * 100)
        assert s[("M", "W")] == approx(20 / 30 * 100)
        assert s[("F", "E")] == approx(70 / 120 * 100)
        assert s[("F", "W")] == approx(50 / 120 * 100)

    def test_denom_matching_a_measure_name_uses_full_grouping_ratio(self, df):
        # pctsum<income> with itself as the denom token (a measure name)
        # uses the SAME grouping for numerator and denominator, which
        # degenerates to 100% everywhere for a single-measure case.
        r_path_order = [("group", "sex", "sex")]
        s = _compute_custom_pct(
            df,
            r_groups=["sex"],
            c_groups=[],
            var="income",
            stat="PCTSUM",
            denom_def="income",
            groupby=["sex"],
            measure=["income"],
            missing=False,
            r_path_order=r_path_order,
            c_path_order=[],
        )
        assert s["M"] == approx(100.0)
        assert s["F"] == approx(100.0)

    def test_all_token_fallback_gives_grand_total_denominator(self, df):
        r_path_order = [("all", "TOTAL", None)]
        s = _compute_custom_pct(
            df,
            r_groups=[],
            c_groups=[],
            var="income",
            stat="PCTSUM",
            denom_def="all",
            groupby=["sex", "region"],
            measure=["income"],
            missing=False,
            r_path_order=r_path_order,
            c_path_order=[],
        )
        assert s["__total__"] == approx(100.0)


# --------------------------------------------------------------------------
# Invalid usages / edge cases
# --------------------------------------------------------------------------

class TestInvalidUsage:
    def test_mean_without_measure_raises(self, df):
        with pytest.raises(ValueError, match="requires a measure variable"):
            _compute_series(df, ["sex"], None, "MEAN", ["sex"], [], missing=False)

    def test_nmiss_without_measure_raises_with_specific_guidance(self, df):
        with pytest.raises(ValueError, match="NMISS"):
            _compute_series(df, ["sex"], None, "NMISS", ["sex"], [], missing=False)

    def test_pctsum_without_measure_raises(self, df):
        with pytest.raises(ValueError, match="requires a measure variable"):
            _compute_series(df, ["sex"], None, "PCTSUM", ["sex"], [], missing=False)

    def test_unknown_statistic_raises(self, df):
        with pytest.raises(ValueError, match="Unknown statistic"):
            _compute_series(df, ["sex"], "income", "NOTASTAT", ["sex"], [], missing=False)

    def test_all_series_mean_without_measure_raises(self, df):
        with pytest.raises(ValueError, match="requires a measure variable"):
            _compute_all_series(
                df, ["region"], None, "MEAN", missing=False, r_groups=[], c_groups=["region"]
            )

    def test_n_count_size_do_not_require_a_measure(self, df):
        # These three should NOT raise even though var is None.
        for stat in ("N", "COUNT", "SIZE"):
            s = _compute_series(df, ["sex"], None, stat, ["sex"], [], missing=False)
            assert s["M"] > 0

    def test_zero_denominator_in_row_pct_yields_nan_not_error(self):
        # A row group with a subtotal of 0 (all values happen to sum to 0)
        # should degrade gracefully to NaN rather than raising or infinity.
        d = pd.DataFrame(
            {"sex": ["M", "M"], "region": ["E", "W"], "income": [0.0, 0.0]}
        )
        s = _compute_series(
            d, ["sex", "region"], "income", "ROWPCTSUM", ["sex"], ["region"], missing=False
        )
        assert s.isna().all()

    def test_empty_dataframe_grand_total_is_nan_or_zero_not_an_exception(self):
        d = pd.DataFrame({"sex": pd.Series([], dtype=object), "income": pd.Series([], dtype=float)})
        s = _compute_series(d, [], "income", "SUM", [], [], missing=False)
        # sum of empty series is 0.0 in pandas
        assert s["__total__"] == 0.0
        n = _compute_series(d, [], None, "N", [], [], missing=False)
        assert n["__total__"] == 0
