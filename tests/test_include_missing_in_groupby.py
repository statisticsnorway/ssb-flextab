"""Tests for ``include_missing_in_groupby`` in ``flextab``.

With ``include_missing_in_groupby=False`` rows with a missing groupby value
must be removed from the data before the table is built. They must therefore
be absent from every part of the table, including totals, subtotals, and
percentage denominators, and not only from the grouped cells.
"""

import numpy as np
import pandas as pd
import pytest

from ssb_flextab import flextab


@pytest.fixture
def df() -> pd.DataFrame:
    """Return a frame with missing values in several groupby columns.

    Row 0 has a missing ``region`` and row 4 has a missing ``sex``. Row 9 has
    a missing ``age_group``. The measure columns also hold missing values to
    show that these are unrelated to the groupby selection.
    """
    return pd.DataFrame(
        {
            "sex": ["1", "1", "2", "1", None, "2", "2", "1", "2", "2"],
            "age_group": ["2", "1", "2", "3", "1", "2", "2", "2", "3", None],
            "region": [None, "2", "1", "2", "2", "3", "3", "2", "2", "1"],
            "education": ["3", "2", "3", "1", "3", "3", "3", "1", "3", "2"],
            "income": [300, 100, 450, 200, 650, 750, None, 850, 400, 350],
            "tax": [100, 10, 200, 90, 340, 370, 30, None, 150, 150],
            "weight": [1.5, 3.2, 1.7, 2.2, 6.1, 4.2, 1.9, 4.8, None, 8.2],
        }
    )


def _cell(result: pd.DataFrame, row_value: str, column: int = 0) -> float:
    """Return the value of the row whose last index level is ``row_value``."""
    for position, key in enumerate(result.index):
        last = key[-1] if isinstance(key, tuple) else key
        if last == row_value:
            return float(result.iloc[position, column])
    msg = f"No row ending in {row_value!r} in {list(result.index)}"
    raise AssertionError(msg)


def test_total_excludes_rows_with_missing_groupby(df: pd.DataFrame) -> None:
    """The total must not count the row whose groupby value is missing."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income", "tax"],
        table="""
        (total region='')
        ,
        size
        """,
        include_missing_in_groupby=False,
    )

    assert _cell(result, "TOTAL") == 9.0
    assert df["region"].isna().sum() == 1
    assert len(df) == 10


def test_total_equals_sum_of_groups(df: pd.DataFrame) -> None:
    """The total row is the sum of the group rows when nothing is missing."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), size",
        include_missing_in_groupby=False,
    )

    groups = sum(_cell(result, level) for level in ("1", "2", "3"))
    assert _cell(result, "TOTAL") == groups == 9.0


def test_no_missing_level_is_created(df: pd.DataFrame) -> None:
    """No row for the missing level appears when it is excluded."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), size",
        include_missing_in_groupby=False,
    )

    last_levels = [key[-1] if isinstance(key, tuple) else key for key in result.index]
    assert "nan" not in last_levels
    assert last_levels.count("TOTAL") == 1


def test_sum_total_excludes_measure_of_dropped_row(df: pd.DataFrame) -> None:
    """The income of the dropped row must not end up in the total."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), income*sum",
        include_missing_in_groupby=False,
    )

    # Row 0 has income 300 and a missing region, so the total is 4050 - 300.
    assert _cell(result, "TOTAL") == 3750.0
    assert _cell(result, "TOTAL") == sum(_cell(result, v) for v in ("1", "2", "3"))


def test_percentages_use_denominator_without_dropped_rows(df: pd.DataFrame) -> None:
    """Percentages of the total must add up to 100 over the kept groups."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), pctn",
        include_missing_in_groupby=False,
    )

    assert _cell(result, "TOTAL") == pytest.approx(100.0)
    parts = [_cell(result, v) for v in ("1", "2", "3")]
    assert sum(parts) == pytest.approx(100.0)
    assert parts == pytest.approx([200 / 9, 500 / 9, 200 / 9])


def test_any_missing_groupby_column_excludes_the_row(df: pd.DataFrame) -> None:
    """With several groupby columns, a missing value in any of them excludes the row.

    Row 0 (missing region) and row 4 (missing sex) are both removed, which
    leaves 8 rows in the grand total.
    """
    result = flextab(
        data=df,
        groupby=["sex", "region"],
        measure=["income"],
        table="sex*(region total) total, size",
        include_missing_in_groupby=False,
    )

    index = list(result.index)
    grand_total = float(result.iloc[index.index(("", "", "", "TOTAL")), 0])
    assert grand_total == 8.0


def test_subtotals_exclude_rows_with_missing_groupby(df: pd.DataFrame) -> None:
    """The subtotal within sex 1 must not include the row with missing region."""
    result = flextab(
        data=df,
        groupby=["sex", "region"],
        measure=["income"],
        table="sex*(region total) total, size",
        include_missing_in_groupby=False,
    )

    index = list(result.index)
    sex1_subtotal = float(result.iloc[index.index(("sex", "1", "", "TOTAL")), 0])
    sex1_regions = float(result.iloc[index.index(("sex", "1", "region", "2")), 0])
    # Sex 1 has four rows, one of which has a missing region.
    assert sex1_subtotal == sex1_regions == 3.0


def test_include_missing_true_keeps_all_rows(df: pd.DataFrame) -> None:
    """The default behaviour is unchanged: the missing level is kept as a group."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), size",
        include_missing_in_groupby=True,
    )

    assert _cell(result, "TOTAL") == 10.0
    assert _cell(result, "nan") == 1.0


def test_default_keeps_missing_rows(df: pd.DataFrame) -> None:
    """Omitting the argument must behave like ``include_missing_in_groupby=True``."""
    result = flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), size",
    )

    assert _cell(result, "TOTAL") == 10.0


def test_input_data_is_not_modified(df: pd.DataFrame) -> None:
    """The caller's frame must keep all its rows."""
    before = df.copy()

    flextab(
        data=df,
        groupby="region",
        measure=["income"],
        table="(region total), size",
        include_missing_in_groupby=False,
    )

    pd.testing.assert_frame_equal(df, before)


def test_no_missing_values_gives_same_result_for_both_settings() -> None:
    """Without missing groupby values the setting makes no difference."""
    data = pd.DataFrame(
        {"region": ["a", "a", "b", "c"], "income": [1.0, 2.0, 3.0, np.nan]}
    )
    kwargs = {
        "data": data,
        "groupby": "region",
        "measure": ["income"],
        "table": "(region total), (n sum)*income",
    }

    kept = flextab(**kwargs, include_missing_in_groupby=True)
    dropped = flextab(**kwargs, include_missing_in_groupby=False)

    pd.testing.assert_frame_equal(
        pd.DataFrame(kept.to_numpy(), index=kept.index, columns=kept.columns),
        pd.DataFrame(dropped.to_numpy(), index=dropped.index, columns=dropped.columns),
    )


def test_table_without_groupby_is_unaffected(df: pd.DataFrame) -> None:
    """A table with no groupby column has nothing to exclude."""
    result = flextab(
        data=df,
        measure=["income"],
        table="size",
        include_missing_in_groupby=False,
    )

    assert float(result.iloc[0, 0]) == 10.0
