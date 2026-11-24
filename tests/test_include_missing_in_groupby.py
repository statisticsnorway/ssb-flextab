"""Tests for ``include_missing_in_groupby`` in ``flextab``.

With ``include_missing_in_groupby=False`` rows with a missing groupby value
must be removed from the data before the table is built. They must therefore
be absent from every part of the table, including totals, subtotals, and
percentage denominators, and not only from the grouped cells.
"""

from typing import Any

import pandas as pd
import pytest

from ssb_flextab import FlextabResult
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


def _run(
    data: pd.DataFrame,
    table: str,
    groupby: str | list[str] | None = "region",
    include_missing: bool = False,
) -> FlextabResult:
    """Build a table for the ``income`` measure with the given settings."""
    return flextab(
        data=data,
        groupby=groupby,
        measure=["income"],
        table=table,
        include_missing_in_groupby=include_missing,
    )


def _by_key(result: FlextabResult) -> dict[Any, float]:
    """Map every row key to the value in the first column."""
    matrix = result.to_numpy(dtype=float)
    return {key: float(matrix[pos, 0]) for pos, key in enumerate(result.index)}


def _by_last_level(result: FlextabResult) -> dict[Any, float]:
    """Map the last index level of every row to the value in the first column."""
    return {
        (key[-1] if isinstance(key, tuple) else key): value
        for key, value in _by_key(result).items()
    }


def test_total_excludes_rows_with_missing_groupby(df: pd.DataFrame) -> None:
    """Reproduce the reported case: the total is 9, not 10."""
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

    assert len(df) == 10
    assert _by_last_level(result)["TOTAL"] == 9.0


@pytest.mark.parametrize(
    ("table", "total", "groups"),
    [
        pytest.param("(region total), size", 9.0, [2.0, 5.0, 2.0], id="size"),
        # Row 0 has income 300 and a missing region, so the total is 4050 - 300.
        pytest.param(
            "(region total), income*sum", 3750.0, [800.0, 2200.0, 750.0], id="sum"
        ),
        pytest.param(
            "(region total), pctn",
            100.0,
            [200 / 9, 500 / 9, 200 / 9],
            id="pctn",
        ),
    ],
)
def test_total_and_groups_exclude_dropped_row(
    df: pd.DataFrame, table: str, total: float, groups: list[float]
) -> None:
    """Totals, group cells, and percentages are all based on the kept rows."""
    values = _by_last_level(_run(df, table))

    assert values["TOTAL"] == pytest.approx(total)
    assert [values[level] for level in ("1", "2", "3")] == pytest.approx(groups)
    assert values["TOTAL"] == pytest.approx(sum(groups))


def test_no_missing_level_is_created(df: pd.DataFrame) -> None:
    """No row for the missing level appears when it is excluded."""
    result = _run(df, "(region total), size")

    last_levels = [key[-1] if isinstance(key, tuple) else key for key in result.index]
    assert "nan" not in last_levels
    assert last_levels.count("TOTAL") == 1


def test_any_missing_groupby_column_excludes_the_row(df: pd.DataFrame) -> None:
    """A missing value in any groupby column removes the row everywhere.

    Row 0 (missing region) and row 4 (missing sex) are both removed. That
    leaves 8 rows in the grand total, and 3 rather than 4 rows in the
    subtotal for sex 1, which equals its only region cell.
    """
    values = _by_key(
        _run(
            df,
            "sex*(region total) total, size",
            groupby=["sex", "region"],
        )
    )

    assert values[("", "", "", "TOTAL")] == 8.0
    assert values[("sex", "1", "", "TOTAL")] == 3.0
    assert values[("sex", "1", "region", "2")] == 3.0


@pytest.mark.parametrize("include_missing", [True, None])
def test_missing_rows_are_kept_by_default(
    df: pd.DataFrame, include_missing: bool | None
) -> None:
    """The ``True`` setting and the default keep the missing level as a group."""
    table = "(region total), size"
    if include_missing is None:
        result = flextab(data=df, groupby="region", measure=["income"], table=table)
    else:
        result = _run(df, table, include_missing=include_missing)

    values = _by_last_level(result)
    assert values["TOTAL"] == 10.0
    assert values["nan"] == 1.0


def test_input_data_is_not_modified(df: pd.DataFrame) -> None:
    """The caller's frame must keep all its rows."""
    before = df.copy()

    _run(df, "(region total), size")

    pd.testing.assert_frame_equal(df, before)


def test_no_missing_values_gives_same_result_for_both_settings() -> None:
    """Without missing groupby values the setting makes no difference."""
    data = pd.DataFrame(
        {"region": ["a", "a", "b", "c"], "income": [1.0, 2.0, 3.0, float("nan")]}
    )
    table = "(region total), (n sum)*income"

    kept = _run(data, table, include_missing=True)
    dropped = _run(data, table, include_missing=False)

    assert list(kept.index) == list(dropped.index)
    assert list(kept.columns) == list(dropped.columns)
    pd.testing.assert_frame_equal(
        pd.DataFrame(kept.to_numpy(dtype=float)),
        pd.DataFrame(dropped.to_numpy(dtype=float)),
    )


def test_table_without_groupby_is_unaffected(df: pd.DataFrame) -> None:
    """A table with no groupby column has nothing to exclude."""
    result = _run(df, "size", groupby=None)

    assert result.to_numpy(dtype=float)[0, 0] == 10.0
