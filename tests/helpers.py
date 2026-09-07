from typing import Any

import pandas as pd


def cell(
    result: pd.DataFrame,
    row: str | tuple[str, ...],
    col: str | tuple[str, ...],
) -> Any:
    return result.loc[row, col]
