# SSB flextab

[![PyPI](https://img.shields.io/pypi/v/ssb-flextab.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-flextab.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-flextab)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-flextab)][license]

[![Documentation](https://github.com/statisticsnorway/ssb-flextab/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/ssb-flextab/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-flextab&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-flextab&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-flextab/
[documentation]: https://statisticsnorway.github.io/ssb-flextab
[tests]: https://github.com/statisticsnorway/ssb-flextab/actions?workflow=Tests
[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-flextab
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-flextab
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

## Features

Flextab is a Python package for making *flex*ible *tab*ulation. It is made with Claude AI, models Sonnet 4.5 and Sonnet 5.

Flextab is inspired by the Sas procedure [proc tabulate]( https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/proc/n00yutbvvckjwrn1ldg5xkvjy1pu.htm) which again was based on the [tpl language](https://en.wikipedia.org/wiki/TPL_Tables)

It makes flexible tables in two dimensions, rows an columns, based on Pandas dataframes.

## Requirements

- pandas
- numpy
- openpyxl when exporting tables to Excel

## Installation

You can install _SSB flextab_ via [pip] from [PyPI]:

```console
pip install ssb-flextab
```

## Usage


Import pandas and flextab.

```python
import pandas as pd
from ssb_flextab import flextab
```

Demo dataframe.

```python
df = pd.DataFrame({
    "sex":        ["1", "1", "2", "1", None, "2", "2", "1", "2", "2"],
    "age_group":  ["2", "1", "2", "3", "1", "2", "2", "2", "3", None],
    "region":     [None, "2", "1", "2", "2", "3", "3", "2", "2", "1"],
    "education":  ["3", "2", "3", "1", "3", "3", "3", "1", "3", "2"],
    "income":     [300, 100, 450, 200, 650, 750, None, 850, 400, 350],
    "tax":        [100, 10, 200, 90, 340, 370, 30, None, 150, 150],
    "weight":     [1.5, 3.2, 1.7, 2.2, 6.1, 4.2, 1.9, 4.8, None, 8.2],
})
```

Two dimensions with totals at the/left.

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    table="""
    total region
    ,
    total sex
    """
)
```

Nested columns in the row and column dimensions with sub-totals.

```python
flextab(
    data=df,
    groupby=["region", "sex", "age_group", "education"],
    table="""
    total region * (total age_group)
    ,
    total sex * (total education)
    """
)
```

Stacked columns in the row dimension and introduction of measure columns with different statsistics.

```python
flextab(
    data=df,
    groupby=["region", "sex", "age_group"],
    measure="income",
    table="""
    total region age_group
    ,
    income * (count sum mean) * (total sex)
    """
)
```

Different percentages:
- Percentage of grand total
- Row percentages
- Column percentages

The cells in the table are formatted with one decimal and decimal comma.

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    table="""
    total region
    ,
    (pctn rowpctn colpctn) *format=7,1 * (total sex)
    """
)
```

A comprehensive [tutorial](https://github.com/statisticsnorway/ssb-flextab/blob/main/docs/flextab_tutorial.md) is found in the `docs` folder at the Statistics Norway github pages.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB flextab_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-flextab/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-flextab/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-flextab/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-flextab/reference.html
