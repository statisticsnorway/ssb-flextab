# Flextab Tutorial

`flextab` is a Python function that builds cross-tabulations the way SAS's
`PROC TABULATE` does — statistics crossed with categorical breakdowns, laid
out with a compact table expression instead of pivot-table boilerplate.
This tutorial walks through it feature by feature, from the simplest
one-line call to styled, exportable tables.

## Setup

```python
import pandas as pd
import sys
from pathlib import Path

# Works both in VS Code and JupyterLab
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    # JupyterLab
    project_root = Path.cwd().parent

src_path = project_root / "src"

if src_path.exists():
    sys.path.append(str(src_path))

from ssb_flextab.flextab import flextab, flextab_to_string, FlextabResult
```

## The data set

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
df
```

Four categorical columns (`sex`, `age_group`, `region`, `education`), two
numeric columns to summarize (`income`, `tax`), and a `weight` column for
weighted statistics. Every column has at least one missing value, which
matters for several of the examples below.

We'll also use a `labels` dictionary throughout, mapping the numeric codes
to display text:

```python
labels = {
    "sex": {
        "1": "Males",
        "2": "Females",
    },
    "age_group": {
        "1": "0-19",
        "2": "20-66",
        "3": "67+",
    },
    "region": {
        "1": "West",
        "2": "East",
        "3": "Central",
    },
    "education": {
        "3": "Higher education",
        "2": "Secondary school",
        "1": "Elementary school",
    },
}
```

## Default tables

Before touching the `table=` argument at all, `flextab` can build a
sensible table from just `groupby=` and/or `measure=`.

### Just a `groupby` column

With only a grouping column given, it counts the number of observations
(`N`) per value:

```python
flextab(data=df, groupby="region")
```

```text
              N
region nan  1.0
       1    2.0
       2    5.0
       3    2.0
```

### Just a `measure` column

With only a measure column given, it counts observations and computes the
mean:

```python
flextab(data=df, measure="income")
```

```text
 income
      N   MEAN
    9.0  450.0
```

### `measure` + `groupby`

Combine the two and you get N and the mean of the measure, broken down by
the grouping column:

```python
flextab(data=df, measure="income", groupby="region")
```

```text
           income
                N   MEAN
region nan    1.0  300.0
       1      2.0  400.0
       2      5.0  440.0
       3      1.0  750.0
```

### Several `groupby` columns

Pass a list, and the groupby columns are **stacked** — first a breakdown
by the first column, then by the second, one under the other:

```python
flextab(data=df, measure="income", groupby=["region", "sex"])
```

```text
           income
                N   MEAN
region nan    1.0  300.0
       1      2.0  400.0
       2      5.0  440.0
       3      1.0  750.0
sex    nan    1.0  650.0
       1      4.0  362.5
       2      4.0  487.5
```

This also works with several measure columns at once — each gets its own
N/MEAN block automatically:

```python
flextab(data=df, measure=["income", "tax"], groupby="sex")
```

```text
        income         tax
             N   MEAN    N   MEAN
sex nan    1.0  650.0  1.0  340.0
    1      4.0  362.5  3.0   66.7
    2      4.0  487.5  5.0  180.0
```

## The `table` argument

Default tables only get you so far. The `table=` argument gives full
control over layout and statistics, using a small expression language:

- Put the expression in a **string** (triple-quoted strings are handy for
  multi-line layouts).
- Column names in the expression are written **without** quotes.
- Row and column dimensions are separated by a **comma**.
- `*` **nests** two things together; a space **stacks** them side by side
  (or one under the other, on rows).
- `( ... )` groups several tokens so they act as one unit for `*` and
  `format=`.
- If you need quotes *inside* the table string (for a label, see below),
  use a different quote style than the one wrapping the whole string —
  e.g. double quotes around the argument, single quotes inside it.

### A single dimension

With no comma, the whole expression goes into the table's single
dimension — which is placed in the **columns**, not the rows:

```python
flextab(
    data=df,
    groupby="region",
    table="""
    region
    """
)
```

```text
 region
    nan    1    2    3
    1.0  2.0  5.0  2.0
```

### Statistics on a measure column

A measure column is crossed with one or more statistic keywords using `*`
and parentheses:

```python
flextab(
    data=df,
    measure="income",
    table="""
    income * (n nmiss sum mean)
    """
)
```

```text
 income
      N NMISS     SUM   MEAN
    9.0   1.0  4050.0  450.0
```

### Combining `groupby` and `measure`

```python
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    income * region * (n nmiss sum mean)
    """
)
```

```text
 income
 region
    nan                        1                        2                         3
      N NMISS    SUM   MEAN    N NMISS    SUM   MEAN    N NMISS     SUM   MEAN    N NMISS    SUM   MEAN
    1.0   0.0  300.0  300.0  2.0   0.0  800.0  400.0  5.0   0.0  2200.0  440.0  1.0   1.0  750.0  750.0
```

This crosses `income`'s statistics with every value of `region`, giving a
full N / NMISS / SUM / MEAN block per region, all still in a single
(column) dimension.

### Two dimensions: rows and columns

Adding a comma splits the expression into a **row** dimension and a
**column** dimension:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    table="""
    region
    ,
    sex
    """
)
```

```text
            sex
            nan    1    2
region nan    .  1.0    .
       1      .    .  2.0
       2    1.0  3.0  1.0
       3      .    .  2.0
```

`region` is now the rows, `sex` the columns, and each cell is the count of
observations (N) for that combination. Note that combinations with *no*
observations show up as blank (`.`), not `0` — flextab only fills in
cells that actually occur in the data.

### Nesting with `*`

Nest a second groupby column into the rows with `*`. For every value of
`region` you now get a breakdown by `education`:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    region * education
    ,
    sex
    """
)
```

```text
                        sex
                        nan    1    2
region nan education 3    .  1.0    .
       1   education 2    .    .  1.0
                     3    .    .  1.0
       2   education 1    .  2.0    .
                     2    .  1.0    .
                     3  1.0    .  1.0
       3   education 3    .    .  2.0
```

Nest a second dimension into the columns the same way:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education", "age_group"],
    table="""
    region * education
    ,
    sex * age_group
    """
)
```

```text
                             sex
                             nan         1                   2
                       age_group age_group           age_group
                               1         1    2    3       nan    2    3
region nan education 3         .         .  1.0    .         .    .    .
       1   education 2         .         .    .    .       1.0    .    .
                     3         .         .    .    .         .  1.0    .
       2   education 1         .         .  1.0  1.0         .    .    .
                     2         .       1.0    .    .         .    .    .
                     3       1.0         .    .    .         .    .  1.0
       3   education 3         .         .    .    .         .  2.0    .
```

### Stacking instead of nesting

Drop the `*` and the two groupby columns are stacked side by side (or one
block after another) instead of nested inside each other:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education", "age_group"],
    table="""
    region education
    ,
    sex age_group
    """
)
```

```text
               sex           age_group
               nan    1    2       nan    1    2    3
region    nan    .  1.0    .         .    .  1.0    .
          1      .    .  2.0       1.0    .  1.0    .
          2    1.0  3.0  1.0         .  2.0  1.0  2.0
          3      .    .  2.0         .    .  2.0    .
education 1      .  2.0    .         .    .  1.0  1.0
          2      .  1.0  1.0       1.0  1.0    .    .
          3    1.0  1.0  4.0         .  1.0  4.0  1.0
```

Now `region` and `education` each get their own independent breakdown in
the rows, and `sex`/`age_group` each get their own breakdown in the
columns — four separate blocks in total, rather than one deeply nested
one.

## Totals and subtotals

The `TOTAL` keyword (an alias for `ALL` — they're interchangeable) adds a
row or column of totals.

### Totals stacked alongside a breakdown

```python
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    total region * (sex education)
    ,
    n
    """
)
```

```text
                               N
                     TOTAL  10.0
region nan sex       1       1.0
           education 3       1.0
       1   sex       2       2.0
           education 2       1.0
                     3       1.0
       2   sex       nan     1.0
                     1       3.0
                     2       1.0
           education 1       2.0
                     2       1.0
                     3       2.0
       3   sex       2       2.0
           education 3       2.0
```

### Nested totals give subtotals

When `total` is *nested* into a groupby column (with `*`), you get a
sub-total per value of that column instead of just a grand total:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    total region * (total education)
    ,
    total sex
    """
)
```

```text
                                  sex
                           TOTAL  nan    1    2
                     TOTAL  10.0  1.0  4.0  5.0
region nan           TOTAL   1.0    .  1.0    .
           education 3       1.0    .  1.0    .
       1             TOTAL   2.0    .    .  2.0
           education 2       1.0    .    .  1.0
                     3       1.0    .    .  1.0
       2             TOTAL   5.0  1.0  3.0  1.0
           education 1       2.0    .  2.0    .
                     2       1.0    .  1.0    .
                     3       2.0  1.0    .  1.0
       3             TOTAL   2.0    .    .  2.0
           education 3       2.0    .    .  2.0
```

Each region row now gets its own `TOTAL` sub-row, in addition to the
overall `TOTAL` at the top.

### Crossing totals with parentheses

Wrapping `total` and a groupby column together in parentheses — `(total
region)` — crosses the total with whatever comes next in the chain, so you
also see the breakdown (here, by education) computed *for* the total, not
just for each individual region:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    (total region) * (total education)
    ,
    total sex
    """
)
```

```text
                                    sex
                             TOTAL  nan    1    2
       TOTAL           TOTAL  10.0  1.0  4.0  5.0
             education 1       2.0    .  2.0    .
                       2       2.0    .  1.0  1.0
                       3       6.0  1.0  1.0  4.0
region nan             TOTAL   1.0    .  1.0    .
             education 3       1.0    .  1.0    .
       1               TOTAL   2.0    .    .  2.0
             education 2       1.0    .    .  1.0
                       3       1.0    .    .  1.0
       2               TOTAL   5.0  1.0  3.0  1.0
             education 1       2.0    .  2.0    .
                       2       1.0    .  1.0    .
                       3       2.0  1.0    .  1.0
       3               TOTAL   2.0    .    .  2.0
             education 3       2.0    .    .  2.0
```

### Totals with a measure column

By default a `total` in one dimension isn't linked to a measure in the
other dimension — it just counts observations:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    total sex * income * (n sum mean)
    """
)
```

```text
                      sex
                      nan                    1                     2
                   income               income                income
             TOTAL      N    SUM   MEAN      N     SUM   MEAN      N     SUM   MEAN
       TOTAL  10.0    1.0  650.0  650.0    4.0  1450.0  362.5    4.0  1950.0  487.5
region nan     1.0      .      .      .    1.0   300.0  300.0      .       .      .
       1       2.0      .      .      .      .       .      .    2.0   800.0  400.0
       2       5.0    1.0  650.0  650.0    3.0  1150.0  383.3    1.0   400.0  400.0
       3       2.0      .      .      .      .       .      .    1.0   750.0  750.0
```

The `TOTAL` column on the left is a plain count (no income statistics).
Wrap `total sex` in parentheses and connect it into the `* income * (...)`
chain, and the total row now gets the full set of income statistics too:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (total sex) * income * (n sum mean)
    """
)
```

```text
                                      sex
              TOTAL                   nan                    1                     2
             income                income               income                income
                  N     SUM   MEAN      N    SUM   MEAN      N     SUM   MEAN      N     SUM   MEAN
       TOTAL    9.0  4050.0  450.0    1.0  650.0  650.0    4.0  1450.0  362.5    4.0  1950.0  487.5
region nan      1.0   300.0  300.0      .      .      .    1.0   300.0  300.0      .       .      .
       1        2.0   800.0  400.0      .      .      .      .       .      .    2.0   800.0  400.0
       2        5.0  2200.0  440.0    1.0  650.0  650.0    3.0  1150.0  383.3    1.0   400.0  400.0
       3        1.0   750.0  750.0      .      .      .      .       .      .    1.0   750.0  750.0
```

## Statistics reference

Here are most of the statistics available on a measure column in one go:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p99 qrange)
    """
)
```

```text
             income
                  N NMISS  SIZE     SUM MEDIAN   MEAN  GMEAN  HMEAN    MIN    MAX    STD STDERR      VAR     P1    P99 QRANGE
       TOTAL    9.0   1.0  10.0  4050.0  400.0  450.0  377.8  300.4  100.0  850.0  252.5   84.2  63750.0  108.0  842.0  350.0
region nan      1.0   0.0   1.0   300.0  300.0  300.0  300.0  300.0  300.0  300.0      .      .        .  300.0  300.0    0.0
       1        2.0   0.0   2.0   800.0  400.0  400.0  396.9  393.8  350.0  450.0   70.7   50.0   5000.0  351.0  449.0   50.0
       2        5.0   0.0   5.0  2200.0  400.0  440.0  338.1  247.3  100.0  850.0  311.0  139.1  96750.0  104.0  842.0  450.0
       3        1.0   1.0   2.0   750.0  750.0  750.0  750.0  750.0  750.0  750.0      .      .        .  750.0  750.0    0.0
```

| Statistic | Meaning |
|---|---|
| `n` | Count of **non-missing** values of the measure |
| `count` | Same as `n` |
| `size` | Count of **all** rows in the group, missing or not |
| `nmiss` | Count of missing values of the measure |
| `sum` | Sum |
| `mean` | Arithmetic mean |
| `median` | 50th percentile |
| `gmean` | Geometric mean (positive values only) |
| `hmean` | Harmonic mean (non-zero values only) |
| `min` / `max` | Minimum / maximum |
| `std` | Sample standard deviation |
| `stderr` | Standard error of the mean |
| `var` | Sample variance |
| `p1` `p5` `p10` `p25` `p75` `p90` `p95` `p99` | Percentiles |
| `qrange` | Interquartile range (P75 − P25) |

`n`/`count` and `size` diverge as soon as the measure column itself has
missing values within a group — `size` still counts those rows, `n` and
`count` don't.

## Swapping rows and columns

There's nothing special about which dimension goes where — swap the two
sides of the comma and the same table appears transposed:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p99 qrange)
    ,
    total region
    """
)
```

```text
                       region
                 TOTAL    nan       1        2      3
income N           9.0    1.0     2.0      5.0    1.0
       NMISS       1.0    0.0     0.0      0.0    1.0
       SIZE       10.0    1.0     2.0      5.0    2.0
       SUM      4050.0  300.0   800.0   2200.0  750.0
       MEDIAN    400.0  300.0   400.0    400.0  750.0
       MEAN      450.0  300.0   400.0    440.0  750.0
       GMEAN     377.8  300.0   396.9    338.1  750.0
       HMEAN     300.4  300.0   393.8    247.3  750.0
       MIN       100.0  300.0   350.0    100.0  750.0
       MAX       850.0  300.0   450.0    850.0  750.0
       STD       252.5      .    70.7    311.0      .
       STDERR     84.2      .    50.0    139.1      .
       VAR     63750.0      .  5000.0  96750.0      .
       P1        108.0  300.0   351.0    104.0  750.0
       P99       842.0  300.0   449.0    842.0  750.0
       QRANGE    350.0    0.0    50.0    450.0    0.0
```

## Weighted statistics

Add `weight=` to compute weighted figures. Rows with a **missing** weight
are excluded from *all* calculations — including plain counts — so a
weighted table can show smaller totals than the unweighted version.

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    weight="weight",
    table="""
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p25 p75 p99 qrange)
    ,
    total region
    """
)
```

```text
                       region
                 TOTAL    nan       1         2       3
income N           8.0    1.0     2.0       4.0     1.0
       NMISS       1.0    0.0     0.0       0.0     1.0
       SIZE        9.0    1.0     2.0       4.0     2.0
       SUM     16040.0  450.0  3635.0    8805.0  3150.0
       MEDIAN    450.0  300.0   350.0     650.0   750.0
       MEAN      502.8  300.0   367.2     540.2   750.0
       GMEAN     425.0  300.0   365.4     415.5   750.0
       HMEAN     332.9  300.0   363.9     280.9   750.0
       MIN       100.0  300.0   350.0     100.0   750.0
       MAX       850.0  300.0   450.0     850.0   750.0
       STD       269.9      .    70.7     347.3       .
       STDERR     47.8      .    22.5      86.0       .
       VAR     72847.8      .  5000.0  120642.7       .
       P1        100.0  300.0   350.0     100.0   750.0
       P25       350.0  300.0   350.0     200.0   750.0
       P75       750.0  300.0   350.0     850.0   750.0
       P99       850.0  300.0   450.0     850.0   750.0
       QRANGE    400.0    0.0     0.0     650.0     0.0
```

A couple of things worth knowing about weighting:

- `n`, `count`, `size`, and `nmiss` are **never** weighted — they always
  report plain row counts, weighted or not.
- `min` and `max` are also unaffected by weighting — a weight doesn't
  change which observation is smallest or largest.
- A weight of `0` still counts the row in `n`/`count`/`size`, but
  contributes nothing to weighted sums, means, etc.
- A **negative** weight is treated as `0` — again, still counted, but
  contributing nothing to weighted statistics.
- A **missing** weight excludes the row entirely, from every statistic,
  including the plain counts.
- `std`, `stderr`, and `var` use a reliability-weights variance estimator,
  which is **not** the same formula SAS uses — expect small numeric
  differences from SAS output for those three statistics specifically.

## Percentages

Two families of percentage statistics are available:

- `pctn` — a count as a percentage
- `pctsum` — a measure's sum as a percentage

Each comes in three flavors:

| Variant | Denominator |
|---|---|
| `pctn` / `pctsum` | Grand total |
| `rowpctn` / `rowpctsum` | Row subtotal |
| `colpctn` / `colpctsum` | Column subtotal |

### Grand-total percentages

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (total sex) * (pctn income*pctsum)
    """
)
```

```text
                                    sex          sex          sex
                     TOTAL          nan            1            2
              TOTAL income   nan income     1 income     2 income
               PCTN PCTSUM  PCTN PCTSUM  PCTN PCTSUM  PCTN PCTSUM
       TOTAL  100.0  100.0  10.0   16.0  40.0   35.8  50.0   48.1
region nan     10.0    7.4     .      .  10.0    7.4     .      .
       1       20.0   19.8     .      .     .      .  20.0   19.8
       2       50.0   54.3  10.0   16.0  30.0   28.4  10.0    9.9
       3       20.0   18.5     .      .     .      .  20.0   18.5
```

### Row and column percentages

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (rowpctn income * rowpctsum) * (total sex)
    """
)
```

```text
                                            income
             ROWPCTN                     ROWPCTSUM
                       sex                           sex
               TOTAL   nan      1      2     TOTAL   nan      1      2
       TOTAL   100.0  10.0   40.0   50.0     100.0  16.0   35.8   48.1
region nan     100.0     .  100.0      .     100.0     .  100.0      .
       1       100.0     .      .  100.0     100.0     .      .  100.0
       2       100.0  20.0   60.0   20.0     100.0  29.5   52.3   18.2
       3       100.0     .      .  100.0     100.0     .      .  100.0
```

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (colpctn income * colpctsum) * (total sex)
    """
)
```

```text
                                             income
             COLPCTN                      COLPCTSUM
                        sex                            sex
               TOTAL    nan      1      2     TOTAL    nan      1      2
       TOTAL   100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0
region nan      10.0      .   25.0      .       7.4      .   20.7      .
       1        20.0      .      .   40.0      19.8      .      .   41.0
       2        50.0  100.0   75.0   20.0      54.3  100.0   79.3   20.5
       3        20.0      .      .   40.0      18.5      .      .   38.5
```

Percentages respect `weight=` too, just like any other statistic:

```python
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    weight="weight",
    table="""
    total region
    ,
    (colpctn income * colpctsum) * (total sex)
    """
)
```

```text
                                             income
             COLPCTN                      COLPCTSUM
                        sex                            sex
               TOTAL    nan      1      2     TOTAL    nan      1      2
       TOTAL   100.0  100.0  100.0  100.0     100.0  100.0  100.0  100.0
region nan      11.1      .   25.0      .       2.8      .    8.5      .
       1        22.2      .      .   50.0      22.7      .      .   53.6
       2        44.4  100.0   75.0      .      54.9  100.0   91.5      .
       3        22.2      .      .   50.0      19.6      .      .   46.4
```

### Custom denominators

Instead of the built-in total/row/column choices, you can pick the
denominator yourself with `<...>` right after `pctn` or `pctsum`.

**A groupby column as denominator.** Here every `region` row is scaled
so the regions add up to 100% — both for the grand total and for each
value of `sex`:

```python
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    (total sex) * (total region)
    ,
    (total education)*pctn<region>
    """
)
```

```text
                              education
                        TOTAL         1      2      3
                         PCTN      PCTN   PCTN   PCTN
    TOTAL        TOTAL  100.0     100.0  100.0  100.0
          region nan     10.0         .      .   16.7
                 1       20.0         .   50.0   16.7
                 2       50.0     100.0   50.0   33.3
                 3       20.0         .      .   33.3
sex nan          TOTAL  100.0         .      .  100.0
          region 2      100.0         .      .  100.0
    1            TOTAL  100.0     100.0  100.0  100.0
          region nan     25.0         .      .  100.0
                 2       75.0     100.0  100.0      .
    2            TOTAL  100.0         .  100.0  100.0
          region 1       40.0         .  100.0   25.0
                 2       20.0         .      .   25.0
                 3       40.0         .      .   50.0
```

**A measure column as denominator.** Here `tax` is expressed as a
percentage of `income` in the same breakdown — a ratio between two
measures rather than a share of a total:

```python
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) pctsum<income> * tax
    """
)
```

```text
                 SUM         PCTSUM
              income     tax    tax
       TOTAL  4050.0  1440.0   35.6
region nan     300.0   100.0   33.3
       1       800.0   350.0   43.8
       2      2200.0   590.0   26.8
       3       750.0   400.0   53.3
```

## Formatting numbers

Attach `*format=W.D` (or a comma variant for European style) right after
any token or group to control how its numbers are displayed. `W` is
accepted for SAS compatibility but ignored — only `D`, the number of
decimals, and the separator style actually matter.

**American style, with a thousands separator (`_`):**

```python
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) * format=9.0_ pctsum<income> * tax * format=9.1
    """
)
```

```text
                SUM        PCTSUM
             income    tax    tax
       TOTAL  4,050  1,440   35.6
region nan      300    100   33.3
       1        800    350   43.8
       2      2,200    590   26.8
       3        750    400   53.3
```

**European style — decimal comma, and `s` for a space thousands
separator:**

```python
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) * format=9,0s pctsum<income> * tax * format=9,1
    """
)
```

```text
                SUM        PCTSUM
             income    tax    tax
       TOTAL  4 050  1 440   35,6
region nan      300    100   33,3
       1        800    350   43,8
       2      2 200    590   26,8
       3        750    400   53,3
```

### Formatting several statistics at once

Attach `*format=...` to a parenthesized **group** instead of a single
token, and it applies to every statistic inside:

```python
flextab(
    data=df,
    measure="income",
    table="income*(mean gmean)*format=7,1"
)
```

```text
 income
   MEAN  GMEAN
  450,0  377,8
```

A statistic given its own explicit `format=` inside the group still wins
over the group-level one — so `(mean*format=9,3 gmean)*format=7,1` gives
`mean` 3 decimals and `gmean` the outer 1 decimal.

### A default format for the whole table

`format=` in the table expression only covers the cells it's attached to.
For everything else, `flextab()`'s own `fmt=` argument sets the fallback
Python format string (default `"{:.1f}"`):

```python
flextab(data=df, groupby="region", measure="income", fmt="{:.2f}")
```

```text
           income
                N    MEAN
region nan   1.00  300.00
       1     2.00  400.00
       2     5.00  440.00
       3     1.00  750.00
```

## Changing or hiding labels

Any token can be relabeled inline with `name='Label'`, or hidden entirely
with `name=''`. Since the whole `table=` string is already wrapped in
double quotes here, the inline labels use single quotes instead:

```python
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    total region=''
    ,
    sum='' * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Region"
)
```

```text
       Income    Tax Tax %
Region
TOTAL   4 050  1 440  35,6
nan       300    100  33,3
1         800    350  43,8
2       2 200    590  26,8
3         750    400  53,3
```

`row_header=` names the row index itself — here it turns the blank corner
of the table into "Region".

## Excluding missing values

By default, a missing value in a `groupby` column gets its own `nan` row.
Set `include_missing_in_groupby=False` to drop those rows from the table
entirely instead:

```python
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region='')
    ,
    sum='' * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Region",
    include_missing_in_groupby=False
)
```

```text
       Income    Tax Tax %
Region
TOTAL   4 050  1 440  35,6
1         800    350  43,8
2       2 200    590  26,8
3         750    400  53,3
```

Note this only controls *rows built from missing groupby values* — it has
no effect on missing values in the measure column itself (those are
handled per-statistic, e.g. by `nmiss`, or simply skipped by `sum`/`mean`).

## Replacing codes with text: the `labels` argument

`labels` maps the raw codes in your groupby columns to display text. It's
a nested dictionary — outer key is the column name, inner dictionary maps
each code to its label. The underlying grouping still happens on the raw
codes; the mapping is purely cosmetic. By default rows are still sorted
by the raw code (`sort_by='code'`):

```python
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    (total education='')
    ,
    (total region='') * income='' * rowpctsum='' * format=9,1
    """,
    row_header="Education",
    labels=labels,
    sort_by="code"
)
```

```text
                   TOTAL   nan  West   East Central
Education
TOTAL              100,0   7,4  19,8   54,3    18,5
Elementary school  100,0     .     .  100,0       .
Secondary school   100,0     .  77,8   22,2       .
Higher education   100,0  11,8  17,6   41,2    29,4
```

### `sort_by`

Change how rows/columns with a label are ordered with `sort_by`:

- `"code"` (default) — sort by the underlying raw code
- `"label"` — sort alphabetically by the display text
- `"index"` — sort by the order the codes appear in the `labels`
  dictionary itself, so you control the order explicitly

```python
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    (total education='')
    ,
    (total region='') * income='' * rowpctsum='' * format=9,1
    """,
    row_header="Education",
    labels=labels,
    sort_by="label"
)
```

```text
                   TOTAL Central   East  West   nan
Education
TOTAL              100,0    18,5   54,3  19,8   7,4
Elementary school  100,0       .  100,0     .     .
Higher education   100,0    29,4   41,2  17,6  11,8
Secondary school   100,0       .   22,2  77,8     .
```

```python
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    (total education='')
    ,
    (total region='') * income='' * rowpctsum='' * format=9,1
    """,
    row_header="Education",
    labels=labels,
    sort_by="index"
)
```

```text
                   TOTAL  West   East Central   nan
Education
TOTAL              100,0  19,8   54,3    18,5   7,4
Higher education   100,0  17,6   41,2    29,4  11,8
Secondary school   100,0  77,8   22,2       .     .
Elementary school  100,0     .  100,0       .     .
```

With `sort_by="index"`, `region` follows the order it was written in
`labels["region"]` (West, East, Central), and `education` follows
`labels["education"]`'s order (Higher education, Secondary school,
Elementary school) — regardless of what the raw codes were.

`sort_by="label"` and `sort_by="index"` only change the order of columns
that actually have an entry in `labels`. A groupby column with **no**
entry in `labels` — or values not covered by its entry — always falls
back to sorting by the raw code, no matter what `sort_by` is set to.

## Changing the missing-cell marker

Empty cells (no observations for that combination) print as `.` by
default. Override that with `na_rep`:

```python
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    (total education='')
    ,
    (total region='') * income='' * rowpctsum='' * format=9,1
    """,
    row_header="Education",
    labels=labels,
    sort_by="index",
    na_rep="-"
)
```

```text
                   TOTAL  West   East Central   nan
Education
TOTAL              100,0  19,8   54,3    18,5   7,4
Higher education   100,0  17,6   41,2    29,4  11,8
Secondary school   100,0  77,8   22,2       -     -
Elementary school  100,0     -  100,0       -     -
```

**This is worth knowing before you use it:** `na_rep=` on `flextab()`
itself doesn't just change how missing cells are *displayed* — it
converts the whole result to `object` dtype and writes the replacement
text directly into the cells. That's fine for a table you're only going
to look at or export, but it means the result is no longer numeric, and
things like `result.sum()` will raise an error afterwards.

If you want a custom missing-cell marker for *display only*, while
keeping the returned object numeric and usable in further calculations,
call `flextab()` without `na_rep` and pass it to `flextab_to_string()`
instead when you render it:

```python
tab = flextab(data=df, groupby=["region", "sex"], table="region , sex")

tab.sum().sum()
```

```text
10.0
```

```python
print(flextab_to_string(tab, na_rep="MISSING"))
```

```text
                sex
                nan        1        2
region nan  MISSING    1.000  MISSING
       1    MISSING  MISSING    2.000
       2      1.000    3.000    1.000
       3    MISSING  MISSING    2.000
```

## Styling

`style=` accepts a dictionary of colours that get applied when the table
renders in a notebook and when it's exported to Excel:

| Key | Colours |
|---|---|
| `header_bg` / `header_fg` | Background / text colour for the column header cells |
| `row_bg` / `row_fg` | Background / text colour for the row index cells |
| `row_header_bg` / `row_header_fg` | Background / text colour for the `row_header=` corner cell |
| `cell_bg` / `cell_fg` | Background / text colour for the data cells |

Every one of these accepts either a single colour, applied to every row,
or a `(colour0, colour1)` 2-tuple that **cycles** through rows: row 0 gets
`colour0`, row 1 gets `colour1`, row 2 gets `colour0` again, and so on —
handy for alternating "zebra stripe" shading.

Colours can be given as:

- a **named colour** — `header_bg='pink'`
- a **hex string** — `cell_bg=('#ecfeed', '#ffffff')`
- an **RGB tuple** — `row_bg=(110, 230, 30)`

```python
tabstyle = {
    "header_bg": "#ecfeed",
    "row_header_bg": "#ecfeed",
    "row_bg": (110, 230, 30),
    "cell_bg": ("#ecfeed", "#ffffff"),
}

flextab(
    data=df,
    groupby="education",
    measure=["income", "tax"],
    table="""
    (total education='')
    ,
    sum * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Education",
    include_missing_in_groupby=False,
    labels=labels,
    sort_by="index",
    style=tabstyle
)
```

```text
                     SUM
                  Income    Tax Tax %
Education
TOTAL              4 050  1 440  35,6
Higher education   2 550  1 190  46,7
Secondary school     450    160  35,6
Elementary school  1 050     90   8,6
```

(The values are identical to the unstyled version above — `style=` only
changes colours in the notebook display and Excel export, which plain
text obviously can't show here.)

Each colour key only affects the cells it names — an unset key never
inherits colour from a neighbouring one, so it's safe to set just the
keys you care about and leave the rest unstyled.

## Working with the result

A `flextab()` result is a `FlextabResult` — a `pd.DataFrame` subclass. Two
things follow from that:

**It stays numeric.** Even though `print()`/`repr()` and the notebook
display apply `format=` styling and show `.` for missing cells, the
underlying values are still real floats. All the usual DataFrame
operations — arithmetic, `.sum()`, slicing, feeding it into another
calculation — work exactly as they would on a plain DataFrame:

```python
tab = flextab(data=df, groupby="sex", measure="income", table="sex, income=''*SUM")
tab["SUM"].sum()
```

```text
4050.0
```

(The one exception is if you pass `na_rep=` to `flextab()` itself — see
[Changing the missing-cell marker](#changing-the-missing-cell-marker)
above — which does convert the data to text.)

**`flextab_to_string()` gives you an independent, one-off rendering.**
Rather than the formatting baked in when the table was built (`fmt=`, and
whatever `.attrs` the object carries), you can render the *same* result
with different settings any time, without rebuilding the table:

```python
flextab_to_string(tab, fmt="{:.0f}", na_rep="n/a")
```

```text
          SUM
sex nan   650
    1    1450
    2    1950
```

This is the function to reach for when you want a plain string to `print`
or write to a log/text file, with formatting independent from how the
table looks in a notebook.

## Exporting

### To Excel

`FlextabResult` is a `pd.DataFrame` subclass, so it carries a `to_excel()`
method — with the added bonus that both the `format=` number formats and
any `style=` colours are preserved in the output file:

```python
tab = flextab(
    data=df,
    groupby="education",
    measure=["income", "tax"],
    table="""
    (total education='')
    ,
    sum * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Education",
    include_missing_in_groupby=False,
    labels=labels,
    sort_by="index",
    style=tabstyle
)

excel_filename = "../reports/tab1.xlsx"
tab.to_excel(excel_filename)
```

### To Markdown

Because it's a plain DataFrame under the hood, the standard
`.to_markdown()` also works — but it's not flextab-aware:

```python
tab.to_markdown()
```

```text
| Education         |   ('SUM', 'Income') |   ('SUM', 'Tax') |   ('', 'Tax %') |
|:------------------|---------------------:|------------------:|-----------------:|
| TOTAL             |                 4050 |              1440 |         35.5556  |
| Higher education  |                 2550 |              1190 |         46.6667  |
| Secondary school  |                  450 |               160 |         35.5556  |
| Elementary school |                 1050 |                90 |          8.57143 |
```

Two problems: it prints raw column-header tuples like `('SUM', 'Income')`
instead of anything readable, and it ignores `format=` entirely, falling
back to pandas' own default number formatting.

Neither problem is really about `.to_markdown()` being careless, though
— it comes down to a real limitation of Markdown tables themselves:
**standard Markdown has no way to merge cells (no `colspan`/`rowspan`)
and no way to stack more than one header row.** The nested, visually
"merged" header layout you see in a notebook simply can't be expressed
in a plain Markdown table — there's nowhere to put it.

There are two practical ways around that, depending on what you need:

**Option 1 — a genuine plain-Markdown table, with flattened headers.**
`flextab_to_markdown()` fixes both problems above without trying to fake
cell-merging: it applies `format=` the same way `flextab_to_string()`
does, and flattens each column's levels into one readable label joined
by `sep` (default `" / "`), dropping blank levels along the way:

```python
from ssb_flextab.flextab import flextab_to_markdown

flextab_to_markdown(tab)
```

```text
| Education | SUM / Income | SUM / Tax | Tax % |
|---|---|---|---|
| TOTAL | 4 050 | 1 440 | 35,6 |
| Higher education | 2 550 | 1 190 | 46,7 |
| Secondary school | 450 | 160 | 35,6 |
| Elementary school | 1 050 | 90 | 8,6 |
```

That's a real Markdown table — it'll render correctly anywhere, and the
numbers use the same formatting you'd see in the notebook. It just can't
visually merge "SUM" over two columns the way the notebook display does;
that header level is folded into each column's label instead.

**Option 2 — embed the actual HTML for an exact merged-header look.**
If you need the table to look *exactly* like the notebook render —
merged headers and all — write out the HTML instead of a Markdown table.
Most Markdown processors (GitHub, GitLab, MkDocs, Jupyter Book, Pandoc)
pass raw HTML straight through untouched:

```python
markdown_filename = "../reports/tab1.md"
with open(markdown_filename, "w", encoding="utf-8") as f:
    f.write(tab._repr_html_())
```

This keeps the real `colspan`-merged header structure. The one caveat:
some renderers — GitHub included — strip inline `style` attributes from
embedded HTML for security, so a `style=` colour scheme may not survive
even though the table structure and merged headers will.

## Quick reference

| I want to... | Use |
|---|---|
| Count observations per group | `groupby=` alone, or `table="col"` |
| Summarize a numeric column | `measure=` alone, or `col*(stat stat ...)` |
| Put a breakdown in rows vs columns | `row_expr , col_expr` |
| Nest one breakdown inside another | `a * b` |
| Put two breakdowns side by side | `a b` |
| Add a grand total | `total` / `all` (interchangeable) |
| Add sub-totals within a group | `col * (total ...)` |
| Include the total in cross-calculations | wrap it: `(total col)` |
| Rename or hide a label | `col='New label'` / `col=''` |
| Format one or more cells | `*format=W.D` (US) or `*format=W,D` (EU) |
| Format a whole group of statistics | `(stat1 stat2)*format=W.D` |
| Set a fallback format for the whole table | `fmt="{:.2f}"` |
| Percent of grand total / row / column | `pctn`, `rowpctn`, `colpctn` (or `*sum` variants) |
| Percent of a custom group or measure | `pctn<col>` / `pctsum<measure>` |
| Weighted statistics | `weight="col"` |
| Map codes to display text | `labels={...}` |
| Control sort order | `sort_by='code' \| 'label' \| 'index'` |
| Drop missing-value rows | `include_missing_in_groupby=False` |
| Change the blank-cell marker (display only, stays numeric) | `flextab_to_string(result, na_rep='-')` |
| Change the blank-cell marker (baked into the data) | `flextab(..., na_rep='-')` |
| Render as a plain string with custom formatting | `flextab_to_string(result, fmt=..., na_rep=...)` |
| Do further math with the result | just use it — it's a real DataFrame |
| Colour the table | `style={...}` |
| Save the table | `.to_excel(path)` / `.to_markdown()` |
