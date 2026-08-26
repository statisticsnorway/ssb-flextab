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

tab.sum().sum()  # 10.0 - still numeric, works fine

print(flextab_to_string(tab, na_rep="MISSING"))  # custom marker, display only
```

## Styling

`style=` accepts a dictionary of colors that get applied when the table
renders in a notebook and when it's exported to Excel:

| Key | Colors |
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
tab["SUM"].sum()  # 4050.0
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

excel_filename = "../../reports/tab1.xlsx"
tab.to_excel(excel_filename)
```

### To Markdown

Because it's a plain DataFrame under the hood, the standard
`.to_markdown()` also works:

```python
tab = flextab(
    data=df,
    groupby=["education", "sex"],
    measure=["income", "tax"],
    table="""
    (total education='') * (total sex='')
    ,
    sum * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Education",
    include_missing_in_groupby=False,
    labels=labels,
    sort_by="index",
    style=tabstyle
)

tab_md = tab.to_markdown()

markdown_filename = "../../reports/tab1.md"
with open(markdown_filename, "w", encoding="utf-8") as f:
    f.write(tab_md)
```

Keep in mind `.to_markdown()` is plain pandas, not flextab-aware: it
prints the raw column-header tuples (e.g. `('SUM', 'Income')`) rather than
the nicely merged headers you see in a notebook, and it ignores any
`format=` decimal/thousands settings — you get pandas' default numeric
formatting instead. It's a convenient plain-text export, not a substitute
for the notebook or Excel rendering.

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
| Color the table | `style={...}` |
| Save the table | `.to_excel(path)` / `.to_markdown()` |
