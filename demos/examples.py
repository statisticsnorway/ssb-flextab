# %% [markdown]
# # Flextab Examples
# These examples are the ones used in the overview of table types

# %%
import numpy as np
import pandas as pd

from ssb_flextab import flextab

# %% [markdown]
# ## Examples for index.html

# %%

df = pd.DataFrame(
    {
        "education": [
            "Lower secondary",
            "Upper secondary",
            "Higher education",
            "Lower secondary",
            "Upper secondary",
            "Higher education",
        ],
        "employment": [
            "Employed",
            "Employed",
            "Employed",
            "Not employed",
            "Not employed",
            "Not employed",
        ],
        "count": [320, 760, 690, 280, 340, 110],
    }
)

df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True).drop(columns="count")


# %% [markdown]
# ### Part 1. Two variables crossed: what runs down, and what runs across

# %%
print("Education across")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    employment
    ,
    education
    """,
)

# %%
print("Employment across")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    education
    ,
    employment
    """,
)

# %% [markdown]
# ### Part 2. Totals are answers, not decoration

# %%
print("Total column")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    employment
    ,
    total education
    """,
)

# %%
print("Total row")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    education
    """,
)

# %%
print("Total row and total column")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    total education
    """,
)

# %% [markdown]
# ### Part 3. One distribution, two distributions, or one inside another

# %%
print("One distribution")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    count
    """,
)

# %%
print("One distribution")
flextab(
    data=df,
    groupby="employment",
    table="""
    total employment
    ,
    count
    """,
)

# %%
print("Stacking")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    total education
    ,
    count
    """,
)

# %%
print("Nesting")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    education * (total employment)
    ,
    count
    """,
)

# %% [markdown]
# ### Part 4. Three percentages, three different questions

# %%
print("Count")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    total education
    """,
)

# %%
print("Column percentages")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    (total education) * colpctn
    """,
)

# %%
print("Row percentages")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    (total education) * rowpctn
    """,
)

# %%
print("Total percentages")
flextab(
    data=df,
    groupby=["education", "employment"],
    table="""
    total employment
    ,
    (total education) * pctn
    """,
)

# %% [markdown]
# ## Examples from bygge.html

# %%
df = pd.DataFrame(
    {
        "region": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "sex": ["1", "1", "2", "2", "1", "1", "2", "2"],
        "education": ["1", "2", "1", "2", "1", "2", "1", "2"],
        "count": [40, 60, 30, 70, 50, 90, 45, 115],
        "income": [
            4500000,
            6500000,
            3500000,
            126000000,
            75000000,
            14900000,
            12300000,
            7000000,
        ],
        "tax": [
            1500000,
            3000000,
            1200000,
            45900000,
            30000000,
            6900000,
            4440000,
            2250000,
        ],
    }
)

rng = np.random.default_rng(12345)

# Keep original values
df = df.copy()
df["income_total"] = df["income"]
df["tax_total"] = df["tax"]

# One row for each person
df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True).drop(columns="count")


# Distribute income for person in each category
def distribute_value(x: pd.Series) -> np.ndarray:
    """Distribute a total value across observations using lognormal shares.

    Parameters
    ----------
    x : pd.Series
        Series containing the total valuee for a group. The first value
        is used as the total value to distribute.

    Returns
    -------
    np.ndarray
        Array containing the distributed value for each observation.
        The sum of the returned values equals the total value in the
        first value of ``x``.
    """
    shares = rng.lognormal(mean=0, sigma=0.8, size=len(x))
    shares /= shares.sum()
    return np.asarray(shares * x.iloc[0], dtype=float)


df["income"] = df.groupby(["region", "sex", "education"])["income_total"].transform(
    distribute_value
)


# tax rate: 20 % - 50 %
# Low income -> Approx. 20 %
# High income -> Approx. 50 %
income_min = df["income"].min()
income_max = df["income"].max()

df["taxrate"] = 0.20 + 0.30 * (df["income"] - income_min) / (income_max - income_min)

df["tax"] = df["income"] * df["taxrate"]

df = df.drop(columns=["income_total", "tax_total", "taxrate"])

# %%
labels_no = {
    "sex": {"1": "Menn", "2": "Kvinner"},
    "region": {"1": "Vest", "2": "Øst"},
    "education": {
        "2": "Høyere",
        "1": "Grunnskole",
    },
}


# %%
labels = {
    "sex": {"1": "Males", "2": "Females"},
    "region": {"1": "West", "2": "East"},
    "education": {
        "2": "Higher education",
        "1": "Elementary school",
    },
}


# %% [markdown]
# ### Part 1. Nesting, stacking — and the parentheses that bind them

# %%
print("All nested\n")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * sex * education
    ,
    count
    """,
    labels=labels,
)

# %%
print("All nested, with subtotals\n")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (total sex * (total education))
    ,
    count
    """,
    labels=labels,
)

# %%
print("All stacked\n")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region sex education
    ,
    count
    """,
    labels=labels,
)

# %%
print("All stacked, with totals\n")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    total region total sex total education
    ,
    count
    """,
    labels=labels,
)

# %%
print("Two stacked distributions within each nested")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (sex education)
    ,
    count
    """,
    labels=labels,
)

# %%
print("Two stacked distributions within each nested, with subtotals")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (total sex * (total education))
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution within every stacked")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    (region sex) * education
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution within every stacked, with totals")
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    total (region sex) * (total education)
    ,
    count
    """,
    labels=labels,
)

# %% [markdown]
# ### Part 2. Subtotals: a total for each group

# %%
print("One nested distribution")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region * sex
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution, with subtotals")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region * (total sex)
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution, with totals")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    total region * (total sex)
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution, with subtotals distributed")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    (total region) * sex
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution, with subtotals")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region * (total sex)
    ,
    count
    """,
    labels=labels,
)

# %%
print("One nested distribution, with totals distributed")
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    (total region) * (total sex)
    ,
    count
    """,
    labels=labels,
)

# %% [markdown]
# ### Part 3. Same table, turned: rows or columns?

# %%
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    sex
    ,
    region
    """,
    labels=labels,
)

# %%
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region
    ,
    sex
    """,
    labels=labels,
)

# %%
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    region
    ,
    education
    """,
    labels=labels,
)

# %%
print("Column totals")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    region
    ,
    total education
    """,
    labels=labels,
)

# %%
print("Row totals")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    education
    """,
    labels=labels,
)

# %%
print("Both row and column totals")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    total education
    """,
    labels=labels,
)

# %%
print("One distribution")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    count
    """,
    labels=labels,
)

# %%
print("Stacking")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region education
    ,
    count
    """,
    labels=labels,
)

# %%
print("Nesting")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    region * (total education)
    ,
    count
    """,
    labels=labels,
)

# %%
print("Percent of total")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * pctn
    """,
    labels=labels,
)

# %%
print("Percent of column")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * colpctn
    """,
    labels=labels,
)

# %%
print("Percent of row")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * rowpctn
    """,
    labels=labels,
)

# %%
print("Column percent of measure variable")
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    total region
    ,
    income * (colpctn colpctsum) * (total education)
    """,
    labels=labels,
)

# %%
print("Percent with custom denominator")
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    (total region) *
    (total education)
    ,
    pctn<education>
    """,
    labels=labels,
)

# %%
print("Mean, median and sum income by group column in the rows and in the columns")
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    education
    ,
    region * income * (mean median sum)
    """,
    labels=labels,
)

# %% [markdown]
# ## Examples from statistikk.html

# %%
df = pd.DataFrame(
    {
        "region": ["Oslo", "Vestland", "Trøndelag", "Nord-Norge"],
        "count": [100, 120, 80, 100],
        "income": [65, 60, 34, 41],
        "tax": [22, 18, 10, 12],
    }
)
df = df.copy()
df["income_total"] = df["income"]
df["tax_total"] = df["tax"]

# One row for each person
df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True).drop(columns="count")

df["income"] = df.groupby("region")["income_total"].transform(distribute_value)
df["tax"] = df.groupby("region")["tax_total"].transform(distribute_value)
df = df.drop(columns=["income_total", "tax_total"])

# %% [markdown]
# ### Part 1. A cell can be more than a count

# %%
print("Count of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * count
    """,
    labels=labels,
)

# %%
print("Sum of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * sum
    """,
    labels=labels,
)

# %%
print("Mean of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * mean
    """,
    labels=labels,
)

# %%
print("Count and sum of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * (count sum)
    """,
    labels=labels,
)

# %%
print("Count and mean of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * (count mean)
    """,
    labels=labels,
)

# %%
print("Sum and mean of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * (sum mean)
    """,
    labels=labels,
)

# %%
print("Count, sum and mean of income")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * (count sum mean)
    """,
    labels=labels,
)

# %% [markdown]
# ### Part 2. Several statistics, several measures — side by side

# %%
print("Sum income by group column in the rows")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * sum
    """,
    labels=labels,
)

# %%
print("Sum income and tax by group column in the rows")
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    total region
    ,
    (income tax) * sum
    """,
    labels=labels,
)

# %%
print("Sum and mean of income and tax by group column in the rows")
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    total region
    ,
    (income tax) * (sum mean)
    """,
    labels=labels,
)

# %%
print("Count, sum and mean of income and tax by group column in the rows")
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    total region
    ,
    (income tax) * (count sum mean)
    """,
    labels=labels,
)

# %% [markdown]
# ### Part 3. Percent of how many, or percent of how much?

# %%
print("Percent of count")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * pctn
    """,
    labels=labels,
)

# %%
print("Percent of sum")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * pctsum
    """,
    labels=labels,
)

# %%
print("Percent of sum")
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    total region
    ,
    income * (pctn pctsum)
    """,
    labels=labels,
)

# %%
print("Percentage of another measure column")
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    region
    ,
    (income tax) * sum tax * pctsum<income>
    """,
    labels=labels,
)
