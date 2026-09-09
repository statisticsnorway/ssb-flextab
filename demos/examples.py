# %% [markdown]
# # Flextab Examples
# These examples are the as used in the overview of table types

# %%
import numpy as np
import pandas as pd

from ssb_flextab import flextab

# %%
df = pd.DataFrame(
    {
        "region": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "sex": ["1", "1", "2", "2", "1", "1", "2", "2"],
        "education": ["1", "2", "1", "2", "1", "2", "1", "2"],
        "count": [40, 60, 30, 70, 50, 90, 45, 115],
        "income": [4500000, 6500000, 3500000, 126000000, 75000000, 14900000, 12300000, 7000000],
        "tax": [1500000, 3000000, 1200000, 45900000, 30000000, 6900000, 4440000, 2250000]
    }
)

rng = np.random.default_rng(12345)

# Keep original values
df = df.copy()
df["income_total"] = df["income"]
df["tax_total"] = df["tax"]

# One row for each person
df = (
    df.loc[df.index.repeat(df["count"])]
      .reset_index(drop=True)
      .drop(columns="count")
)


# Distribute income for person in each category
def distribute_income(x: pd.Series) -> np.ndarray:
    """Distribute a total income across observations using lognormal shares.

    Parameters
    ----------
    x : pd.Series
        Series containing the total income for a group. The first value
        is used as the total income to distribute.

    Returns
    -------
    np.ndarray
        Array containing the distributed income for each observation.
        The sum of the returned values equals the total income in the
        first value of ``x``.
    """
    shares = rng.lognormal(mean=0, sigma=0.8, size=len(x))
    shares /= shares.sum()
    return shares * x.iloc[0]



df["income"] = (
    df.groupby(
        ["region", "sex", "education"]
    )["income_total"]
    .transform(distribute_income)
)


# tax rate: 20 % - 50 %
# Low income -> Approx. 20 %
# High income -> Approx. 50 %
income_min = df["income"].min()
income_max = df["income"].max()

df["taxrate"] = (
    0.20
    + 0.30
    * (df["income"] - income_min)
    / (income_max - income_min)
)

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


# %%
print('All nested\n')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * sex * education
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('All nested, with subtotals\n')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (total sex * (total education))
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('All stacked\n')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region sex education
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('All stacked, with totals\n')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    total region sex education
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('All stacked, with totals with distribution of lower level\n')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    total='Hele landet' region total='Begge kjønn' sex total='Alle' education
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('Two stacked distributions within each nested')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (sex education)
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('Two stacked distributions within each nested, with subtotals')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    region * (total sex * (total education))
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution within every stacked')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    (region sex) * education 
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution within every stacked, with totals')
flextab(
    data=df,
    groupby=["education", "sex", "region"],
    table="""
    total='I alt' (region='' sex='') * (total='' education='')
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution')
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region='' * sex=''
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution, with subtotals')
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    region='' * (total sex='')
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution, with totals')
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    total region='' * (total='' sex='')
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution, with totals distributed')
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    (total region='') * (total='' sex='')
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('One nested distribution, with subtotals distributed')
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    (total region='') * sex=''
    ,
    n='count'
    """,
    labels=labels
)

# %%
flextab(
    data=df,
    groupby=["sex", "region"],
    table="""
    sex
    ,
    region
    """,
    labels=labels
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
    labels=labels
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
    labels=labels
)

# %%
print('Column totals')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    region
    ,
    total education
    """,
    labels=labels
)

# %%
print('Row totals')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    education
    """,
    labels=labels
)

# %%
print('Both row and column totals')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    total education
    """,
    labels=labels
)

# %%
print('One distribution')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('Stacking')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region education
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('Nesting')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    region * (total education)
    ,
    n='count'
    """,
    labels=labels
)

# %%
print('Percent of total')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * pctn
    """,
    labels=labels
)

# %%
print('Percent of column')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * colpctn
    """,
    labels=labels
)

# %%
print('Percent of row')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    total region
    ,
    (total education) * rowpctn
    """,
    labels=labels
)

# %%
print('Column percent of measure variable')
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    total region
    ,
    income * (colpctn colpctsum) * (total education) 
    """,
    labels=labels
)

# %%
print('Percent with custom denominator')
flextab(
    data=df,
    groupby=["education", "region"],
    table="""
    (total region) *
    (total education) 
    ,
    pctn<education>
    """,
    labels=labels
)

# %% [markdown]
# ## With measure columns

# %%
print('Mean, median and sum income')
flextab(
    data=df,
    measure="income",
    table="""
    income
    ,
    mean median sum
    """,
    labels=labels
)

# %%
print('Mean, median and sum income by group column in the rows')
flextab(
    data=df,
    groupby="education",
    measure="income",
    table="""
    education
    ,
    income * (mean median sum)
    """,
    labels=labels
)

# %%
print('Mean, median and sum income by group column in the rows and in the columns')
flextab(
    data=df,
    groupby=["education", "region"],
    measure="income",
    table="""
    education
    ,
    region * income * (mean median sum)
    """,
    labels=labels
)

# %%
print('Percentage of another measure column')
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    region
    ,
    (income tax) * sum tax * pctsum<income>
    """,
    labels=labels
)

# %%
