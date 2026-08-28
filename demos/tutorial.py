# %% [markdown]
# # Flextab Tutorial

# %%
from pprint import pprint

import pandas as pd

from ssb_flextab import flextab

# %% [markdown]
# ## Data set for tutorial

# %%
df = pd.DataFrame(
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

labels = {
    "sex": {"1": "Males", "2": "Females"},
    "age_group": {"1": "0-19", "2": "20-66", "3": "67+"},
    "region": {"1": "West", "2": "East", "3": "Central"},
    "education": {
        "3": "Higher education",
        "2": "Secondary school",
        "1": "Elementary school",
    },
}
pprint(df)
pprint(labels)

# %% [markdown]
# With just defining a groupby column it will count the number of observations (N)

# %%
flextab(data=df, groupby="region")

# %% [markdown]
# With just a measure column it will count the number of observations (N) and mean of the measure column

# %%
flextab(data=df, measure="income")

# %% [markdown]
# When we add a measure column it will count the number of observations with values and the mean for the measure column and group it by the groupby column

# %%
flextab(data=df, measure="income", groupby="region")

# %% [markdown]
# When we add a groupby column they must be put within a list. It will make a stacked table first grouped by the first groupby column and then by second, and count the number of observations and average for the measure column.

# %%
flextab(data=df, measure="income", groupby=["region", "sex"])

# %% [markdown]
# So far the tables are made as default tables. Now, let us introduce the table argument. With it we define the table content and layout ourselves in a very flexible way. The table argument must be put within quotes. When column names are used in the table argument they shall **not** be within qoutes. Beware that when we use qoutes within the table argument it should use another quote (like double quotes for the argument and single quotes within the argument).
#
# We start with a simple table in one dimension where make a distribution of the regions. The default count for groupby columns is the number of observations (n)

# %%
flextab(
    data=df,
    groupby="region",
    table="""
    region
    """,
)

# %% [markdown]
# Next, we use a measure column instead. For measure columns we have several different statistics to choose from. Here is an example.

# %%
flextab(
    data=df,
    measure="income",
    table="""
    income * (n nmiss sum mean)
    """,
)

# %% [markdown]
# We can combine groupby and measure columns in our table.

# %%
flextab(
    data=df,
    groupby="region",
    measure="income",
    table="""
    income * region * (n nmiss sum mean)
    """,
)

# %% [markdown]
# We can define two dimensions in the table argument, rows and columns, and they are separated by a comma. Here is a table where we put *region* in the rows and *sex* in the columns. The default count is the number of observations (N).

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    table="""
    region
    ,
    sex
    """,
)

# %% [markdown]
# We can add a groupby column and nest it in to the region in the rows with an asterisk (*). For each value of region there will be a distribution of education

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    region * education
    ,
    sex
    """,
)

# %% [markdown]
# We can add another groupby column and nest it in to the sex in the rows with an asterisk (*) as well

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education", "age_group"],
    table="""
    region * education
    ,
    sex * age_group
    """,
)

# %% [markdown]
# Instead of nesting columns we can stack them, we just remove the asterisk

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education", "age_group"],
    table="""
    region education
    ,
    sex age_group
    """,
)

# %% [markdown]
# Now we introduce the `total`keyword for adding totals to our table.
# We can also combine nesting and stacking. In the next example there will be stacked *sex* and *education* for each region

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    total region * (sex education)
    ,
    n
    """,
)

# %% [markdown]
# Now we can change to nested column in the rows, with totals. When `total` is nested to a group column, there will be made sub-totals.

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    total region * (total education)
    ,
    total sex
    """,
)

# %% [markdown]
# By adding parenthesis around the `total` and *region* we cross the total with education as well. This means there will be distribution of education for the total as well as for each region.

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    (total region) * (total education)
    ,
    total sex
    """,
)

# %% [markdown]
# Now, let us introduce a measure column again and some calculations. The `total` in the columns is not connected to the income and calcualtions, hence only the number of observations will be counted for the total.

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    total sex * income * (n sum mean)
    """,
)

# %% [markdown]
# When we put parenthesis around `total` and *region*, and connect it to *income* and calculations, we get all the statistics for the total as well.

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (total sex) * income * (n sum mean)
    """,
)

# %% [markdown]
# Some of the calculations we can do on the measure columns is used below.

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p99 qrange)
    """,
)

# %% [markdown]
# We can swap the rows and columns

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p99 qrange)
    ,
    total region
    """,
)

# %% [markdown]
# We can calculate weighted figures by adding the `weight` argument. Beware that rows with missing values for the weight column will be excluded from all the calculations.
#
# **NB!** The std, stderr and var is different from the sas results

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    weight="weight",
    table="""
    income * (n nmiss size sum median mean gmean hmean min max std stderr var p1 p25 p75 p99 qrange)
    ,
    total region
    """,
)

# %% [markdown]
# We can calculate percentages of the number of observations or the sum of a measure column
# - pctn
# - pctsum
#
# There are 3 basic percentages:
# - Total
# - Row
# - Column
#
# These two can be combined to:
# - rowpctn
# - rowpctsum
# - colpctn
# - colpctsum

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (total sex) * (pctn income*pctsum)
    """,
)

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (pctn income*pctsum) * (total sex)
    """,
)

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (rowpctn income * rowpctsum) * (total sex)
    """,
)

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    table="""
    total region
    ,
    (colpctn income * colpctsum) * (total sex)
    """,
)

# %%
flextab(
    data=df,
    groupby=["region", "sex"],
    measure="income",
    weight="weight",
    table="""
    total region
    ,
    (colpctn income * colpctsum) * (total sex)
    """,
)

# %% [markdown]
# Besides the default percentages we can decide the denominator ourselves by putting it within angle brackets. Here we want the regions two add up to 100 percent for the total and each value of sex.

# %%
flextab(
    data=df,
    groupby=["region", "sex", "education"],
    table="""
    (total sex) * (total region)
    ,
    (total education)*pctn<region>
    """,
)

# %% [markdown]
# We can use a measure column as the denominator to another measure column, which will be the numerator.

# %%
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) pctsum<income> * tax
    """,
)

# %% [markdown]
# We can format the numbers in the table cells by adding formats within the table argument. First American with thousands separator.

# %%
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) * format=9.0_ pctsum<income> * tax * format=9.1
    """,
)

# %% [markdown]
# Then European with thousands separator.

# %%
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    (total region)
    ,
    sum * (income tax) * format=9,0s pctsum<income> * tax * format=9,1
    """,
)

# %% [markdown]
# we can change or omit label texts.

# %%
flextab(
    data=df,
    groupby="region",
    measure=["income", "tax"],
    table="""
    total region=''
    ,
    sum='' * (income='Income' tax='Tax') * format=9,0s pctsum=''<income> * tax='Tax %' * format=9,1
    """,
    row_header="Region",
)

# %% [markdown]
# We can exclude missing values in groupby columns.

# %%
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
    include_missing_in_groupby=False,
)

# %% [markdown]
# We use the `labels` argument to replace the codes with texts. These labels must exist in a nested dictionary. The default is to sort by the code.

# %%
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
    sort_by="code",
)

# %% [markdown]
# We use the `sort_by` argument to change the sorting order.

# %%
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
    sort_by="label",
)

# %%
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
)

# %% [markdown]
# We can change the dot in the cells with no values

# %%
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
    na_rep="-",
)

# %% [markdown]
# There are also some styling options to add some colours to the table. These must be defined in dictionary which on ore more of these keys:
# - **header_bg**   background colour for the column header rows
# - **header_fg**   foreground (text) colour for column header rows
# - **row_bg**    background colour for the row index cells
# - **row_fg**    foreground colour for the row index cells
# - **row_bg**      background colour for data cells
# - **row_fg**      foreground colour for data cells
# - **cell_bg**  alternating row background colours — either a single
#                 colour (applied to every other row) or a 2-tuple
#                 (colour0, colour1) that cycles through all rows:
#                 row 0 → colour0, row 1 → colour1, row 2 → colour0, …
# - **cell_fg**  same as cell_bg but for foreground (text) colour
#
# The colours may be specified in any of these ways:
# - **named colour** Most common colours can be specified by name, header_bg='pink'
# - **Hexadecimal colors** Colours represented by hexadecimal figures, cell_bg=('#ecfeed', '#ffffff')
# - **RGB tuples** Colours represented as Red, Green, Blue in a tuple, row_bg=(110, 230, 30)

# %%
tabstyle = {
    "header_bg": "#ecfeed",
    "row_header_bg": "#ecfeed",
    #    'row_bg': '#ecfeed',
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
    style=tabstyle,
)

# %% [markdown]
# We can export the table to different formats, like excel or markdown. Here is an example on export to excel.

# %%
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
    style=tabstyle,
)

excel_filename = "../reports/tab1.xlsx"

tab.to_excel(excel_filename)

# %% [markdown]
# We can export the table to a markdown file. However, the layout will not be as nice as for the output in the browser.

# %%
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
    style=tabstyle,
)

tab_md = tab.to_markdown()

markdown_filename = "../reports/tab1.md"
with open(markdown_filename, "w", encoding="utf-8") as f:
    f.write(tab_md)

# %% [markdown]
# When we use `_repr_html`, the table will be rendered as html and the layout will be kept.

# %%
markdown_filename = "../reports/tab1b.md"
with open(markdown_filename, "w", encoding="utf-8") as f:
    f.write(tab._repr_html_())
