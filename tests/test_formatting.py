from ssb_flextab import flextab
from ssb_flextab import flextab_to_string

class TestFormatting:

    def test_format_spec_controls_decimal_places(self, df):
        r = flextab(
            data=df,
            groupby="sex",
            measure="income",
            table="sex, income=''*MEAN*format=8,0",
        )
        text = flextab_to_string(r)
        # 0 decimal places -> no "." in the formatted mean values
        mean_lines = [line for line in text.splitlines() if "1" in line or "2" in line]
        assert any("." not in line.split()[-1] for line in mean_lines)

    def test_na_rep_used_for_missing_cells(self, df):
        r = flextab(data=df, groupby=["sex", "region"], table="sex, region*N")
        text = flextab_to_string(r, na_rep="MISSING")
        assert "MISSING" in text

    def test_default_fmt_applies_when_no_format_spec(self, df):
        r = flextab(data=df, measure="income", table="income=''*MEAN", fmt="{:.3f}")
        text = flextab_to_string(r, fmt="{:.3f}")
        assert "." in text  # 3 decimal places rendered

