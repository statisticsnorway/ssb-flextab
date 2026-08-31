# TABLE parser, statistics & formatting test suite

`tabulate/parser.py`, `tabulate/statistics.py`, and `tabulate/formatting.py`
are your uploaded files, unchanged. `tabulate/result.py` has one fix applied
(see below) relative to what you uploaded.

Run everything with:

    cd tabulate_parser_tests
    pip install pytest pandas numpy openpyxl
    pytest tests/ -v

Or just one file:

    pytest tests/test_parser.py -v
    pytest tests/test_statistics.py -v
    pytest tests/test_formatting.py -v

## Bug fixed in result.py

`FlextabResult._repr_html_` had an indentation slip: `_style_one_th` and
the `for line in lines:` loop sat OUTSIDE the `if any_style:` block that
defines `lines`. Whenever `style=` was empty/unset, `any_style` was
`False`, `lines` was never assigned, the loop raised `NameError`, and the
surrounding bare `except Exception` silently fell back to plain pandas
HTML — meaning `default_fmt` was completely ignored (raw floats leaked
through) any time no style key was set.

Fix: re-indented `_style_one_th` through `html = "\n".join(out)` one level
deeper so they're nested inside `if any_style:`. The `return html`
statement right after picks up whichever `html` value is live in each
branch (the freshly formatted `formatted.to_html(border=0)` when
`any_style` is False, the re-styled version when it's True), so both
paths now render correctly with no fallback.

`tests/test_formatting.py::TestReprHtml` now has a regression test
(`test_default_fmt_is_applied_even_when_no_style_key_is_set`) confirming
default_fmt is honored with no style set, plus a companion test
confirming the plain no-style/no-fmt case never hits the exception
fallback either.
