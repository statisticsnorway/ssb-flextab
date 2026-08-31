# TABLE parser, statistics & formatting test suite

`tabulate/parser.py`, `tabulate/statistics.py`, `tabulate/result.py`, and
`tabulate/formatting.py` are your uploaded files, unchanged.

Run everything with:

    cd tabulate_parser_tests
    pip install pytest pandas numpy openpyxl
    pytest tests/ -v

Or just one file:

    pytest tests/test_parser.py -v
    pytest tests/test_statistics.py -v
    pytest tests/test_formatting.py -v

## Heads-up: a real bug pinned by a test

`tests/test_formatting.py::TestReprHtml::test_bug_default_fmt_is_ignored_when_no_style_key_is_set`
documents a bug in `FlextabResult._repr_html_` (result.py): `_style_one_th`
and the `for line in lines:` loop are indented one level too shallow, so
they sit OUTSIDE the `if any_style:` block that defines `lines`. When
`style=` is empty/unset, `any_style` is False, `lines` is never assigned,
the loop raises `NameError`, and the surrounding bare `except Exception`
silently falls back to plain pandas HTML — meaning `default_fmt` is
completely ignored (raw floats show up) any time no `style` key is set.
The moment you fix the indentation, that one test will start failing —
that's expected; just update it to assert the fixed behavior.
