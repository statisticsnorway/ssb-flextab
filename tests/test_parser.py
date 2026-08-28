import pytest

from ssb_flextab.parser import (
    DimNode,
    _expand_node,
    _expand_node_with_branch,
    _split_dimensions,
    _tokenize,
    parse_table,
)

def test_tokenize_simple_expression():
    tokens = _tokenize("sex, income*MEAN")

    assert tokens

def test_split_dimensions_two_dimensions():
    dims = _split_dimensions("sex, income*MEAN")

    assert len(dims) == 2

def test_split_dimensions_single_dimension():
    dims = _split_dimensions("N")

    assert len(dims) == 1

def test_parse_single_variable():
    row, col = parse_table("sex")

    assert row is not None
    assert col is None
    assert row.kind == "var"
    assert row.name == "sex"

def test_parse_cross_expression():
    row, col = parse_table("sex*region")

    assert row.kind == "cross"
    assert len(row.children) == 2
    assert row.children[0].name == "sex"
    assert row.children[1].name == "region"

def test_parse_concat_expression():
    row, col = parse_table("sex region")

    assert row.kind == "concat"
    assert [child.name for child in row.children] == ["sex", "region"]

def test_parse_group_expression():
    row, col = parse_table("(sex region)*income")

    assert row.kind == "cross"
    assert row.children[0].kind == "group"

@pytest.mark.parametrize("keyword", ["ALL", "TOTAL"])
def test_parse_total_keyword(keyword):
    row, col = parse_table(keyword)

    assert row.kind == "all"

def test_parse_label():
    row, col = parse_table("sex='Gender'")

    assert row.kind == "var"
    assert row.label == "Gender"

def test_parse_format_spec():
    row, col = parse_table("MEAN*format=7,1")

    assert row.fmt == "7,1"

def test_parse_custom_denominator():
    row, col = parse_table("PCTN<gender all>")

    assert row.denom == "gender all"

def test_expand_concat_produces_separate_paths():
    row, _ = parse_table("sex region")

    paths = _expand_node(row)

    assert len(paths) == 2

def test_expand_cross_combines_paths():
    row, _ = parse_table("sex*region")

    paths = _expand_node(row)

    assert len(paths) == 1
    assert [node.name for node in paths[0]] == ["sex", "region"]

def test_expand_with_branch_numbers_top_level_concat():
    row, _ = parse_table("sex region")

    expanded = _expand_node_with_branch(row)

    assert [branch for branch, _ in expanded] == [0, 1]

def test_expand_cross_combines_paths():
    row, _ = parse_table("sex*region")

    paths = _expand_node(row)

    assert len(paths) == 1
    assert [node.name for node in paths[0]] == ["sex", "region"]

def test_expand_with_branch_numbers_top_level_concat():
    row, _ = parse_table("sex region")

    expanded = _expand_node_with_branch(row)

    assert [branch for branch, _ in expanded] == [0, 1]

def test_expand_with_branch_cross_uses_single_branch():
    row, _ = parse_table("sex*region")

    expanded = _expand_node_with_branch(row)

    assert all(branch == 0 for branch, _ in expanded)

@pytest.mark.parametrize(
    "expr",
    [
        "(",
        "sex*",
        "*sex",
        "sex,,region",
        "sex)",
    ],
)
def test_invalid_syntax_raises(expr):
    with pytest.raises(ValueError):
        parse_table(expr)

def test_invalid_character_is_rejected():
    with pytest.raises(ValueError):
        parse_table("sex @ region")