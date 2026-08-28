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
    tokens = _tokenize("income*MEAN")

    assert tokens

def test_split_dimensions_two_dimensions():
    dims = _split_dimensions("sex, income*MEAN")

    assert len(dims) == 2

def test_split_dimensions_single_dimension():
    dims = _split_dimensions("N")

    assert len(dims) == 1

def test_parse_single_variable():
    row = parse_table("sex")[0]

    assert row is not None
    assert row.kind == "var"
    assert row.name == "sex"

def test_parse_cross_expression():
    node = parse_table("sex*region")[0]

    assert node.kind == "cross"
    assert len(node.children) == 2
    assert node.children[0].name == "sex"
    assert node.children[1].name == "region"

def test_parse_concat_expression():
    node = parse_table("sex region")[0]

    assert node.kind == "concat"
    assert [child.name for child in node.children] == ["sex", "region"]

def test_parse_group_expression():
    node = parse_table("(sex region)*income")[0]

    assert node.kind == "cross"
    assert node.children[0].kind == "group"
    assert node.children[1].name == "income"

@pytest.mark.parametrize("keyword", ["ALL", "TOTAL"])
def test_parse_total_keyword(keyword):
    node = parse_table(keyword)[0]

    assert node.kind == "all"
    assert node.name == "ALL"

def test_parse_label():
    node = parse_table("sex='Gender'")[0]

    assert node.kind == "var"
    assert node.name == "sex"
    assert node.label == "Gender"

def test_parse_format_spec():
    node = parse_table("MEAN*format=7,1")[0]

    assert node.fmt == "7,1"

def test_parse_custom_denominator():
    node = parse_table("PCTN<gender all>")[0]

    assert node.denom == "gender all"

def test_expand_concat_produces_separate_paths():
    node = parse_table("sex region")[0]

    paths = _expand_node(node)

    assert len(paths) == 2

def test_expand_cross_combines_paths():
    node = parse_table("sex*region")[0]

    paths = _expand_node(node)

    assert len(paths) == 1
    assert [n.name for n in paths[0]] == ["sex", "region"]

def test_expand_with_branch_numbers_top_level_concat():
    node = parse_table("sex region")[0]

    expanded = _expand_node_with_branch(node)

    assert [branch for branch, _ in expanded] == [0, 1]

def test_expand_with_branch_cross_uses_single_branch():
    node = parse_table("sex*region")[0]

    expanded = _expand_node_with_branch(node)

    assert all(branch == 0 for branch, _ in expanded)

@pytest.mark.parametrize(
    "expr",
    [
        "(",
        "sex*",
        "*sex",
        "sex)",
    ],
)
def test_invalid_syntax_raises(expr):
    with pytest.raises(SyntaxError):
        parse_table(expr)

def test_invalid_character_is_rejected():
    with pytest.raises(SyntaxError):
        parse_table("sex @ region")

def test_too_many_dimensions_raises():
    with pytest.raises(ValueError, match="at most 2 dimensions"):
        parse_table("sex,,region")
