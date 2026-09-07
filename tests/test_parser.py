"""
Pytest suite for ssb_flextab.parser (the TABLE-expression parser).

Covers:
  * tokenizing / parsing of representative expressions
      "income*mean"
      "sex, income*MEAN"
      "(sex region)*income"
      "(sex region)*income*format=9,1s"
      "(sex region)*pctsum=''<region>*income*format=9,1s"
  * structural checks via _expand_node / _expand_node_with_branch
  * semantic classification via _classify_path
  * invalid expressions (syntax errors, dimension-count errors,
    unknown tokens, malformed format specs, unmatched parens, etc.)

Run with:  pytest tests/test_parser.py -v
"""

from __future__ import annotations

import pytest

from ssb_flextab.parser import DimNode
from ssb_flextab.parser import _classify_path
from ssb_flextab.parser import _expand_node
from ssb_flextab.parser import _expand_node_with_branch
from ssb_flextab.parser import _split_dimensions
from ssb_flextab.parser import _tokenize
from ssb_flextab.parser import parse_table

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def leaf_names(path: list[DimNode]) -> list[str]:
    """Pull the .name off each node in an expanded leaf path."""
    return [n.name for n in path]


def only_path(node: DimNode) -> list[DimNode]:
    """Expand a node that is expected to produce exactly one leaf path."""
    paths: list[list[DimNode]] = _expand_node(node)
    assert len(paths) == 1
    return paths[0]


# --------------------------------------------------------------------------
# "income*mean"
# --------------------------------------------------------------------------


class TestSimpleCross:
    def test_single_dimension_returned(self) -> None:
        dims = parse_table("income*mean")
        assert len(dims) == 1

    def test_tree_shape_is_cross_of_two_vars(self) -> None:
        (root,) = parse_table("income*mean")
        assert root.kind == "cross"
        assert [c.kind for c in root.children] == ["var", "var"]
        assert [c.name for c in root.children] == ["income", "mean"]

    def test_expands_to_single_leaf_path(self) -> None:
        (root,) = parse_table("income*mean")
        path = only_path(root)
        assert leaf_names(path) == ["income", "mean"]

    def test_classification_recognizes_measure_and_stat(self) -> None:
        (root,) = parse_table("income*mean")
        path = only_path(root)
        spec = _classify_path(path, measure_list=["income"], groupby_list=["sex"])
        assert spec["var"] == "income"
        assert spec["stat"] == "MEAN"
        assert spec["group_keys"] == []
        assert spec["has_all"] is False

    def test_repr_round_trips_readably(self) -> None:
        (root,) = parse_table("income*mean")
        assert repr(root) == "income * mean"


# --------------------------------------------------------------------------
# "sex, income*MEAN"
# --------------------------------------------------------------------------


class TestTwoDimensionsWithComma:
    def test_comma_splits_into_row_and_column(self) -> None:
        row, col = parse_table("sex, income*MEAN")
        assert row.kind == "var"
        assert row.name == "sex"
        assert col.kind == "cross"
        assert [c.name for c in col.children] == ["income", "MEAN"]

    def test_stat_name_case_is_preserved_on_node_but_normalized_on_classify(
        self,
    ) -> None:
        _row, col = parse_table("sex, income*MEAN")
        path = only_path(col)
        # the raw token text keeps its original case...
        assert path[1].name == "MEAN"
        # ...but classification matches case-insensitively against ALL_STATS
        spec = _classify_path(path, measure_list=["income"], groupby_list=["sex"])
        assert spec["stat"] == "MEAN"
        assert spec["stat_label"] == "MEAN"

    def test_row_classifies_as_a_bare_groupby(self) -> None:
        row, _col = parse_table("sex, income*MEAN")
        path = only_path(row)
        spec = _classify_path(path, measure_list=["income"], groupby_list=["sex"])
        assert spec["group_keys"] == [("sex", "sex")]
        assert spec["var"] is None
        assert spec["stat"] is None


# --------------------------------------------------------------------------
# "(sex region)*income"
# --------------------------------------------------------------------------


class TestGroupCrossedWithMeasure:
    def test_tree_shape(self) -> None:
        (root,) = parse_table("(sex region)*income")
        assert root.kind == "cross"
        group, measure = root.children
        assert group.kind == "group"
        assert measure.kind == "var" and measure.name == "income"
        inner = group.children[0]
        assert inner.kind == "concat"
        assert [c.name for c in inner.children] == ["sex", "region"]

    def test_distributes_into_two_leaf_paths(self) -> None:
        (root,) = parse_table("(sex region)*income")
        paths = _expand_node(root)
        assert [leaf_names(p) for p in paths] == [
            ["sex", "income"],
            ["region", "income"],
        ]

    def test_branch_index_is_zero_when_root_is_not_a_top_level_concat(self) -> None:
        # The outermost node here is a "cross" (group * income), not a
        # "concat", so every expanded path should share branch 0 and rely
        # on positional order alone.
        (root,) = parse_table("(sex region)*income")
        branches = [b for b, _p in _expand_node_with_branch(root)]
        assert branches == [0, 0]

    def test_classification_of_each_branch(self) -> None:
        (root,) = parse_table("(sex region)*income")
        paths = _expand_node(root)
        specs = [
            _classify_path(p, measure_list=["income"], groupby_list=["sex", "region"])
            for p in paths
        ]
        assert [s["group_keys"] for s in specs] == [
            [("sex", "sex")],
            [("region", "region")],
        ]
        assert all(s["var"] == "income" for s in specs)


# --------------------------------------------------------------------------
# "(sex region)*income*format=9,1s"
# --------------------------------------------------------------------------


class TestFormatSuffixOnLeaf:
    def test_format_attaches_to_preceding_measure_not_a_new_atom(self) -> None:
        (root,) = parse_table("(sex region)*income*format=9,1s")
        # root is still a 2-child cross: (group) * income -- format=9,1s
        # was consumed as a suffix, not parsed as a third crossed atom.
        assert root.kind == "cross"
        assert len(root.children) == 2

    def test_format_spec_is_recorded_on_the_measure_node(self) -> None:
        (root,) = parse_table("(sex region)*income*format=9,1s")
        paths = _expand_node(root)
        for path in paths:
            income_node = next(n for n in path if n.name == "income")
            assert income_node.fmt == "9,1s"

    def test_classification_surfaces_the_format(self) -> None:
        (root,) = parse_table("(sex region)*income*format=9,1s")
        for path in _expand_node(root):
            spec = _classify_path(
                path, measure_list=["income"], groupby_list=["sex", "region"]
            )
            assert spec["fmt"] == "9,1s"

    def test_format_on_a_group_propagates_to_every_leaf(self) -> None:
        (root,) = parse_table("(mean gmean)*format=7,1")
        paths = _expand_node(root)
        assert {n.fmt for p in paths for n in p if n.kind == "var"} == {"7,1"}

    def test_explicit_leaf_format_wins_over_outer_group_format(self) -> None:
        (root,) = parse_table("(mean*format=7,1 gmean)*format=8,2")
        paths = _expand_node(root)
        fmts = {n.name: n.fmt for p in paths for n in p}
        assert fmts["mean"] == "7,1"
        assert fmts["gmean"] == "8,2"


# --------------------------------------------------------------------------
# "(sex region)*pctsum<region>=''*income*format=9,1s"
# --------------------------------------------------------------------------


class TestDenominatorAndSuppressedLabel:
    def test_as_literally_written_this_is_invalid(self) -> None:
        # Grammar requires the label (name='...') to appear *before* the
        # <denom> clause, e.g. pctsum=''<region>, not pctsum<region>=''.
        # Writing the denom first leaves a bare "=''" that no token can
        # start with, which is a tokenizer error.
        with pytest.raises(SyntaxError):
            parse_table("(sex region)*pctsum<region>=''*income*format=9,1s")

    def test_correct_order_parses_with_denom_and_suppressed_label(self) -> None:
        (root,) = parse_table("(sex region)*pctsum=''<region>*income*format=9,1s")
        paths = _expand_node(root)
        assert len(paths) == 2
        for path in paths:
            pct_node = next(n for n in path if n.name == "pctsum")
            assert pct_node.denom == "region"
            assert pct_node.label == ""  # explicitly suppressed, not "unset"
            income_node = next(n for n in path if n.name == "income")
            assert income_node.fmt == "9,1s"

    def test_denom_without_label_suppression_also_works(self) -> None:
        (root,) = parse_table("pctsum<region>")
        path = only_path(root)
        assert path[0].denom == "region"
        assert path[0].label is None

    def test_classification_surfaces_denom_def(self) -> None:
        (root,) = parse_table("(sex region)*pctsum=''<region>*income*format=9,1s")
        for path in _expand_node(root):
            spec = _classify_path(
                path, measure_list=["income"], groupby_list=["sex", "region"]
            )
            assert spec["denom_def"] == "region"
            assert spec["stat"] == "PCTSUM"
            assert spec["stat_label"] == ""


# --------------------------------------------------------------------------
# Miscellaneous valid syntax: ALL/TOTAL keyword, quoted labels, top-level
# concat ordering.
# --------------------------------------------------------------------------


class TestAllKeywordAndLabels:
    def test_all_keyword_recognized_case_insensitively(self) -> None:
        for kw in ("ALL", "all", "TOTAL", "total"):
            (root,) = parse_table(f"sex {kw}")
            all_node = root.children[1]
            assert all_node.kind == "all"

    def test_all_with_double_quoted_label(self) -> None:
        (root,) = parse_table('sex ALL="Grand Total"')
        all_node = root.children[1]
        assert all_node.label == "Grand Total"

    def test_all_with_single_quoted_label(self) -> None:
        (root,) = parse_table("sex ALL='Grand Total'")
        all_node = root.children[1]
        assert all_node.label == "Grand Total"

    def test_top_level_concat_preserves_branch_order(self) -> None:
        (root,) = parse_table("sex ALL='Subtotal'")
        branches = [b for b, _p in _expand_node_with_branch(root)]
        assert branches == [0, 1]

    def test_renamed_groupby_label_is_tracked_separately_from_original_name(
        self,
    ) -> None:
        (root,) = parse_table("sex='Gender'")
        path = only_path(root)
        spec = _classify_path(path, measure_list=[], groupby_list=["sex"])
        assert spec["group_keys"] == [("sex", "Gender")]
        # path_order keeps the *original* column name as its 3rd element
        assert spec["path_order"] == [("group", "Gender", "sex")]


# --------------------------------------------------------------------------
# _split_dimensions: comma handling incl. the format=W,D decimal comma
# --------------------------------------------------------------------------


class TestSplitDimensions:
    def test_splits_on_top_level_comma(self) -> None:
        assert _split_dimensions("sex, income") == ["sex", " income"]

    def test_does_not_split_comma_inside_parens(self) -> None:
        assert _split_dimensions("(sex, region)*income") == ["(sex, region)*income"]

    def test_does_not_split_decimal_comma_in_format_spec(self) -> None:
        assert _split_dimensions("sex, income*format=9,1s") == [
            "sex",
            " income*format=9,1s",
        ]

    def test_does_not_split_comma_inside_quoted_label(self) -> None:
        assert _split_dimensions("sex='a, b', income") == ["sex='a, b'", " income"]


# --------------------------------------------------------------------------
# Invalid expressions
# --------------------------------------------------------------------------


class TestInvalidExpressions:
    def test_empty_string_yields_no_dimensions(self) -> None:
        # Not an error: no dimensions is a degenerate-but-legal TABLE spec.
        assert parse_table("") == ()

    def test_more_than_two_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most 2 dimensions"):
            parse_table("sex, region, income")

    def test_unclosed_paren_raises(self) -> None:
        with pytest.raises(IndexError):
            parse_table("(sex")

    def test_unopened_paren_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            parse_table("sex)")

    def test_double_star_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            parse_table("sex**income")

    def test_trailing_star_raises(self) -> None:
        with pytest.raises((SyntaxError, IndexError)):
            parse_table("sex*")

    def test_malformed_format_spec_raises_syntax_error(self) -> None:
        # "format=abc" isn't a valid W,D[_s] spec, so "format" is tokenized
        # as a plain NAME and the bare "=" that follows can't start any
        # token.
        with pytest.raises(SyntaxError):
            parse_table("income*format=abc")

    def test_abbreviated_f_equals_is_not_recognized_as_format(self) -> None:
        # Only the literal keyword "format=" is special-cased; "f=9,1s" is
        # not a supported shorthand and fails to tokenize.
        with pytest.raises(SyntaxError):
            parse_table("income*f=9,1s")

    def test_leading_digit_identifier_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            parse_table("1income*mean")

    def test_unterminated_quoted_label_raises(self) -> None:
        with pytest.raises(SyntaxError):
            parse_table("sex='Gender")

    def test_stray_character_raises_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            parse_table("income # mean")

    def test_unknown_token_passes_parsing_but_fails_classification(self) -> None:
        # parse_table has no notion of "known" names, so this parses fine
        # as a plain var...
        (root,) = parse_table("sex*totallynotastat")
        path = only_path(root)
        # ...but classifying it against real measure/groupby/stat lists
        # correctly rejects the unrecognized token.
        with pytest.raises(ValueError, match="not found in measure="):
            _classify_path(path, measure_list=["income"], groupby_list=["sex"])

    @pytest.mark.parametrize(
        "expr",
        [
            "(sex region)*income*format=9,1s)",  # extra trailing paren
            "((sex)*income",  # mismatched nesting
        ],
    )
    def test_various_malformed_expressions_raise(self, expr: str) -> None:
        with pytest.raises((SyntaxError, IndexError)):
            parse_table(expr)

    def test_empty_dimension_between_commas_raises(self) -> None:
        # "sex,,income" splits into three parts ("sex", "", "income"), which
        # trips the >2-dimensions check before the empty middle piece would
        # even get tokenized.
        with pytest.raises(ValueError, match="at most 2 dimensions"):
            parse_table("sex,,income")


# --------------------------------------------------------------------------
# Tokenizer-level sanity checks (independent of the recursive-descent parser)
# --------------------------------------------------------------------------


class TestTokenize:
    def test_simple_cross_tokens(self) -> None:
        tokens = _tokenize("income*mean")
        assert tokens == [
            ("NAME", "income", None),
            ("OP", "*", None),
            ("NAME", "mean", None),
        ]

    def test_labeled_name_token(self) -> None:
        tokens = _tokenize("sex='Gender'")
        assert tokens == [("NAME", "sex", "Gender")]

    def test_denom_token_strips_brackets(self) -> None:
        tokens = _tokenize("pctsum<region>")
        assert ("DENOM", "region", None) in tokens

    def test_format_token_captures_spec(self) -> None:
        tokens = _tokenize("format=9,1s")
        assert tokens == [("FMT", "9,1s", None)]

    def test_redundant_whitespace_is_collapsed(self) -> None:
        tokens = _tokenize("sex   region")
        assert tokens == [
            ("NAME", "sex", None),
            ("SP", " ", None),
            ("NAME", "region", None),
        ]
